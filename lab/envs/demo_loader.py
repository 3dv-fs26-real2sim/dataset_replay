"""Loader for the per-frame demo trajectories the residual-RL env tracks.

A *demo* is a processed product of one raw recording, holding everything the
training env needs to drive and score a single manipulation clip:

    obj_trajectory:   (T, 4, 4)   object SE(3) per frame
    wrist_pos:        (T, 3)      EE-wrist position
    wrist_rot_aa:     (T, 3)      EE-wrist rotation (axis-angle)
    hand_qpos:        (T, 17)     OrcaHand joint angles
    hand_joint_names: (17,)       joint names in the qpos column order
    arm_qpos:         (T, 7)      Franka Panda joint angles (Lula-IK retarget)
    frame:            scalar str  "panda_link0" | legacy demo-world

``arm_qpos`` is the deterministic PD baseline for the arm; ``hand_qpos`` for
the fingers; ``obj_trajectory`` is the object reference the reward tracks. The
file is written by :mod:`lab.scripts.make_demo` (which runs the same Lula IK as
the kinematic-replay scripts) — see that script for how a raw H5 + object-pose
becomes one of these.

Pure numpy/torch — no Isaac Sim imports, safe to import any time. The env's
``startup`` event (:func:`lab.envs.mdp.events.attach_demo`) calls
:func:`load_demo` once ``num_envs`` is known and broadcasts to ``(E, T, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Canonical 17-DOF OrcaHand column order shared with the replay scripts.
from utils.constants import HAND_JOINT_NAMES


# ── Quaternion helpers (pure-torch, wxyz convention) ─────────────────────────
def _aa_to_quat_wxyz(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle ``(..., 3)`` → quaternion wxyz ``(..., 4)``."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    half = theta * 0.5
    small = theta < 1e-8
    sin_term = torch.where(small, torch.full_like(half, 0.5), torch.sin(half) / theta.clamp_min(1e-12))
    xyz = aa * sin_term
    w = torch.cos(half)
    return torch.cat([w, xyz], dim=-1)


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``(..., 3, 3)`` → quaternion wxyz ``(..., 4)``.

    Branchless trace + max-diagonal disambiguation.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    s0 = torch.sqrt(torch.clamp(1.0 + trace, min=1e-12)) * 0.5
    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 0.5
    s2 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12)) * 0.5
    s3 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12)) * 0.5
    idx = torch.stack([s0, s1, s2, s3], dim=-1).argmax(dim=-1)
    qw0 = s0;                     qx0 = (m21 - m12) / (4 * s0); qy0 = (m02 - m20) / (4 * s0); qz0 = (m10 - m01) / (4 * s0)
    qw1 = (m21 - m12) / (4 * s1); qx1 = s1;                     qy1 = (m01 + m10) / (4 * s1); qz1 = (m02 + m20) / (4 * s1)
    qw2 = (m02 - m20) / (4 * s2); qx2 = (m01 + m10) / (4 * s2); qy2 = s2;                     qz2 = (m12 + m21) / (4 * s2)
    qw3 = (m10 - m01) / (4 * s3); qx3 = (m02 + m20) / (4 * s3); qy3 = (m12 + m21) / (4 * s3); qz3 = s3
    qw = torch.stack([qw0, qw1, qw2, qw3], dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    qx = torch.stack([qx0, qx1, qx2, qx3], dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    qy = torch.stack([qy0, qy1, qy2, qy3], dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    qz = torch.stack([qz0, qz1, qz2, qz3], dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return torch.stack([qw, qx, qy, qz], dim=-1)


@dataclass
class Demo:
    """Per-frame demo broadcast to every env (leading dim ``E = num_envs``)."""

    obj_pos:        torch.Tensor   # (E, T, 3)   world-frame object position reference
    obj_quat:       torch.Tensor   # (E, T, 4)   wxyz
    obj_init_pos:   torch.Tensor   # (E, 3)      spawn pose (frame-0 or SAM3-refined)
    obj_init_quat:  torch.Tensor   # (E, 4)      wxyz
    wrist_pos:      torch.Tensor   # (E, T, 3)   recorded EE-wrist position
    wrist_quat:     torch.Tensor   # (E, T, 4)   wxyz (tool convention)
    hand_qpos:      torch.Tensor   # (E, T, 17)  demo column order
    arm_qpos:       torch.Tensor   # (E, T, 7)   Lula-IK retarget (PD baseline)
    hand_joint_names: list[str]    # 17 names, demo column order
    seq_len:        torch.Tensor   # (E,) long
    metadata:       dict[str, Any] = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return self.obj_pos.shape[0]

    @property
    def horizon(self) -> int:
        return self.obj_pos.shape[1]

    def clamp_t(self, t: torch.Tensor) -> torch.Tensor:
        """Clamp a per-env frame index into ``[0, seq_len - 1]``."""
        return torch.minimum(t, self.seq_len - 1).clamp_min(0)


def peek_demo_length(npz_path: str | Path) -> int:
    """Return the demo's frame count ``T`` without loading the full arrays.

    Used by the train/play scripts to size ``episode_length_s`` to the clip
    (``T · demo_dt + buffer``) — the env cfg can't know ``T`` at construction
    because the demo only loads at the startup event.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        return int(d["arm_qpos"].shape[0])


def load_demo(
    npz_path: str | Path,
    *,
    num_envs: int,
    device: torch.device | str,
    max_seq_len: int | None = None,
    world_z_offset: float = 0.0,
    wrist_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    obj_world_offset: tuple[float, float, float] | None = None,
    use_sam3_init: bool = False,
) -> Demo:
    """Load a demo npz and broadcast it to ``num_envs``.

    Object/wrist placement depends on the npz ``frame`` marker:

    * ``frame == "panda_link0"`` — trajectories are in the robot-base frame, so
      the full ``obj_world_offset`` / ``wrist_world_offset`` (xyz) is added.
      Pass the env's ``mount_xyz`` so the rig moves together.
    * legacy demo-world (no marker, table top at z=0) — only ``world_z_offset``
      lifts z to the env's table height; xy are absolute.

    Joint-space fields (``arm_qpos`` / ``hand_qpos``) are frame-independent.
    With ``use_sam3_init`` and a ``sam3_init_pose`` in the npz, the spawn pose
    is the SAM3-refined pose; the tracking reference is unchanged either way.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    frame = str(data["frame"]) if "frame" in data.files else "demo_world"

    obj_traj = torch.as_tensor(data["obj_trajectory"], dtype=torch.float32, device=device)
    T = obj_traj.shape[0] if max_seq_len is None else min(obj_traj.shape[0], max_seq_len)
    obj_traj = obj_traj[:T]

    def _to_world(pos: torch.Tensor) -> torch.Tensor:
        """Apply the frame-appropriate world offset to a position tensor."""
        if frame == "panda_link0":
            off = obj_world_offset if obj_world_offset is not None else (0.0, 0.0, 0.0)
            return pos + torch.tensor(off, dtype=pos.dtype, device=pos.device)
        out = pos.clone()
        out[..., 2] += world_z_offset
        return out

    obj_pos = _to_world(obj_traj[..., :3, 3]).contiguous()
    obj_quat = _rotmat_to_quat_wxyz(obj_traj[..., :3, :3])

    if use_sam3_init and "sam3_init_pose" in data.files:
        sam3 = torch.as_tensor(data["sam3_init_pose"], dtype=torch.float32, device=device)
        init_pos = _to_world(sam3[:3, 3])
        init_quat = _rotmat_to_quat_wxyz(sam3[:3, :3])
        init_src = "sam3_init_pose"
    else:
        init_pos = obj_pos[0].clone()
        init_quat = obj_quat[0].clone()
        init_src = "obj_trajectory[0]"

    wrist_pos = torch.as_tensor(data["wrist_pos"][:T], dtype=torch.float32, device=device).clone()
    wrist_pos += torch.tensor(wrist_world_offset, dtype=wrist_pos.dtype, device=device)
    wrist_quat = _aa_to_quat_wxyz(
        torch.as_tensor(data["wrist_rot_aa"][:T], dtype=torch.float32, device=device))

    hand_qpos = torch.as_tensor(data["hand_qpos"][:T], dtype=torch.float32, device=device)
    arm_qpos = torch.as_tensor(data["arm_qpos"][:T], dtype=torch.float32, device=device)
    hand_joint_names = [str(n) for n in data["hand_joint_names"]]

    # Sanity check the demo's hand column order against the canonical one. The
    # action/reward terms remap by name, so a permutation is fine, but a missing
    # joint would break the remap — fail loud here rather than mid-episode.
    if set(hand_joint_names) != set(HAND_JOINT_NAMES):
        raise ValueError(
            f"{npz_path.name}: hand_joint_names do not match utils.constants."
            f"HAND_JOINT_NAMES.\n  demo:      {hand_joint_names}\n"
            f"  canonical: {list(HAND_JOINT_NAMES)}"
        )

    def _bcast(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(num_envs, *t.shape).contiguous()

    return Demo(
        obj_pos=_bcast(obj_pos),
        obj_quat=_bcast(obj_quat),
        obj_init_pos=_bcast(init_pos),
        obj_init_quat=_bcast(init_quat),
        wrist_pos=_bcast(wrist_pos),
        wrist_quat=_bcast(wrist_quat),
        hand_qpos=_bcast(hand_qpos),
        arm_qpos=_bcast(arm_qpos),
        hand_joint_names=hand_joint_names,
        seq_len=torch.full((num_envs,), T, device=device, dtype=torch.long),
        metadata={
            "npz_path": str(npz_path),
            "T": T,
            "frame": frame,
            "dataset": str(data["dataset"]) if "dataset" in data.files else "?",
            "arm_qpos_source": str(data["arm_qpos_source"]) if "arm_qpos_source" in data.files else "?",
            "obj_init_source": init_src,
        },
    )
