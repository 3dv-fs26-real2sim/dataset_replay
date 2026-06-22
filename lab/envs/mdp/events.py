"""Event terms: load the demo once at startup, reset to its frame 0 each episode.

* :func:`attach_demo` (startup) — load the demo npz once, stash on ``env.demo``;
  the action / reward / observation terms read from it.
* :func:`reset_robot_to_demo_frame_0` (reset) — snap arm + hand joint state to
  the recorded frame-0 qpos.
* :func:`reset_object_to_demo_frame_0` (reset) — write the object root pose to
  frame 0 of the trajectory, with optional xy/yaw randomization.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ..demo_loader import Demo, load_demo

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── demo attach (startup) ─────────────────────────────────────────────────────
def attach_demo(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | None,
    *,
    npz_path: str,
    demo_dt: float = 0.02,
    max_seq_len: int | None = None,
    world_z_offset: float = 0.0,
    wrist_world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    obj_world_offset: tuple[float, float, float] | None = None,
    use_sam3_init: bool = False,
) -> None:
    """Startup-only: load ``npz_path`` and stash it on ``env.demo``.

    ``demo_dt`` is the demo's sample period (1/fps): 0.02 for 50 Hz EgoVerse,
    0.10 for 10 Hz MAPLE. Stored on the env so the rate-aware ``demo_index``
    plays the reference back at real-time speed.
    """
    if getattr(env, "demo", None) is not None:
        return
    env.demo_dt = float(demo_dt)  # type: ignore[attr-defined]
    demo = load_demo(
        npz_path,
        num_envs=env.num_envs,
        device=env.device,
        max_seq_len=max_seq_len,
        world_z_offset=world_z_offset,
        wrist_world_offset=wrist_world_offset,
        obj_world_offset=obj_world_offset,
        use_sam3_init=use_sam3_init,
    )
    env.demo = demo  # type: ignore[attr-defined]
    print(f"[demo] {demo.metadata['dataset']} | frame={demo.metadata['frame']} | "
          f"T={demo.metadata['T']} | arm_qpos={demo.metadata['arm_qpos_source']} | "
          f"obj_init={demo.metadata['obj_init_source']}", flush=True)


def _resolve_env_ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


# ── quaternion helpers (wxyz) ─────────────────────────────────────────────────
def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    w, z = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(w)
    return torch.stack([w, zero, zero, z], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ── robot reset ───────────────────────────────────────────────────────────────
def reset_robot_to_demo_frame_0(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | None,
    *,
    asset_cfg: SceneEntityCfg,
    arm_joint_names_expr: str = "panda_joint.*",
    hand_joint_names_expr: str = "right_.*",
) -> None:
    """Snap arm + hand joint positions (and zero velocities) to demo frame 0."""
    demo: Demo | None = getattr(env, "demo", None)
    if demo is None:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids)

    arm_ids, _ = asset.find_joints(arm_joint_names_expr, preserve_order=True)
    hand_ids, hand_names = asset.find_joints(hand_joint_names_expr, preserve_order=False)

    arm_qpos = demo.arm_qpos[ids, 0]
    hand_qpos = demo.hand_qpos[ids, 0]

    art_hand_index = {name: i for i, name in enumerate(hand_names)}
    perm = torch.tensor([art_hand_index[n] for n in demo.hand_joint_names], dtype=torch.long, device=env.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=env.device)
    hand_qpos_art = hand_qpos[:, inv]

    joint_ids = arm_ids + hand_ids
    qpos = torch.cat([arm_qpos, hand_qpos_art], dim=-1)
    asset.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), joint_ids=joint_ids, env_ids=ids)


# ── object reset ──────────────────────────────────────────────────────────────
def reset_object_to_demo_frame_0(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | None,
    *,
    object_cfg: SceneEntityCfg,
    position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_noise_rad: float = 0.0,
    extra_z_clearance: float = 0.0,
) -> None:
    """Reset the object to demo frame 0 (+ optional xy/yaw noise + z clearance).

    ``demo.obj_init_pos`` is world-anchored without the per-env origin, so the
    env origin is added before writing the absolute world pose (otherwise every
    env's object piles at env 0). ``extra_z_clearance`` lifts the spawn to avoid
    interpenetrating the table — keep it 0 to spawn exactly where the demo rests
    (the open-loop baseline only grasps from the resting pose).
    """
    demo: Demo | None = getattr(env, "demo", None)
    if demo is None:
        return
    obj: RigidObject = env.scene[object_cfg.name]
    ids = _resolve_env_ids(env, env_ids)

    pos = demo.obj_init_pos[ids].clone()
    quat = demo.obj_init_quat[ids].clone()

    if extra_z_clearance:
        pos[:, 2] += extra_z_clearance

    B = pos.shape[0]
    noise_bound = torch.tensor(position_noise, device=env.device, dtype=pos.dtype)
    if noise_bound.abs().sum() > 0:
        pos = pos + (torch.rand(B, 3, device=env.device, dtype=pos.dtype) * 2 - 1) * noise_bound
    if yaw_noise_rad > 0:
        dyaw = (torch.rand(B, device=env.device, dtype=pos.dtype) * 2 - 1) * yaw_noise_rad
        quat = _quat_mul_wxyz(_yaw_to_quat_wxyz(dyaw), quat)

    pos = pos + env.scene.env_origins[ids]
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(B, 6, device=env.device), env_ids=ids)
