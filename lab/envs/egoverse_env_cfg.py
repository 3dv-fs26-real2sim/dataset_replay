"""EgoVerse residual-RL teleop env (Aria duck-grasp demos).

EgoVerse recordings run at 50 Hz; the manipulated object is the duck (the base
default), so this variant only pins the dataset-specific knobs and leaves the
shared scene / managers from :class:`TeleopBaseEnvCfg` untouched.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from utils.config import select_config

from . import mdp
from .objects import bowl_spawn_cfg, object_cfg
from .teleop_base_env_cfg import TeleopBaseEnvCfg

# Robot-base (panda_link0) world pose from dataset_replay's EgoVerse rig config
# — the measured/overlay-calibrated base, shared with kinematic_replay_egoverse.
_MOUNT = tuple(select_config("egoverse").robot.mount_xyz)


def attach_bowl(cfg: "EgoverseTeleopEnvCfg", pose_npz: str, frame: str | None = None,
                static: bool = False) -> None:
    """Spawn the purple bowl in the EgoVerse scene at a pose from ``pose_npz``.

    ``pose_npz`` holds the bowl SE(3) pose — a single ``(4,4)`` or a ``(N,4,4)``
    trajectory (frame 0 is used). It is composed into world like the object:

    * ``aria_camera`` (default): ``T_world_bowl = mount · ARIA_EXTRINSICS_RIGHT · T_cam_bowl``
    * ``panda_link0``: ``T_world_bowl = mount · T_link0_bowl``

    ``static=True`` (default) → a fixed kinematic container (immovable, like the
    maple props); ``static=False`` → a dynamic rigid body re-settled each episode.
    Callable post-construction (train/play ``--bowl-pose``).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    from utils.constants import ARIA_EXTRINSICS_RIGHT

    data = np.load(pose_npz, allow_pickle=True)
    T = np.asarray(data[data.files[0]], dtype=float)
    T = T[0] if T.ndim == 3 else T                       # (4,4)
    frame = frame or "aria_camera"
    T_link0 = ARIA_EXTRINSICS_RIGHT @ T if frame == "aria_camera" else T
    pos = tuple(float(T_link0[i, 3] + cfg.mount_xyz[i]) for i in range(3))
    qx, qy, qz, qw = Rotation.from_matrix(T_link0[:3, :3]).as_quat()

    cfg.scene.bowl = object_cfg("{ENV_REGEX_NS}/Bowl", bowl_spawn_cfg(static=static), pos, (qw, qx, qy, qz))
    if not static:
        # Dynamic bowl: re-settle at its spawn pose each episode (free rigid body).
        cfg.events.reset_bowl = EventTerm(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"asset_cfg": SceneEntityCfg("bowl"), "pose_range": {}, "velocity_range": {}},
        )
    print(f"[egoverse] {'static' if static else 'dynamic'} bowl spawned at world "
          f"{tuple(round(p, 3) for p in pos)} (frame={frame})", flush=True)


@configclass
class EgoverseTeleopEnvCfg(TeleopBaseEnvCfg):
    """Franka + OrcaHand residual-RL on an EgoVerse (Aria) duck-grasp demo."""

    # Robot base = the EgoVerse rig's measured panda_link0 pose. The grasp is
    # mount-invariant (object/wrist/arm baseline all anchor here and move
    # together); this only places the rig over the fixed table.
    mount_xyz: tuple[float, float, float] = _MOUNT
    # 50 Hz Aria recordings → one demo frame per control step.
    demo_dt: float = 0.02
    # episode_length_s is set per-demo by the train/play scripts (clip length +
    # 2 s buffer); the default here is a safe fallback if not overridden.
    episode_length_s: float = 26.0
