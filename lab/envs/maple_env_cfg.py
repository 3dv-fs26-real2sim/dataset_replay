"""MAPLE residual-RL teleop env (OAK-D pan-manipulation demos).

MAPLE recordings run at 10 Hz and the manipulated object is the textured pan.
Optional static scene props (box / carpet / heater) can be spawned as fixed
kinematic obstacles at their measured poses from a props npz
(``maple_props_*.npz``, one (4,4) panda_link0-frame pose per prop).

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from utils.config import select_config

from .objects import maple_prop_cfg, pan_spawn_cfg
from .teleop_base_env_cfg import TeleopBaseEnvCfg

_PROP_NAMES = ("box", "carpet", "heater")
# Robot-base (panda_link0) world pose from dataset_replay's MAPLE rig config.
_MOUNT = tuple(select_config("maple").robot.mount_xyz)


def attach_maple_props(cfg: "MapleTeleopEnvCfg", npz_path: str) -> list[str]:
    """Spawn the MAPLE static props (box/carpet/heater) from ``npz_path``.

    Each prop is a fixed kinematic collider at its measured panda_link0-frame
    pose. Callable post-construction (the train/play scripts use it for the
    ``--maple-props`` flag, since the cfg's ``__post_init__`` already ran).
    """
    import numpy as np

    cfg.maple_props_npz = npz_path
    d = np.load(npz_path, allow_pickle=True)
    spawned = []
    for name in _PROP_NAMES:
        if name in d.files:
            setattr(cfg.scene, f"prop_{name}", maple_prop_cfg(name, d[name], cfg.mount_xyz))
            spawned.append(name)
    print(f"[maple] static props spawned (kinematic): {spawned}", flush=True)
    return spawned


@configclass
class MapleTeleopEnvCfg(TeleopBaseEnvCfg):
    """Franka + OrcaHand residual-RL on a MAPLE (OAK-D) pan demo."""

    # Robot base = the MAPLE rig's measured panda_link0 pose (mount-invariant grasp).
    mount_xyz: tuple[float, float, float] = _MOUNT
    # 10 Hz OAK-D recordings → hold each demo frame for 5 control steps.
    demo_dt: float = 0.10
    # episode_length_s is set per-demo by the train/play scripts (clip length +
    # 2 s buffer); the default here is a safe fallback if not overridden.
    episode_length_s: float = 14.0
    # Optional: path to a maple_props_*.npz to spawn static obstacles. None = off.
    maple_props_npz: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # Swap the duck for the textured pan.
        self.scene.object.spawn = pan_spawn_cfg()
        # Spawn props if set programmatically before construction (CLI uses the
        # standalone attach_maple_props after construction instead).
        if self.maple_props_npz:
            attach_maple_props(self, self.maple_props_npz)
