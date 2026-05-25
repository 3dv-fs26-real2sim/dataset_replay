"""Egoverse-specific scene config: Aria camera, no walls, no tag.

Pure-Python, no Isaac Sim imports — safe to import before SimulationApp.

The Aria pose is computed at runtime as
``T_world_cam = T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT`` (with optional
SAM-table-mask refinement via :mod:`utils.calibrate_table`); the right-base
extrinsic carries verbatim from the main-branch capture rig because the
rig sits in the same place relative to the robot across sessions.

Geometry sketch (top-down)
--------------------------
::

                              +X (robot faces this direction)
                                ^
                                |
              ==================================
              |                                |
              |                                |
              |                  panda_link0   |      Combined surface:
              |  LEFT TABLE      RIGHT TABLE   |      X in [-0.50, +0.50]  (1.00 m)
              |  70 x 100 cm     70 x 100 cm   |      Y in [-0.70, +0.70]  (1.40 m)
       +Y <---|                                |---> -Y      top at Z = 0.75 m
              |                                |
              |                                |
              |                                |
              ==================================
                                |
                                v
                              -X  (Aria rig sits here, looking +X)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BaseCameraConfig, BaseSceneConfig, RobotMountConfig
from .constants import ARIA_INTRINSICS, ARIA_EXTRINSICS_RIGHT


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AriaCameraConfig(BaseCameraConfig):
    """Aria glasses camera, posed relative to the right-arm base.

    Unlike maple's world-fixed OakD, the egoverse Aria pose is computed at
    runtime as

        T_world_cam = T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT

    using constants in :mod:`utils.constants`. A per-session SAM-mask-driven
    refinement (see :func:`utils.calibrate_table.refine_aria_extrinsic`)
    absorbs head-pose drift between sessions; the replay script auto-calls
    it when a mask is present at ``data/egoverse/desk/<h5_stem>_desk.npz``.
    """
    name:   str  = "aria_rgb_cam"
    width:  int  = ARIA_INTRINSICS["width"]
    height: int  = ARIA_INTRINSICS["height"]

    intrinsics: dict = field(default_factory=lambda: {
        "fx": ARIA_INTRINSICS["fx"],
        "fy": ARIA_INTRINSICS["fy"],
        "cx": ARIA_INTRINSICS["cx"],
        "cy": ARIA_INTRINSICS["cy"],
    })

    # OpenCV (k1, k2, p1, p2, k3). The Aria pipeline rectifies upstream, so
    # the simulated pinhole render uses zero distortion.
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def t_base_cam(self) -> np.ndarray:
        """4×4 column-vector transform: camera in the right-arm-base frame."""
        return np.asarray(ARIA_EXTRINSICS_RIGHT, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EgoverseSceneConfig(BaseSceneConfig):
    """Egoverse rig: Aria glasses, open desk (no walls / no AprilTag).

    Mount xyz `(-0.246, -0.350, 0.75)` was chosen by visual overlay comparison.
    """
    dataset: str = "egoverse"
    camera:  AriaCameraConfig = field(default_factory=AriaCameraConfig)
    robot:   RobotMountConfig = field(
        default_factory=lambda: RobotMountConfig(mount_xyz=(-0.246, -0.350, 0.75)))

    def viewport_size(self) -> tuple[int, int]:
        # Aria intrinsics are 640×480 (4:3). Use a 2× viewport so the
        # rendered window is not microscopic on a 4K monitor.
        return (ARIA_INTRINSICS["width"]  * 2,
                ARIA_INTRINSICS["height"] * 2)


