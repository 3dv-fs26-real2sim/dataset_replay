"""Scene configuration for the egoverse (single-arm, Aria, no walls/AprilTag) branch.

Pure-Python, no Isaac Sim imports — safe to import before SimulationApp is
created. The SceneConfig dataclass is the *single source of truth* for every
scene parameter the procedural builder, robot setup, and camera setup read.

Compared to ``maple`` this branch:
  * drops the U-shape walls (no ``WallsConfig``);
  * drops the AprilTag plane (no ``AprilTagConfig`` and no AprilTag-based PnP);
  * swaps the world-fixed OakD camera for an Aria glasses camera whose pose
    is computed at runtime as ``T_world_base @ ARIA_EXTRINSICS_RIGHT`` from
    :mod:`utils.constants`, optionally refined on the fly from a SAM table
    mask via ``utils.calibrate_table.refine_aria_extrinsic`` (the replay
    script auto-finds the mask under ``data/egoverse/desk/<stem>_desk.npz``).

Coordinate convention
---------------------
* World frame: Z-up, 1 m units.
* Combined table surface centred at world origin in XY, top at ``Z = top_z``.
* ``+X`` points away from the operator/camera (toward where the back wall
  used to be in maple); the robot's ``panda_link0`` faces ``+X``.
* ``-X`` points toward the open side, where the Aria wearer sits.
* ``+Y`` points toward the **left** table cell; ``-Y`` toward the **right**.

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
                              -X  (Aria wearer sits here, looking +X)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import ARIA_EXTRINSICS_RIGHT, ARIA_INTRINSICS


# ── Path anchors ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # dataset_replay/
ASSETS = PROJECT_ROOT / "assets"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TableConfig:
    """Two tables placed side-by-side along the Y axis.

    Default: each cell is 1.00 m (X) × 0.70 m (Y), so the combined surface
    is 1.00 × 1.40 m centred at the world origin.
    """
    single_size_xy: tuple[float, float] = (1.00, 0.70)   # (Lx, Ly) of ONE table
    n_tables:       int = 2                              # tiled along +Y/-Y
    top_thickness:  float = 0.05                         # m
    top_z:          float = 0.75                         # world Z of top surface
    centre_xy:      tuple[float, float] = (0.0, 0.0)    # of the combined surface
    texture_path:   Path = ASSETS / "textures" / "wood.jpg"
    uv_repeat:      tuple[float, float] = (2.0, 3.0)    # texture tiles per surface

    # ------------------------------------------------------------- derived
    @property
    def combined_size_xy(self) -> tuple[float, float]:
        return (self.single_size_xy[0],
                self.single_size_xy[1] * self.n_tables)

    @property
    def x_extent(self) -> tuple[float, float]:
        Lx = self.combined_size_xy[0]
        return (self.centre_xy[0] - Lx / 2, self.centre_xy[0] + Lx / 2)

    @property
    def y_extent(self) -> tuple[float, float]:
        Ly = self.combined_size_xy[1]
        return (self.centre_xy[1] - Ly / 2, self.centre_xy[1] + Ly / 2)

    @property
    def right_table_centre_y(self) -> float:
        """Y-centre of the right (-Y) table cell."""
        return self.centre_xy[1] - self.single_size_xy[1] / 2


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotMountConfig:
    """World pose of the wrapper Xform that references the USD's /Root.

    Defaults place ``panda_link0`` at ``(-0.246, -0.350, 0.75)`` — the
    maple-branch value, chosen after a 5-variant sweep (see
    ``3dv/experiments/table_align_egoverse/run_mount_sweep.sh``) gave it
    the best sim-vs-real Aria overlay alignment on the egoverse
    ``20250804_104715`` session.

    Origin of the number: measured "back of robot puck at world x = -0.40"
    plus the 154 mm link0-mesh extent (the Panda's ``panda_link0`` visual
    mesh sticks 154 mm behind the kinematic origin, measured from
    ``pandaorca_description/meshes/franka/fer/visual/link0.dae``).
    Main's baked-USD value (-0.262, -0.386, 1.0) was an approximation
    that ran ~36 mm off in Y and ~16 mm off in X in overlay.

    Relative to the parametric table:
      * X: 0.254 m in from the back edge (x=+0.5).
      * Y: at the geometric centre of the right cell
        (Y = -table.single_size_xy[1] / 2 = -0.35).

    NOTE: ``panda_link0`` has a non-trivial local translate inside the
    orcav1b USD ``(-0.00761, -0.00027, -0.47602)``. The wrapper transform
    must subtract this so panda_link0's *origin* lands at ``mount_xyz``.
    Both the mount xyz and the USD-internal offset are exposed here so
    the math is fully inspectable.
    """
    mount_xyz: tuple[float, float, float] = (-0.246, -0.350, 0.75)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda_link0_local_translate: tuple[float, float, float] = (
        -0.007610592991113663,
        -0.00026992621133103967,
        -0.4760153889656067,
    )


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """Aria glasses camera, posed relative to the right-arm base.

    Unlike the ``maple`` OakD setup (world-fixed, loaded from an AprilTag-PnP
    NPZ), the egoverse Aria pose is computed at runtime as

        T_world_cam = T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT

    using the constants in :mod:`utils.constants`. The wearer sits in the
    same place relative to the robot across sessions, so the right-base
    extrinsic carries verbatim from the main-branch capture rig — only the
    absolute table-top Z (0.75 vs main's 1.0) differs, and that's absorbed
    by the world composition.

    A per-session refinement (desk-based, SAM-mask driven) absorbs the
    head-pose drift between sessions — see
    ``utils.calibrate_table.refine_aria_extrinsic``. The replay scripts
    auto-call it when a mask is present at
    ``data/egoverse/desk/<h5_stem>_desk.npz``.
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

    # The base-relative Aria pose. Stored as a method on the dataclass so
    # changing it (e.g., re-measuring) doesn't require touching every call
    # site. ``T_base_cam`` is what gets left-multiplied by ``T_world_base``.
    def t_base_cam(self) -> np.ndarray:
        """4×4 column-vector transform: camera in the right-arm-base frame."""
        return np.asarray(ARIA_EXTRINSICS_RIGHT, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SceneConfig:
    """Top-level config aggregating all scene parameters."""
    table:  TableConfig      = field(default_factory=TableConfig)
    robot:  RobotMountConfig = field(default_factory=RobotMountConfig)
    camera: CameraConfig     = field(default_factory=CameraConfig)

    robot_asset_path: Path = ASSETS / "orcav1b_franka_vmnt_v10_flattened.usd"
    urdf_path:        Path = ASSETS / "urdf" / "panda_arm.urdf"
    lula_descriptor:  Path = ASSETS / "lula" / "panda_arm_descriptor.yaml"

    # ------------------------------------------------------------------ helpers
    def default_mount_xyz(self) -> tuple[float, float, float]:
        """Recompute the default mount xyz from current table dims.

        Useful when the user changes ``table.single_size_xy`` or ``n_tables``
        and wants to reset ``robot.mount_xyz`` to the canonical "centre of
        right table, 10 cm from back edge" position. Not auto-called — kept
        explicit so the user sees the dependency.
        """
        return (
            self.table.x_extent[0] + 0.10,           # 10 cm from back edge
            self.table.right_table_centre_y,         # centre of right table
            self.table.top_z,
        )
