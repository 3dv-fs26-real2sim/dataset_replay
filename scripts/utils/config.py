"""Scene configuration for the minimal single-arm OAK-D branch.

Pure-Python, no Isaac Sim imports — safe to import before SimulationApp is
created. The SceneConfig dataclass is the *single source of truth* for every
scene parameter the procedural builder, robot setup, camera setup, and
calibration scripts read.

Coordinate convention
---------------------
* World frame: Z-up, 1 m units.
* Combined table surface centred at world origin in XY, top at ``Z = top_z``.
* ``+X`` points toward the operator / camera (the open side of the U-walls).
* ``+Y`` points toward the **left** table; ``-Y`` toward the **right** table.
* Robot's ``panda_link0`` faces ``+X`` by default (identity yaw).

Geometry sketch (top-down)
--------------------------
::

                              +X (toward operator / camera)
                                ^
                                |
          /==================================\\
          ||                                ||
          ||  LEFT TABLE      RIGHT TABLE    ||      Combined surface:
          ||  70 x 100 cm     70 x 100 cm    ||      X in [-0.50, +0.50]  (1.00 m)
          ||                                ||      Y in [-0.70, +0.70]  (1.40 m)
          ||                       AprilTag ||      top at  Z = 0.75 m
          ||                                ||
          ||                  panda_link0   ||
          ||                                ||
   +Y <---||                                ||---> -Y      (back wall  -- at -X)
          ||                                ||             (left wall  -- at +Y)
          \\==================================/             (right wall -- at -Y)
                                |                          (open       -- at +X)
                                v
                              -X (back, far from operator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
class WallsConfig:
    """U-shape opening toward +X. Walls span the full combined-table edge.

    All heights default to 1.0 m to block background; tweak per-wall as needed.
    """
    back_height:  float = 1.0   # wall at -X edge (full Y span)
    left_height:  float = 1.0   # wall at +Y edge (full X span)
    right_height: float = 1.0   # wall at -Y edge (full X span)
    thickness:    float = 0.02
    texture_path: Path = ASSETS / "textures" / "wood.jpg"
    uv_repeat:    tuple[float, float] = (2.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AprilTagConfig:
    """tag36h11 id=5, 7.5 cm edge, top-right corner of the combined table.

    Offsets are measured *inward* from the corner: horizontal in +Y from the
    right (-Y) edge, vertical in +X from the back (-X) edge. The 1 mm Z lift
    avoids z-fighting with the table-top mesh.
    """
    family:    str = "tag36h11"
    tag_id:    int = 5                                  # tag36_11_00005
    edge_size: float = 0.075                            # m (black square edge)
    image_path: Path = ASSETS / "textures" / "apriltag_5.png"
    horizontal_offset: float = 0.165                    # m, +Y from right edge
    vertical_offset:   float = 0.135                    # m, +X from back edge
    z_offset_above_table: float = 0.001                 # m


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotMountConfig:
    """World pose of the wrapper Xform that references the USD's /Root.

    Defaults place ``panda_link0`` at:

      * laterally: centre of the RIGHT table (Y = -table.single_size_xy[1] / 2)
      * axially:   10 cm forward (+X) of the back edge (X = x_min + 0.10)
      * vertically: on the table top (Z = top_z)

    yielding ``(-0.40, -0.35, 0.75)`` with the default table.

    NOTE: ``panda_link0`` has a non-trivial local translate inside the USD
    ``(-0.00761, -0.00027, -0.47602)``. The wrapper transform must subtract
    this so panda_link0's *origin* lands at ``mount_xyz``. Both the mount xyz
    and the USD-internal offset are exposed here so the math is fully
    inspectable.
    """
    mount_xyz: tuple[float, float, float] = (-0.40, -0.35, 0.75)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda_link0_local_translate: tuple[float, float, float] = (
        -0.007610592991113663,
        -0.00026992621133103967,
        -0.4760153889656067,
    )


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """OAK-D Pro AF mounted above the open (+X) edge of the table, fixed in world.

    Intrinsics are a plugin slot (Q6 in NEW_BRANCH_PLAN.md) — zero defaults
    will trigger an explicit error in setup_camera until the factory
    calibration is filled in. Distortion coeffs are exposed for the PnP step
    in calibrate_extrinsic.py — they don't affect the simulated pinhole render.
    """
    name:   str = "oakd_front_view"
    width:  int = 1280
    height: int = 720

    # Q6: zero defaults trigger an explicit error in setup_camera.
    intrinsics: dict = field(default_factory=lambda: {
        "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,
    })
    # OpenCV (k1, k2, p1, p2, k3). Zero distortion is the safe default for the
    # simulated render; the calibration script accepts the real values via CLI.
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    extrinsic_path: Path = ASSETS / "calibration" / "oakd_extrinsic.npz"

    # Nominal pose (used as PnP initial guess and as fallback only when no
    # extrinsic file exists). Position above the +X edge of the combined table,
    # 0.6 m above the table top, looking at the table-top centre.
    nominal_position: tuple[float, float, float] = (0.50, 0.0, 1.35)
    nominal_lookat:   tuple[float, float, float] = (0.0, 0.0, 0.75)
    nominal_up:       tuple[float, float, float] = (0.0, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SceneConfig:
    """Top-level config aggregating all scene parameters."""
    table:    TableConfig      = field(default_factory=TableConfig)
    walls:    WallsConfig      = field(default_factory=WallsConfig)
    apriltag: AprilTagConfig   = field(default_factory=AprilTagConfig)
    robot:    RobotMountConfig = field(default_factory=RobotMountConfig)
    camera:   CameraConfig     = field(default_factory=CameraConfig)

    robot_asset_path: Path = ASSETS / "orcav1b_franka_vmnt_v10_flattened.usd"
    urdf_path:        Path = ASSETS / "urdf" / "panda_arm.urdf"
    lula_descriptor:  Path = ASSETS / "lula" / "panda_arm_descriptor.yaml"

    # ------------------------------------------------------------------ helpers
    def apriltag_world_pose(self) -> np.ndarray:
        """Resolve AprilTag centre to a 4×4 world-frame pose (identity rotation)."""
        x = self.table.x_extent[0] + self.apriltag.vertical_offset
        y = self.table.y_extent[0] + self.apriltag.horizontal_offset
        z = self.table.top_z + self.apriltag.z_offset_above_table
        T = np.eye(4)
        T[:3, 3] = (x, y, z)
        return T

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
