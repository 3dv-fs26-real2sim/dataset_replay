"""Base scene configuration shared by both Maple and Egoverse datasets.

Pure-Python — safe to import before SimulationApp is created.

This module defines only the *base* dataclasses (`TableConfig`,
`RobotMountConfig`, `BaseCameraConfig`, `BaseSceneConfig`) and a
`select_config()` dispatcher. Concrete configs live in:

  * ``utils.config_maple``     — OakD camera, walls, AprilTag
  * ``utils.config_egoverse``  — Aria camera, open desk, SAM table refinement

Shared modules (`scene.py`, `camera.py`, `robot.py`, …) consume
`BaseSceneConfig` and dispatch on its decision hooks
(`has_walls()`, `has_apriltag()`, `viewport_size()`).

Coordinate convention (identical for both rigs)
-----------------------------------------------
* World frame: Z-up, 1 m units.
* Combined table surface centred at world origin in XY, top at ``Z = top_z``.
* ``+X`` points away from the operator/camera; the robot's ``panda_link0``
  faces ``+X``. Maple's open side (and egoverse's wearer) sit at ``-X``.
* ``+Y`` points toward the **left** table cell; ``-Y`` toward the **right**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Path anchors ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # dataset_replay/
ASSETS = PROJECT_ROOT / "assets"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TableConfig:
    """Two tables placed side-by-side along the Y axis.

    Default: each cell is 1.00 m (X) × 0.70 m (Y), so the combined surface
    is 1.00 × 1.40 m centred at the world origin. Identical across rigs.
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

    Per-dataset defaults are set by `MapleSceneConfig` / `EgoverseSceneConfig`
    (they construct this dataclass with different `mount_xyz` values because
    the May-2026 overlay comparison showed each rig aligns best at its own
    value: maple at `-0.255` X, egoverse at `-0.246` X). The field default
    below is just a placeholder so the dataclass is instantiable on its own;
    production scene configs always override it.

    NOTE: ``panda_link0`` has a non-trivial local translate inside the
    orcav1b USD ``(-0.00761, -0.00027, -0.47602)``. The wrapper transform
    must subtract this so panda_link0's *origin* lands at ``mount_xyz``.
    Both the mount xyz and the USD-internal offset are exposed here so
    the math is fully inspectable.
    """
    mount_xyz: tuple[float, float, float] = (-0.255, -0.35, 0.75)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda_link0_local_translate: tuple[float, float, float] = (
        -0.007610592991113663,
        -0.00026992621133103967,
        -0.4760153889656067,
    )


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BaseCameraConfig:
    """Common pinhole-camera fields. Subclasses add pose-resolution hooks.

    Concrete subclasses live in `config_maple.OakDCameraConfig` (carries a
    nominal world-fixed lookat pose) and `config_egoverse.AriaCameraConfig`
    (carries a base-relative `t_base_cam()` extrinsic).
    """
    name:       str = ""
    width:      int = 0
    height:     int = 0
    intrinsics: dict = field(default_factory=dict)            # {"fx","fy","cx","cy"}
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BaseSceneConfig:
    """Top-level scene config.

    Concrete subclasses set the camera type, flip the
    `has_walls() / has_apriltag()` switches, and override `viewport_size()`
    if the rig wants something other than 16:9.
    """
    dataset: str = ""                                    # "maple" | "egoverse"
    table:   TableConfig      = field(default_factory=TableConfig)
    robot:   RobotMountConfig = field(default_factory=RobotMountConfig)
    camera:  BaseCameraConfig = field(default_factory=BaseCameraConfig)

    robot_asset_path: Path = ASSETS / "orcav1b_franka_vmnt_v10_flattened.usd"
    urdf_path:        Path = ASSETS / "urdf" / "panda_arm.urdf"
    lula_descriptor:  Path = ASSETS / "lula" / "panda_arm_descriptor.yaml"

    # ── Decision hooks shared modules read ───────────────────────────────────
    def has_walls(self) -> bool:
        return False

    def has_apriltag(self) -> bool:
        return False

    def viewport_size(self) -> tuple[int, int]:
        return (1280, 720)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def default_mount_xyz(self) -> tuple[float, float, float]:
        """Canonical "centre of right table, 10 cm from back edge" position.

        Useful when the user changes ``table.single_size_xy`` or ``n_tables``
        and wants a reasonable starting `robot.mount_xyz`. Not auto-called —
        kept explicit so the user sees the dependency.
        """
        return (
            self.table.x_extent[0] + 0.10,
            self.table.right_table_centre_y,
            self.table.top_z,
        )


# ─────────────────────────────────────────────────────────────────────────────
def select_config(dataset: str) -> BaseSceneConfig:
    """Dispatch a dataset name → concrete SceneConfig instance.

    Imports the concrete module lazily so neither dataset's module is
    imported until its config is actually requested.
    """
    if dataset == "maple":
        from .config_maple import MapleSceneConfig
        return MapleSceneConfig()
    if dataset == "egoverse":
        from .config_egoverse import EgoverseSceneConfig
        return EgoverseSceneConfig()
    raise ValueError(f"unknown dataset: {dataset!r}")
