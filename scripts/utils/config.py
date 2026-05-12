"""Scene configuration for the minimal single-arm OAK-D branch.

Pure-Python, no Isaac Sim imports — safe to import before SimulationApp is
created. The SceneConfig dataclass is the *single source of truth* for every
scene parameter the procedural builder, robot setup, camera setup, and
calibration scripts read.

Coordinate convention
---------------------
* World frame: Z-up, 1 m units.
* Combined table surface centred at world origin in XY, top at ``Z = top_z``.
* ``+X`` points toward the **back wall** — i.e., away from the operator/camera.
* ``-X`` points toward the open side of the U, where the OAK-D is mounted.
* ``+Y`` points toward the **left** table; ``-Y`` toward the **right** table.
* Robot's ``panda_link0`` faces ``+X`` by default (identity yaw) — i.e., it
  faces the back wall.

Geometry sketch (top-down)
--------------------------
::

                              +X (back wall — robot faces this direction)
                                ^
                                |
          /==================================\\
          ||                       AprilTag ||
          ||                                ||
          ||                  panda_link0   ||      Combined surface:
          ||  LEFT TABLE      RIGHT TABLE    ||      X in [-0.50, +0.50]  (1.00 m)
          ||  70 x 100 cm     70 x 100 cm    ||      Y in [-0.70, +0.70]  (1.40 m)
   +Y <---||                                ||---> -Y      top at Z = 0.75 m
          ||                                ||
          ||                                ||             (back wall  -- at +X)
          ||                                ||             (left wall  -- at +Y)
          \\==================================/             (right wall -- at -Y)
                                |                          (open       -- at -X)
                                v
                              -X (camera mounted here, looking +X)
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
    """tag36h11 id=5, lying flat on the table top.

    Two physical sizes matter for this tag, and they are NOT the same:

    * ``edge_size``      — the **black** square's edge (what pupil-apriltags
                            detects; what PnP uses as the marker side
                            length). This is the size you measure with a
                            ruler between the outer black-border edges of
                            the printed tag.
    * ``printed_edge_size`` — the **full printed quad**'s edge, i.e. black
                            + the outer **white border** that is part of
                            the PNG. This is the size the simulated quad
                            mesh is built at, so the texture maps 1:1 onto
                            it. For the bundled ``tag36_11_00005.png``
                            (10 px total, 8 px black) this is
                            ``edge_size × 10/8 = 1.25 × edge_size``.
                            Update ``printed_to_black_ratio`` if you swap
                            to a PNG with a different border layout.

    The corner offsets are measured on the real table with a ruler from
    the table edge to the **outer printed corner** (i.e. the white-border
    corner you can see touching the table). They therefore describe
    ``printed_edge_size``'s back-right corner, not the black corner.

    Coordinate reminder (see TableConfig diagram):
      * +X = toward the back wall (away from operator/camera)
      * -Y = toward the right table half (where the robot lives)

    The fields below are GROUND TRUTH measured from the real table.
    PnP-based calibration moves only the *camera* to match this fixed
    AprilTag pose; it never re-derives these.
    """
    family:    str   = "tag36h11"
    tag_id:    int   = 5                                # tag36_11_00005

    # PHYSICAL black-square edge (ruler measurement of the printed tag's
    # outer black-border edges). What pupil-apriltags detects.
    edge_size: float = 0.075                            # m (black square edge)

    image_path: Path = ASSETS / "textures" / "tag36_11_00005.png"

    # Ratio of the full printed quad edge to the black-square edge. The
    # bundled tag36_11_00005.png is 10 px wide with 8 px of black, so the
    # white border adds 1 px (= 1/8 of the black edge) on every side and
    # the full printed edge is 10/8 = 1.25 × edge_size. Adjust if you
    # swap to a PNG with a different white-border thickness.
    printed_to_black_ratio: float = 10.0 / 8.0          # = 1.25

    # Distance from the **outer printed corner** of the back-right corner
    # of the tag inward to the table edges. Read these as: "the +X-most,
    # -Y-most corner of the **printed quad including its white border**
    # is N m from the +X (back) edge, and M m from the -Y (right) edge".
    back_right_corner_to_back_wall:  float = 0.135      # m, in -X direction
    back_right_corner_to_right_edge: float = 0.165      # m, in +Y direction

    z_offset_above_table: float = 0.001                 # m, anti-z-fighting lift

    # Yaw of the printed tag about world Z, in degrees. 0 = unrotated quad
    # (intrinsic +Y "up in image" along world +X). -90° = physical tag mounted
    # rotated 90° clockwise viewed from above. The corner offsets above are
    # in WORLD frame and do NOT change when ``rotation_z_deg`` changes.
    rotation_z_deg: float = -90.0

    # ---------------------------------------------------------------- derived
    @property
    def printed_edge_size(self) -> float:
        """Full printed quad edge (black + outer white border)."""
        return self.edge_size * self.printed_to_black_ratio


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
    # Corrected from (-0.40, -0.35, 0.75) by +0.154 m in X: the previous
    # value treated mount_xyz as the back-of-link0 reference, but
    # ``utils.scene._add_robot`` actually places the kinematic
    # panda_link0 origin at this point. The back of the link0 mesh
    # extends 0.154 m in -X from the origin (measured from
    # pandaorca_description/meshes/franka/fer/visual/link0.dae); the
    # corrected value puts the back of the physical robot puck at the
    # previously-measured -0.40 m reference.
    mount_xyz: tuple[float, float, float] = (-0.246, -0.35, 0.75)
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    panda_link0_local_translate: tuple[float, float, float] = (
        -0.007610592991113663,
        -0.00026992621133103967,
        -0.4760153889656067,
    )


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """OAK-D Pro AF mounted above the open (-X) edge of the table, fixed in world.

    Defaults reflect the recording in ``data/h5/20250922_143954.h5``: 480×270
    image with the K matrix stored alongside it. The recorded T_robot_cam is
    available via ``utils.h5_loader.read_h5_extrinsic`` and converts to world
    via the configured ``robot.mount_xyz``. The nominal pose below is a
    sensible fallback that roughly mirrors the recorded camera placement.

    Distortion coeffs are exposed for the PnP step in calibrate_extrinsic.py;
    they don't affect the simulated pinhole render.
    """
    name:   str = "oakd_front_view"
    width:  int = 480
    height: int = 270

    # OAK-D Pro AF spec-sheet derived K at 480×270 (IMX378 sensor, 4.81 mm
    # lens, 1.55 μm pixel pitch, scaled from native 4056×3040). Empirically
    # this gave the cleanest sim/real alignment in the visualizer overlay
    # vs the H5-stored K (fx≈299 — likely mis-scaled at recording time).
    # See assets/calibration/oakd_proaf_doc_intrinsics.json for the
    # derivation. To revert to the H5-stored K: fx=fy≈299.24, cx≈244.32,
    # cy≈138.16.
    intrinsics: dict = field(default_factory=lambda: {
        "fx": 367.16,
        "fy": 367.16,
        "cx": 240.0,
        "cy": 135.0,
    })
    # OpenCV (k1, k2, p1, p2, k3). Zero distortion is the safe default for the
    # simulated render; the calibration script accepts real values via CLI.
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    extrinsic_path: Path = ASSETS / "calibration" / "oakd_extrinsic.npz"

    # Nominal pose (used as PnP initial guess and as fallback when no extrinsic
    # file is on disk and no H5 is supplied). Roughly matches the H5-recorded
    # T_world_cam: above the -X edge of the table, slightly +Y, ~1.7 m up,
    # looking +X+slight-Z toward the workspace.
    nominal_position: tuple[float, float, float] = (-0.32, 0.10, 1.71)
    nominal_lookat:   tuple[float, float, float] = (0.20, 0.10, 0.75)
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
    def apriltag_back_right_printed_corner_world(self) -> np.ndarray:
        """World position of the **outer printed corner** of the tag's
        back-right corner (+X, -Y in world).

        This is the *measured* anchor — what a ruler reads from the back
        wall and the right edge to the corner of the printed tag,
        including its outer white border. It is independent of the black
        edge size and of ``rotation_z_deg`` (rotation just permutes which
        intrinsic corner ends up at this world location).
        """
        x = self.table.x_extent[1] - self.apriltag.back_right_corner_to_back_wall
        y = self.table.y_extent[0] + self.apriltag.back_right_corner_to_right_edge
        z = self.table.top_z + self.apriltag.z_offset_above_table
        return np.array([x, y, z], dtype=float)

    # Legacy alias — kept so existing callers (and a few debug snippets)
    # don't break. New code should use the explicit name above.
    apriltag_back_right_corner_world = apriltag_back_right_printed_corner_world

    def apriltag_world_pose(self) -> np.ndarray:
        """AprilTag *centre* pose (4×4) in world frame with identity rotation.

        The centre of the printed quad and the centre of the black square
        coincide (concentric), so we derive it once from the measured
        printed corner: move one **printed** half-edge inward in -X and
        +Y. Identity rotation here; the printed tag's ``rotation_z_deg``
        is applied separately in
        ``utils.calibration._tag_world_pose_with_rotation`` and in
        ``utils.apriltag.add_apriltag_plane`` (USD).
        """
        half_printed = self.apriltag.printed_edge_size / 2.0
        corner = self.apriltag_back_right_printed_corner_world()
        centre = np.array([corner[0] - half_printed,
                           corner[1] + half_printed,
                           corner[2]])
        T = np.eye(4)
        T[:3, 3] = centre
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
