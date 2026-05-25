"""Maple-specific scene config: walls, AprilTag, OakD camera.

Pure-Python, no Isaac Sim imports — safe to import before SimulationApp.

The OakD pose is calibrated at runtime by detecting an AprilTag in the H5
image stream and running joint PnP+RANSAC against its measured world pose
(see :mod:`utils.calibrate_april`). A nominal lookat pose lives on
``OakDCameraConfig`` as the PnP seed / no-H5 fallback.

Geometry sketch (top-down)
--------------------------
::

                              +X (back wall)
                                ^
                                |
              ==================================
              |  BACK WALL                     |
              |                                |
              |                  panda_link0   |      Combined surface:
              |  LEFT TABLE      RIGHT TABLE   |      X in [-0.50, +0.50]  (1.00 m)
              |  70 x 100 cm     70 x 100 cm   |      Y in [-0.70, +0.70]  (1.40 m)
       +Y <-(LEFT WALL)         (RIGHT WALL)-> -Y     top at Z = 0.75 m
              |                                |
              |                                |
              |                                |
              ==================================
                                |
                                v
                              -X  (open side — OakD lives above this edge)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    ASSETS,
    BaseCameraConfig,
    BaseSceneConfig,
    RobotMountConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WallsConfig:
    """U-shape opening toward -X. Walls span the full combined-table edge.

    All heights default to 1.0 m to block background; tweak per-wall as needed.
    """
    back_height:  float = 1.0   # wall at +X edge (full Y span)
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

    Coordinate reminder:
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
class OakDCameraConfig(BaseCameraConfig):
    """OAK-D Pro AF mounted above the open (-X) edge, world-fixed.

    Carries a nominal lookat pose used as:
      * PnP seed inside :func:`utils.calibrate_april.calibrate_from_h5`,
      * fallback when ``kinematic_replay_maple.py`` runs without ``--h5``
        or with ``--no-calibrate``, or when the AprilTag isn't detected.

    K is the OAK-D Pro AF **spec-sheet ("doc") K** at 480×270 (IMX378
    sensor, 4.81 mm lens, 1.55 μm pixel pitch, scaled from native
    4056×3040). This one K is used for BOTH the PnP extrinsic and the
    sim-camera render FOV; they MUST share a single K, or the sim feed is
    zoomed relative to the H5 in overlay/sidebyside videos.

    Use doc K, NOT the H5-stored K. Overlay comparisons in earlier sessions
    showed doc K gives the cleanest sim/real alignment; the H5-stored K
    (fx=fy≈299.24, cx≈244.32, cy≈138.16) is misleading — it was mis-scaled
    at recording time. Do not "prefer the H5 K because it's the real sensor
    calibration": that reasoning has been checked and rejected here.

    Distortion coeffs are exposed for the PnP step; they don't affect the
    simulated pinhole render.
    """
    name:   str = "oakd_front_view"
    width:  int = 480
    height: int = 270

    intrinsics: dict = field(default_factory=lambda: {
        "fx": 367.16,
        "fy": 367.16,
        "cx": 240.0,
        "cy": 135.0,
    })
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    # Nominal pose — above the -X edge of the table, slightly +Y, ~1.7 m up,
    # looking +X+slight-Z toward the workspace.
    nominal_position: tuple[float, float, float] = (-0.32, 0.10, 1.71)
    nominal_lookat:   tuple[float, float, float] = (0.20, 0.10, 0.75)
    nominal_up:       tuple[float, float, float] = (0.0, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MapleSceneConfig(BaseSceneConfig):
    """Maple rig: walls + AprilTag + OakD camera.

    Mount xyz `(-0.255, -0.35, 0.75)` was the OakD-bringup measurement,
    re-confirmed by overlay comparison..
    """
    dataset:  str               = "maple"
    walls:    WallsConfig       = field(default_factory=WallsConfig)
    apriltag: AprilTagConfig    = field(default_factory=AprilTagConfig)
    camera:   OakDCameraConfig  = field(default_factory=OakDCameraConfig)
    robot:    RobotMountConfig  = field(
        default_factory=lambda: RobotMountConfig(mount_xyz=(-0.255, -0.35, 0.75)))

    def has_walls(self)    -> bool: return True
    def has_apriltag(self) -> bool: return True

    # ── Maple-specific helpers (used by scene/apriltag/calibrate_april) ──
    def apriltag_back_right_printed_corner_world(self) -> np.ndarray:
        """World position of the **outer printed corner** of the tag's
        back-right corner (+X, -Y in world).

        This is the *measured* anchor — what a ruler reads from the back
        wall and the right edge to the corner of the printed tag,
        including its outer white border. It is independent of the black
        edge size and of ``rotation_z_deg``.
        """
        x = self.table.x_extent[1] - self.apriltag.back_right_corner_to_back_wall
        y = self.table.y_extent[0] + self.apriltag.back_right_corner_to_right_edge
        z = self.table.top_z + self.apriltag.z_offset_above_table
        return np.array([x, y, z], dtype=float)

    # Legacy alias — kept so existing callers don't break.
    apriltag_back_right_corner_world = apriltag_back_right_printed_corner_world

    def apriltag_world_pose(self) -> np.ndarray:
        """AprilTag *centre* pose (4×4) in world frame with identity rotation.

        The printed quad's centre and the black square's centre coincide,
        so move one **printed** half-edge inward in -X and +Y from the
        measured printed corner. ``rotation_z_deg`` is applied separately
        in :func:`utils.calibrate_april._tag_world_pose_with_rotation` and
        in :func:`utils.apriltag.add_apriltag_plane`.
        """
        half_printed = self.apriltag.printed_edge_size / 2.0
        corner = self.apriltag_back_right_printed_corner_world()
        centre = np.array([corner[0] - half_printed,
                           corner[1] + half_printed,
                           corner[2]])
        T = np.eye(4)
        T[:3, 3] = centre
        return T
