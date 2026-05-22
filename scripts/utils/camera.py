"""Aria camera setup for the egoverse branch.

The world pose is computed at runtime as ``T_world_panda_link0 @ T_base_cam``
where ``T_base_cam`` (= ``ARIA_EXTRINSICS_RIGHT``) is the wearer-rig
calibration carried over from the original main-branch capture setup. The
robot base sits in the same place relative to the wearer in egoverse as it
did in main, so the base-relative transform is invariant; only the table-top
Z differs (0.75 m here vs 1.0 m in main), and that's absorbed by the world
composition.

For per-session corrections, ``scripts/calibrate_extrinsic_table.py`` produces
a refined ``T_world_cam`` NPZ by aligning projected table edges to SAM
masks of the recorded video. Pass it through ``world_pose_override`` here
to override the base-relative computation.

Depends on ``pxr`` and ``omni.usd`` — must be imported after SimulationApp
is created.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .config import CameraConfig
from .constants import ROBOT_BASE_PRIM_PATH

# CV cameras look along +Z; USD cameras look along -Z (OpenGL convention).
# Diag(1, -1, -1, 1) flips Y and Z to convert between the two.
_T_USD_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ── Public API ───────────────────────────────────────────────────────────────
def setup_camera(stage: Usd.Stage, cfg: CameraConfig,
                 *, prim_path: str | None = None,
                 world_pose_override: np.ndarray | None = None) -> str:
    """Create a UsdGeom.Camera at the Aria pose and set it as the active
    viewport camera. Returns the camera prim path.

    If ``world_pose_override`` is supplied, it is used verbatim — bypassing
    the base-relative computation. Use this to plug in a refined
    ``T_world_cam`` produced by ``scripts/calibrate_extrinsic_table.py``.
    """
    _check_intrinsics(cfg.intrinsics)

    if world_pose_override is not None:
        world_pose = np.asarray(world_pose_override, dtype=float)
        if world_pose.shape != (4, 4):
            raise ValueError(
                f"world_pose_override must be 4×4, got {world_pose.shape}"
            )
        print(f"[camera] {cfg.name}: using world_pose_override")
    else:
        world_pose = compute_aria_world_pose(stage, cfg)
    pos = world_pose[:3, 3]
    print(f"[camera] {cfg.name}: world position [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    prim_path = prim_path or f"/World/Cameras/{_camel(cfg.name)}"
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    create_camera_prim(stage, prim_path, world_pose, cfg)
    set_viewport_camera(prim_path)
    return prim_path


def compute_aria_world_pose(stage: Usd.Stage, cfg: CameraConfig) -> np.ndarray:
    """Return ``T_world_cam = T_world_panda_link0 @ cfg.t_base_cam()``.

    Reads the right-arm base pose from the live USD stage so any shift of
    ``RobotMountConfig.mount_xyz`` propagates without extra wiring.
    """
    T_world_base = get_prim_world_transform(stage, ROBOT_BASE_PRIM_PATH)
    return T_world_base @ cfg.t_base_cam()


def create_camera_prim(stage: Usd.Stage, prim_path: str,
                       world_pose: np.ndarray, cfg: CameraConfig) -> str:
    """Create a UsdGeom.Camera with the given world pose and pinhole intrinsics.

    The world pose is in OpenCV camera convention (+Z forward); a CV→USD
    rotation flip is applied so the resulting USD camera looks at the same
    target as the input pose suggests.
    """
    camera = UsdGeom.Camera.Define(stage, prim_path)

    # Apply CV→USD orientation flip, then convert to USD row-vector convention.
    pose_usd = world_pose @ _T_USD_FROM_CV
    gf_mat = Gf.Matrix4d(*pose_usd.T.flatten().tolist())

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(gf_mat)

    # Map pinhole intrinsics → USD camera attributes.
    K = cfg.intrinsics
    focal_length_mm = 24.0   # arbitrary — only the ratio aperture/focal matters
    h_aperture = focal_length_mm * cfg.width  / K["fx"]
    v_aperture = focal_length_mm * cfg.height / K["fy"]

    camera.GetFocalLengthAttr().Set(focal_length_mm)
    camera.GetHorizontalApertureAttr().Set(h_aperture)
    camera.GetVerticalApertureAttr().Set(v_aperture)

    # Principal-point offset (0 = centred).
    cx_offset = (K["cx"] - cfg.width  / 2.0) / K["fx"] * focal_length_mm
    cy_offset = (K["cy"] - cfg.height / 2.0) / K["fy"] * focal_length_mm
    camera.GetHorizontalApertureOffsetAttr().Set(cx_offset)
    camera.GetVerticalApertureOffsetAttr().Set(cy_offset)

    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return prim_path


def set_viewport_camera(camera_path: str) -> None:
    """Switch the active viewport to render from the given camera prim."""
    from .viewport import get_viewport_utility
    viewport_utility = get_viewport_utility()
    if viewport_utility is None:
        print("[camera] Warning: omni.kit.viewport.utility unavailable")
        return
    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        print("[camera] Warning: no active viewport found")
        return
    viewport.set_active_camera(camera_path)


def get_prim_world_transform(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    """Return the 4x4 column-vector-convention world transform of a USD prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    gf_mat = cache.GetLocalToWorldTransform(prim)
    return np.array(gf_mat, dtype=float).T


# ── Internal helpers ─────────────────────────────────────────────────────────
def _check_intrinsics(K: dict) -> None:
    missing = [k for k in ("fx", "fy", "cx", "cy") if K.get(k, 0.0) == 0.0]
    if missing:
        raise ValueError(
            f"Aria intrinsics keys {missing} are zero. Fill "
            f"SceneConfig.camera.intrinsics from the factory calibration "
            f"before running."
        )


def _camel(name: str) -> str:
    """Convert ``foo_bar_baz`` → ``FooBarBaz`` for use as a prim path segment."""
    return "".join(part.capitalize() for part in name.split("_"))
