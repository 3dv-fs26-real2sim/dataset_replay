"""OAK-D Pro AF camera setup — single, world-fixed pinhole camera.

The world pose is loaded from a saved extrinsic (output of
``calibrate_extrinsic.py``), or computed from the nominal position+lookat+up
in ``CameraConfig`` as a fallback.

Depends on ``pxr`` and ``omni.usd`` — must be imported after SimulationApp
is created.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .config import CameraConfig

# CV cameras look along +Z; USD cameras look along -Z (OpenGL convention).
# Diag(1, -1, -1, 1) flips Y and Z to convert between the two.
_T_USD_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ── Public API ───────────────────────────────────────────────────────────────
def setup_camera(stage: Usd.Stage, cfg: CameraConfig,
                 *, prim_path: str | None = None) -> str:
    """Create a UsdGeom.Camera at the calibrated (or nominal) world pose and
    set it as the active viewport camera. Returns the camera prim path."""
    _check_intrinsics(cfg.intrinsics)

    world_pose = _resolve_world_pose(cfg)
    pos = world_pose[:3, 3]
    print(f"[camera] {cfg.name}: world position [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    prim_path = prim_path or f"/World/Cameras/{_camel(cfg.name)}"
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    create_camera_prim(stage, prim_path, world_pose, cfg)
    set_viewport_camera(prim_path)
    return prim_path


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


# ── World-pose resolution ────────────────────────────────────────────────────
def _resolve_world_pose(cfg: CameraConfig) -> np.ndarray:
    """Prefer the saved calibration; fall back to the nominal lookat pose."""
    path: Path = cfg.extrinsic_path
    if path.exists():
        T = np.load(path)["T_world_cam"]
        print(f"[camera] Using calibrated extrinsic from {path}")
        return T

    warnings.warn(
        f"No OAK-D extrinsic file at {path}. Using nominal pose; "
        f"run scripts/calibrate_extrinsic.py to save a real one."
    )
    return lookat_to_T_world_cam(
        cfg.nominal_position, cfg.nominal_lookat, cfg.nominal_up,
    )


def lookat_to_T_world_cam(
    position: tuple[float, float, float] | np.ndarray,
    lookat:   tuple[float, float, float] | np.ndarray,
    up:       tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Build a 4x4 OpenCV-convention camera-in-world transform from the
    (position, lookat, up) triple.

    Camera axes (in world frame): +Z = forward (toward ``lookat``),
    +X = right, +Y = down. With world-up = +Z, "down in image" is roughly
    -world-up, so the y-axis is the down-cross-forward direction.
    """
    pos    = np.asarray(position, dtype=float)
    lookat = np.asarray(lookat,   dtype=float)
    up     = np.asarray(up,       dtype=float)

    forward = lookat - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        raise ValueError("nominal lookat direction is parallel to nominal up vector")
    right /= n

    down = np.cross(forward, right)

    R = np.column_stack([right, down, forward])    # columns = camera axes in world
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pos
    return T


# ── Internal helpers ─────────────────────────────────────────────────────────
def _check_intrinsics(K: dict) -> None:
    missing = [k for k in ("fx", "fy", "cx", "cy") if K.get(k, 0.0) == 0.0]
    if missing:
        raise ValueError(
            f"OAK-D intrinsics keys {missing} are zero. Fill "
            f"SceneConfig.camera.intrinsics from the factory calibration "
            f"JSON before running."
        )


def _camel(name: str) -> str:
    """Convert ``foo_bar_baz`` → ``FooBarBaz`` for use as a prim path segment."""
    return "".join(part.capitalize() for part in name.split("_"))
