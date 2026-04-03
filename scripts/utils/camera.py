"""Camera setup for Isaac Sim viewport from real-world calibration data.

Positions the viewport camera to match a calibrated real-world camera
(e.g., Aria glasses) using extrinsic transforms relative to robot arm bases.

Depends on Isaac Sim / pxr (deferred import).
Must be imported after SimulationApp is created.
"""

import importlib

import numpy as np
from scipy.spatial.transform import Rotation

from pxr import Gf, Usd, UsdGeom

from .constants import CAMERA_CONFIGS

# CV cameras look along +Z; USD cameras look along -Z (OpenGL convention).
# Rotate 180° around X to convert between the two.
_T_USD_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ── Public API ───────────────────────────────────────────────────────────────


def setup_camera(stage, camera_name, mode, left_prim_path, right_prim_path):
    """Create a calibrated camera and set it as the active viewport camera.

    Args:
        stage: USD stage.
        camera_name: Key into CAMERA_CONFIGS (e.g. "aria").
        mode: "single" or "dual".
        left_prim_path: Prim path of the left robot arm (ignored in single mode).
        right_prim_path: Prim path of the right robot arm.

    Returns:
        The USD prim path of the created camera (e.g. "/World/AriaCamera").
    """
    config = CAMERA_CONFIGS[camera_name]
    extrinsics = config["extrinsics"]
    intrinsics = config["intrinsics"]

    left_path = left_prim_path if mode == "dual" else None
    world_pose = compute_camera_world_pose(stage, extrinsics, left_path, right_prim_path)

    prim_path = f"/World/{camera_name.capitalize()}Camera"
    create_camera_prim(stage, prim_path, world_pose, intrinsics)
    set_viewport_camera(prim_path)
    return prim_path


def compute_camera_world_pose(stage, extrinsics, left_prim_path, right_prim_path):
    """Compute the camera world pose by transforming through robot base poses.

    In dual mode both arm bases are used and the results averaged.
    In single mode (left_prim_path is None) only the right arm is used.

    Returns:
        4x4 homogeneous matrix — camera pose in world frame.
    """
    poses = []

    # Right arm (always available).
    T_world_base_r = get_prim_world_transform(stage, right_prim_path)
    T_world_cam_r = T_world_base_r @ extrinsics["right"]
    poses.append(T_world_cam_r)

    # Left arm (dual mode only).
    if left_prim_path is not None:
        T_world_base_l = get_prim_world_transform(stage, left_prim_path)
        T_world_cam_l = T_world_base_l @ extrinsics["left"]
        poses.append(T_world_cam_l)

    if len(poses) == 1:
        result = poses[0]
    else:
        result = _average_poses(poses)

    pos = result[:3, 3]
    print(f"[camera] Camera world position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    return result


def get_prim_world_transform(stage, prim_path):
    """Get the 4x4 column-vector-convention world transform of a USD prim.

    USD's Gf.Matrix4d uses row-vector convention (p' = p * M).
    We transpose to standard column-vector convention (p' = M * p).
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    gf_mat = xform_cache.GetLocalToWorldTransform(prim)
    # Gf.Matrix4d → numpy.  GetArray() returns row-major; transpose for column-vector.
    return np.array(gf_mat, dtype=float).T


def create_camera_prim(stage, prim_path, world_pose, intrinsics):
    """Create a UsdGeom.Camera with the given world pose and pinhole intrinsics.

    Args:
        stage: USD stage.
        prim_path: Where to define the camera (e.g. "/World/AriaCamera").
        world_pose: 4x4 column-vector-convention transform (camera-in-world).
        intrinsics: dict with keys width, height, fx, fy, cx, cy.
    """
    camera = UsdGeom.Camera.Define(stage, prim_path)

    # Apply CV→USD orientation flip, then convert to USD row-vector convention.
    pose_usd = world_pose @ _T_USD_FROM_CV
    gf_mat = Gf.Matrix4d(*pose_usd.T.flatten().tolist())

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(gf_mat)

    # Map pinhole intrinsics to USD camera attributes.
    # USD uses focalLength (mm) and aperture (mm).  We pick an arbitrary
    # focal-length value and derive the aperture so that the projection
    # matches the pinhole model.
    focal_length_mm = 24.0  # arbitrary — only the ratio matters
    h_aperture = focal_length_mm * intrinsics["width"]  / intrinsics["fx"]
    v_aperture = focal_length_mm * intrinsics["height"] / intrinsics["fy"]

    camera.GetFocalLengthAttr().Set(focal_length_mm)
    camera.GetHorizontalApertureAttr().Set(h_aperture)
    camera.GetVerticalApertureAttr().Set(v_aperture)

    # Principal point offset (0 = centered).
    cx_offset = (intrinsics["cx"] - intrinsics["width"]  / 2.0) / intrinsics["fx"] * focal_length_mm
    cy_offset = (intrinsics["cy"] - intrinsics["height"] / 2.0) / intrinsics["fy"] * focal_length_mm
    camera.GetHorizontalApertureOffsetAttr().Set(cx_offset)
    camera.GetVerticalApertureOffsetAttr().Set(cy_offset)

    # Near/far clipping planes (meters).
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    return prim_path


def set_viewport_camera(camera_path):
    """Set the active viewport to render from the given camera prim."""
    viewport_utility = _import_viewport_utility()
    if viewport_utility is None:
        print("[camera] Warning: could not set viewport camera (omni.kit.viewport.utility unavailable)")
        return
    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        print("[camera] Warning: no active viewport found")
        return
    viewport.set_active_camera(camera_path)


# ── Private helpers ──────────────────────────────────────────────────────────


def _average_poses(poses):
    """Average multiple 4x4 poses (position mean + rotation mean)."""
    positions = np.array([p[:3, 3] for p in poses])
    avg_pos = positions.mean(axis=0)

    rotations = Rotation.from_matrix([p[:3, :3] for p in poses])
    avg_rot = rotations.mean().as_matrix()

    result = np.eye(4)
    result[:3, :3] = avg_rot
    result[:3, 3] = avg_pos
    return result


def _import_viewport_utility():
    """Import omni.kit.viewport.utility, enabling the extension if needed."""
    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        pass
    try:
        app = importlib.import_module("omni.kit.app")
        app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.kit.viewport.utility", True,
        )
    except Exception:
        return None
    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        return None
