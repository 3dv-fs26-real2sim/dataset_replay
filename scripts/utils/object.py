"""Object spawning, pose updates, and collision filtering for Isaac Sim scenes.

Depends on pxr / Isaac Sim types — must be imported after SimulationApp is created.
"""

import importlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


#* THe mode does not support dual. Current dataset only uses single but fix if needed. 
def load_object_world_trajectory(
    stage,
    object_name: str,
    object_poses_override: str | None,
    object_pose_camera: str,
    n_frames: int,
    mode: str,
    right_prim_path: str,
    object_pose_version: str | None = None,
    T_world_cam_override=None,
):
    """Load an object's 6D pose trajectory and transform it to world frame.

    Handles the full pipeline: resolve the .npz path, load the camera-frame
    trajectory, compute the camera-to-world transform via robot-base extrinsics,
    and warn if the trajectory length doesn't match the H5 frame count.

    Args:
        stage: USD stage (needed for robot-base world transforms).
        object_name: Object name (e.g. ``"duck"``).  ``"none"`` skips loading.
        object_poses_override: Explicit .npz path, or ``None`` for default lookup.
        object_pose_camera: Camera name whose extrinsics define the pose frame.
        n_frames: Expected frame count from the H5 dataset.
        mode: Replay mode (``"single"`` or ``"dual"``).
        right_prim_path: Prim path of the right robot arm.
        object_pose_version: Pose-trajectory version key (e.g. ``"vda"``,
            ``"depthpro"``).  Ignored when ``object_poses_override`` is set.
            When ``None``, falls back to ``OBJECT_POSE_DEFAULT_VERSION``.
        T_world_cam_override: Optional 4x4 camera-in-world pose. When supplied,
            bypasses the nominal base-to-camera extrinsics and uses this pose
            to map the camera-frame trajectory into world frame — used by
            ``kinematic_replay.py --refined-extrinsic`` so the viewport
            camera and the object trajectory share the same world frame.

    Returns:
        ``(N, 4, 4)`` array of world-frame transforms, or ``None`` if the
        object should not be spawned.
    """
    from .constants import CAMERA_CONFIGS
    from .poses import load_pose_trajectory, transform_trajectory
    from .camera import compute_camera_world_pose

    if object_name == "none":
        print("[object] Object spawning disabled (--object none).")
        return None
    if mode != "single":
        print(f"[object] Skipping object spawn — only available in --mode single (got {mode})")
        return None

    npz_path = resolve_object_pose_path(
        object_name, object_poses_override, object_pose_version,
    )
    if npz_path is None:
        print(
            f"[object] No default pose trajectory for '{object_name}' "
            f"(version={object_pose_version!r}). "
            f"Pass --object-poses to provide one. Skipping spawn."
        )
        return None

    traj_cam = load_pose_trajectory(npz_path)
    print(f"[object] Loaded {traj_cam.shape[0]} poses from {npz_path}")

    # Camera frame -> world frame.
    if T_world_cam_override is not None:
        T_world_cam = np.asarray(T_world_cam_override, dtype=float)
        if T_world_cam.shape != (4, 4):
            raise ValueError(
                f"T_world_cam_override must be 4x4, got {T_world_cam.shape}"
            )
        print("[object] Using refined T_world_cam for camera→world trajectory mapping")
    else:
        cam_cfg = CAMERA_CONFIGS[object_pose_camera]
        T_world_cam = compute_camera_world_pose(
            stage, cam_cfg["extrinsics"],
            left_prim_path=None,  # single mode
            right_prim_path=right_prim_path,
        )
    traj_world = transform_trajectory(T_world_cam, traj_cam)

    if traj_world.shape[0] != n_frames:
        print(
            f"[object] WARNING: trajectory has {traj_world.shape[0]} frames, "
            f"H5 has {n_frames}. Will use min(N, n_frames) frames during replay."
        )

    return traj_world


def resolve_object_pose_path(
    object_name: str,
    object_poses_override: str | None = None,
    object_pose_version: str | None = None,
):
    """Find the .npz pose trajectory for the chosen object, or None to skip spawn.

    Args:
        object_name: Object name (must be a key in ``OBJECT_POSE_PATHS``, or
            any string when ``object_poses_override`` is given).
        object_poses_override: Explicit path to a .npz file.  When provided,
            this takes precedence over the default lookup.
        object_pose_version: Which pose-estimator variant to pick from
            ``OBJECT_POSE_PATHS[object_name]`` (e.g. ``"vda"``, ``"depthpro"``).
            When ``None``, falls back to ``OBJECT_POSE_DEFAULT_VERSION``.

    Returns:
        ``Path`` to the .npz file, or ``None`` if no trajectory is available.
    """
    from .constants import OBJECT_POSE_PATHS, OBJECT_POSE_DEFAULT_VERSION

    if object_poses_override is not None:
        return Path(object_poses_override)

    versions = OBJECT_POSE_PATHS.get(object_name)
    if versions is None:
        return None

    version = object_pose_version or OBJECT_POSE_DEFAULT_VERSION
    if version not in versions:
        raise ValueError(
            f"Unknown pose version '{version}' for object '{object_name}'. "
            f"Available: {sorted(versions.keys())}"
        )
    return versions[version]


def spawn_object(
    stage,
    object_name: str,
    position,
    scale: float,
    objects_dir: Path,
    *,
    kinematic: bool = False,
    collision: bool = True,
) -> str:
    """Spawn an OBJ mesh into the scene with physics enabled.

    Args:
        stage: USD stage.
        object_name: Name of the object directory under ``objects_dir``.
        position: Initial ``(x, y, z)`` world position.
        scale: Uniform scale factor.
        objects_dir: Root directory containing per-object subfolders with ``.obj`` files.
        kinematic: When ``True``, the rigid body is *kinematic*: gravity is disabled
            and the body will not respond to forces. Use with ``set_object_world_pose``
            to drive it from an external trajectory (e.g. a 6D pose estimator).
        collision: When ``False``, no collision geometry is added — the object is
            purely visual and cannot interact with any physics body.

    Returns:
        The prim path of the spawned object.
    """
    try:
        kit_commands = importlib.import_module("omni.kit.commands")
    except Exception as exc:
        raise RuntimeError("omni.kit.commands extension is required to spawn OBJ assets") from exc

    import omni.usd

    obj_path = (objects_dir / object_name / f"{object_name}.obj").resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {obj_path}")

    object_root_path = "/World/spawned_objects"
    UsdGeom.Xform.Define(stage, object_root_path)

    prim_path = f"{object_root_path}/{object_name}_instance"
    UsdGeom.Xform.Define(stage, prim_path)

    ref_path = f"{prim_path}/visual"
    kit_commands.execute(
        "CreateReferenceCommand",
        usd_context=omni.usd.get_context(),
        path_to=ref_path,
        asset_path=str(obj_path),
        instanceable=False,
    )

    # The OBJ importer adds a rotateX:unitsResolve = 90° to the visual sub-prim
    # (Y-up → Z-up conversion), but our Artec-scanned meshes are already Z-up.
    # Zero out the spurious rotation while keeping the cm→m unit-conversion scale.
    visual_prim = stage.GetPrimAtPath(ref_path)
    if visual_prim and visual_prim.IsValid():
        xformable = UsdGeom.Xformable(visual_prim)
        for op in xformable.GetOrderedXformOps():
            if "rotate" in op.GetOpName().lower():
                op.Set(0.0)

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Failed to create object prim at {prim_path}")

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
    xform_api.SetScale(Gf.Vec3f(scale, scale, scale))
    _setup_rigid_body(prim, kinematic=kinematic, collision=collision)
    return prim_path


def set_object_world_pose(stage, prim_path: str, T_world_obj: np.ndarray) -> None:
    """Update an object's world pose from a 4x4 homogeneous transform.

    Translation and rotation are written via ``XformCommonAPI``; the scale set
    at spawn time is left untouched. Designed for per-frame trajectory replay
    of kinematic rigid bodies.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Object prim not found: {prim_path}")

    pos = T_world_obj[:3, 3]
    eul_xyz_deg = Rotation.from_matrix(T_world_obj[:3, :3]).as_euler("XYZ", degrees=True)

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform_api.SetRotate(
        Gf.Vec3f(float(eul_xyz_deg[0]), float(eul_xyz_deg[1]), float(eul_xyz_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


#* For dynamic_replay.py, but very much experimental only. We need a better idea.
def create_d6_tracking_joint(
    stage,
    object_prim_path: str,
    initial_pose: "np.ndarray",
    stiffness: float = 500.0,
    damping: float = 100.0,
) -> str:
    """Create a kinematic anchor and D6 joint with spring-damper drives.

    The anchor follows the estimated trajectory each frame (via
    ``set_object_world_pose``); the D6 joint's spring-damper drives pull the
    dynamic object toward the anchor while PhysX resolves collisions, gravity,
    and contact forces.

    Args:
        stage: USD stage.
        object_prim_path: Prim path of the **dynamic** rigid body object.
        initial_pose: 4x4 homogeneous transform for the anchor's starting pose.
        stiffness: Spring stiffness (N/m for translation, N*m/rad for rotation).
        damping: Damping coefficient for all 6 DOF drives.

    Returns:
        The prim path of the kinematic anchor.
    """
    from scipy.spatial.transform import Rotation as R

    anchor_path = "/World/spawned_objects/pose_anchor"
    anchor_prim = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()

    # Make the anchor a kinematic rigid body (no gravity, no forces).
    rb_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
    rb_api.CreateRigidBodyEnabledAttr(True)
    rb_api.CreateKinematicEnabledAttr(True)
    try:
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(anchor_prim)
        physx_rb.CreateDisableGravityAttr(True)
    except Exception:
        pass

    # Place anchor at the initial pose.
    pos = initial_pose[:3, 3]
    eul_xyz_deg = R.from_matrix(initial_pose[:3, :3]).as_euler("XYZ", degrees=True)
    xform_api = UsdGeom.XformCommonAPI(anchor_prim)
    xform_api.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform_api.SetRotate(
        Gf.Vec3f(float(eul_xyz_deg[0]), float(eul_xyz_deg[1]), float(eul_xyz_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    # Create D6 joint connecting anchor (body0) -> dynamic object (body1).
    joint_path = "/World/spawned_objects/d6_joint"
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([anchor_path])
    joint.CreateBody1Rel().SetTargets([object_prim_path])

    # Configure spring-damper drives on all 6 DOFs.
    joint_prim = joint.GetPrim()
    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateTargetPositionAttr(0.0)

    print(
        f"[d6] Created tracking joint: {anchor_path} -> {object_prim_path} "
        f"(Kp={stiffness}, Kd={damping})"
    )
    return anchor_path


def filter_collision_pair(stage, prim_path_a: str, prim_path_b: str) -> None:
    """Disable collision between two prims (and their descendants).

    Uses ``UsdPhysics.FilteredPairsAPI`` so the rest of the scene's collision
    behaviour is unchanged. Useful for replaying trajectories where two bodies
    intentionally interpenetrate (e.g. robot hand passing through a kinematic
    object whose pose came from an offline 6D estimator).
    """
    prim_a = stage.GetPrimAtPath(prim_path_a)
    if not prim_a.IsValid():
        print(f"[collision] WARNING: {prim_path_a} not found in stage")
        return

    api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
    rel = api.CreateFilteredPairsRel()
    rel.AddTarget(Sdf.Path(prim_path_b))
    print(f"[collision] Filtered collision pair: {prim_path_a} <-> {prim_path_b}")


# ── Internal helpers ────────────────────────────────────────────────────────


def _setup_rigid_body(root_prim: Usd.Prim, *, kinematic: bool, collision: bool = True) -> None:
    """Apply rigid-body + collision APIs.

    For kinematic bodies, gravity is disabled and the kinematic flag is set,
    so PhysX follows externally-prescribed transforms instead of integrating
    forces. The object can still collide with other dynamic bodies (e.g. the
    table) — kinematic bodies push dynamics but ignore forces themselves.

    When ``collision=False``, no ``CollisionAPI`` is applied to any mesh prim,
    making the object purely visual with no physics interaction.
    """
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb_api.CreateRigidBodyEnabledAttr(True)
    if kinematic:
        rb_api.CreateKinematicEnabledAttr(True)

    try:
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_rb_api.CreateDisableGravityAttr(kinematic)
    except Exception:
        pass

    if collision:
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.CreateApproximationAttr("convexHull")
