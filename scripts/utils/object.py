"""Object spawning + per-frame world-pose updates for kinematic replay.

Simplified for the new minimal branch:
* Objects are spawned in **kinematic, no-collision** mode by default —
  trajectories drive them; PhysX never integrates contact for them.
* No object-pose-version registry, no D6 tracking joints, no collision-pair
  filtering. ``set_object_world_pose`` consumes a **world-frame** ``T_world_obj``;
  the replay scripts compose camera-frame trajectories (the default — a 6D
  pose estimator outputs ``T_cam_obj``) with ``T_world_cam`` before calling
  here.

Depends on pxr / Isaac Sim — must be imported after SimulationApp is created.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


# ── Public API ───────────────────────────────────────────────────────────────
def spawn_object(
    stage: Usd.Stage,
    object_name: str,
    objects_dir: Path,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 0.1,
    kinematic: bool = True,
    collision: bool = False,
) -> str:
    """Spawn an OBJ-mesh object under ``/World/Objects/<name>``.

    Args:
        stage: USD stage.
        object_name: Folder name under ``objects_dir``; the OBJ is loaded from
            ``objects_dir/<name>/<name>.obj``.
        objects_dir: Root containing per-object subfolders.
        position: Initial world position; trajectory replay overrides this.
        scale: Uniform scale factor.
        kinematic: When ``True``, gravity is disabled and the rigid body is
            kinematic. PhysX follows externally-prescribed transforms instead
            of integrating forces.
        collision: When ``False``, no ``CollisionAPI`` is applied — purely
            visual, no physics interaction. Default for kinematic replay.

    Returns the prim path of the spawned object.
    """
    try:
        kit_commands = importlib.import_module("omni.kit.commands")
    except Exception as exc:
        raise RuntimeError("omni.kit.commands is required to spawn OBJ assets") from exc

    obj_path = (objects_dir / object_name / f"{object_name}.obj").resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {obj_path}")

    UsdGeom.Xform.Define(stage, "/World/Objects")
    prim_path = f"/World/Objects/{object_name}"
    UsdGeom.Xform.Define(stage, prim_path)

    ref_path = f"{prim_path}/visual"
    kit_commands.execute(
        "CreateReferenceCommand",
        usd_context=omni.usd.get_context(),
        path_to=ref_path,
        asset_path=str(obj_path),
        instanceable=False,
    )

    # The OBJ importer adds a rotateX:unitsResolve = 90° (Y-up → Z-up). Our
    # scanned meshes are already Z-up, so zero out the spurious rotation
    # while keeping any cm→m scale.
    visual_prim = stage.GetPrimAtPath(ref_path)
    if visual_prim and visual_prim.IsValid():
        for op in UsdGeom.Xformable(visual_prim).GetOrderedXformOps():
            if "rotate" in op.GetOpName().lower():
                op.Set(0.0)

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Failed to create object prim at {prim_path}")

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*position))
    xform_api.SetScale(Gf.Vec3f(scale, scale, scale))
    _setup_rigid_body(prim, kinematic=kinematic, collision=collision)
    return prim_path


def set_object_world_pose(
    stage: Usd.Stage, prim_path: str, T_world_obj: np.ndarray,
) -> None:
    """Update an object's world pose from a 4×4 homogeneous transform."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Object prim not found: {prim_path}")

    pos = T_world_obj[:3, 3]
    eul_xyz_deg = Rotation.from_matrix(T_world_obj[:3, :3]).as_euler("XYZ", degrees=True)

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xform_api.SetRotate(
        Gf.Vec3f(float(eul_xyz_deg[0]),
                 float(eul_xyz_deg[1]),
                 float(eul_xyz_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────
def _setup_rigid_body(
    root_prim: Usd.Prim, *, kinematic: bool, collision: bool,
) -> None:
    """Apply rigid-body + (optional) collision APIs.

    Mirrors the existing pattern: kinematic ⇒ gravity disabled +
    KinematicEnabledAttr(True); collision=False ⇒ skip the per-mesh
    CollisionAPI / MeshCollisionAPI apply loop entirely.
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
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
