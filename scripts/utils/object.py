"""Object spawning and physics setup for Isaac Sim scenes.

Depends on pxr types — must be imported after SimulationApp is created.
"""

import importlib
from pathlib import Path

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def spawn_object(
    stage,
    object_name: str,
    position: list[float],
    scale: float,
    objects_dir: Path,
) -> str:
    """Spawn an OBJ mesh into the scene with physics enabled.

    Returns the prim path of the spawned object.
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

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Failed to create object prim at {prim_path}")

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
    xform_api.SetScale(Gf.Vec3f(scale, scale, scale))
    _enable_collision_and_gravity(prim)
    return prim_path


def _enable_collision_and_gravity(root_prim: Usd.Prim) -> None:
    """Make object dynamic and affected by gravity."""
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb_api.CreateRigidBodyEnabledAttr(True)

    try:
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_rb_api.CreateDisableGravityAttr(False)
    except Exception:
        pass

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr("convexHull")
