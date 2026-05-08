"""Programmatic scene construction.

Builds the simulation stage from scratch — Z-up, metric — then adds a
ground plane, the table cube, lighting, a physics scene, and one robot
arm per side via reference to the flattened USD asset. Replaces the
static ``pandaorca_*.usda`` scene files.

Depends on ``pxr`` and ``omni.usd`` — must be imported after
``SimulationApp`` is created.
"""

from typing import Iterable

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from .constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    PANDA_LINK0_INTERNAL_OFFSET,
    ROBOT_ASSET_PATH,
    ROBOT_BASE_WORLD_POSITIONS,
    TABLE_PRIM_PATH,
)

_ARM_PRIM_PATHS = {"right": FRANKA_RIGHT_PATH, "left": FRANKA_LEFT_PATH}

# Cube geometry (mirrors the original /World/Cube exactly).
_CUBE_POINTS = [
    (-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5),
    (-0.5,  0.5,  0.5), ( 0.5,  0.5,  0.5),
    (-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5),
    (-0.5,  0.5, -0.5), ( 0.5,  0.5, -0.5),
]
_CUBE_FACE_VERTEX_COUNTS = [4, 4, 4, 4, 4, 4]
_CUBE_FACE_VERTEX_INDICES = [
    0, 1, 3, 2,  4, 6, 7, 5,  6, 2, 3, 7,
    4, 5, 1, 0,  4, 0, 2, 6,  5, 7, 3, 1,
]
_CUBE_TRANSLATE = (0.0, 0.0, 0.5)
_CUBE_SCALE     = (1.0, 1.4, 1.0)


def build_scene(mode: str, *, robot_collision: bool = True) -> Usd.Stage:
    """Create a fresh stage and populate it for the requested replay mode.

    Args:
        mode: ``"single"`` (right arm only) or ``"dual"`` (both arms).
        robot_collision: When ``False``, ``PhysicsCollisionAPI`` is
            disabled on every descendant of the robot wrapper xforms. Use
            this for kinematic replay, where joint positions are
            teleported each frame — leaving collisions on causes PhysX to
            integrate contact forces wherever the new pose overlaps the
            table, the floor, or itself, which manifests as visible
            jitter. Dynamic replay needs this on.

    Returns:
        The active ``Usd.Stage``.
    """
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    _set_stage_metadata(stage)
    _define_world(stage)
    _add_physics_scene(stage)
    _add_ground_plane(stage)
    _add_table(stage)
    _add_lighting(stage)

    sides = ("right", "left") if mode == "dual" else ("right",)
    for side in sides:
        _add_robot(stage, side)

    if not robot_collision:
        _disable_robot_collisions(
            stage, tuple(_ARM_PRIM_PATHS[s] for s in sides),
        )

    return stage


# ── Building blocks ──────────────────────────────────────────────────────────


def _set_stage_metadata(stage: Usd.Stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def _define_world(stage: Usd.Stage) -> None:
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())


def _add_physics_scene(stage: Usd.Stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))


def _add_ground_plane(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/GroundPlane")

    mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/CollisionMesh")
    mesh.CreatePointsAttr([
        Gf.Vec3f(-25, -25, 0), Gf.Vec3f( 25, -25, 0),
        Gf.Vec3f( 25,  25, 0), Gf.Vec3f(-25,  25, 0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
    mesh.CreateDoubleSidedAttr(False)

    plane = UsdGeom.Plane.Define(stage, "/World/GroundPlane/CollisionPlane")
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())


def _add_table(stage: Usd.Stage) -> None:
    """Mirror the original /World/Cube exactly: 1 m × 1.4 m × 1 m, top at z=1.0."""
    cube = UsdGeom.Mesh.Define(stage, TABLE_PRIM_PATH)
    cube.CreatePointsAttr([Gf.Vec3f(*p) for p in _CUBE_POINTS])
    cube.CreateFaceVertexCountsAttr(_CUBE_FACE_VERTEX_COUNTS)
    cube.CreateFaceVertexIndicesAttr(_CUBE_FACE_VERTEX_INDICES)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    cube.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    _set_translate_scale(cube.GetPrim(), _CUBE_TRANSLATE, _CUBE_SCALE)

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim())
    mesh_col.CreateApproximationAttr("convexHull")


def _add_lighting(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, "/Environment")
    light = UsdLux.DistantLight.Define(stage, "/Environment/defaultLight")
    light.CreateAngleAttr(1.0)
    light.CreateIntensityAttr(3000.0)


def _add_robot(stage: Usd.Stage, side: str) -> None:
    """Reference the robot USD as ``/World/Robot{Right|Left}`` with placement.

    The new USD's ``/Root`` has ``panda_link0`` at internal translate
    ``z = -0.476`` (and a small xy offset). To land panda_link0 at the
    legacy world position (e.g. ``(-0.262, -0.386, 1.0)`` for the right
    arm — flush with the table top), the wrapper Xform's translate is
    set to ``world_pos − panda_link0_internal_offset``. Without this
    compensation the visible robot base would float ~0.5 m above the
    table.

    Camera and object math are anchored at ``panda_link0`` (see
    ``FRANKA_*_BASE_PATH``) so they observe the same numerical base
    pose as the legacy setup.
    """
    arm_path = _ARM_PRIM_PATHS[side]
    target_world = ROBOT_BASE_WORLD_POSITIONS[side]
    wrapper_translate = tuple(
        target_world[i] - PANDA_LINK0_INTERNAL_OFFSET[i] for i in range(3)
    )

    xform = UsdGeom.Xform.Define(stage, arm_path)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(
        assetPath=str(ROBOT_ASSET_PATH),
        primPath=Sdf.Path("/Root"),
    )
    _override_translate(prim, wrapper_translate)


def _disable_robot_collisions(stage: Usd.Stage, arm_paths: tuple[str, ...]) -> None:
    """Disable every PhysX collision interaction the robot can take part in.

    Walking ``Usd.PrimRange`` and clearing ``CollisionAPI`` doesn't work for
    this asset because each link's collision meshes live inside instanceable
    Xforms that reference ``/Flattened_Prototype_*`` — instances are
    read-only from a referencing stage, so the ``CollisionAPI`` on the leaf
    meshes is unreachable. Two knobs that DO reach inside instances:

    1. ``physxArticulation:enabledSelfCollisions = False`` on each
       articulation root — kills finger-vs-finger and link-vs-link
       contacts (the symptom the user observed: hand jitter mid-replay).
    2. ``UsdPhysics.FilteredPairsAPI`` on each wrapper xform, targeting
       the table, ground plane, and (in dual mode) the other arm — kills
       external contacts so the teleported pose can't push against the
       static scene either.
    """
    env_targets = (TABLE_PRIM_PATH, "/World/GroundPlane")
    for arm_path in arm_paths:
        prim = stage.GetPrimAtPath(arm_path)
        prim.CreateAttribute(
            "physxArticulation:enabledSelfCollisions",
            Sdf.ValueTypeNames.Bool,
        ).Set(False)

        pair_api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = pair_api.CreateFilteredPairsRel()
        for target in env_targets:
            rel.AddTarget(Sdf.Path(target))
        for other in arm_paths:
            if other != arm_path:
                rel.AddTarget(Sdf.Path(other))


# ── Xform helpers ────────────────────────────────────────────────────────────


def _set_translate_scale(prim: Usd.Prim, translate, scale) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _override_translate(prim: Usd.Prim, translate: Iterable[float]) -> None:
    """Set the local translate value on a referenced prim without disturbing
    the inherited xformOpOrder (the reference brings translate/orient/scale
    ops; we only want to override the translate value)."""
    xformable = UsdGeom.Xformable(prim)
    translate_op = next(
        (op for op in xformable.GetOrderedXformOps()
         if op.GetOpName() == "xformOp:translate"),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translate))
