"""Programmatic scene construction (single-arm, Aria, procedural — egoverse).

Builds the simulation stage from scratch: Z-up, metric, with a ground plane,
two side-by-side tables, and the right-arm Panda + OrcaHand referenced from
the orcav1b USD. No walls, no AprilTag — the egoverse capture is an open
desk in front of an Aria wearer.

Depends on ``pxr`` and ``omni.usd`` — must be imported after
``SimulationApp`` is created.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from .config import SceneConfig
from .constants import ROBOT_PRIM_PATH
from .textures import bind_image_texture, define_box_mesh, set_box_planar_uvs


# ── Public API ───────────────────────────────────────────────────────────────
def build_scene(cfg: SceneConfig, *, robot_collision: bool = False) -> Usd.Stage:
    """Create a fresh stage and populate it from ``cfg``.

    With ``robot_collision=False`` (the default for kinematic replay), the
    robot's self-collisions and its filtered pairs against table/ground are
    disabled. PhysX therefore tracks the teleported joint positions without
    integrating contact forces — the robot does not jitter when the replayed
    pose intersects scene geometry.

    Returns the active ``Usd.Stage``.
    """
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    _set_stage_metadata(stage)
    _define_world(stage)
    _add_physics_scene(stage)
    _add_ground_plane(stage)
    _add_lighting(stage)
    _add_tables(stage, cfg)
    _add_robot(stage, cfg)

    if not robot_collision:
        _disable_robot_collisions(stage)

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
    """Visual-quad + collision-plane at z = 0. Floor extends well past the table."""
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


def _add_lighting(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, "/Environment")
    light = UsdLux.DistantLight.Define(stage, "/Environment/defaultLight")
    light.CreateAngleAttr(1.0)
    light.CreateIntensityAttr(3000.0)


def _add_tables(stage: Usd.Stage, cfg: SceneConfig) -> None:
    """Build ``cfg.table.n_tables`` cuboid table cells side-by-side along Y.

    Each cell is its own prim under ``/World/Tables`` so the wood material
    can be bound independently per-cell — useful if the user later wants
    different finishes per table.
    """
    UsdGeom.Xform.Define(stage, "/World/Tables")

    Lx, Ly = cfg.table.single_size_xy
    Lz     = cfg.table.top_thickness
    z_top  = cfg.table.top_z
    cz     = z_top - Lz / 2

    # Tile along Y starting at +Y (left) → -Y (right).
    y_lo = cfg.table.y_extent[0]
    for i in range(cfg.table.n_tables):
        cy = y_lo + Ly * (i + 0.5)
        prim_path = f"/World/Tables/Cell_{i}"
        mesh = define_box_mesh(
            stage, prim_path,
            size_xyz=(Lx, Ly, Lz),
            centre_xyz=(cfg.table.centre_xy[0], cy, cz),
            display_color=(0.55, 0.40, 0.25),
        )
        set_box_planar_uvs(
            mesh, extent_xyz=(Lx, Ly, Lz),
            uv_repeat=cfg.table.uv_repeat,
        )
        if cfg.table.texture_path.exists():
            bind_image_texture(stage, prim_path, cfg.table.texture_path,
                               material_name="wood_table_mat")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        col = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        col.CreateApproximationAttr("convexHull")


def _add_robot(stage: Usd.Stage, cfg: SceneConfig) -> None:
    """Reference the orcav1b USD as ``/World/Robot`` with the wrapper transform
    that places ``panda_link0`` at ``cfg.robot.mount_xyz``.

    The orcav1b USD's ``panda_link0`` has a non-trivial local translate
    inside ``/Root`` (≈``z = -0.476``). The wrapper's translate compensates so
    the visible base lands exactly at ``mount_xyz`` rather than ~0.5 m above
    the table.
    """
    target_world = np.asarray(cfg.robot.mount_xyz, dtype=float)
    internal     = np.asarray(cfg.robot.panda_link0_local_translate, dtype=float)
    wrapper_translate = tuple((target_world - internal).tolist())

    xform = UsdGeom.Xform.Define(stage, ROBOT_PRIM_PATH)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(
        assetPath=str(cfg.robot_asset_path),
        primPath=Sdf.Path("/Root"),
    )

    # Set the wrapper's local translate. mount_rpy is currently identity by
    # default; if you set non-zero RPY in the config, extend this to add a
    # rotation op too (the existing default identity orientation is fine for
    # the new minimal setup).
    if any(r != 0.0 for r in cfg.robot.mount_rpy):
        # Build a quaternion (wxyz) from RPY (XYZ extrinsic).
        from scipy.spatial.transform import Rotation
        q = Rotation.from_euler("XYZ", cfg.robot.mount_rpy).as_quat()  # xyzw
        q_wxyz = (q[3], q[0], q[1], q[2])
        _override_translate_orient(prim, wrapper_translate, q_wxyz)
    else:
        _override_translate(prim, wrapper_translate)


def _disable_robot_collisions(stage: Usd.Stage) -> None:
    """Disable PhysX self-collisions on the articulation and filter against
    static scene geometry (tables + ground). Standard kinematic-replay setup.

    Walking ``Usd.PrimRange`` and clearing ``CollisionAPI`` doesn't work
    against the orcav1b USD's instanceable Xforms; the two knobs that DO
    reach inside instances are:

    1. ``physxArticulation:enabledSelfCollisions = False`` on the articulation
       root (the wrapper Xform that holds the reference) — kills
       finger-vs-finger and link-vs-link contacts.
    2. ``UsdPhysics.FilteredPairsAPI`` on the articulation root, targeting
       the table cells and the ground plane.
    """
    prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    prim.CreateAttribute(
        "physxArticulation:enabledSelfCollisions",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)

    pair_api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = pair_api.CreateFilteredPairsRel()
    rel.AddTarget(Sdf.Path("/World/GroundPlane"))
    # Filter against every table cell.
    tables_root = stage.GetPrimAtPath("/World/Tables")
    if tables_root.IsValid():
        for child in tables_root.GetChildren():
            rel.AddTarget(child.GetPath())


# ── Xform helpers ────────────────────────────────────────────────────────────
def _override_translate(prim: Usd.Prim, translate: Iterable[float]) -> None:
    """Set the local translate on a referenced prim without disturbing the
    inherited xformOpOrder (the reference brings translate/orient/scale ops;
    we only want to override the translate value)."""
    xformable = UsdGeom.Xformable(prim)
    translate_op = next(
        (op for op in xformable.GetOrderedXformOps()
         if op.GetOpName() == "xformOp:translate"),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translate))


def _override_translate_orient(
    prim: Usd.Prim, translate: Iterable[float], orient_wxyz: Iterable[float]
) -> None:
    """Same as ``_override_translate``, plus a quaternion orient op."""
    xformable = UsdGeom.Xformable(prim)
    translate_op = next(
        (op for op in xformable.GetOrderedXformOps()
         if op.GetOpName() == "xformOp:translate"),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translate))

    orient_op = next(
        (op for op in xformable.GetOrderedXformOps()
         if op.GetOpName() == "xformOp:orient"),
        None,
    )
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    w, x, y, z = orient_wxyz
    orient_op.Set(Gf.Quatf(w, x, y, z))
