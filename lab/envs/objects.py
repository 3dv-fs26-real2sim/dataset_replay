"""Spawn / RigidObject configs for the manipulated objects and static props.

Object meshes live under ``dataset_replay/assets/objects/<name>/``. The grasped
objects use convex-DECOMPOSITION (VHACD) colliders so the hand can actually
enclose them (a single convex hull is bloated and ungraspable). Artec-scanned
meshes (duck/pan) are authored in millimetres → spawn at scale 0.001.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from utils.config import ASSETS

_OBJ = ASSETS / "objects"

# Shared rigid-body solver/contact settings tuned for a stable in-hand grasp.
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=12,
    solver_velocity_iteration_count=4,
    sleep_threshold=0.005,
    stabilization_threshold=0.0025,
)
_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True, contact_offset=0.005, rest_offset=0.0,
)


def duck_spawn_cfg() -> sim_utils.UsdFileCfg:
    """EgoVerse duck — VHACD collider, mm-scale mesh (→ ~12.5 cm duck)."""
    return sim_utils.UsdFileCfg(
        usd_path=str((_OBJ / "duck" / "duck_vhacd.usd").resolve()),
        scale=(0.001, 0.001, 0.001),
        rigid_props=_RIGID_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=_COLLISION_PROPS,
    )


def pan_spawn_cfg() -> sim_utils.UsdFileCfg:
    """MAPLE pan — textured VHACD collider, mm-scale mesh (→ ~26 cm pan)."""
    return sim_utils.UsdFileCfg(
        usd_path=str((_OBJ / "pan" / "pan_vhacd.usd").resolve()),
        scale=(0.001, 0.001, 0.001),
        rigid_props=_RIGID_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=_COLLISION_PROPS,
    )


def bowl_spawn_cfg(mass: float = 0.1, static: bool = False) -> sim_utils.UsdFileCfg:
    """Purple bowl — metre-scale VHACD collider (keeps the cavity open).

    ``static=False`` (default) → a light (~100 g) dynamic rigid body that can be
    knocked/moved; ``static=True`` → a fixed kinematic collider. Spawned only
    when a bowl pose is supplied (``attach_bowl``).
    """
    if static:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        return sim_utils.UsdFileCfg(
            usd_path=str((_OBJ / "bowl" / "bowl_vhacd.usd").resolve()),
            scale=(1.0, 1.0, 1.0),
            rigid_props=rigid_props,
            collision_props=_COLLISION_PROPS,
        )
    return sim_utils.UsdFileCfg(
        usd_path=str((_OBJ / "bowl" / "bowl_vhacd.usd").resolve()),
        scale=(1.0, 1.0, 1.0),
        rigid_props=_RIGID_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=_COLLISION_PROPS,
    )


def ball_spawn_cfg(radius: float = 0.048) -> sim_utils.SphereCfg:
    """Duck-sized analytic sphere (``--object ball``). radius 0.048 ≈ duck height."""
    return sim_utils.SphereCfg(
        radius=radius,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1), roughness=0.6),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0),
        rigid_props=_RIGID_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=_COLLISION_PROPS,
    )


def object_cfg(prim_path: str, spawn, init_pos, init_rot=(1.0, 0.0, 0.0, 0.0)) -> RigidObjectCfg:
    """A dynamic manipulated-object RigidObjectCfg (pose rewritten by reset event)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(init_pos), rot=tuple(init_rot)),
        spawn=spawn,
    )


def maple_prop_cfg(name: str, T_link0, mount_xyz) -> RigidObjectCfg:
    """A static (kinematic) MAPLE scene prop at its measured pose.

    ``T_link0`` is the prop's 4×4 pose in the panda_link0 frame, so its env world
    pose is ``T_link0 + mount_xyz`` — the same convention the demo loader uses for
    the object/wrist. ``kinematic_enabled`` makes it an immovable obstacle.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    T = np.asarray(T_link0, dtype=float).reshape(4, 4)
    pos = tuple(float(T[i, 3] + mount_xyz[i]) for i in range(3))
    qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Prop_" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(qw, qx, qy, qz)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str((_OBJ / name / f"{name}.usd").resolve()),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=_COLLISION_PROPS,
        ),
    )
