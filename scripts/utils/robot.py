"""Robot articulation setup, DOF resolution, and collision control.

Imports Isaac Sim types — must be imported after SimulationApp is created.
"""

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

from .constants import (
    ARM_CONFIGS, ARM_JOINT_NAMES, HAND_HOME_JOINT_VALUES,
    LULA_DESCRIPTOR_PATH,
    WRIST_HOME_POSITION, WRIST_HOME_ROTATION, EE_WRIST_OFFSET_IN_LINK8,
)
from .rotation import rotation_matrix_to_wxyz
from .ik import create_ik_solver, solve_ik_for_pose, make_ik_position_setter


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
    """Create a robot articulation from a USD prim path and add it to the world."""
    name = prim_path.lstrip("/").replace("/", "_")
    art = SingleArticulation(prim_path=prim_path, name=name)
    world.scene.add(art)
    return art


def print_dof_info(label: str, art: SingleArticulation) -> None:
    """Print DOF names and indices for debugging."""
    print(f"\n[DOF] {label}: {art.num_dof} DOFs")
    for i, name in enumerate(art.dof_names):
        print(f"      [{i:2d}] {name}")


def resolve_dof_indices(
    art: SingleArticulation, names: list[str], label: str
) -> np.ndarray:
    """Map canonical joint names to articulation DOF indices."""
    dof_names = list(art.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    indices = []
    for name in names:
        if name not in name_to_idx:
            raise RuntimeError(
                f"[DOF] Cannot find '{name}' in {label} DOFs: {dof_names}"
            )
        indices.append(name_to_idx[name])
    return np.array(indices, dtype=int)


def add_articulations(world: World, mode: str) -> dict:
    """Add articulations to the world (must be called before world.reset()).

    Returns ``{side: {"articulation": art, "config": cfg}}``.
    """
    sides = ["right", "left"] if mode == "dual" else ["right"]
    arms = {}
    for side in sides:
        cfg = ARM_CONFIGS[side]
        art = setup_articulation(cfg["prim_path"], world)
        arms[side] = {"articulation": art, "config": cfg}
    return arms


def setup_arms_ik(arms: dict) -> None:
    """Create IK solvers, resolve DOFs, set home pose, build position setters.

    Must be called after ``world.reset()``.
    Modifies ``arms`` in-place, adding ``"set_positions"`` to each entry.
    """
    home_wrist_quat = rotation_matrix_to_wxyz(WRIST_HOME_ROTATION)
    home_link8_pos = WRIST_HOME_POSITION - WRIST_HOME_ROTATION @ EE_WRIST_OFFSET_IN_LINK8

    for side, arm in arms.items():
        cfg = arm["config"]
        art = arm["articulation"]

        ik_solver = create_ik_solver(cfg["urdf_path"], LULA_DESCRIPTOR_PATH, side)

        home_arm_joints, _ = solve_ik_for_pose(
            ik_solver, cfg["ee_frame"],
            home_link8_pos, home_wrist_quat,
        )
        if home_arm_joints is None:
            raise RuntimeError(
                f"IK failed for {side} arm home wrist pose "
                f"(link8_pos={home_link8_pos}, quat={home_wrist_quat}). "
                f"Check ee_frame='{cfg['ee_frame']}' and the Lula descriptor."
            )
        print(f"[IK] Home arm joints {side.upper():5s} (rad): {home_arm_joints}")

        print_dof_info(f"franka_{side}", art)
        arm_idx = resolve_dof_indices(art, ARM_JOINT_NAMES, f"franka_{side}")
        hand_idx = resolve_dof_indices(art, cfg["hand_joint_names"], f"franka_{side}")

        # Set home pose.
        q_home = art.get_joint_positions().copy()
        q_home[arm_idx] = home_arm_joints
        q_home[hand_idx] = HAND_HOME_JOINT_VALUES
        art.set_joint_positions(q_home)

        # Build per-frame position setter.
        arm["set_positions"] = make_ik_position_setter(
            art, arm_idx, hand_idx,
            ik_solver, cfg["ee_frame"],
            HAND_HOME_JOINT_VALUES, home_arm_joints,
            ee_wrist_offset=EE_WRIST_OFFSET_IN_LINK8,
        )
