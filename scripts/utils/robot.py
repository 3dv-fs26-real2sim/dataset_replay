"""Single-arm Panda + 17-DOF OrcaHand articulation setup.

Imports Isaac Sim types — must be imported after SimulationApp is created.
"""

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

from .config import BaseSceneConfig
from .constants import (
    ARM_JOINT_NAMES, EE_FRAME_NAME, EE_WRIST_OFFSET_IN_LINK8,
    HAND_HOME_JOINT_VALUES, HAND_JOINT_NAMES, ROBOT_PRIM_PATH,
    WRIST_HOME_POSITION, WRIST_HOME_ROTATION,
)
from .ik import create_ik_solver, make_ik_position_setter, solve_ik_for_pose
from .rotation import rotation_matrix_to_wxyz


def setup_robot(world: World, cfg: BaseSceneConfig) -> dict:
    """Add the right-arm articulation to the world, resolve DOFs, set the
    home pose via IK, and build a per-frame position setter.

    Must be called in this order::

        world  = World()
        robot  = setup_robot(world, cfg)
        # ... after this returns the world is reset and the home pose is set.

    Returns a dict with keys:
      ``articulation``       -- the SingleArticulation
      ``arm_dof_indices``    -- (7,)  indices into get_joint_positions()
      ``hand_dof_indices``   -- (17,) indices into get_joint_positions()
      ``set_positions``      -- callable(wrist_pose, hand_q) -> None
      ``home_arm_joints``    -- (7,)  the IK-solved home joint values
    """
    # ── 1. Create articulation, register with world, reset.
    art = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="panda_orca_right")
    world.scene.add(art)
    world.reset()

    # ── 2. Build IK solver, solve home pose.
    ik_solver = create_ik_solver(cfg.urdf_path, cfg.lula_descriptor, "right")

    home_wrist_quat = rotation_matrix_to_wxyz(WRIST_HOME_ROTATION)
    home_link8_pos  = WRIST_HOME_POSITION - WRIST_HOME_ROTATION @ EE_WRIST_OFFSET_IN_LINK8

    home_arm_joints, ok = solve_ik_for_pose(
        ik_solver, EE_FRAME_NAME, home_link8_pos, home_wrist_quat,
    )
    if not ok or home_arm_joints is None:
        raise RuntimeError(
            f"IK failed for home wrist pose "
            f"(link8_pos={home_link8_pos}, quat={home_wrist_quat}). "
            f"Check the Lula descriptor at {cfg.lula_descriptor}."
        )
    print(f"[IK] Home arm joints (rad): {home_arm_joints}")

    # ── 3. Resolve DOF indices.
    print_dof_info(art)
    arm_idx  = _resolve_dof_indices(art, ARM_JOINT_NAMES,  "arm")
    hand_idx = _resolve_dof_indices(art, HAND_JOINT_NAMES, "hand")

    # ── 4. Set home pose on the live articulation.
    q_home = art.get_joint_positions().copy()
    q_home[arm_idx]  = home_arm_joints
    q_home[hand_idx] = HAND_HOME_JOINT_VALUES
    art.set_joint_positions(q_home)

    # ── 5. Build the per-frame position setter.
    set_positions = make_ik_position_setter(
        art, arm_idx, hand_idx,
        ik_solver, EE_FRAME_NAME,
        HAND_HOME_JOINT_VALUES, home_arm_joints,
        ee_wrist_offset=EE_WRIST_OFFSET_IN_LINK8,
    )

    return {
        "articulation":     art,
        "arm_dof_indices":  arm_idx,
        "hand_dof_indices": hand_idx,
        "set_positions":    set_positions,
        "home_arm_joints":  home_arm_joints,
    }


# ── Diagnostics ──────────────────────────────────────────────────────────────
def print_dof_info(art: SingleArticulation) -> None:
    """Print DOF names and indices for debugging."""
    print(f"\n[DOF] {art.name}: {art.num_dof} DOFs")
    for i, name in enumerate(art.dof_names):
        print(f"      [{i:2d}] {name}")


# ── Internal helpers ─────────────────────────────────────────────────────────
def _resolve_dof_indices(
    art: SingleArticulation, names: list[str], label: str
) -> np.ndarray:
    """Map canonical joint names to articulation DOF indices."""
    name_to_idx = {n: i for i, n in enumerate(art.dof_names)}
    indices = []
    for n in names:
        if n not in name_to_idx:
            raise RuntimeError(
                f"[DOF] Cannot find {n!r} in {label} DOFs. "
                f"Available: {list(art.dof_names)}"
            )
        indices.append(name_to_idx[n])
    return np.array(indices, dtype=int)
