"""Inverse kinematics solver creation, solving, and position-setter factory.

Depends on rotation.py and Isaac Sim Lula solver.
Must be imported after SimulationApp is created.
"""

from pathlib import Path

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

from .rotation import tool_quat_to_urdf, wxyz_to_rotation_matrix


def create_ik_solver(
    urdf_path: Path, descriptor_path: Path, label: str
) -> LulaKinematicsSolver:
    """Create a Lula IK solver for the FER arm using the given URDF."""
    solver = LulaKinematicsSolver(
        robot_description_path=str(descriptor_path.resolve()),
        urdf_path=str(urdf_path.resolve()),
    )
    print(f"[IK] Solver ({label}) created. Active joints: {solver.get_joint_names()}")
    print(f"[IK] Available frames: {solver.get_all_frame_names()}")
    return solver


def solve_ik_for_pose(
    solver: LulaKinematicsSolver,
    ee_frame_name: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    warm_start: np.ndarray = None,
) -> tuple[np.ndarray | None, bool]:
    """Solve IK for a target wrist pose.

    Returns:
        (joint_positions, success) — joint_positions is None on failure.
    """
    joint_positions, success = solver.compute_inverse_kinematics(
        frame_name=ee_frame_name,
        target_position=position,
        target_orientation=orientation_wxyz,
        warm_start=warm_start,
    )
    return joint_positions if success else None, success


def make_ik_position_setter(
    art: SingleArticulation,
    arm_idx: np.ndarray,
    hand_idx: np.ndarray,
    ik_solver: LulaKinematicsSolver,
    ee_frame_name: str,
    hand_home: np.ndarray,
    home_arm_joints: np.ndarray,
    ee_wrist_offset: np.ndarray = None,
):
    """Return a callable that solves IK for each wrist pose and sets joint positions.

    Args:
        ee_wrist_offset: offset from ee_frame (e.g. panda_link8) to the EE wrist in
            the frame's local coordinates.  When provided, the IK target position
            is shifted so that the physical EE wrist lands on the requested pose:
            ``ik_position = ee_wrist_pos - R_frame @ ee_wrist_offset``.

    The returned callable has an attribute ``get_ik_failure_count()`` for stats.
    """
    base = art.get_joint_positions().copy()
    buf = base.copy()
    state = {
        "prev_arm_joints": home_arm_joints.copy(),
        "ik_failures": 0,
    }

    def set_positions(wrist_pose: np.ndarray, q_hand: np.ndarray):
        """Set joint positions from a wrist pose and hand joint angles.

        Args:
            wrist_pose: [x, y, z, qw, qx, qy, qz] — EE wrist pose in tool
                        convention (identity=down). Quaternion is converted to
                        URDF convention via Rx(180°) before IK.  If
                        ``ee_wrist_offset`` was supplied at construction time,
                        the IK target is shifted from the EE wrist to the
                        ee_frame origin (e.g. panda_link8).
            q_hand: hand joint angle offsets (added to hand_home).
        """
        position = wrist_pose[:3]
        orientation_wxyz = tool_quat_to_urdf(wrist_pose[3:])

        if ee_wrist_offset is not None:
            R = wxyz_to_rotation_matrix(orientation_wxyz)
            ik_position = position - R @ ee_wrist_offset
        else:
            ik_position = position

        arm_joints, _ = solve_ik_for_pose(
            ik_solver, ee_frame_name, ik_position, orientation_wxyz,
            warm_start=state["prev_arm_joints"],
        )

        buf[:] = base
        if arm_joints is not None:
            buf[arm_idx] = arm_joints
            state["prev_arm_joints"] = arm_joints.copy()
        else:
            state["ik_failures"] += 1
            buf[arm_idx] = state["prev_arm_joints"]

        buf[hand_idx] = hand_home + q_hand
        art.set_joint_positions(buf)

    def get_ik_failure_count() -> int:
        return state["ik_failures"]

    set_positions.get_ik_failure_count = get_ik_failure_count
    return set_positions
