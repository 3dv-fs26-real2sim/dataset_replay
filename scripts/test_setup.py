"""Test robot setup — load scene, optionally spawn an object, hold IK home pose.

Usage:
    python dataset_replay/scripts/test_setup.py
    python dataset_replay/scripts/test_setup.py --mode dual --camera aria
    python dataset_replay/scripts/test_setup.py --object none
"""

import argparse

from utils.app import add_common_args, create_app, resolve_usd_path
from utils.constants import CAMERA_CONFIGS, OBJECT_CHOICES, OBJECT_DEFAULT_SCALE

parser = argparse.ArgumentParser(
    description="Test robot setup — load scene, optionally spawn object, hold IK home pose",
)
add_common_args(parser)
parser.set_defaults(mode="single")
parser.add_argument(
    "--camera", type=str, default=None, choices=list(CAMERA_CONFIGS.keys()),
    help="Set viewport to a calibrated camera (default: use Isaac Sim default viewport)",
)
parser.add_argument(
    "--object", type=str, default="duck", choices=OBJECT_CHOICES + ["none"],
    help="Object to spawn (default: duck). Use 'none' to skip object spawning.",
)
parser.add_argument(
    "--position", type=float, nargs=3, default=[0.0, 0.0, 1.0],
    metavar=("X", "Y", "Z"),
    help="Object spawn position in meters (default: 0.0 0.0 1.0)",
)
parser.add_argument(
    "--scale", type=float, default=OBJECT_DEFAULT_SCALE,
    help=f"Object uniform scale factor (default: {OBJECT_DEFAULT_SCALE})",
)
args = parser.parse_args()

simulation_app = create_app(args)

# Isaac Sim imports must come after SimulationApp creation.
import omni.usd
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    ARM_JOINT_NAMES, HAND_LEFT_JOINT_NAMES, HAND_RIGHT_JOINT_NAMES,
    HAND_HOME_JOINT_VALUES,
    EE_FRAME_NAME_LEFT, EE_FRAME_NAME_RIGHT,
    LULA_DESCRIPTOR_PATH, URDF_PATH_LEFT, URDF_PATH_RIGHT,
    WRIST_HOME_POSITION, WRIST_HOME_ROTATION,
    EE_WRIST_OFFSET_IN_LINK8,
)
from utils.rotation import rotation_matrix_to_wxyz
from utils.robot import setup_articulation, resolve_dof_indices, print_dof_info
from utils.ik import create_ik_solver, solve_ik_for_pose
from utils.camera import setup_camera
from utils.object import spawn_object
from utils.constants import OBJECTS_DIR


def main():
    omni.usd.get_context().open_stage(str(resolve_usd_path(args.mode)))

    world = World()
    franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
    if args.mode == "dual":
        franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
    world.reset()

    stage = omni.usd.get_context().get_stage()

    if args.camera is not None:
        camera_prim_path = setup_camera(
            stage, args.camera, args.mode,
            FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
        )
        print(f"[camera] Viewport set to {camera_prim_path}")

    # Create IK solvers.
    ik_solver_right = create_ik_solver(URDF_PATH_RIGHT, LULA_DESCRIPTOR_PATH, "right")
    if args.mode == "dual":
        ik_solver_left = create_ik_solver(URDF_PATH_LEFT, LULA_DESCRIPTOR_PATH, "left")

    # Compute home arm joint values via IK.
    home_wrist_quat = rotation_matrix_to_wxyz(WRIST_HOME_ROTATION)
    home_fer_link8_pos = WRIST_HOME_POSITION - WRIST_HOME_ROTATION @ EE_WRIST_OFFSET_IN_LINK8
    home_arm_joints, _ = solve_ik_for_pose(
        ik_solver_right, EE_FRAME_NAME_RIGHT,
        home_fer_link8_pos, home_wrist_quat,
    )
    if home_arm_joints is None:
        raise RuntimeError(
            f"IK failed for home wrist pose "
            f"(fer_link8_pos={home_fer_link8_pos}, quat={home_wrist_quat}). "
            f"Check EE_FRAME_NAME_RIGHT='{EE_FRAME_NAME_RIGHT}' and the Lula descriptor."
        )
    print(f"[IK] Home arm joints RIGHT (rad): {home_arm_joints}")

    if args.mode == "dual":
        home_arm_joints_left, _ = solve_ik_for_pose(
            ik_solver_left, EE_FRAME_NAME_LEFT,
            home_fer_link8_pos, home_wrist_quat,
        )
        if home_arm_joints_left is None:
            raise RuntimeError(
                f"IK failed for left arm home wrist pose "
                f"(fer_link8_pos={home_fer_link8_pos}, quat={home_wrist_quat}). "
                f"Check EE_FRAME_NAME_LEFT='{EE_FRAME_NAME_LEFT}' and the Lula descriptor."
            )
        print(f"[IK] Home arm joints LEFT  (rad): {home_arm_joints_left}")

    # Resolve DOF indices.
    print_dof_info("franka_right", franka_right)
    arm_idx_right = resolve_dof_indices(franka_right, ARM_JOINT_NAMES, "franka_right")
    hand_idx_right = resolve_dof_indices(franka_right, HAND_RIGHT_JOINT_NAMES, "franka_right")
    if args.mode == "dual":
        print_dof_info("franka_left", franka_left)
        arm_idx_left = resolve_dof_indices(franka_left, ARM_JOINT_NAMES, "franka_left")
        hand_idx_left = resolve_dof_indices(franka_left, HAND_LEFT_JOINT_NAMES, "franka_left")

    # Set home pose.
    q_home_right = franka_right.get_joint_positions().copy()
    q_home_right[arm_idx_right] = home_arm_joints
    q_home_right[hand_idx_right] = HAND_HOME_JOINT_VALUES
    franka_right.set_joint_positions(q_home_right)
    if args.mode == "dual":
        q_home_left = franka_left.get_joint_positions().copy()
        q_home_left[arm_idx_left] = home_arm_joints_left
        q_home_left[hand_idx_left] = HAND_HOME_JOINT_VALUES
        franka_left.set_joint_positions(q_home_left)

    # Spawn object.
    if args.object != "none":
        spawned = spawn_object(stage, args.object, args.position, args.scale, OBJECTS_DIR)
        print(f"[setup] Spawned '{args.object}' at {args.position} scale {args.scale} -> {spawned}")

    # Hold home pose until window is closed.
    print("[setup] Holding home pose (close window to exit)")
    while simulation_app.is_running():
        world.step(render=True)

    print("[setup] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
