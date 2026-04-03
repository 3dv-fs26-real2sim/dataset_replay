import argparse

from utils.app import add_common_args, create_app, resolve_usd_path, resolve_h5_path
# constants.py has no Isaac Sim deps — safe to import before SimulationApp.
from utils.constants import CAMERA_CONFIGS

parser = argparse.ArgumentParser(description="Replay H5 trajectory with IK")
add_common_args(parser)
parser.add_argument(
    "--use-actions", action="store_true",
    help="Use actions_* instead of observations/qpos_* for replay",
)
#* Recording works, but VERY slow.
parser.add_argument(
    "--record", action=argparse.BooleanOptionalAction, default=False,
    help="Record replay video to outputs/H5NAME_replay.mp4 (default: False)",
)
parser.add_argument(
    "--no-collision", action="store_true",
    help="Disable collision between robots and the table (/World/Cube)",
)
parser.add_argument(
    "--camera", type=str, default=None, choices=list(CAMERA_CONFIGS.keys()),
    help="Set viewport to a calibrated camera (default: use Isaac Sim default viewport)",
)
args = parser.parse_args()

# Match the app/capture resolution to the camera's aspect ratio so the
# recorded video isn't stretched.  Target ~720 px on the shorter side.
_TARGET_SHORT = 720
if args.camera is not None:
    _intr = CAMERA_CONFIGS[args.camera]["intrinsics"]
    _cam_w, _cam_h = _intr["width"], _intr["height"]
    if _cam_w >= _cam_h:
        APP_HEIGHT = _TARGET_SHORT
        APP_WIDTH  = round(_TARGET_SHORT * _cam_w / _cam_h)
    else:
        APP_WIDTH  = _TARGET_SHORT
        APP_HEIGHT = round(_TARGET_SHORT * _cam_h / _cam_w)
else:
    APP_WIDTH  = 1280
    APP_HEIGHT = 720
simulation_app = create_app(args, width=APP_WIDTH, height=APP_HEIGHT)

# Isaac Sim imports must come after SimulationApp creation.
import numpy as np
import omni.usd
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    ARM_JOINT_NAMES, HAND_LEFT_JOINT_NAMES, HAND_RIGHT_JOINT_NAMES,
    HAND_HOME_JOINT_VALUES, HOME_HOLD_SECONDS, OUTPUT_DIR,
    EE_FRAME_NAME_LEFT, EE_FRAME_NAME_RIGHT,
    LULA_DESCRIPTOR_PATH, URDF_PATH_LEFT, URDF_PATH_RIGHT,
    WRIST_HOME_POSITION, WRIST_HOME_ROTATION,
    EE_WRIST_OFFSET_IN_LINK8,
)
from utils.rotation import rotation_matrix_to_wxyz, detect_quaternion_order
from utils.h5_loader import load_h5
from utils.robot import setup_articulation, resolve_dof_indices, print_dof_info, set_collision_enabled
from utils.ik import create_ik_solver, solve_ik_for_pose, make_ik_position_setter
from utils.capture import setup_capture, capture_frame_to_writer, close_recorder
from utils.camera import setup_camera


def main():
    h5_path = resolve_h5_path(args.mode)
    data = load_h5(h5_path, args.use_actions, args.mode)
    n_frames = data["n_frames"]

    # Auto-detect quaternion order and reorder to wxyz (scalar-first).
    data["arm_right"] = detect_quaternion_order(data["arm_right"], "arm_right")
    if data["arm_left"] is not None:
        data["arm_left"] = detect_quaternion_order(data["arm_left"], "arm_left")

    # First frame diagnostics.
    sample = data["arm_right"][0]
    print(f"[debug] First frame arm_right: {sample}")
    print(f"[debug]   Position (xyz): {sample[:3]}")
    print(f"[debug]   Quaternion (wxyz): {sample[3:]}")
    print(f"[debug]   Quaternion norm: {np.linalg.norm(sample[3:]):.4f}")

    # Load USD scene.
    omni.usd.get_context().open_stage(str(resolve_usd_path(args.mode)))

    world = World()
    franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
    if args.mode == "dual":
        franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
    world.reset()

    stage = omni.usd.get_context().get_stage()

    if args.no_collision:
        set_collision_enabled(stage, "/World/Cube", False)

    if args.camera is not None:
        camera_prim_path = setup_camera(
            stage, args.camera, args.mode,
            FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
        )
        print(f"[camera] Viewport set to {camera_prim_path}")

    # Create per-side IK solvers.
    ik_solver_right = create_ik_solver(URDF_PATH_RIGHT, LULA_DESCRIPTOR_PATH, "right")
    if args.mode == "dual":
        ik_solver_left = create_ik_solver(URDF_PATH_LEFT, LULA_DESCRIPTOR_PATH, "left")

    # Compute home arm joint values via IK.
    # WRIST_HOME_POSITION is the desired EE wrist location; shift to fer_link8 frame.
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

    # Create position setters with IK warm-start tracking.
    set_right = make_ik_position_setter(
        franka_right, arm_idx_right, hand_idx_right,
        ik_solver_right, EE_FRAME_NAME_RIGHT,
        HAND_HOME_JOINT_VALUES, home_arm_joints,
        ee_wrist_offset=EE_WRIST_OFFSET_IN_LINK8,
    )
    if args.mode == "dual":
        set_left = make_ik_position_setter(
            franka_left, arm_idx_left, hand_idx_left,
            ik_solver_left, EE_FRAME_NAME_LEFT,
            HAND_HOME_JOINT_VALUES, home_arm_joints_left,
            ee_wrist_offset=EE_WRIST_OFFSET_IN_LINK8,
        )

    # Video capture setup.
    hold_frames = max(1, int(round(HOME_HOLD_SECONDS * args.fps)))
    total_capture_frames = hold_frames + n_frames
    recorder, output_path = (None, None)
    if args.record:
        video_name = f"{h5_path.stem}_replay.mp4"
        recorder, output_path = setup_capture(
            total_capture_frames, OUTPUT_DIR / video_name,
            args.fps, APP_WIDTH, APP_HEIGHT,
        )
    else:
        print("[capture] Recording disabled by --no-record")
    captured_frames = 0

    # Hold home pose.
    print(f"[replay] Holding home pose for {HOME_HOLD_SECONDS:.1f}s ({hold_frames} frames)")
    for _ in range(hold_frames):
        if not simulation_app.is_running():
            close_recorder(recorder)
            simulation_app.close()
            return
        world.step(render=True)
        if recorder is not None and capture_frame_to_writer(recorder, simulation_app):
            captured_frames += 1

    # Replay loop.
    print(f"\n[replay] Starting {n_frames} frames at {args.fps} fps...  (Ctrl-C to stop)\n")
    try:
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                break

            if args.mode == "dual":
                set_left(data["arm_left"][frame_idx], data["hand_left"][frame_idx])
            set_right(data["arm_right"][frame_idx], data["hand_right"][frame_idx])

            world.step(render=True)
            if recorder is not None and capture_frame_to_writer(recorder, simulation_app):
                captured_frames += 1

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / n_frames
                print(f"  frame {frame_idx:5d}/{n_frames}  ({pct:.1f}%)")
    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.")

    # Report IK failure stats.
    ik_failures_right = set_right.get_ik_failure_count()
    if args.mode == "dual":
        ik_failures_left = set_left.get_ik_failure_count()
        print(f"[IK] Failures: right={ik_failures_right}/{n_frames}, "
              f"left={ik_failures_left}/{n_frames}")
    else:
        print(f"[IK] Failures: {ik_failures_right}/{n_frames}")

    close_recorder(recorder)
    if recorder is not None:
        print(f"[capture] Saved replay video to {output_path} ({captured_frames} frames)")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
