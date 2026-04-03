import argparse

from utils.app import add_common_args, create_app, resolve_usd_path, resolve_h5_path
# constants.py has no Isaac Sim deps — safe to import before SimulationApp.
from utils.constants import CAMERA_CONFIGS, H5_IMAGE_PATHS, H5_DEFAULT_CAMERA
from utils.h5_loader import get_h5_image_dims

parser = argparse.ArgumentParser(description="Replay H5 trajectory with IK")
add_common_args(parser)
parser.add_argument(
    "--use-actions", action="store_true",
    help="Use actions_* instead of observations/qpos_* for replay",
)
parser.add_argument(
    "--no-collision", action="store_true",
    help="Disable collision between robots and the table (/World/Cube)",
)
parser.add_argument(
    "--camera", type=str, default=None, choices=list(CAMERA_CONFIGS.keys()),
    help="Set viewport to a calibrated camera (default: use Isaac Sim default viewport)",
)

# ── Recording flags (independent, combinable) ───────────────────────────────
parser.add_argument(
    "--record-sim", action="store_true",
    help="Record Isaac Sim viewport to MP4",
)
parser.add_argument(
    "--record-comparison", action="store_true",
    help="Record side-by-side comparison (Isaac Sim left, H5 original right)",
)
parser.add_argument(
    "--h5-camera", type=str, default=H5_DEFAULT_CAMERA,
    choices=list(H5_IMAGE_PATHS.keys()),
    help=f"Which H5 camera to use for --record-comparison (default: {H5_DEFAULT_CAMERA})",
)
parser.add_argument(
    "--no-fast-record", action="store_true",
    help="Disable deferred encoding (slower but uses less memory)",
)
parser.add_argument(
    "--sim-resolution", type=str, default=None,
    help="Override Isaac Sim render resolution as WxH (default: match H5 image dims)",
)
args = parser.parse_args()

# ── Resolve app resolution ──────────────────────────────────────────────────
# Default: derive from H5 image dims so sim output matches H5 video.
if args.sim_resolution is not None:
    _parts = args.sim_resolution.lower().split("x")
    APP_WIDTH, APP_HEIGHT = int(_parts[0]), int(_parts[1])
else:
    h5_path_for_dims = resolve_h5_path(args.mode)
    _w, _h = get_h5_image_dims(h5_path_for_dims, args.h5_camera)
    if _w is not None:
        APP_WIDTH, APP_HEIGHT = _w, _h
    else:
        APP_WIDTH, APP_HEIGHT = 640, 480

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
from utils.capture import (
    setup_capture, capture_frame_to_writer, close_recorder,
    setup_sidebyside, capture_sidebyside_frame, close_sidebyside,
)  # H5 video export lives in record_h5.py (no Isaac Sim needed)
from utils.camera import setup_camera


def _build_video_suffix(args) -> str:
    """Build a descriptive suffix from the active flags.

    Examples:
        actions_nocol       --use-actions --no-collision
        qpos                (defaults)
        qpos_nocol          --no-collision
        actions             --use-actions
    """
    parts = []
    parts.append("actions" if args.use_actions else "qpos")
    if args.no_collision:
        parts.append("nocol")
    return "_".join(parts)


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

    # ── Video capture setup ─────────────────────────────────────────────────
    hold_frames = max(1, int(round(HOME_HOLD_SECONDS * args.fps)))
    total_capture_frames = hold_frames + n_frames
    needs_viewport_capture = args.record_sim or args.record_comparison
    deferred = not args.no_fast_record
    suffix = _build_video_suffix(args)

    recorder, sim_output_path = (None, None)
    if needs_viewport_capture:
        video_name = f"{h5_path.stem}_replay_{suffix}.mp4"
        recorder, sim_output_path = setup_capture(
            total_capture_frames, OUTPUT_DIR / video_name,
            args.fps, APP_WIDTH, APP_HEIGHT, deferred=deferred,
        )
    captured_frames = 0

    sbs_recorder, sbs_output_path = (None, None)
    if args.record_comparison:
        sbs_name = f"{h5_path.stem}_comparison_{suffix}.mp4"
        sbs_recorder, sbs_output_path = setup_sidebyside(
            total_capture_frames, OUTPUT_DIR / sbs_name,
            args.fps, APP_WIDTH, APP_HEIGHT, h5_path, args.h5_camera,
        )

    if not needs_viewport_capture:
        print("[capture] No recording flags set.")

    # Hold home pose.
    print(f"[replay] Holding home pose for {HOME_HOLD_SECONDS:.1f}s ({hold_frames} frames)")
    for _ in range(hold_frames):
        if not simulation_app.is_running():
            close_recorder(recorder)
            close_sidebyside(sbs_recorder)
            simulation_app.close()
            return
        world.step(render=True)
        if recorder is not None and capture_frame_to_writer(recorder, simulation_app):
            captured_frames += 1
            if sbs_recorder is not None:
                capture_sidebyside_frame(sbs_recorder, recorder["last_frame"], 0)

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
                if sbs_recorder is not None:
                    capture_sidebyside_frame(sbs_recorder, recorder["last_frame"], frame_idx)

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

    # Finalize recordings.
    close_recorder(recorder)
    if recorder is not None and args.record_sim:
        print(f"[capture] Saved sim replay to {sim_output_path} ({captured_frames} frames)")
    elif recorder is not None and not args.record_sim:
        # Viewport was captured for comparison only; remove standalone file.
        import os
        try:
            os.remove(sim_output_path)
        except OSError:
            pass

    close_sidebyside(sbs_recorder)
    if sbs_recorder is not None:
        print(f"[capture] Saved comparison to {sbs_output_path} "
              f"({sbs_recorder['frames_written']} frames)")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
