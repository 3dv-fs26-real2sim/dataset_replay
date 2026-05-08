import argparse

from utils.app import add_common_args, create_app, resolve_h5_path
# constants.py / poses.py have no Isaac Sim deps — safe to import before SimulationApp.
from utils.constants import (
    CAMERA_CONFIGS, H5_IMAGE_PATHS, H5_DEFAULT_CAMERA,
    OBJECT_CHOICES, OBJECT_DEFAULT_SCALE,
    OBJECT_POSE_DEFAULT_CAMERA, OBJECT_POSE_PATHS, OBJECT_POSE_DEFAULT_VERSION,
)
from utils.h5_loader import get_h5_image_dims

parser = argparse.ArgumentParser(description="Kinematic replay of H5 trajectory with IK")
add_common_args(parser)
parser.set_defaults(mode="single")
parser.add_argument(
    "--use-actions", action="store_true",
    help="Use actions_* instead of observations/qpos_* for replay",
)
parser.add_argument(
    "--camera", type=str, default="aria", choices=list(CAMERA_CONFIGS.keys()),
    help="Set viewport to a calibrated camera (default: aria)",
)
parser.add_argument(
    "--refined-extrinsic", type=str, default=None,
    help="Path to an NPZ produced by scripts/refine_camera_extrinsic.py. "
         "When set, its 'T_world_cam' overrides the nominal camera pose "
         "computed from CAMERA_CONFIGS — applied to both the viewport "
         "camera and the object-trajectory world frame so they stay aligned.",
)

# ── Object spawning + 6D pose trajectory ────────────────────────────────────
parser.add_argument(
    "--object", type=str, default="duck", choices=OBJECT_CHOICES + ["none"],
    help="Object to spawn and animate via 6D pose trajectory (default: duck). "
         "Use 'none' to skip object spawning. Only honoured in --mode single.",
)
parser.add_argument(
    "--object-scale", type=float, default=OBJECT_DEFAULT_SCALE,
    help=f"Object uniform scale (default: {OBJECT_DEFAULT_SCALE})",
)
parser.add_argument(
    "--object-poses", type=str, default=None,
    help="Override path to object pose .npz (default: lookup OBJECT_POSE_PATHS for --object). "
         "Must contain a single (N, 4, 4) array of camera-frame transforms.",
)
parser.add_argument(
    "--object-pose-version", type=str, default=OBJECT_POSE_DEFAULT_VERSION,
    help=f"Pose-estimator version to load from OBJECT_POSE_PATHS[--object] "
         f"(default: {OBJECT_POSE_DEFAULT_VERSION}). Ignored when --object-poses is set. "
         f"Available per object: "
         + ", ".join(f"{k}=[{'/'.join(sorted(v.keys()))}]" for k, v in OBJECT_POSE_PATHS.items()),
)
parser.add_argument(
    "--object-pose-camera", type=str, default=OBJECT_POSE_DEFAULT_CAMERA,
    choices=list(CAMERA_CONFIGS.keys()),
    help=f"Camera frame the object pose .npz is expressed in (default: {OBJECT_POSE_DEFAULT_CAMERA}). "
         "The trajectory is transformed into world frame using this camera's extrinsics.",
)

# ── Recording flags (independent, combinable) ───────────────────────────────
parser.add_argument(
    "--record-sim", action="store_true",
    help="Record Isaac Sim viewport to MP4",
)
parser.add_argument(
    "--record-sidebyside", action="store_true",
    help="Record side-by-side comparison (Isaac Sim left, H5 original right)",
)
parser.add_argument(
    "--record-overlay", type=str, nargs="?", default=None, const="0.3,0.5,0.7",
    help="Record alpha-blended overlay of Isaac Sim and H5 video. Accepts a "
         "comma-separated list of sim-opacity values in [0, 1] (one output MP4 "
         "per value). Bare flag defaults to '0.3,0.5,0.7'.",
)
parser.add_argument(
    "--h5-camera", type=str, default=H5_DEFAULT_CAMERA,
    choices=list(H5_IMAGE_PATHS.keys()),
    help=f"Which H5 camera to use for --record-sidebyside / --record-overlay "
         f"(default: {H5_DEFAULT_CAMERA})",
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
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_BASE_PATH, FRANKA_RIGHT_BASE_PATH, OBJECTS_DIR,
)
from utils.rotation import detect_quaternion_order
from utils.h5_loader import load_h5
from utils.robot import add_articulations, setup_arms_ik
from utils.capture import (
    setup_recording, capture_frame_to_writer, close_recorder,
    capture_sidebyside_frame, close_sidebyside,
    capture_overlay_frame, close_overlay,
)
from utils.camera import setup_camera
from utils.object import load_object_world_trajectory, spawn_object, set_object_world_pose
from utils.scene import build_scene


# ── Helper functions ────────────────────────────────────────────────────────


def _build_video_suffix(args) -> str:
    """Build a descriptive suffix from the active flags.

    Examples:
        qpos                       (defaults)
        actions                    --use-actions
        qpos_duck_vda              --object duck (default version)
        qpos_duck_depthpro         --object-pose-version depthpro
        qpos_duck_<stem>           --object-poses /path/to/<stem>.npz
    """
    from pathlib import Path as _Path

    parts = []
    parts.append("actions" if args.use_actions else "qpos")
    if args.object != "none" and args.mode == "single":
        parts.append(args.object)
        if args.object_poses is not None:
            parts.append(_Path(args.object_poses).stem)
        else:
            parts.append(args.object_pose_version)
    return "_".join(parts)


def _setup_object_replay(args, stage, n_frames, T_world_cam_override=None):
    """Spawn the chosen object and prepare its world-frame pose trajectory.

    Returns ``(prim_path, traj_world)`` where ``traj_world`` is an ``(N, 4, 4)``
    array of world-frame transforms (one per H5 frame), or ``(None, None)`` if
    no object should be spawned.
    """
    traj_world = load_object_world_trajectory(
        stage, args.object, args.object_poses,
        args.object_pose_camera, n_frames,
        args.mode, FRANKA_RIGHT_BASE_PATH,
        object_pose_version=args.object_pose_version,
        T_world_cam_override=T_world_cam_override,
    )
    if traj_world is None:
        return None, None

    # Spawn at the trajectory's first pose.
    initial_pose = traj_world[0]
    prim_path = spawn_object(
        stage, args.object,
        position=initial_pose[:3, 3].tolist(),
        scale=args.object_scale,
        objects_dir=OBJECTS_DIR,
        kinematic=True,
        collision=False,
    )
    set_object_world_pose(stage, prim_path, initial_pose)
    print(f"[object] Spawned '{args.object}' (kinematic, no collision) at {prim_path}")

    return prim_path, traj_world


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

    # Build the scene programmatically (ground, table, light, physics, robots).
    # Robot collisions are disabled because joint positions are teleported each
    # frame — leaving them on lets PhysX integrate spurious contact forces.
    stage = build_scene(args.mode, robot_collision=False)

    world = World()
    arms = add_articulations(world, args.mode)
    world.reset()
    setup_arms_ik(arms)

    # Resolve the optional refined extrinsic once, before it's needed by either
    # the object trajectory (camera→world mapping) or the viewport camera.
    # The same T_world_cam must be used for both, otherwise the rendered object
    # would drift relative to the refined camera by the refinement delta
    # (typically 60-100 mm in world frame for the Aria extrinsic).
    refined_T_world_cam = None
    refined_camera_name = None
    if args.refined_extrinsic is not None:
        from pathlib import Path as _Path
        ref_path = _Path(args.refined_extrinsic)
        with np.load(ref_path) as _ref:
            if "T_world_cam" not in _ref.files:
                raise SystemExit(
                    f"{ref_path} missing required key 'T_world_cam' "
                    f"(found: {_ref.files})"
                )
            refined_T_world_cam = _ref["T_world_cam"]
            refined_camera_name = str(_ref["camera"]) if "camera" in _ref.files else None
        print(f"[refine] Loaded refined extrinsic from {ref_path} "
              f"(camera={refined_camera_name})")
        with np.printoptions(precision=6, suppress=True):
            print(f"[refine] T_world_cam =\n{refined_T_world_cam}")

    # Apply the override to the object trajectory only when the refinement
    # camera matches --object-pose-camera (otherwise fall back to nominal).
    object_override = (
        refined_T_world_cam
        if refined_camera_name is None or refined_camera_name == args.object_pose_camera
        else None
    )
    if refined_T_world_cam is not None and object_override is None:
        print(f"[refine] Skipping object-trajectory override — refinement camera "
              f"'{refined_camera_name}' != --object-pose-camera '{args.object_pose_camera}'")

    # Spawn the trajectory-driven object (single mode only) and load its
    # world-frame pose trajectory. Must happen after world.reset() so the
    # robot bases have valid world transforms for camera-frame conversion.
    object_prim_path, object_traj_world = _setup_object_replay(
        args, stage, n_frames, T_world_cam_override=object_override,
    )

    if args.camera is not None:
        # Apply the override to the viewport camera only when its name matches
        # the refinement camera.
        camera_override = (
            refined_T_world_cam
            if refined_camera_name is None or refined_camera_name == args.camera
            else None
        )
        if refined_T_world_cam is not None and camera_override is None:
            print(f"[refine] Skipping viewport-camera override — refinement camera "
                  f"'{refined_camera_name}' != --camera '{args.camera}'")

        camera_prim_path = setup_camera(
            stage, args.camera, args.mode,
            FRANKA_LEFT_BASE_PATH, FRANKA_RIGHT_BASE_PATH,
            world_pose_override=camera_override,
        )
        print(f"[camera] Viewport set to {camera_prim_path}")

    # ── Video capture setup ─────────────────────────────────────────────────
    suffix = _build_video_suffix(args)
    recorder, sim_output_path, sbs_recorder, sbs_output_path, overlay_recorders = (
        setup_recording(args, h5_path, n_frames, suffix, APP_WIDTH, APP_HEIGHT)
    )
    captured_frames = 0

    # ── Replay loop ─────────────────────────────────────────────────────────
    print(f"\n[replay] Starting {n_frames} frames at {args.fps} fps...  (Ctrl-C to stop)\n")
    n_object_frames = object_traj_world.shape[0] if object_traj_world is not None else 0
    try:
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                break

            for side, arm in arms.items():
                arm["set_positions"](
                    data[f"arm_{side}"][frame_idx],
                    data[f"hand_{side}"][frame_idx],
                )

            if object_prim_path is not None and frame_idx < n_object_frames:
                set_object_world_pose(
                    stage, object_prim_path, object_traj_world[frame_idx],
                )

            world.step(render=True)
            if recorder is not None and capture_frame_to_writer(recorder, simulation_app):
                captured_frames += 1
                if sbs_recorder is not None:
                    capture_sidebyside_frame(sbs_recorder, recorder["last_frame"], frame_idx)
                for ov in overlay_recorders:
                    capture_overlay_frame(ov, recorder["last_frame"], frame_idx)

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / n_frames
                print(f"  frame {frame_idx:5d}/{n_frames}  ({pct:.1f}%)")
    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.")

    # ── Report & cleanup ────────────────────────────────────────────────────
    failure_parts = []
    for side, arm in arms.items():
        count = arm["set_positions"].get_ik_failure_count()
        failure_parts.append(f"{side}={count}/{n_frames}")
    print(f"[IK] Failures: {', '.join(failure_parts)}")

    close_recorder(recorder)
    if recorder is not None and sim_output_path is not None:
        print(f"[capture] Saved sim replay to {sim_output_path} ({captured_frames} frames)")

    close_sidebyside(sbs_recorder)
    if sbs_recorder is not None:
        print(f"[capture] Saved side-by-side to {sbs_output_path} "
              f"({sbs_recorder['frames_written']} frames)")

    for ov in overlay_recorders:
        close_overlay(ov)
        print(f"[capture] Saved overlay to {ov['output_path']} "
              f"(alpha={ov['alpha']:.2f}, {ov['frames_written']} frames)")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
