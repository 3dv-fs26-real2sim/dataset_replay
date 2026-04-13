"""
Dynamic replay which has collision enabled for the object and robot and table. 
Experimental idea based on docs/dynamic_replay_strategy.html option A. 
Does not work. We need a better idea for how to do dynamic replay with physics in Isaac Sim.
"""

import argparse

from utils.app import add_common_args, create_app, resolve_usd_path, resolve_h5_path
# constants.py / poses.py have no Isaac Sim deps — safe to import before SimulationApp.
from utils.constants import (
    CAMERA_CONFIGS, H5_IMAGE_PATHS, H5_DEFAULT_CAMERA,
    OBJECT_CHOICES, OBJECT_DEFAULT_SCALE,
    OBJECT_POSE_DEFAULT_CAMERA,
)
from utils.h5_loader import get_h5_image_dims

parser = argparse.ArgumentParser(
    description="Dynamic replay of H5 trajectory — object follows physics with D6 spring-damper tracking",
)
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
    "--object-pose-camera", type=str, default=OBJECT_POSE_DEFAULT_CAMERA,
    choices=list(CAMERA_CONFIGS.keys()),
    help=f"Camera frame the object pose .npz is expressed in (default: {OBJECT_POSE_DEFAULT_CAMERA}). "
         "The trajectory is transformed into world frame using this camera's extrinsics.",
)

# ── D6 joint physics parameters ────────────────────────────────────────────
parser.add_argument(
    "--stiffness", type=float, default=500.0,
    help="D6 joint spring stiffness for all 6 DOF (default: 500.0). "
         "Higher values track the trajectory more tightly.",
)
parser.add_argument(
    "--damping", type=float, default=100.0,
    help="D6 joint damping coefficient for all 6 DOF (default: 100.0)",
)
parser.add_argument(
    "--object-mass", type=float, default=0.1,
    help="Override object mass in kg (default: derive from mesh density)",
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

from utils.constants import FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH, OBJECTS_DIR
from utils.rotation import detect_quaternion_order
from utils.h5_loader import load_h5
from utils.robot import add_articulations, setup_arms_ik
from utils.capture import (
    setup_recording, capture_frame_to_writer, close_recorder,
    capture_sidebyside_frame, close_sidebyside,
)
from utils.camera import setup_camera
from pxr import UsdPhysics as _UsdPhysics
from utils.object import (
    load_object_world_trajectory, spawn_object, set_object_world_pose,
    create_d6_tracking_joint,
)


# ── Helper functions ────────────────────────────────────────────────────────


def _build_video_suffix(args) -> str:
    """Build a descriptive suffix from the active flags.

    Examples:
        qpos_duck_dynamic           (defaults)
        actions_duck_dynamic        --use-actions
    """
    parts = []
    parts.append("actions" if args.use_actions else "qpos")
    if args.object != "none" and args.mode == "single":
        parts.append(args.object)
    parts.append("dynamic")
    return "_".join(parts)


def _setup_dynamic_object_replay(args, stage, n_frames):
    """Spawn a dynamic object with D6 joint anchor and prepare its trajectory.

    The object is a dynamic rigid body (gravity, collisions enabled). A kinematic
    anchor follows the estimated trajectory each frame; a D6 joint with spring-damper
    drives pulls the dynamic object toward the anchor while PhysX resolves physics.

    Returns ``(anchor_path, object_path, traj_world)`` or ``(None, None, None)``.
    """
    traj_world = load_object_world_trajectory(
        stage, args.object, args.object_poses,
        args.object_pose_camera, n_frames,
        args.mode, FRANKA_RIGHT_PATH,
    )
    if traj_world is None:
        return None, None, None

    # Spawn as dynamic rigid body (physics-enabled, gravity on).
    initial_pose = traj_world[0]
    object_path = spawn_object(
        stage, args.object,
        position=initial_pose[:3, 3].tolist(),
        scale=args.object_scale,
        objects_dir=OBJECTS_DIR,
        kinematic=False,
    )
    set_object_world_pose(stage, object_path, initial_pose)
    print(f"[object] Spawned '{args.object}' (dynamic) at {object_path}")

    # Optionally override mass.
    if args.object_mass is not None:
        obj_prim = stage.GetPrimAtPath(object_path)
        mass_api = _UsdPhysics.MassAPI.Apply(obj_prim)
        mass_api.CreateMassAttr(args.object_mass)
        print(f"[object] Set mass to {args.object_mass} kg")

    # Create kinematic anchor + D6 joint for spring-damper tracking.
    anchor_path = create_d6_tracking_joint(
        stage, object_path, initial_pose,
        stiffness=args.stiffness,
        damping=args.damping,
    )

    return anchor_path, object_path, traj_world


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
    arms = add_articulations(world, args.mode)
    world.reset()
    setup_arms_ik(arms)

    stage = omni.usd.get_context().get_stage()

    # Spawn the dynamic object with D6 tracking joint (single mode only).
    # Must happen after world.reset() so the robot bases have valid world
    # transforms for camera-frame conversion.
    anchor_prim_path, object_prim_path, object_traj_world = (
        _setup_dynamic_object_replay(args, stage, n_frames)
    )

    if args.camera is not None:
        camera_prim_path = setup_camera(
            stage, args.camera, args.mode,
            FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
        )
        print(f"[camera] Viewport set to {camera_prim_path}")

    # ── Video capture setup ─────────────────────────────────────────────────
    suffix = _build_video_suffix(args)
    recorder, sim_output_path, sbs_recorder, sbs_output_path = (
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

            if anchor_prim_path is not None and frame_idx < n_object_frames:
                set_object_world_pose(
                    stage, anchor_prim_path, object_traj_world[frame_idx],
                )

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
        print(f"[capture] Saved comparison to {sbs_output_path} "
              f"({sbs_recorder['frames_written']} frames)")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
