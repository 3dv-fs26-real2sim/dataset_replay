"""Test harness: replay an Isaac Gym RL rollout in Isaac Sim.

This is a self-contained near-copy of ``scripts/kinematic_replay.py`` with two
surgical changes — the original pipeline is untouched.

What's different from kinematic_replay.py
-----------------------------------------
1. **No H5, no IK.**  The H5 file + IK-from-wrist-pose path is replaced with a
   direct joint-position replay.  The arm and hand DOFs come straight from the
   gym ``dof_pos`` array (split 7 + 17), so we skip the IK solver entirely and
   write joint targets to the articulation each frame.

2. **Defaults point at preprocessed gym data in this folder.**
   - ``--gym-joints``  → ``test_gym/gym_joints.npz``       (built from
     ``data/object_picked_up.npz``'s ``dof_pos``: ``arm_right`` = first 7,
     ``hand_right`` = last 17).
   - ``--object-poses`` → ``test_gym/poses_duck_a_233_n289_cam.npz``
     (already in Aria camera frame, ready for ``--object-pose-camera aria``).

Why this matters
----------------
Gym stores absolute arm joint angles, not a wrist 6D pose.  Round-tripping them
through IK would lose information (any IK failure or null-space ambiguity gets
papered over).  Driving joints directly is also the simplest end-to-end check
that the gym rollout (joints + object trajectory) is internally consistent
with the Isaac scene — if the duck and the hand line up here, both the
joint convention and the gym→world→cam frame chain are right.

Limitations
-----------
* ``--record-sidebyside`` and ``--record-overlay`` need H5 image streams and
  won't work here. Use ``--record-sim`` only.
* Only ``--mode single`` is meaningful (gym rollout is single-arm).
"""

import argparse
import sys
from pathlib import Path

# Make scripts/utils importable when running from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from utils.app import add_common_args, create_app, resolve_usd_path, resolve_h5_path
from utils.constants import (
    CAMERA_CONFIGS,
    H5_IMAGE_PATHS, H5_DEFAULT_CAMERA,
    OBJECT_CHOICES, OBJECT_DEFAULT_SCALE,
    OBJECT_POSE_DEFAULT_CAMERA,
)


_HERE = Path(__file__).resolve().parent
DEFAULT_GYM_JOINTS = _HERE / "gym_joints.npz"
DEFAULT_OBJECT_POSES = _HERE / "poses_duck_a_233_n289_cam.npz"
DEFAULT_REFINED_EXTRINSIC = _REPO_ROOT / "data" / "sam_masks_aria_extrinsic.npz"


parser = argparse.ArgumentParser(
    description="Gym RL rollout replay in Isaac Sim — direct joint setting (no IK)."
)
add_common_args(parser)
parser.set_defaults(mode="single")

parser.add_argument(
    "--gym-joints", type=Path, default=DEFAULT_GYM_JOINTS,
    help=f"NPZ with 'arm_right' (N, 7) and 'hand_right' (N, 17) joint trajectories "
         f"(default: {DEFAULT_GYM_JOINTS.name})",
)
parser.add_argument(
    "--camera", type=str, default="aria", choices=list(CAMERA_CONFIGS.keys()),
    help="Set viewport to a calibrated camera (default: aria)",
)
parser.add_argument(
    "--refined-extrinsic", type=Path, default=DEFAULT_REFINED_EXTRINSIC,
    help=f"NPZ from scripts/refine_camera_extrinsic.py — its 'T_world_cam' "
         f"overrides both the viewport camera and the object trajectory's "
         f"camera→world mapping (default: {DEFAULT_REFINED_EXTRINSIC.name}). "
         f"Pass an empty string or 'none' to disable.",
)

# ── Object spawning + 6D pose trajectory ────────────────────────────────────
parser.add_argument(
    "--object", type=str, default="duck", choices=OBJECT_CHOICES + ["none"],
    help="Object to spawn and animate via 6D pose trajectory (default: duck)",
)
parser.add_argument(
    "--object-scale", type=float, default=OBJECT_DEFAULT_SCALE,
    help=f"Object uniform scale (default: {OBJECT_DEFAULT_SCALE})",
)
parser.add_argument(
    "--object-poses", type=Path, default=DEFAULT_OBJECT_POSES,
    help=f"Cam-frame 6D pose trajectory NPZ (default: {DEFAULT_OBJECT_POSES.name})",
)
parser.add_argument(
    "--object-pose-camera", type=str, default=OBJECT_POSE_DEFAULT_CAMERA,
    choices=list(CAMERA_CONFIGS.keys()),
    help=f"Camera frame the object poses are expressed in "
         f"(default: {OBJECT_POSE_DEFAULT_CAMERA}). "
         "Used to derive the camera→world transform applied to the trajectory.",
)

# ── Recording ───────────────────────────────────────────────────────────────
parser.add_argument(
    "--record-sim", action="store_true",
    help="Record Isaac Sim viewport to MP4",
)
parser.add_argument(
    "--record-sidebyside", action="store_true",
    help="Record side-by-side comparison (Isaac Sim left, H5 original right)",
)
parser.add_argument(
    "--record-overlay", type=str, nargs="?", default="0.5", const="0.3,0.5,0.7",
    help="Record alpha-blended overlay of Isaac Sim and H5 video. "
         "Comma-separated sim opacities in [0, 1] (one MP4 per value). "
         "Default: '0.5' (single 50/50 blend, enabled out of the box). "
         "Bare flag with no value: '0.3,0.5,0.7'. Pass 'none' to disable.",
)
parser.add_argument(
    "--h5-path", type=Path, default=None,
    help="H5 file used as the overlay/sbs source AND to set the loop length "
         "(default: utils.constants.H5_PATH_SINGLE / H5_PATH_DUAL via --mode)",
)
parser.add_argument(
    "--h5-camera", type=str, default=H5_DEFAULT_CAMERA,
    choices=list(H5_IMAGE_PATHS.keys()),
    help=f"Which H5 camera stream to overlay against (default: {H5_DEFAULT_CAMERA})",
)
parser.add_argument(
    "--no-fast-record", action="store_true",
    help="Disable deferred encoding (slower but uses less memory)",
)
parser.add_argument(
    "--sim-resolution", type=str, default="640x480",
    help="Isaac Sim render resolution as WxH (default: 640x480, matches Aria)",
)
args = parser.parse_args()

# Resolve "none" / empty string as "disabled" for path-like flags.
if args.refined_extrinsic is not None and str(args.refined_extrinsic).lower() in ("", "none"):
    args.refined_extrinsic = None
if args.record_overlay is not None and str(args.record_overlay).lower() in ("", "none"):
    args.record_overlay = None

# Default H5 path comes from the mode-specific constant.
if args.h5_path is None:
    args.h5_path = resolve_h5_path(args.mode)

_w, _h = args.sim_resolution.lower().split("x")
APP_WIDTH, APP_HEIGHT = int(_w), int(_h)

simulation_app = create_app(args, width=APP_WIDTH, height=APP_HEIGHT)

# Isaac Sim imports must come after SimulationApp creation.
import numpy as np
import omni.usd
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH, OBJECTS_DIR,
    ARM_JOINT_NAMES,
)
from utils.robot import add_articulations, resolve_dof_indices
from utils.capture import (
    setup_recording, capture_frame_to_writer, close_recorder,
    capture_sidebyside_frame, close_sidebyside,
    capture_overlay_frame, close_overlay,
)
from utils.camera import setup_camera
from utils.h5_loader import get_available_cameras
from utils.object import load_object_world_trajectory, spawn_object, set_object_world_pose


# ── Helper functions ────────────────────────────────────────────────────────


def load_gym_joints(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load arm + hand joint trajectories saved by the preprocessing step."""
    with np.load(path) as f:
        arm = f["arm_right"]
        hand = f["hand_right"]
        n = int(f["n_frames"]) if "n_frames" in f.files else arm.shape[0]
    if arm.shape[0] != hand.shape[0]:
        raise ValueError(
            f"arm/hand frame counts disagree: {arm.shape[0]} vs {hand.shape[0]}"
        )
    print(f"[gym] Loaded {arm.shape[0]} frames from {path}  "
          f"(arm={arm.shape}, hand={hand.shape})")
    return arm, hand, n


def make_direct_position_setter(art, arm_idx: np.ndarray, hand_idx: np.ndarray):
    """Write absolute joint positions directly to the articulation — no IK.

    Returns a callable matching the signature of the IK setter
    (``set_positions(arm_joints, hand_joints)``) plus a no-op
    ``get_ik_failure_count`` so the existing reporting code keeps working.
    """
    base = art.get_joint_positions().copy()
    buf = base.copy()

    def set_positions(arm_joints: np.ndarray, hand_joints: np.ndarray) -> None:
        buf[:] = base
        buf[arm_idx] = arm_joints
        buf[hand_idx] = hand_joints
        art.set_joint_positions(buf)

    set_positions.get_ik_failure_count = lambda: 0
    return set_positions


def _setup_object_replay(args, stage, n_frames, object_T_world_cam_override=None):
    """Spawn the chosen object and load its world-frame trajectory.

    Same contract as kinematic_replay's helper — the object NPZ is
    interpreted in ``--object-pose-camera`` frame and pre-multiplied by
    ``T_world_cam`` internally by ``load_object_world_trajectory``.
    """
    traj_world = load_object_world_trajectory(
        stage, args.object, str(args.object_poses) if args.object_poses else None,
        args.object_pose_camera, n_frames,
        args.mode, FRANKA_RIGHT_PATH,
        T_world_cam_override=object_T_world_cam_override,
    )
    if traj_world is None:
        return None, None

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
    arm_traj, hand_traj, n_frames = load_gym_joints(args.gym_joints)

    print(f"[debug] First frame arm joints (rad): {arm_traj[0]}")
    print(f"[debug] First frame hand joints (rad): {hand_traj[0]}")

    # Load USD scene.
    omni.usd.get_context().open_stage(str(resolve_usd_path(args.mode)))
    world = World()
    arms = add_articulations(world, args.mode)
    world.reset()

    # Build direct (non-IK) joint setters for each arm.
    setters = {}
    for side, arm in arms.items():
        cfg = arm["config"]
        art = arm["articulation"]
        arm_idx = resolve_dof_indices(art, ARM_JOINT_NAMES, f"franka_{side}")
        hand_idx = resolve_dof_indices(art, cfg["hand_joint_names"], f"franka_{side}")
        setters[side] = make_direct_position_setter(art, arm_idx, hand_idx)

    stage = omni.usd.get_context().get_stage()

    # Resolve the optional refined extrinsic once, before it's needed by either
    # the object trajectory (camera→world mapping) or the viewport camera.
    refined_T_world_cam = None
    refined_camera_name = None
    if args.refined_extrinsic is not None:
        with np.load(args.refined_extrinsic) as _ref:
            if "T_world_cam" not in _ref.files:
                raise SystemExit(
                    f"{args.refined_extrinsic} missing required key 'T_world_cam' "
                    f"(found: {_ref.files})"
                )
            refined_T_world_cam = _ref["T_world_cam"]
            refined_camera_name = str(_ref["camera"]) if "camera" in _ref.files else None
        print(f"[refine] Loaded refined extrinsic from {args.refined_extrinsic} "
              f"(camera={refined_camera_name})")
        with np.printoptions(precision=6, suppress=True):
            print(f"[refine] T_world_cam =\n{refined_T_world_cam}")

    # Apply the override only when the refinement camera matches the consumer.
    object_override = (
        refined_T_world_cam
        if refined_camera_name is None or refined_camera_name == args.object_pose_camera
        else None
    )
    if refined_T_world_cam is not None and object_override is None:
        print(f"[refine] Skipping object-trajectory override — refinement camera "
              f"'{refined_camera_name}' != --object-pose-camera '{args.object_pose_camera}'")

    # Spawn the trajectory-driven object — must happen after world.reset()
    # so robot bases have valid world transforms for the cam→world mapping.
    object_prim_path, object_traj_world = _setup_object_replay(
        args, stage, n_frames, object_T_world_cam_override=object_override,
    )

    if args.camera is not None:
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
            FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
            world_pose_override=camera_override,
        )
        print(f"[camera] Viewport set to {camera_prim_path}")

    # ── Determine total loop length ─────────────────────────────────────────
    # The gym rollout has a fixed N_gym; the H5 video usually has more frames.
    # When overlay/sbs is on we want to keep stepping past N_gym so the H5
    # stream can play out under a frozen-sim view. This is the only place that
    # decision lives — `n_total` drives both the loop length and the recorder
    # buffer size.
    n_gym = n_frames
    n_h5 = 0
    cameras_avail = get_available_cameras(args.h5_path)
    if args.h5_camera in cameras_avail:
        n_h5 = int(cameras_avail[args.h5_camera][0])
        print(f"[h5] {args.h5_path.name} '{args.h5_camera}' has {n_h5} frames "
              f"(gym has {n_gym})")
    extends_with_h5 = (
        (args.record_overlay is not None or args.record_sidebyside)
        and n_h5 > n_gym
    )
    n_total = max(n_gym, n_h5) if extends_with_h5 else n_gym
    if extends_with_h5:
        print(f"[loop] Extending {n_gym} → {n_total} frames; gym state will be "
              f"frozen on the last gym frame while H5 video continues.")

    # ── Video capture setup ─────────────────────────────────────────────────
    # Pass the H5 path so sbs/overlay can open the image stream; output names
    # also derive from its stem.
    suffix = "gym"
    recorder, sim_output_path, sbs_recorder, sbs_output_path, overlay_recorders = (
        setup_recording(args, args.h5_path, n_total, suffix, APP_WIDTH, APP_HEIGHT)
    )
    captured_frames = 0

    # ── Replay loop ─────────────────────────────────────────────────────────
    print(f"\n[replay] Starting {n_total} frames at {args.fps} fps...  (Ctrl-C to stop)\n")
    n_object_frames = object_traj_world.shape[0] if object_traj_world is not None else 0
    try:
        for frame_idx in range(n_total):
            if not simulation_app.is_running():
                break

            # Drive the sim only while gym data lasts; afterwards we just keep
            # stepping (joints/object stay at the last setpoint = frozen).
            if frame_idx < n_gym:
                for side, setter in setters.items():
                    setter(arm_traj[frame_idx], hand_traj[frame_idx])
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
                pct = 100 * frame_idx / n_total
                tag = "" if frame_idx < n_gym else "  [frozen]"
                print(f"  frame {frame_idx:5d}/{n_total}  ({pct:.1f}%){tag}")
    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.")

    close_recorder(recorder)
    if recorder is not None and sim_output_path is not None:
        print(f"[capture] Saved sim replay to {sim_output_path} ({captured_frames} frames)")

    close_sidebyside(sbs_recorder)
    if sbs_recorder is not None:
        print(f"[capture] Saved side-by-side to {sbs_output_path} "
              f"({sbs_recorder['frames_written']} frames)")

    for ov in overlay_recorders:
        close_overlay(ov)
        print(f"[capture] Saved overlay (alpha={ov['alpha']:.2f}) to "
              f"{ov['output_path']} ({ov['frames_written']} frames)")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
