"""Single-arm kinematic replay against a maple OakD-recorded H5.

Loads a wrist+hand trajectory from H5, solves IK per frame, drives the
articulation kinematically, and (optionally) replays a kinematic object
trajectory.

Usage::

    # No H5 → build the scene, set home pose, hold (test-setup behaviour).
    python scripts/kinematic_replay_maple.py

    # Full replay; AprilTag-PnP refines the OakD pose at startup by scanning
    # every H5 frame (~5s, depends on frame count).
    python scripts/kinematic_replay_maple.py --h5 data/maple/h5/<session>.h5

    # Headless side-by-side recording.
    python scripts/kinematic_replay_maple.py --headless --h5 ... \\
        --record-sim --record-sidebyside --record-overlay 0.3 0.6

The OakD camera world pose is **auto-calibrated per session** from the
AprilTag visible in the H5 video. The scan is full-frame (~30 ms per
480×270 frame) and runs once at startup. Pass ``--no-calibrate`` to use
the configured nominal lookat pose instead.

Object trajectories are expected in **camera frame** by default (a 6D pose
estimator outputs ``T_cam_obj`` per frame). They are composed with the OakD
``T_world_cam`` at replay time so the sim object lands where the real object
did relative to the camera. Pass ``--object-traj-frame world`` if your NPZ is
already world-frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.app import add_common_args, create_app
from utils.config import PROJECT_ROOT
from utils.config_maple import MapleSceneConfig
from utils.h5_loader import H5Reader, SCHEMA

DATASET = "maple"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser, dataset=DATASET)
parser.add_argument("--h5", type=Path, default=None,
                    help="Path to the H5 trajectory file. If omitted, the "
                         "script builds the scene, sets the home pose, and "
                         "holds (useful for inspecting the scene).")
parser.add_argument("--h5-camera", "--camera-name", dest="h5_camera",
                    type=str, default=SCHEMA[DATASET].default_camera,
                    help=f"Image dataset key in H5 (default: "
                         f"{SCHEMA[DATASET].default_camera})")
parser.add_argument("--object", type=str, default=None,
                    help="Object name to spawn (folder under objects/); "
                         "pass empty string to disable (default: none)")
parser.add_argument("--object-traj", type=Path, default=None,
                    help="(N,4,4) trajectory .npz; frame set by "
                         "--object-traj-frame (camera by default)")
parser.add_argument("--object-traj-frame", choices=("camera", "world"),
                    default="camera",
                    help="Reference frame of --object-traj. 'camera' (default) "
                         "composes each T_cam_obj with T_world_cam; 'world' "
                         "uses the NPZ verbatim.")
parser.add_argument("--object-scale", type=float, default=0.1,
                    help="Object scale (default: 0.1)")
parser.add_argument("--no-camera", action="store_true",
                    help="Skip OakD camera setup")
parser.add_argument("--no-calibrate", action="store_true",
                    help="Skip the startup AprilTag-PnP scan and use the "
                         "configured nominal OakD lookat pose. Useful when "
                         "the tag isn't visible in the recording.")

# Recording flags.
parser.add_argument("--record-sim", action="store_true",
                    help="Save the Isaac Sim viewport as <h5stem>_replay_<suffix>.mp4")
parser.add_argument("--record-sidebyside", action="store_true",
                    help="Save sim+H5 side-by-side as <h5stem>_sidebyside_<suffix>.mp4")
parser.add_argument("--record-overlay", nargs="*", type=float, default=None,
                    help="Alpha-blend sim+H5; pass one or more alphas in (0, 1).")
parser.add_argument("--sample-every", type=int, default=1,
                    help="Record every Nth simulated frame. Output video fps "
                         "becomes args.fps / N (default: 1, no subsample).")
parser.add_argument("--no-fast-record", action="store_true",
                    help="Encode each frame inline (slower but lower memory).")

args = parser.parse_args()

if args.sample_every < 1:
    parser.error("--sample-every must be >= 1")
if args.sample_every > 1:
    effective_fps = args.fps / args.sample_every
    print(f"[kinematic_replay/maple] sample-every={args.sample_every} → "
          f"output video {effective_fps:.2f} fps (sim runs at full source rate)")
    args.fps = effective_fps

if args.object is not None and args.object.strip() == "":
    args.object = None

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
cfg = MapleSceneConfig()
APP_W, APP_H = cfg.viewport_size()
simulation_app = create_app(args, width=APP_W, height=APP_H)

from isaacsim.core.api import World                          # noqa: E402

from utils.calibrate_april import calibrate_from_h5, nominal_oakd_pose  # noqa: E402
from utils.camera import setup_camera                        # noqa: E402
from utils.capture import (                                  # noqa: E402
    capture_frame_to_writer,
    capture_overlay_frame,
    capture_sidebyside_frame,
    close_overlay,
    close_recorder,
    close_sidebyside,
    setup_recording,
)
from utils.object import set_object_world_pose, spawn_object # noqa: E402
from utils.poses import load_pose_trajectory                 # noqa: E402
from utils.robot import setup_robot                          # noqa: E402
from utils.rotation import detect_quaternion_order           # noqa: E402
from utils.scene import build_scene                          # noqa: E402


# capture.py's setup_recording expects record_overlay as a comma-separated
# string; argparse gives us a list of floats. Translate.
args.record_overlay = (
    ",".join(f"{a:.4f}" for a in args.record_overlay)
    if args.record_overlay else None
)


# ── Build scene + robot ──────────────────────────────────────────────────────
stage = build_scene(cfg, robot_collision=False)

world = World()
robot = setup_robot(world, cfg)

# ── Resolve T_world_cam (AprilTag-calib if H5 + tag; else nominal) ───────────
T_world_cam: np.ndarray | None = None
if not args.no_camera:
    if args.h5 is not None and not args.no_calibrate:
        try:
            result = calibrate_from_h5(args.h5, cfg, camera=args.h5_camera)
            T_world_cam = result["T_world_cam"]
            dp = result["delta_pos_mm"]
            print(
                f"[kinematic_replay/maple] AprilTag-calib from "
                f"{args.h5.name}: "
                f"{result['n_inlier_frames']}/{result['n_frames_detected']} "
                f"inlier frames "
                f"(scanned {result['n_frames_scanned']}), "
                f"rms={result['residual_rms_px']:.2f}px, "
                f"Δpos=|{np.linalg.norm(dp):.1f}|mm, "
                f"Δrot={result['delta_rot_deg']:.2f}°"
            )
        except (KeyError, ValueError, RuntimeError, ImportError) as e:
            print(f"[kinematic_replay/maple] WARN: calibration failed ({e}); "
                  f"using nominal OakD pose")
    elif args.no_calibrate:
        print(f"[kinematic_replay/maple] --no-calibrate: using nominal OakD pose")
    else:
        print(f"[kinematic_replay/maple] no --h5: using nominal OakD pose")

    if T_world_cam is None:
        T_world_cam = nominal_oakd_pose(cfg)

    try:
        setup_camera(stage, cfg.camera, world_pose_override=T_world_cam)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[kinematic_replay/maple] Skipping camera setup: {e}")


# ── Spawn object (independent of --h5 so the scene can be inspected) ──────────
# Placed at the table centre, lifted slightly above the top surface. When an
# --object-traj is given (H5 path below) the per-frame poses override this.
object_prim_path: str | None = None
object_traj: np.ndarray | None = None
if args.object is not None:
    objects_dir = PROJECT_ROOT / "objects"
    object_prim_path = spawn_object(
        stage, args.object, objects_dir,
        position=(cfg.table.centre_xy[0], cfg.table.centre_xy[1],
                  cfg.table.top_z + 0.05),
        scale=args.object_scale,
        kinematic=True, collision=False,
    )
    print(f"[kinematic_replay/maple] spawned '{args.object}' at table centre "
          f"({cfg.table.centre_xy[0]:.3f}, {cfg.table.centre_xy[1]:.3f}, "
          f"{cfg.table.top_z + 0.05:.3f})")


# ── No --h5 → just hold home pose ────────────────────────────────────────────
if args.h5 is None:
    print("[kinematic_replay/maple] no --h5; holding home pose. "
          "Ctrl+C to exit.")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()
    sys.exit(0)


# ── Load H5 trajectory ───────────────────────────────────────────────────────
with H5Reader(args.h5, dataset=DATASET, camera=args.h5_camera) as h5:
    traj = h5.load_trajectories()

    arm_xyz_quat = detect_quaternion_order(traj["arm_xyz_quat"], "arm")
    hand_q       = traj["hand_q"]
    n_frames     = traj["n_frames"]

    # ── Optional object trajectory (object already spawned above) ──────────
    if object_prim_path is not None and args.object_traj is not None:
        object_traj = load_pose_trajectory(args.object_traj)
        if args.object_traj_frame == "camera":
            if T_world_cam is None:
                raise SystemExit(
                    "[kinematic_replay/maple] --object-traj-frame=camera "
                    "requires the OakD camera (drop --no-camera, or pass "
                    "--object-traj-frame world)."
                )
            object_traj = np.einsum("ij,njk->nik", T_world_cam, object_traj)
            print(f"[kinematic_replay/maple] composed "
                  f"{object_traj.shape[0]} object poses with T_world_cam")
        if object_traj.shape[0] != n_frames:
            print(f"[kinematic_replay/maple] Object trajectory has "
                  f"{object_traj.shape[0]} frames; H5 has {n_frames}. "
                  f"Replay uses min(N, n_frames).")

    # ── Recording setup ────────────────────────────────────────────────────
    video_suffix = "qpos" + (f"_{args.object}" if args.object else "")
    recorder, _, sbs_recorder, _, overlay_recorders = setup_recording(
        args, args.h5, n_frames, video_suffix, APP_W, APP_H,
        dataset=DATASET,
    )

    # ── Main replay loop ───────────────────────────────────────────────────
    print(f"[kinematic_replay/maple] Replaying {n_frames} frames from "
          f"{args.h5.name}")
    n = min(n_frames, object_traj.shape[0] if object_traj is not None else n_frames)
    progress_width = len(str(n))
    for i in range(n):
        robot["set_positions"](arm_xyz_quat[i], hand_q[i])
        if object_prim_path is not None and object_traj is not None:
            set_object_world_pose(stage, object_prim_path, object_traj[i])
        world.step(render=True)

        record_this_frame = (i % args.sample_every == 0)
        if recorder is not None and record_this_frame:
            ok = capture_frame_to_writer(recorder, simulation_app)
            if ok:
                sim_frame = recorder["last_frame"]
                if sbs_recorder is not None:
                    capture_sidebyside_frame(sbs_recorder, sim_frame, i)
                for ov in overlay_recorders:
                    capture_overlay_frame(ov, sim_frame, i)

        sys.stdout.write(f"\r[replay] {i + 1:>{progress_width}}/{n} frames")
        sys.stdout.flush()

        if not simulation_app.is_running():
            break
    sys.stdout.write("\n")
    sys.stdout.flush()

    if recorder is not None:
        close_recorder(recorder)
    if sbs_recorder is not None:
        close_sidebyside(sbs_recorder)
    for ov in overlay_recorders:
        close_overlay(ov)

    fail_count = robot["set_positions"].get_ik_failure_count()
    if fail_count:
        print(f"[kinematic_replay/maple] IK failures across run: {fail_count}")

simulation_app.close()
