"""Single-arm kinematic replay against an Aria-recorded H5 (egoverse).

Loads a wrist+hand trajectory from H5, solves IK per frame, drives the
articulation kinematically, and (optionally) replays a kinematic object
trajectory in world frame.

Usage::

    python scripts/kinematic_replay.py --h5 data/h5/session.h5
    python scripts/kinematic_replay.py --h5 ... --object duck --object-traj data/poses/foo.npz
    python scripts/kinematic_replay.py --h5 ... --record-sim
    python scripts/kinematic_replay.py --h5 ... --record-sidebyside --record-overlay 0.3 0.6

The Aria camera viewport pose is by default computed at runtime as
``T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT`` (constants in
``utils.constants``). Pass ``--refined-extrinsic PATH`` to override with a
per-session refined ``T_world_cam`` produced by
``scripts/calibrate_extrinsic_table.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.app import add_common_args, create_app
from utils.config import PROJECT_ROOT, SceneConfig
from utils.h5_loader import H5Reader

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--h5", type=Path, required=True,
                    help="Path to the H5 trajectory file")
parser.add_argument("--h5-camera", "--camera-name", dest="h5_camera",
                    type=str, default=H5Reader.DEFAULT_CAMERA,
                    help=f"Image dataset key in H5 (default: {H5Reader.DEFAULT_CAMERA})")
parser.add_argument("--use-actions", action="store_true",
                    help="Drive replay from top-level actions_arm / "
                         "actions_hand instead of observations/qpos_arm / "
                         "observations/qpos_hand.")
parser.add_argument("--object", type=str, default=None,
                    help="Object name to spawn (folder under objects/)")
parser.add_argument("--object-traj", type=Path, default=None,
                    help="World-frame (N,4,4) trajectory .npz")
parser.add_argument("--object-scale", type=float, default=0.1,
                    help="Object scale (default: 0.1)")
parser.add_argument("--no-camera", action="store_true",
                    help="Skip Aria camera setup")
parser.add_argument("--refined-extrinsic", type=Path, default=None,
                    help="Path to an NPZ produced by "
                         "scripts/calibrate_extrinsic_table.py. When set, its "
                         "'T_world_cam' overrides the nominal Aria pose.")

# Recording flags. Capture wiring requires --no-camera off and a viewport.
parser.add_argument("--record-sim", action="store_true",
                    help="Save the Isaac Sim viewport as <h5stem>_replay_<suffix>.mp4")
parser.add_argument("--record-sidebyside", action="store_true",
                    help="Save sim+H5 side-by-side as <h5stem>_sidebyside_<suffix>.mp4")
parser.add_argument("--record-overlay", nargs="*", type=float, default=None,
                    help="Alpha-blend sim+H5; pass one or more alphas in (0, 1).")
parser.add_argument("--no-fast-record", action="store_true",
                    help="Encode each frame inline (slower but lower memory).")

args = parser.parse_args()

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
APP_W, APP_H = 1280, 720
simulation_app = create_app(args, width=APP_W, height=APP_H)

from isaacsim.core.api import World                          # noqa: E402

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
cfg = SceneConfig()
stage = build_scene(cfg, robot_collision=False)

world = World()
robot = setup_robot(world, cfg)

# ── Camera (Aria, optionally refined) ────────────────────────────────────────
if not args.no_camera:
    try:
        from utils.camera import setup_camera
        world_pose_override = None
        if args.refined_extrinsic is not None:
            with np.load(args.refined_extrinsic) as d:
                if "T_world_cam" not in d.files:
                    raise KeyError(f"{args.refined_extrinsic} missing 'T_world_cam'")
                world_pose_override = np.asarray(d["T_world_cam"], dtype=float)
                refined_camera = str(d["camera"]) if "camera" in d.files else "?"
            if refined_camera not in ("?", cfg.camera.name):
                print(f"[kinematic_replay] refined extrinsic camera "
                      f"{refined_camera!r} != cfg.camera.name "
                      f"{cfg.camera.name!r}; falling back to nominal")
                world_pose_override = None
            else:
                print(f"[kinematic_replay] using refined extrinsic from "
                      f"{args.refined_extrinsic.name}")
        setup_camera(stage, cfg.camera, world_pose_override=world_pose_override)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[kinematic_replay] Skipping camera setup: {e}")

# ── Load H5 trajectory ───────────────────────────────────────────────────────
with H5Reader(args.h5, camera=args.h5_camera, use_actions=args.use_actions) as h5:
    traj = h5.load_trajectories()

    arm_xyz_quat = detect_quaternion_order(traj["arm_xyz_quat"], "arm")
    hand_q       = traj["hand_q"]
    n_frames     = traj["n_frames"]

    # ── Optional object spawn + trajectory ─────────────────────────────────
    object_prim_path: str | None = None
    object_traj: np.ndarray | None = None
    if args.object is not None:
        objects_dir = PROJECT_ROOT / "objects"
        object_prim_path = spawn_object(
            stage, args.object, objects_dir,
            position=(0.0, 0.0, cfg.table.top_z + 0.05),
            scale=args.object_scale,
            kinematic=True, collision=False,
        )
        if args.object_traj is not None:
            object_traj = load_pose_trajectory(args.object_traj)
            if object_traj.shape[0] != n_frames:
                print(f"[kinematic_replay] Object trajectory has "
                      f"{object_traj.shape[0]} frames; H5 has {n_frames}. "
                      f"Replay uses min(N, n_frames).")

    # ── Recording setup ────────────────────────────────────────────────────
    video_suffix = "qpos" + (f"_{args.object}" if args.object else "")
    recorder, _, sbs_recorder, _, overlay_recorders = setup_recording(
        args, args.h5, n_frames, video_suffix, APP_W, APP_H,
    )

    # ── Main replay loop ───────────────────────────────────────────────────
    print(f"[kinematic_replay] Replaying {n_frames} frames from {args.h5.name}")
    n = min(n_frames, object_traj.shape[0] if object_traj is not None else n_frames)
    for i in range(n):
        robot["set_positions"](arm_xyz_quat[i], hand_q[i])
        if object_prim_path is not None and object_traj is not None:
            set_object_world_pose(stage, object_prim_path, object_traj[i])
        world.step(render=True)

        if recorder is not None:
            ok = capture_frame_to_writer(recorder, simulation_app)
            if ok:
                sim_frame = recorder["last_frame"]
                if sbs_recorder is not None:
                    capture_sidebyside_frame(sbs_recorder, sim_frame, i)
                for ov in overlay_recorders:
                    capture_overlay_frame(ov, sim_frame, i)

        if not simulation_app.is_running():
            break

    # ── Close recorders (encode deferred frames + close H5 handles) ────────
    if recorder is not None:
        close_recorder(recorder)
    if sbs_recorder is not None:
        close_sidebyside(sbs_recorder)
    for ov in overlay_recorders:
        close_overlay(ov)

    fail_count = robot["set_positions"].get_ik_failure_count()
    if fail_count:
        print(f"[kinematic_replay] IK failures across run: {fail_count}")

simulation_app.close()
