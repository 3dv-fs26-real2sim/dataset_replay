"""Single-arm kinematic replay against an Aria-recorded H5 (egoverse).

Loads a wrist+hand trajectory from H5, solves IK per frame, drives the
articulation kinematically, and (optionally) replays a kinematic object
trajectory.

Usage::

    # No H5 → build the scene, set home pose, hold (test-setup behaviour).
    python scripts/kinematic_replay_egoverse.py

    # Full replay; SAM table-mask refines the Aria pose at startup if
    # data/egoverse/desk/<stem>_desk.npz exists, else nominal pose with
    # a warning.
    python scripts/kinematic_replay_egoverse.py --h5 data/egoverse/h5/<session>.h5

    # Headless side-by-side recording.
    python scripts/kinematic_replay_egoverse.py --headless --h5 ... \\
        --record-sim --record-sidebyside --record-overlay 0.3 0.6

The Aria camera viewport pose is **auto-refined per session** from a SAM
table-mask NPZ. Convention: drop the mask at
``data/egoverse/desk/<h5_stem>_desk.npz`` and the replay finds it
automatically (or pass ``--sam-mask <path>``). When the mask is missing the
replay warns and falls back to the nominal pose
``T_world_cam = T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT``.

Object trajectories are expected in **camera frame** by default (egoverse's
pose estimator outputs ``T_cam_obj`` per frame). They are composed with the
viewport's ``T_world_cam`` at replay time so the sim object lands where the
real object did relative to the camera. Pass ``--object-traj-frame world``
if your NPZ is already world-frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.app import add_common_args, create_app
from utils.config import ASSETS
from utils.config_egoverse import EgoverseSceneConfig
from utils.h5_loader import H5Reader, SCHEMA

DATASET = "egoverse"

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
parser.add_argument("--use-actions", action="store_true",
                    help="Drive replay from top-level actions_arm / "
                         "actions_hand instead of observations/qpos_arm / "
                         "observations/qpos_hand.")
parser.add_argument("--object", type=str, default=None,
                    help="Object name to spawn (folder under assets/objects/); "
                         "pass empty string to disable (default: none)")
parser.add_argument("--object-traj", type=Path, default=None,
                    help="(N,4,4) trajectory .npz")
parser.add_argument("--object-traj-frame", choices=("camera", "world"),
                    default="camera",
                    help="Reference frame of --object-traj. 'camera' (default "
                         "for egoverse) composes each T_cam_obj with "
                         "T_world_cam; 'world' uses the NPZ verbatim.")
parser.add_argument("--object-scale", type=float, default=0.1,
                    help="Object scale (default: 0.1)")
parser.add_argument("--no-camera", action="store_true",
                    help="Skip Aria camera setup")
parser.add_argument("--sam-mask", type=Path, default=None,
                    help="Path to a SAM table-mask NPZ (key 'mask', shape "
                         "(N, H, W)) used to refine the Aria extrinsic on "
                         "the fly. Default: data/egoverse/desk/"
                         "<h5_stem>_desk.npz next to the H5. If the file is "
                         "missing the replay warns and falls back to the "
                         "nominal Aria pose.")
parser.add_argument("--no-refine", action="store_true",
                    help="Skip SAM refinement even if the mask file exists; "
                         "use the nominal Aria pose directly.")

# Recording flags.
parser.add_argument("--record-sim", action="store_true",
                    help="Save the Isaac Sim viewport as <h5stem>_replay_<suffix>.mp4")
parser.add_argument("--record-sidebyside", action="store_true",
                    help="Save sim+H5 side-by-side as <h5stem>_sidebyside_<suffix>.mp4")
parser.add_argument("--record-overlay", nargs="*", type=float, default=None,
                    help="Alpha-blend sim+H5; pass one or more alphas in (0, 1).")
parser.add_argument("--sample-every", type=int, default=1,
                    help="Record every Nth simulated frame. Output video fps "
                         "becomes args.fps / N — e.g. --sample-every 5 on 50Hz "
                         "data writes a 10Hz video. (default: 1, no subsample)")
parser.add_argument("--no-fast-record", action="store_true",
                    help="Encode each frame inline (slower but lower memory).")

args = parser.parse_args()

if args.sample_every < 1:
    parser.error("--sample-every must be >= 1")

if args.sample_every > 1:
    effective_fps = args.fps / args.sample_every
    print(f"[kinematic_replay/egoverse] sample-every={args.sample_every} → "
          f"output video {effective_fps:.2f} fps "
          f"(sim runs at full source rate)")
    args.fps = effective_fps

if args.object is not None and args.object.strip() == "":
    args.object = None

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
# Aria K is 4:3 — use the rig-supplied 2× viewport so the rendered table
# edges land on the row the pinhole projection predicts.
cfg = EgoverseSceneConfig()
APP_W, APP_H = cfg.viewport_size()
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
from utils.calibrate_table import (                          # noqa: E402
    compute_nominal_aria_pose,
    refine_aria_extrinsic,
)
from utils.camera import setup_camera                        # noqa: E402
from utils.object import set_object_world_pose, spawn_object # noqa: E402
from utils.poses import load_pose_trajectory                 # noqa: E402
from utils.robot import setup_robot                          # noqa: E402
from utils.rotation import detect_quaternion_order           # noqa: E402
from utils.scene import build_scene                          # noqa: E402


def _default_sam_mask_path(h5_path: Path) -> Path:
    """Convention: ``<h5_root>/desk/<h5_stem>_desk.npz`` next to ``<h5_root>/h5/``."""
    return h5_path.parent.parent / "desk" / f"{h5_path.stem}_desk.npz"


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

# ── Resolve T_world_cam (SAM-refine if mask present; else nominal) ───────────
T_world_cam: np.ndarray | None = None
if not args.no_camera:
    sam_mask_path: Path | None = None
    if args.h5 is not None:
        sam_mask_path = args.sam_mask if args.sam_mask is not None \
            else _default_sam_mask_path(args.h5)

    if args.no_refine or sam_mask_path is None:
        reason = "--no-refine" if args.no_refine else "no --h5"
        print(f"[kinematic_replay/egoverse] {reason}: using nominal Aria pose "
              f"(skipping SAM refinement)")
    elif sam_mask_path.exists():
        try:
            result = refine_aria_extrinsic(sam_mask_path, cfg)
            T_world_cam = result["T_world_cam"]
            dp = result["delta_pos_mm"]
            print(f"[kinematic_replay/egoverse] refined Aria pose from "
                  f"{sam_mask_path.name}: Δpos=|"
                  f"{np.linalg.norm(dp):.1f}|mm, "
                  f"Δrot={result['delta_rot_deg']:.2f}°, "
                  f"rms={result['residual_rms_px']:.3f}px")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[kinematic_replay/egoverse] WARN: SAM refinement failed "
                  f"({e}); falling back to nominal Aria pose")
    else:
        print(f"[kinematic_replay/egoverse] WARN: no SAM mask at "
              f"{sam_mask_path}")
        print(f"[kinematic_replay/egoverse]   → using nominal T_world_cam = "
              f"T_world_base @ ARIA_EXTRINSICS_RIGHT (likely a few cm off).")
        print(f"[kinematic_replay/egoverse]   To refine, drop a SAM mask NPZ "
              f"at that path or pass --sam-mask <path>.")

    if T_world_cam is None:
        T_world_cam = compute_nominal_aria_pose(cfg)

    try:
        setup_camera(stage, cfg.camera, world_pose_override=T_world_cam)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[kinematic_replay/egoverse] Skipping camera setup: {e}")


# ── Spawn object (independent of --h5 so the scene can be inspected) ──────────
# Placed at the table centre, lifted slightly above the top surface. When an
# --object-traj is given (H5 path below) the per-frame poses override this.
object_prim_path: str | None = None
object_traj: np.ndarray | None = None
if args.object is not None:
    objects_dir = ASSETS / "objects"
    object_prim_path = spawn_object(
        stage, args.object, objects_dir,
        position=(cfg.table.centre_xy[0], cfg.table.centre_xy[1],
                  cfg.table.top_z + 0.05),
        scale=args.object_scale,
        kinematic=True, collision=False,
    )
    print(f"[kinematic_replay/egoverse] spawned '{args.object}' at table centre "
          f"({cfg.table.centre_xy[0]:.3f}, {cfg.table.centre_xy[1]:.3f}, "
          f"{cfg.table.top_z + 0.05:.3f})")


# ── No --h5 → just hold home pose ────────────────────────────────────────────
if args.h5 is None:
    print("[kinematic_replay/egoverse] no --h5; holding home pose. "
          "Ctrl+C to exit.")
    while simulation_app.is_running():
        world.step(render=True)
    simulation_app.close()
    sys.exit(0)


# ── Load H5 trajectory ───────────────────────────────────────────────────────
with H5Reader(args.h5, dataset=DATASET, camera=args.h5_camera,
              use_actions=args.use_actions) as h5:
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
                    "[kinematic_replay/egoverse] --object-traj-frame=camera "
                    "requires the Aria camera (drop --no-camera, or pass "
                    "--object-traj-frame world)."
                )
            object_traj = np.einsum("ij,njk->nik", T_world_cam, object_traj)
            print(f"[kinematic_replay/egoverse] composed "
                  f"{object_traj.shape[0]} object poses with T_world_cam")
        if object_traj.shape[0] != n_frames:
            print(f"[kinematic_replay/egoverse] Object trajectory has "
                  f"{object_traj.shape[0]} frames; H5 has {n_frames}. "
                  f"Replay uses min(N, n_frames).")

    # ── Recording setup ────────────────────────────────────────────────────
    video_suffix = "qpos" + (f"_{args.object}" if args.object else "")
    recorder, _, sbs_recorder, _, overlay_recorders = setup_recording(
        args, args.h5, n_frames, video_suffix, APP_W, APP_H,
        dataset=DATASET,
    )

    # ── Main replay loop ───────────────────────────────────────────────────
    print(f"[kinematic_replay/egoverse] Replaying {n_frames} frames from "
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
        print(f"[kinematic_replay/egoverse] IK failures across run: {fail_count}")

simulation_app.close()
