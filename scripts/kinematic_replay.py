"""Single-arm kinematic replay against an Aria-recorded H5 (egoverse).

Loads a wrist+hand trajectory from H5, solves IK per frame, drives the
articulation kinematically, and (optionally) replays a kinematic object
trajectory.

Usage::

    # Defaults reproduce the duck/20250804_104715 demo end-to-end.
    python scripts/kinematic_replay.py

    # Drive from a different H5; spawn a different object.
    python scripts/kinematic_replay.py --h5 data/egoverse/h5/<session>.h5 \\
        --object duck --object-traj data/egoverse/pose/<...>.npz

    # Headless side-by-side recording.
    python scripts/kinematic_replay.py --headless \\
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
from utils.config import PROJECT_ROOT, SceneConfig
from utils.constants import ARIA_INTRINSICS
from utils.h5_loader import H5Reader

# ── Demo defaults — point at the bundled duck/20250804_104715 sample ─────────
DEFAULT_H5          = PROJECT_ROOT / "data" / "egoverse" / "h5"   / "20250804_104715.h5"
DEFAULT_OBJECT      = "duck"
DEFAULT_OBJECT_TRAJ = PROJECT_ROOT / "data" / "egoverse" / "pose" / "20250804_104715_duck_vdahand.npz"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--h5", type=Path, default=DEFAULT_H5,
                    help=f"Path to the H5 trajectory file (default: {DEFAULT_H5})")
parser.add_argument("--h5-camera", "--camera-name", dest="h5_camera",
                    type=str, default=H5Reader.DEFAULT_CAMERA,
                    help=f"Image dataset key in H5 (default: {H5Reader.DEFAULT_CAMERA})")
parser.add_argument("--use-actions", action="store_true",
                    help="Drive replay from top-level actions_arm / "
                         "actions_hand instead of observations/qpos_arm / "
                         "observations/qpos_hand.")
parser.add_argument("--object", type=str, default=DEFAULT_OBJECT,
                    help=f"Object name to spawn (folder under objects/); "
                         f"pass empty string to disable (default: {DEFAULT_OBJECT})")
parser.add_argument("--object-traj", type=Path, default=DEFAULT_OBJECT_TRAJ,
                    help=f"(N,4,4) trajectory .npz (default: {DEFAULT_OBJECT_TRAJ})")
parser.add_argument("--object-traj-frame", choices=("camera", "world"),
                    default="camera",
                    help="Reference frame of --object-traj. 'camera' (default) "
                         "composes each T_cam_obj with T_world_cam to land in "
                         "world; 'world' uses the NPZ verbatim.")
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
parser.add_argument("--mount-xyz", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Override SceneConfig.robot.mount_xyz (where "
                         "panda_link0 lands in world). Used for sweeping the "
                         "robot mount during alignment diagnostics.")
parser.add_argument("--variant-suffix", type=str, default="",
                    help="Extra suffix appended to output MP4 filenames "
                         "(e.g. 'mountB' produces *_qpos_duck_mountB.mp4).")

# Recording flags. Capture wiring requires --no-camera off and a viewport.
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

# Tell the recording writers about the effective output fps so playback
# duration matches the source recording. The simulation loop still steps
# every frame; only the capture cadence drops.
if args.sample_every > 1:
    effective_fps = args.fps / args.sample_every
    print(f"[kinematic_replay] sample-every={args.sample_every} → "
          f"output video {effective_fps:.2f} fps "
          f"(sim runs at full source rate)")
    args.fps = effective_fps

# Normalise "no object" — accept empty string from default-override.
if args.object is not None and args.object.strip() == "":
    args.object = None

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
# Render at the camera's aspect ratio (Aria K is 4:3). A 16:9 viewport on a
# 4:3 camera aperture forces USD to apply different px/mm scales to the
# horizontal and vertical projection axes, so the rendered table edges land
# off the row the pinhole projection predicts. We pick a 2× scale of the
# native sensor (640×480 → 1280×960) so the recorded video is high-res
# without breaking aspect.
APP_W = ARIA_INTRINSICS["width"]  * 2     # 1280
APP_H = ARIA_INTRINSICS["height"] * 2     # 960
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
cfg = SceneConfig()
if args.mount_xyz is not None:
    cfg.robot.mount_xyz = tuple(args.mount_xyz)
    print(f"[kinematic_replay] mount_xyz override → {cfg.robot.mount_xyz}")
stage = build_scene(cfg, robot_collision=False)

world = World()
robot = setup_robot(world, cfg)

# ── Resolve T_world_cam (SAM-refine if mask present; else nominal) ───────────
T_world_cam: np.ndarray | None = None
if not args.no_camera:
    sam_mask_path = args.sam_mask if args.sam_mask is not None \
        else _default_sam_mask_path(args.h5)

    if args.no_refine:
        print(f"[kinematic_replay] --no-refine: using nominal Aria pose "
              f"(skipping SAM refinement)")
    elif sam_mask_path.exists():
        try:
            result = refine_aria_extrinsic(sam_mask_path, cfg)
            T_world_cam = result["T_world_cam"]
            dp = result["delta_pos_mm"]
            print(f"[kinematic_replay] refined Aria pose from "
                  f"{sam_mask_path.name}: Δpos=|"
                  f"{np.linalg.norm(dp):.1f}|mm, "
                  f"Δrot={result['delta_rot_deg']:.2f}°, "
                  f"rms={result['residual_rms_px']:.3f}px")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[kinematic_replay] WARN: SAM refinement failed ({e}); "
                  f"falling back to nominal Aria pose")
    else:
        print(f"[kinematic_replay] WARN: no SAM mask at {sam_mask_path}")
        print(f"[kinematic_replay]   → using nominal T_world_cam = "
              f"T_world_base @ ARIA_EXTRINSICS_RIGHT (likely a few cm off).")
        print(f"[kinematic_replay]   To refine, drop a SAM mask NPZ at that "
              f"path or pass --sam-mask <path>.")

    if T_world_cam is None:
        T_world_cam = compute_nominal_aria_pose(cfg)

    try:
        setup_camera(stage, cfg.camera, world_pose_override=T_world_cam)
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
            if args.object_traj_frame == "camera":
                if T_world_cam is None:
                    raise SystemExit(
                        "[kinematic_replay] --object-traj-frame=camera requires "
                        "the Aria camera (drop --no-camera, or pass "
                        "--object-traj-frame world)."
                    )
                # T_world_obj[i] = T_world_cam @ T_cam_obj[i]
                object_traj = np.einsum("ij,njk->nik", T_world_cam, object_traj)
                print(f"[kinematic_replay] composed {object_traj.shape[0]} "
                      f"object poses with T_world_cam (camera→world)")
            if object_traj.shape[0] != n_frames:
                print(f"[kinematic_replay] Object trajectory has "
                      f"{object_traj.shape[0]} frames; H5 has {n_frames}. "
                      f"Replay uses min(N, n_frames).")

    # ── Recording setup ────────────────────────────────────────────────────
    video_suffix = "qpos" + (f"_{args.object}" if args.object else "")
    if args.variant_suffix:
        video_suffix += f"_{args.variant_suffix}"
    recorder, _, sbs_recorder, _, overlay_recorders = setup_recording(
        args, args.h5, n_frames, video_suffix, APP_W, APP_H,
    )

    # ── Main replay loop ───────────────────────────────────────────────────
    print(f"[kinematic_replay] Replaying {n_frames} frames from {args.h5.name}")
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

        # In-place progress line. Updates every frame; \r overwrites so the
        # terminal shows one rolling indicator instead of n_frames lines.
        sys.stdout.write(
            f"\r[replay] {i + 1:>{progress_width}}/{n} frames"
        )
        sys.stdout.flush()

        if not simulation_app.is_running():
            break
    sys.stdout.write("\n")
    sys.stdout.flush()

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
