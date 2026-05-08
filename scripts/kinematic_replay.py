"""Single-arm kinematic replay against an H5 recording.

Loads a wrist+hand trajectory from H5, solves IK per frame, drives the
articulation kinematically, and (optionally) replays a kinematic object
trajectory in world frame.

Usage:
    python scripts/kinematic_replay.py --h5 data/h5/session.h5
    python scripts/kinematic_replay.py --h5 ... --object duck --object-traj data/poses/foo.npz

NOTE: The H5 schema is currently a stub (see utils/h5_loader.py). When the
new dataset format lands, update ``H5Reader.ARM_KEY``/``HAND_KEY``/
``IMAGE_KEY_FMT`` to match — the rest of this script needs no change.
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
parser.add_argument("--camera-name", type=str, default=H5Reader.DEFAULT_CAMERA,
                    help=f"Image dataset key in H5 (default: {H5Reader.DEFAULT_CAMERA})")
parser.add_argument("--object", type=str, default=None,
                    help="Object name to spawn (folder under objects/)")
parser.add_argument("--object-traj", type=Path, default=None,
                    help="World-frame (N,4,4) trajectory .npz")
parser.add_argument("--object-scale", type=float, default=0.1,
                    help="Object scale (default: 0.1)")
parser.add_argument("--no-camera", action="store_true",
                    help="Skip OAK-D camera setup")

# Recording flags are accepted but their wiring lands in Phase 7 (when the
# new H5 schema is concrete and we know how to overlay the H5 image stream).
parser.add_argument("--record-sim", action="store_true")
parser.add_argument("--record-sidebyside", action="store_true")
parser.add_argument("--record-overlay", nargs="*", type=float, default=None,
                    help="Overlay alpha(s) in (0, 1) — pass one or more values")

args = parser.parse_args()

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
simulation_app = create_app(args)

from isaacsim.core.api import World                          # noqa: E402

from utils.object import set_object_world_pose, spawn_object # noqa: E402
from utils.poses import load_pose_trajectory                 # noqa: E402
from utils.robot import setup_robot                          # noqa: E402
from utils.rotation import detect_quaternion_order           # noqa: E402
from utils.scene import build_scene                          # noqa: E402


def _ensure_wxyz(arm_xyz_quat: np.ndarray) -> np.ndarray:
    """Reorder quaternion columns to wxyz if the H5 stores xyzw."""
    order = detect_quaternion_order(arm_xyz_quat[:, 3:])
    if order == "xyzw":
        out = arm_xyz_quat.copy()
        out[:, 3:] = arm_xyz_quat[:, [6, 3, 4, 5]]
        return out
    return arm_xyz_quat


# ── Build scene + robot ──────────────────────────────────────────────────────
cfg = SceneConfig()
stage = build_scene(cfg, robot_collision=False)

world = World()
robot = setup_robot(world, cfg)

# ── Optional camera ──────────────────────────────────────────────────────────
if not args.no_camera:
    try:
        from utils.camera import setup_camera
        setup_camera(stage, cfg.camera)
    except (ValueError, FileNotFoundError) as e:
        print(f"[kinematic_replay] Skipping camera setup: {e}")

# ── Load H5 trajectory ───────────────────────────────────────────────────────
with H5Reader(args.h5, camera=args.camera_name) as h5:
    traj = h5.load_trajectories()

    arm_xyz_quat = _ensure_wxyz(traj["arm_xyz_quat"])
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

    if args.record_sim or args.record_sidebyside or args.record_overlay:
        print("[kinematic_replay] Recording flags supplied; the capture "
              "pipeline is a Phase 7 follow-up (it depends on the new H5 "
              "schema for sidebyside/overlay frames). Skipping for now.")

    # ── Main replay loop ───────────────────────────────────────────────────
    print(f"[kinematic_replay] Replaying {n_frames} frames from {args.h5.name}")
    n = min(n_frames, object_traj.shape[0] if object_traj is not None else n_frames)
    for i in range(n):
        robot["set_positions"](arm_xyz_quat[i], hand_q[i])
        if object_prim_path is not None and object_traj is not None:
            set_object_world_pose(stage, object_prim_path, object_traj[i])
        world.step(render=True)
        if not simulation_app.is_running():
            break

    fail_count = robot["set_positions"].get_ik_failure_count()
    if fail_count:
        print(f"[kinematic_replay] IK failures across run: {fail_count}")

simulation_app.close()
