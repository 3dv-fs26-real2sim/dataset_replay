"""Visual smoke test for the procedural scene + robot setup.

Loads ``SceneConfig`` defaults, builds the scene, sets the robot to the
home pose via IK, and holds it. Useful for eyeballing table dimensions,
wall heights, AprilTag placement, and robot mount pose against your
physical setup.

Usage:
    python dataset_replay/scripts/test_setup.py
    python dataset_replay/scripts/test_setup.py --headless --duration 5
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.app import add_common_args, create_app
from utils.config import SceneConfig

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--duration", type=float, default=0.0,
                    help="Hold the home pose for N seconds before exiting "
                         "(0 = run until window closed; default 0)")
parser.add_argument("--no-camera", action="store_true",
                    help="Skip OAK-D camera setup (skip if intrinsics aren't filled in)")
parser.add_argument("--object", type=str, default=None,
                    help="Object name (folder under objects/) to spawn at a "
                         "fixed pose for visual sanity. Default: none — pass "
                         "e.g. `--object duck` to enable.")
parser.add_argument("--from-h5", type=Path, default=None,
                    help="Use this H5 file's stored extrinsic as T_world_cam "
                         "(after composing with the configured mount_xyz). "
                         "Bypasses the saved-extrinsic file + nominal fallback.")
parser.add_argument("--camera-name", type=str, default=None,
                    help="Camera key inside the H5 (default: cfg.camera.name).")
args = parser.parse_args()

# ── Boot Isaac Sim FIRST ─────────────────────────────────────────────────────
simulation_app = create_app(args)

# Now safe to import everything Isaac/pxr.
from isaacsim.core.api import World  # noqa: E402

from utils.config import PROJECT_ROOT  # noqa: E402
from utils.object import spawn_object  # noqa: E402
from utils.robot import setup_robot  # noqa: E402
from utils.scene import build_scene  # noqa: E402

# ── Build scene + robot ──────────────────────────────────────────────────────
cfg = SceneConfig()
print(f"[test_setup] mount xyz   = {cfg.robot.mount_xyz}")
print(f"[test_setup] table dims  = {cfg.table.combined_size_xy} m, top z={cfg.table.top_z}")
print(f"[test_setup] AprilTag XY = {cfg.apriltag_world_pose()[:3, 3]}")

stage = build_scene(cfg, robot_collision=False)

world = World()
robot = setup_robot(world, cfg)
print(f"[test_setup] arm DOFs:  {robot['arm_dof_indices']}")
print(f"[test_setup] hand DOFs: {robot['hand_dof_indices']}")

# ── Optional camera ──────────────────────────────────────────────────────────
if not args.no_camera:
    try:
        from utils.camera import setup_camera
        world_pose_override = None
        if args.from_h5 is not None:
            from utils.h5_loader import read_h5_extrinsic
            cam_name = args.camera_name or cfg.camera.name
            world_pose_override = read_h5_extrinsic(
                args.from_h5, camera=cam_name,
                mount_xyz=cfg.robot.mount_xyz,
                mount_rpy=cfg.robot.mount_rpy,
            )
            print(f"[test_setup] Using H5 extrinsic from {args.from_h5.name} "
                  f"(camera={cam_name})")
        setup_camera(stage, cfg.camera, world_pose_override=world_pose_override)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"[test_setup] Skipping camera setup: {e}")

# ── Optional object spawn (visual sanity check; off by default) ──────────────
if args.object is not None:
    objects_dir = PROJECT_ROOT / "objects"
    obj_obj_path = objects_dir / args.object / f"{args.object}.obj"
    if obj_obj_path.exists():
        prim = spawn_object(stage, args.object, objects_dir,
                            position=(-0.10, -0.20, cfg.table.top_z + 0.05),
                            scale=0.10, kinematic=True, collision=False)
        print(f"[test_setup] Spawned {args.object} at {prim}")
    else:
        print(f"[test_setup] Object asset not found: {obj_obj_path}")

# ── Hold pose ────────────────────────────────────────────────────────────────
print("[test_setup] Holding home pose. Ctrl-C to exit (or pass --duration).")
t0 = time.time()
while simulation_app.is_running():
    world.step(render=True)
    if args.duration > 0 and (time.time() - t0) >= args.duration:
        break

simulation_app.close()
