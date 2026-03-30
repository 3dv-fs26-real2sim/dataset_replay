import argparse
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Test replay script")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--fps", type=float, default=60.0, help="Playback frame rate (default: 60)")
parser.add_argument(
    "--mode",
    type=str,
    default="dual",
    choices=["single", "dual"],
    help="Choose between single arm (right) or dual arm setup (default: dual)",
)
args = parser.parse_args()

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import omni
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

SCRIPT_DIR = Path(__file__).parent
if args.mode == "single":
    USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_single.usd"
elif args.mode == "dual":
    USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_dual.usd"
else:
    raise ValueError(f"Invalid mode: {args.mode}")

# Prim paths in the stage
FRANKA_LEFT_PATH  = "/World/fer_orcahand_left_extended"
FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"

# Number of DOFs expected from the h5 data
N_ARM_DOFS  = 7
N_HAND_DOFS = 17

# Joint names and order. May need fix.
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

HAND_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_thumb_mcp",  "left_thumb_abd",  "left_thumb_pip",  "left_thumb_dip",
    "left_index_abd",  "left_index_mcp",  "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd",   "left_ring_mcp",   "left_ring_pip",
    "left_pinky_abd",  "left_pinky_mcp",  "left_pinky_pip",
]

HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

# From invkin_pose.py
ARM_JOINT_VALUES_INITIAL = np.array([0.184280, -0.225290, -0.193213, -2.701355, -0.069782, 2.478938, 0.050881])
HAND_JOINT_VALUES_INITIAL = np.array([0.0] * N_HAND_DOFS)


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
    name = prim_path.lstrip("/").replace("/", "_")
    art = SingleArticulation(prim_path=prim_path, name=name)
    world.scene.add(art)
    return art


def main():
    # Load USD scene
    omni.usd.get_context().open_stage(str(USD_PATH))
    
    # Create world and add articulations
    world = World()
    franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
    if args.mode == "dual":
        franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
    world.reset()
    
    # Set initial joint values for arm and hand
    franka_right.set_joint_positions(np.concatenate([ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL]))
    if args.mode == "dual":
        franka_left.set_joint_positions(np.concatenate([ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL]))
    
    # Simulation loop
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()


if __name__ == "__main__":
    main()