import argparse
import time

from utils.app import add_common_args, create_app, resolve_usd_path

parser = argparse.ArgumentParser(description="Spawn an object on the pandaorca scene")
add_common_args(parser)
parser.add_argument(
    "--object", type=str, default="grape",
    choices=["ball", "duck", "fish", "grape", "shovel"],
    help="Object to spawn",
)
parser.add_argument(
    "--position", type=float, nargs=3, default=[0.0, 0.0, 1.0],
    metavar=("X", "Y", "Z"),
    help="Spawn position in meters (default: 0 0 1.0; table-top center)",
)
parser.add_argument(
    "--scale", type=float, default=0.1,
    help="Uniform object scale factor (default: 0.1). Due to Artec export scale.",
)
args = parser.parse_args()

simulation_app = create_app(args)

# Isaac Sim imports must come after SimulationApp creation.
import numpy as np
import omni.usd
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL,
    OBJECTS_DIR,
)
from utils.robot import setup_articulation
from utils.object import spawn_object


def main():
    try:
        omni.usd.get_context().open_stage(str(resolve_usd_path(args.mode)))

        world = World()
        franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
        if args.mode == "dual":
            franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
        world.reset()

        q0 = np.concatenate([ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL])
        franka_right.set_joint_positions(q0)
        if args.mode == "dual":
            franka_left.set_joint_positions(q0)

        stage = omni.usd.get_context().get_stage()
        spawned_prim_path = spawn_object(stage, args.object, args.position, args.scale, OBJECTS_DIR)
        print(f"Spawned '{args.object}' at {args.position} with scale {args.scale} -> {spawned_prim_path}")

        dt = 1.0 / max(args.fps, 1e-6)
        while simulation_app.is_running():
            world.step(render=True)
            time.sleep(dt)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
