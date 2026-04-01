import argparse

from utils.app import add_common_args, create_app, resolve_usd_path

parser = argparse.ArgumentParser(description="Test robot setup — load scene and hold home pose")
add_common_args(parser)
args = parser.parse_args()

simulation_app = create_app(args)

# Isaac Sim imports must come after SimulationApp creation.
import numpy as np
import omni.usd
from isaacsim.core.api import World

from utils.constants import (
    FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    ARM_JOINT_VALUES_INITIAL, HAND_JOINT_VALUES_INITIAL,
)
from utils.robot import setup_articulation


def main():
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

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
