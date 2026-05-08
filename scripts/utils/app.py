"""SimulationApp creation and shared argparse helpers.

Only imports ``isaacsim.SimulationApp`` — safe as a top-level import.
"""

import argparse

from isaacsim import SimulationApp


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --headless and --fps arguments shared by all scripts."""
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--fps", type=float, default=50.0,
        help="Simulation frame rate (default: 50.0)",
    )


def create_app(args, width: int = 1280, height: int = 720) -> SimulationApp:
    """Create the Isaac Sim application with standard renderer config."""
    return SimulationApp({
        "headless":  args.headless,
        "renderer":  "RayTracedLighting",
        "width":     width,
        "height":    height,
    })
