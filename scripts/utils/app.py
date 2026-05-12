"""SimulationApp creation and shared argparse helpers.

Only imports ``isaacsim.SimulationApp`` — safe as a top-level import.
"""

import argparse

from isaacsim import SimulationApp

from .constants import H5_DEFAULT_FPS


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--headless`` and ``--fps`` arguments shared by all scripts.

    The ``--fps`` value drives the **output video frame rate** used by
    capture.py (sim viewport, side-by-side, overlay MP4s). For the
    sim/H5 overlays to stay in sync with the source recording, this
    must match the H5 capture rate — see ``constants.H5_DEFAULT_FPS``.
    The kinematic replay loop teleports joints once per H5 frame, so
    the underlying Isaac Sim physics_dt is independent and does not
    need to be retimed.
    """
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--fps", type=float, default=H5_DEFAULT_FPS,
        help=f"Output video frame rate (default: {H5_DEFAULT_FPS}, "
             f"matching H5_DEFAULT_FPS in utils/constants.py)",
    )


def create_app(args, width: int = 1280, height: int = 720) -> SimulationApp:
    """Create the Isaac Sim application with standard renderer config."""
    return SimulationApp({
        "headless":  args.headless,
        "renderer":  "RayTracedLighting",
        "width":     width,
        "height":    height,
    })
