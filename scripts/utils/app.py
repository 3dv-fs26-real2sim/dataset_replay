"""SimulationApp creation and shared argparse helpers.

Only imports ``isaacsim.SimulationApp`` — safe as a top-level import.
"""

import argparse
from pathlib import Path

from isaacsim import SimulationApp

from .constants import USD_PATH_SINGLE, USD_PATH_DUAL, H5_PATH_SINGLE, H5_PATH_DUAL


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --headless, --fps, and --mode arguments shared by all scripts."""
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--fps", type=float, default=50.0, help="Simulation frame rate (default: 50.0)")
    parser.add_argument(
        "--mode",
        type=str,
        default="dual",
        choices=["single", "dual"],
        help="Choose between single arm (right) or dual arm setup (default: dual)",
    )


def create_app(args, width: int = 1280, height: int = 720) -> SimulationApp:
    """Create the Isaac Sim application with standard renderer config."""
    return SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RayTracedLighting",
            "width": width,
            "height": height,
        }
    )


def resolve_usd_path(mode: str) -> Path:
    """Return the USD scene path for the given mode ('single' or 'dual')."""
    if mode == "single":
        return USD_PATH_SINGLE
    elif mode == "dual":
        return USD_PATH_DUAL
    raise ValueError(f"Invalid mode: {mode}")


def resolve_h5_path(mode: str) -> Path:
    """Return the default H5 data path for the given mode."""
    if mode == "single":
        return H5_PATH_SINGLE
    elif mode == "dual":
        return H5_PATH_DUAL
    raise ValueError(f"Invalid mode: {mode}")
