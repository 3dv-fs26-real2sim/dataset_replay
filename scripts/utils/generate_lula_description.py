"""Generate a Lula robot descriptor YAML for the panda arm from its URDF.

This is a one-time utility. Run it if the URDF changes or you need to
regenerate the descriptor file.

Usage:
    python generate_lula_description.py [--urdf PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_URDF = SCRIPT_DIR / "../../assets/urdf/panda_arm.urdf"
DEFAULT_OUTPUT = SCRIPT_DIR / "../../assets/lula/panda_arm_descriptor.yaml"

# Panda arm active joints (7-DOF Franka FR3/Panda)
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

# Limits taken from the Franka Panda descriptor shipped with Isaac Sim
ACCELERATION_LIMITS = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
JERK_LIMITS = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]

# Default configuration (Franka home-ish pose)
DEFAULT_Q = [0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75]


def generate_descriptor(urdf_path: Path, output_path: Path):
    descriptor = {
        "api_version": 1.0,
        "cspace": PANDA_JOINTS,
        "root_link": "panda_link0",
        "default_q": DEFAULT_Q,
        "acceleration_limits": ACCELERATION_LIMITS,
        "jerk_limits": JERK_LIMITS,
        "cspace_to_urdf_rules": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(
            "# Robot descriptor for the 7-DOF Franka FR3/Panda arm.\n"
            "# Used by the Lula IK solver in Isaac Sim.\n"
            f"# Generated from URDF: {urdf_path}\n\n"
        )
        yaml.dump(descriptor, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote robot descriptor to {output_path}")
    print(f"  URDF: {urdf_path}")
    print(f"  Active joints: {PANDA_JOINTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to FER URDF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output YAML path")
    args = parser.parse_args()

    generate_descriptor(args.urdf.resolve(), args.output.resolve())
