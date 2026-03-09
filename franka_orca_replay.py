"""
Replay motion from h5 file in Isaac Sim.

The h5 file contains:
  observations/qpos_arm_left   (N, 7)  - left  Franka arm joint positions  [rad]
  observations/qpos_arm_right  (N, 7)  - right Franka arm joint positions  [rad]
  observations/qpos_hand_left  (N, 17) - left  OrcaHand joint positions    [rad]
  observations/qpos_hand_right (N, 17) - right OrcaHand joint positions    [rad]

Usage:
    conda activate 3dv
    cd /home/shinben0327/3dv
    python franka_orca_replay.py               # with GUI
    python franka_orca_replay.py --headless    # headless mode
    python franka_orca_replay.py --fps 30      # override playback speed
"""

import argparse
from pathlib import Path

# ── SimulationApp must be created before any other omni/isaacsim imports ─────
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Replay Franka+OrcaHand motion from h5")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--fps", type=float, default=60.0, help="Playback frame rate (default: 60)")
parser.add_argument(
    "--h5",
    type=str,
    default="20250829_180500.h5",
    help="Path to h5 file (relative to script dir)",
)
parser.add_argument(
    "--usd",
    type=str,
    default="franka_orca/franka_orca.usd",
    help="Path to USD scene (relative to script dir)",
)
parser.add_argument(
    "--use-actions",
    action="store_true",
    help="Use actions_* instead of observations/qpos_* for replay",
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

# ── Now safe to import omni / isaac modules ───────────────────────────────────
import numpy as np
import h5py
import omni
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
H5_PATH = SCRIPT_DIR / args.h5
USD_PATH = SCRIPT_DIR / args.usd

# Prim paths in the stage
FRANKA_LEFT_PATH  = "/World/franka_left"
FRANKA_RIGHT_PATH = "/World/franka_right"

# Number of DOFs expected from the h5 data
N_ARM_DOFS  = 7
N_HAND_DOFS = 17

# ── Canonical joint names matching h5 column order ────────────────────────────
# Franka arm: panda_joint1..7 (standard Franka URDF ordering)
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# The h5 arm data is stored as joint angles RELATIVE TO the Franka home position.
# The data collection system (MuJoCo) sets this home as q=0, but Isaac Sim's
# Franka USD uses absolute URDF angles.  We must add this offset before replay.
#
# Verification: joint4 in h5 is [+0.93, +1.0] rad, but Franka URDF joint4 is
# always in [-3.07, -0.07].  Adding home offset (-3π/4 ≈ -2.356) gives [-1.42, -1.36]
# which is within limits.  All 7 joints pass after this correction.
FRANKA_HOME = np.array([
    0.0,          # joint1
    -np.pi / 4,   # joint2  (-45°)
    0.0,          # joint3
    -3*np.pi / 4, # joint4  (-135°)
    0.0,          # joint5
    np.pi / 2,    # joint6  (+90°)
    np.pi / 4,    # joint7  (+45°)
])

# OrcaHand: matches URDF joint definition order from orcahand_*_extended.urdf
HAND_JOINT_NAMES_LEFT = [
    "left_wrist",
    "left_thumb_mcp",  "left_thumb_abd",  "left_thumb_pip",  "left_thumb_dip",
    "left_index_abd",  "left_index_mcp",  "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd",   "left_ring_mcp",   "left_ring_pip",
    "left_pinky_abd",  "left_pinky_mcp",  "left_pinky_pip",
]

HAND_JOINT_NAMES_RIGHT = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

# ─────────────────────────────────────────────────────────────────────────────


def load_h5(path: Path, use_actions: bool):
    """Load joint position trajectories from h5 file."""
    prefix = "actions" if use_actions else "observations/qpos"
    with h5py.File(path, "r") as f:
        arm_left   = f[f"{prefix}_arm_left"][()]
        arm_right  = f[f"{prefix}_arm_right"][()]
        hand_left  = f[f"{prefix}_hand_left"][()]
        hand_right = f[f"{prefix}_hand_right"][()]

    n = arm_left.shape[0]
    assert arm_left.shape   == (n, N_ARM_DOFS),  f"arm_left shape:  {arm_left.shape}"
    assert arm_right.shape  == (n, N_ARM_DOFS),  f"arm_right shape: {arm_right.shape}"
    assert hand_left.shape  == (n, N_HAND_DOFS), f"hand_left shape:  {hand_left.shape}"
    assert hand_right.shape == (n, N_HAND_DOFS), f"hand_right shape: {hand_right.shape}"

    print(f"[h5] Loaded {n} frames from {path}")
    print(f"     arm_left:   {arm_left.shape}  arm_right:  {arm_right.shape}")
    print(f"     hand_left:  {hand_left.shape}  hand_right: {hand_right.shape}")
    return arm_left, arm_right, hand_left, hand_right, n


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
    name = prim_path.lstrip("/").replace("/", "_")
    art = SingleArticulation(prim_path=prim_path, name=name)
    world.scene.add(art)
    return art


def print_dof_info(label: str, art: SingleArticulation):
    print(f"\n[DOF] {label}: {art.num_dof} DOFs")
    for i, name in enumerate(art.dof_names):
        print(f"      [{i:2d}] {name}")


def resolve_dof_indices(art: SingleArticulation, names: list[str], label: str) -> np.ndarray:
    """
    Return an int array mapping h5 column i -> articulation DOF index.
    Tries exact match first, then suffix match (handles any namespace prefixes).
    Aborts if any name cannot be resolved.
    """
    dof_names = list(art.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    indices = []
    for name in names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
        else:
            # Try suffix match in case Isaac Sim prepends a namespace
            matches = [dof for dof in dof_names if dof.endswith(name)]
            if len(matches) == 1:
                print(f"[DOF] '{name}' matched via suffix to '{matches[0]}'")
                indices.append(name_to_idx[matches[0]])
            elif len(matches) > 1:
                raise RuntimeError(
                    f"[DOF] Ambiguous suffix match for '{name}' in {label}: {matches}"
                )
            else:
                raise RuntimeError(
                    f"[DOF] Cannot find '{name}' in {label} DOFs: {dof_names}"
                )
    return np.array(indices, dtype=int)


def make_position_setter(art: SingleArticulation, arm_idx: np.ndarray, hand_idx: np.ndarray):
    """Return a callable that scatters (q_arm, q_hand) into the correct DOF slots.

    q_arm is in h5 convention (relative to FRANKA_HOME); we add the offset here
    to convert to absolute URDF angles expected by Isaac Sim.
    """
    n_dof = art.num_dof
    buf = np.zeros(n_dof)

    def set_positions(q_arm: np.ndarray, q_hand: np.ndarray):
        buf[arm_idx]  = q_arm + FRANKA_HOME   # convert relative → absolute
        buf[hand_idx] = q_hand
        art.set_joint_positions(buf)

    return set_positions


def main():
    print(f"[cfg] H5:      {H5_PATH}")
    print(f"[cfg] USD:     {USD_PATH}")
    print(f"[cfg] headless:{args.headless}  fps:{args.fps}  use_actions:{args.use_actions}")

    # ── Load data ─────────────────────────────────────────────────────────────
    arm_left, arm_right, hand_left, hand_right, n_frames = load_h5(H5_PATH, args.use_actions)

    # ── Open stage BEFORE creating World ──────────────────────────────────────
    print(f"\n[usd] Opening stage (may download Franka asset from Nucleus)…")
    omni.usd.get_context().open_stage(str(USD_PATH))
    for _ in range(30):
        simulation_app.update()

    # ── Create World ──────────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)

    # ── Register articulations ────────────────────────────────────────────────
    # orcahand has PhysicsArticulationRootAPI deleted; it merges into franka's
    # articulation via PhysicsFixedJoint.  Only franka roots are articulations.
    franka_l = setup_articulation(FRANKA_LEFT_PATH,  world)
    franka_r = setup_articulation(FRANKA_RIGHT_PATH, world)

    world.reset()

    print_dof_info("franka_left",  franka_l)
    print_dof_info("franka_right", franka_r)

    # ── Build name-based DOF index mappings ───────────────────────────────────
    print("\n[DOF] Resolving arm joint indices for franka_left…")
    arm_idx_l  = resolve_dof_indices(franka_l, ARM_JOINT_NAMES,        "franka_left")
    print(f"      arm  indices: {arm_idx_l.tolist()}")

    print("[DOF] Resolving hand joint indices for franka_left…")
    hand_idx_l = resolve_dof_indices(franka_l, HAND_JOINT_NAMES_LEFT,  "franka_left")
    print(f"      hand indices: {hand_idx_l.tolist()}")

    print("[DOF] Resolving arm joint indices for franka_right…")
    arm_idx_r  = resolve_dof_indices(franka_r, ARM_JOINT_NAMES,        "franka_right")
    print(f"      arm  indices: {arm_idx_r.tolist()}")

    print("[DOF] Resolving hand joint indices for franka_right…")
    hand_idx_r = resolve_dof_indices(franka_r, HAND_JOINT_NAMES_RIGHT, "franka_right")
    print(f"      hand indices: {hand_idx_r.tolist()}")

    set_l = make_position_setter(franka_l, arm_idx_l, hand_idx_l)
    set_r = make_position_setter(franka_r, arm_idx_r, hand_idx_r)

    # ── Replay loop ───────────────────────────────────────────────────────────
    print(f"\n[replay] Starting {n_frames} frames at {args.fps} fps…  (Ctrl-C to stop)\n")

    try:
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                break

            set_l(arm_left[frame_idx],  hand_left[frame_idx])
            set_r(arm_right[frame_idx], hand_right[frame_idx])

            world.step(render=True)

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / n_frames
                print(f"  frame {frame_idx:5d}/{n_frames}  ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.")

    print("[replay] Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
