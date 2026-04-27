"""Convert a world-frame object pose trajectory into the Aria camera frame.

Why this script exists
----------------------
``scripts/kinematic_replay.py --object-poses`` does **not** consume world-frame
poses — it consumes *camera-frame* poses.  At replay time it computes a
``T_world_cam`` from ``--object-pose-camera`` (e.g. ``aria``) and applies::

    T_world_obj = T_world_cam @ T_cam_obj                       (eq. 1)

If you hand it a trajectory that is already in world frame (the gym
``obj_root_state`` re-anchored to Isaac coords, like
``poses_duck_a_233_n289.npz``), eq. 1 composes ``T_world_cam`` on top of an
already-world pose and puts the duck in nonsense — exactly the bug we're
fixing here.

You can confirm this empirically:

* ``poses_duck_vda_palm_rot.npz`` (a known camera-frame trajectory) has
  positions clustered around ``[0.26, 0.08, 0.48]`` with z ∈ [0.37, 0.59].
  That positive z is *depth into the camera*, the OpenCV convention.
* ``poses_duck_a_233_n289.npz`` (world frame) has z ≈ 1.0 — the duck is
  resting on the Isaac table at z=1.0, which is a clear giveaway it's not
  camera-frame.

Where ``T_world_cam`` comes from
--------------------------------
At runtime ``utils/camera.py::compute_camera_world_pose`` builds it as::

    T_world_cam = T_world_base_right @ ARIA_EXTRINSICS["right"]

so we mirror that here using two pieces of static information:

1. **Right-arm base pose in the world.** Hard-coded in
   ``pandaorca_description/usd/pandaorca_single.usda`` (lines 109-111)::

       quatd  xformOp:orient    = (1, 0, 0, 0)         # identity rotation
       double3 xformOp:translate = (-0.262, -0.386, 1)

   That's the only place the gym→Isaac shift "lives" in the simulator —
   gym puts the robot at the world origin, Isaac shifts it to that pose.
   This is exactly why the same gym ``obj_root_state`` shows up at z≈0.43
   (gym frame, table top there) but at z≈1.02 (Isaac frame, table top at z=1).

2. **Camera offset on the base.** ``ARIA_EXTRINSICS["right"]`` from
   ``scripts/utils/constants.py`` is ``T_base_from_cam`` per the comment
   above the constant — it maps points from camera frame to right-base
   frame, which is exactly what we need to compose with the base-in-world
   transform.

The conversion is then a single pre-multiplication::

    T_cam_obj = inv(T_world_cam) @ T_world_obj                  (eq. 2)

so that eq. 1 round-trips back to the original world pose at replay time.

Output
------
``<input_stem>_cam.npz`` — a single ``(N, 4, 4)`` float64 array under the key
``poses``, ready for ``--object-poses`` with ``--object-pose-camera aria``.

Example
-------
::

    python data/world_to_camera_frame.py --input data/poses_duck_a_233_n289.npz
    # then in kinematic_replay:
    #   --object-poses data/poses_duck_a_233_n289_cam.npz --object-pose-camera aria
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Pull the calibrated camera extrinsics from the simulator's single source of
# truth so this script can't drift from runtime behaviour. constants.py has no
# Isaac Sim deps, so importing it without a SimulationApp is safe.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils.constants import ARIA_EXTRINSICS  # noqa: E402

# Right-arm base pose in world frame.
# Source: pandaorca_description/usd/pandaorca_single.usda, lines 109-111.
# Identity orientation, translate (-0.262, -0.386, 1.0).
T_WORLD_BASE_RIGHT = np.eye(4)
T_WORLD_BASE_RIGHT[:3, 3] = [-0.262, -0.386, 1.0]


def compute_T_world_cam_aria_right() -> np.ndarray:
    """Build the Aria camera-in-world pose from the right base + extrinsics.

    Mirrors ``utils/camera.py::compute_camera_world_pose`` for single-mode
    Aria (right base only): ``T_world_cam = T_world_base_right @ T_base_from_cam``.
    """
    return T_WORLD_BASE_RIGHT @ ARIA_EXTRINSICS["right"]


def world_to_camera(traj_world: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Pre-multiply by ``T_cam_world`` so replay's ``T_world_cam @ ...`` round-trips."""
    if traj_world.ndim != 3 or traj_world.shape[1:] != (4, 4):
        raise ValueError(f"traj must be (N, 4, 4), got {traj_world.shape}")
    T_cam_world = np.linalg.inv(T_world_cam)
    return np.einsum("ij,njk->nik", T_cam_world, traj_world)


def load_world_trajectory(input_path: Path) -> np.ndarray:
    """Load a single ``(N, 4, 4)`` world-frame trajectory from a NPZ.

    Mirrors the contract of ``utils/poses.py::load_pose_trajectory`` — the file
    must hold exactly one array of shape ``(N, 4, 4)``.  If you have a
    multi-key gym-format file, run ``change_gym_format.py`` first.
    """
    with np.load(input_path) as data:
        files = list(data.files)
        if len(files) != 1:
            raise ValueError(
                f"Expected a single (N, 4, 4) array in {input_path}, got keys: {files}. "
                f"Run change_gym_format.py first if this is gym-format."
            )
        arr = data[files[0]]
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Expected shape (N, 4, 4) in {input_path}, got {arr.shape}")
    return arr.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-multiply a world-frame (N, 4, 4) trajectory by "
                    "inv(T_world_cam) for the Aria camera, so it loads correctly "
                    "with --object-poses + --object-pose-camera aria.",
    )
    default_input = Path(__file__).resolve().parent / "poses_duck_a_233_n289.npz"
    parser.add_argument(
        "--input", type=Path, default=default_input,
        help=f"World-frame (N, 4, 4) NPZ (default: {default_input})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output NPZ path (default: <input_stem>_cam.npz next to input)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}_cam.npz")

    traj_world = load_world_trajectory(args.input)
    T_world_cam = compute_T_world_cam_aria_right()
    print("[setup] T_world_cam (Aria via right base):")
    print(T_world_cam)
    print(f"        camera world pos: {T_world_cam[:3, 3]}")

    traj_cam = world_to_camera(traj_world, T_world_cam)
    pos = traj_cam[:, :3, 3]
    print(f"[stats] cam-frame pos: mean={pos.mean(axis=0)}  "
          f"z range=[{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, poses=traj_cam)
    print(f"[saved] {output_path}  (shape={traj_cam.shape}, dtype={traj_cam.dtype}, key='poses')")


if __name__ == "__main__":
    main()
