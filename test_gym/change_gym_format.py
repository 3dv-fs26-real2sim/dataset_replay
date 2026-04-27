"""Convert gym-format NPZ files into the (N, 4, 4) trajectory format that
``scripts/kinematic_replay.py --object-poses`` expects.

Source formats supported:
  * Gym sim NPZ — has ``obj_root_state`` of shape (N, 13) where columns are
    ``[pos_xyz(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]``.
  * Already-converted NPZ — a single (N, 4, 4) array (re-saved with the
    canonical key name ``poses``).

Output: an NPZ at ``<input_stem>_replay.npz`` containing one float64 array
of shape (N, 4, 4) under the key ``poses``.

Frame caveat:
    Gym ``obj_root_state`` is in the gym **world** frame. ``kinematic_replay``
    will then apply ``T_world_cam @ poses`` (from ``--object-pose-camera``),
    which is only correct if the saved trajectory is in *camera* frame. If
    the object lands in the wrong place, the saved poses are world-frame and
    need an extra ``T_cam_world`` pre-multiplication — easiest done at replay
    time by adding a pseudo-camera with identity extrinsics.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


GYM_STATE_KEYS = ("obj_root_state", "wrist_state")


def gym_state_to_homogeneous(state: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    """Convert an (N, >=7) gym state matrix to (N, 4, 4) homogeneous transforms.

    Columns 0..2 are position; columns 3..6 are the unit quaternion (gym uses
    xyzw by default). Velocity columns (if present) are dropped.
    """
    if state.ndim != 2 or state.shape[1] < 7:
        raise ValueError(f"Expected (N, >=7) state, got shape {state.shape}")

    pos = state[:, 0:3].astype(np.float64)
    quat = state[:, 3:7].astype(np.float64)

    if quat_order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"quat_order must be 'xyzw' or 'wxyz', got {quat_order!r}")

    R = Rotation.from_quat(quat).as_matrix()
    n = state.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T


def load_trajectory(input_path: Path, state_key: str | None, quat_order: str) -> np.ndarray:
    """Auto-detect the source format and return an (N, 4, 4) trajectory."""
    with np.load(input_path, allow_pickle=True) as data:
        files = list(data.files)

        if state_key is None:
            state_key = next((k for k in GYM_STATE_KEYS if k in files), None)

        if state_key is not None and state_key in files:
            arr = data[state_key]
            print(f"[gym] Using key '{state_key}' (shape={arr.shape}) from {input_path}")
            return gym_state_to_homogeneous(arr, quat_order=quat_order)

        if len(files) == 1:
            arr = data[files[0]]
            if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                print(f"[passthrough] Single (N, 4, 4) array under key '{files[0]}' (shape={arr.shape})")
                return arr.astype(np.float64)

        raise ValueError(
            f"Could not auto-detect a trajectory in {input_path}. "
            f"Available keys: {files}. Pass --state-key to select one."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert gym-format NPZ to kinematic_replay (N, 4, 4) format.",
    )
    default_input = Path(__file__).resolve().parent / "object_picked_up.npz"
    parser.add_argument(
        "--input", type=Path, default=default_input,
        help=f"Source NPZ path (default: {default_input})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output NPZ path (default: <input_stem>_replay.npz next to input)",
    )
    parser.add_argument(
        "--state-key", type=str, default=None,
        help=f"Which key in the source NPZ holds the (N, >=7) state vectors. "
             f"Auto-detected from {GYM_STATE_KEYS} when omitted.",
    )
    parser.add_argument(
        "--quat-order", type=str, default="xyzw", choices=["xyzw", "wxyz"],
        help="Quaternion ordering in the source state (default: xyzw, gym standard).",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}_replay.npz")

    poses = load_trajectory(args.input, args.state_key, args.quat_order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, poses=poses)
    print(f"[saved] {output_path}  (shape={poses.shape}, dtype={poses.dtype}, key='poses')")


if __name__ == "__main__":
    main()
