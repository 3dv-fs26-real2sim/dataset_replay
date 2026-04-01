"""HDF5 trajectory loading utilities.

Only depends on h5py and numpy — no Isaac Sim imports.
"""

from pathlib import Path

import h5py
import numpy as np

from .constants import N_ARM_POSE_DIMS, N_HAND_DOFS


def _read_dataset(f: h5py.File, candidates: list[str], required: bool = True):
    """Try multiple dataset keys, return the first found."""
    for key in candidates:
        if key in f:
            return f[key][()], key
    if required:
        raise KeyError(f"None of these datasets exist: {candidates}")
    return None, None


def load_h5(
    path: Path,
    use_actions: bool,
    mode: str,
    n_arm_pose_dims: int = N_ARM_POSE_DIMS,
    n_hand_dofs: int = N_HAND_DOFS,
) -> dict:
    """Load joint trajectories from either side-specific or generic h5 schemas."""
    if use_actions:
        arm_left_keys = ["actions_arm_left"]
        arm_right_keys = ["actions_arm_right"]
        hand_left_keys = ["actions_hand_left"]
        hand_right_keys = ["actions_hand_right"]
        arm_single_keys = ["actions_arm"]
        hand_single_keys = ["actions_hand"]
    else:
        arm_left_keys = ["observations/qpos_arm_left"]
        arm_right_keys = ["observations/qpos_arm_right"]
        hand_left_keys = ["observations/qpos_hand_left"]
        hand_right_keys = ["observations/qpos_hand_right"]
        arm_single_keys = ["observations/qpos_arm"]
        hand_single_keys = ["observations/qpos_hand"]

    with h5py.File(path, "r") as f:
        if mode == "dual":
            arm_left, arm_left_key = _read_dataset(f, arm_left_keys)
            arm_right, arm_right_key = _read_dataset(f, arm_right_keys)
            hand_left, hand_left_key = _read_dataset(f, hand_left_keys)
            hand_right, hand_right_key = _read_dataset(f, hand_right_keys)

            n = arm_left.shape[0]
            assert arm_left.shape   == (n, n_arm_pose_dims),  f"arm_left shape:  {arm_left.shape}"
            assert arm_right.shape  == (n, n_arm_pose_dims),  f"arm_right shape: {arm_right.shape}"
            assert hand_left.shape  == (n, n_hand_dofs), f"hand_left shape:  {hand_left.shape}"
            assert hand_right.shape == (n, n_hand_dofs), f"hand_right shape: {hand_right.shape}"

            print(f"[h5] Loaded {n} frames from {path}")
            print(f"     arm_left:  {arm_left.shape}  ({arm_left_key})")
            print(f"     arm_right: {arm_right.shape} ({arm_right_key})")
            print(f"     hand_left: {hand_left.shape} ({hand_left_key})")
            print(f"     hand_right:{hand_right.shape} ({hand_right_key})")

            return {
                "arm_left": arm_left,
                "arm_right": arm_right,
                "hand_left": hand_left,
                "hand_right": hand_right,
                "n_frames": n,
            }

        arm_right, arm_right_key = _read_dataset(f, arm_right_keys + arm_single_keys)
        hand_right, hand_right_key = _read_dataset(f, hand_right_keys + hand_single_keys)
        arm_left, arm_left_key = _read_dataset(f, arm_left_keys, required=False)
        hand_left, hand_left_key = _read_dataset(f, hand_left_keys, required=False)

    n = arm_right.shape[0]
    assert arm_right.shape == (n, n_arm_pose_dims), f"arm_right shape: {arm_right.shape}"
    assert hand_right.shape == (n, n_hand_dofs), f"hand_right shape: {hand_right.shape}"

    if arm_left is not None and hand_left is not None:
        assert arm_left.shape == (n, n_arm_pose_dims), f"arm_left shape: {arm_left.shape}"
        assert hand_left.shape == (n, n_hand_dofs), f"hand_left shape: {hand_left.shape}"

    print(f"[h5] Loaded {n} frames from {path}")
    print(f"     arm_right:  {arm_right.shape} ({arm_right_key})")
    print(f"     hand_right: {hand_right.shape} ({hand_right_key})")
    if arm_left is not None and hand_left is not None:
        print(f"     arm_left:   {arm_left.shape} ({arm_left_key})")
        print(f"     hand_left:  {hand_left.shape} ({hand_left_key})")

    return {
        "arm_left": arm_left,
        "arm_right": arm_right,
        "hand_left": hand_left,
        "hand_right": hand_right,
        "n_frames": n,
    }
