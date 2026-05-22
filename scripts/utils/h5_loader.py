"""H5 trajectory loading — egoverse single-arm Aria adapter.

The egoverse H5 schema is single-arm, generic-named (no ``_right`` suffix):

    observations/qpos_arm           (N, 7)  float64  — [x,y,z, qw,qx,qy,qz]
    observations/qpos_hand          (N, 17) float64
    observations/images/aria_rgb_cam/color  (N, 480, 640, 3) uint8
    actions_arm                     (N, 7)  float64  (top level; optional)
    actions_hand                    (N, 17) float64  (top level; optional)

Pass ``use_actions=True`` to ``H5Reader`` (or ``--use-actions`` on the
replay scripts) to drive replay from the top-level ``actions_*`` arrays
instead of the observation qpos arrays.

Unlike maple's OakD recordings, egoverse H5s do **not** carry per-frame
camera intrinsics or extrinsics: the Aria intrinsics are static (stored
in ``utils.constants.ARIA_INTRINSICS``) and the extrinsic is computed at
runtime from the right-arm base via
``utils.calibrate_table.compute_nominal_aria_pose`` — optionally refined
on the fly by ``utils.calibrate_table.refine_aria_extrinsic`` from a SAM
table mask under ``data/egoverse/desk/``.

Only depends on h5py and numpy — no Isaac Sim imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .constants import N_ARM_POSE_DIMS, N_HAND_DOFS


class H5Reader:
    """Adapter between an H5 file and the kinematic replay loop.

    Usage::

        with H5Reader(path) as h5:                          # use qpos
            traj = h5.load_trajectories()
        with H5Reader(path, use_actions=True) as h5:        # use actions_*
            traj = h5.load_trajectories()

        for i in range(traj["n_frames"]):
            arm_pose  = traj["arm_xyz_quat"][i]   # (7,) [x y z qw qx qy qz]
            hand_q    = traj["hand_q"][i]         # (17,)
            rgb_frame = h5.image(i)               # (H, W, 3) uint8
    """

    # ── Schema constants ────────────────────────────────────────────────────
    DEFAULT_CAMERA = "aria_rgb_cam"
    OBS_ARM_KEY     = "observations/qpos_arm"   # (N, 7)  [x y z qw qx qy qz]
    OBS_HAND_KEY    = "observations/qpos_hand"  # (N, 17)
    ACTION_ARM_KEY  = "actions_arm"             # (N, 7)  top-level
    ACTION_HAND_KEY = "actions_hand"            # (N, 17) top-level
    IMAGE_KEY_FMT   = "observations/images/{camera}/color"   # (N, H, W, 3) uint8
    DEPTH_KEY_FMT   = "observations/images/{camera}/depth"   # (N, H, W) float32

    # ── Construction ───────────────────────────────────────────────────────
    def __init__(self, h5_path: str | Path, camera: str = DEFAULT_CAMERA,
                 *, use_actions: bool = False):
        self._path = Path(h5_path)
        self._camera = camera
        self._use_actions = bool(use_actions)
        self._f: h5py.File | None = h5py.File(self._path, "r")

    @property
    def arm_key(self) -> str:
        return self.ACTION_ARM_KEY if self._use_actions else self.OBS_ARM_KEY

    @property
    def hand_key(self) -> str:
        return self.ACTION_HAND_KEY if self._use_actions else self.OBS_HAND_KEY

    @property
    def n_frames(self) -> int:
        return self._f[self.arm_key].shape[0]

    @property
    def camera(self) -> str:
        return self._camera

    # ── Trajectory access ──────────────────────────────────────────────────
    def load_trajectories(self) -> dict:
        """Return a dict with the per-frame trajectories the replay loop needs.

        Keys:
          ``arm_xyz_quat`` -- (N, 7) wrist target [x, y, z, qw, qx, qy, qz]
                              in some quaternion order. Use
                              ``rotation.detect_quaternion_order`` upstream
                              to canonicalise to wxyz.
          ``hand_q``       -- (N, 17) hand joint positions in the order
                              given by ``constants.HAND_JOINT_NAMES``.
          ``n_frames``     -- int, the number of frames N.
        """
        arm_key, hand_key = self.arm_key, self.hand_key
        if arm_key not in self._f:
            raise KeyError(f"H5 file lacks {arm_key!r}; run peek_schema()")
        if hand_key not in self._f:
            raise KeyError(f"H5 file lacks {hand_key!r}; run peek_schema()")

        arm  = self._f[arm_key][()]
        hand = self._f[hand_key][()]

        if arm.shape[1] != N_ARM_POSE_DIMS:
            raise ValueError(
                f"{arm_key} has {arm.shape[1]} cols; expected {N_ARM_POSE_DIMS}"
            )
        if hand.shape[1] != N_HAND_DOFS:
            raise ValueError(
                f"{hand_key} has {hand.shape[1]} cols; expected {N_HAND_DOFS}"
            )
        if arm.shape[0] != hand.shape[0]:
            raise ValueError(
                f"frame mismatch: arm has {arm.shape[0]}, hand has {hand.shape[0]}"
            )

        n = arm.shape[0]
        source = "actions" if self._use_actions else "observations/qpos"
        print(f"[h5] {self._path.name}: {n} frames from {source}, "
              f"arm{arm.shape}, hand{hand.shape}")
        return {"arm_xyz_quat": arm, "hand_q": hand, "n_frames": n}

    # ── Image access ───────────────────────────────────────────────────────
    def image(self, frame_idx: int) -> np.ndarray:
        """Return RGB image for one frame, ``(H, W, 3)`` uint8."""
        key = self.IMAGE_KEY_FMT.format(camera=self._camera)
        return self._f[key][frame_idx]

    def image_dataset(self):
        """Return the open ``h5py.Dataset`` for random-access frame reads.

        The capture pipeline (``utils.capture``) prefers this over per-frame
        ``image()`` calls when overlaying live sim onto recorded H5 frames.
        """
        key = self.IMAGE_KEY_FMT.format(camera=self._camera)
        return self._f[key] if key in self._f else None

    def depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return depth image for one frame if present, else ``None``."""
        key = self.DEPTH_KEY_FMT.format(camera=self._camera)
        return self._f[key][frame_idx] if key in self._f else None

    def image_dims(self) -> tuple[int, int] | tuple[None, None]:
        """Return ``(width, height)`` of the camera images, or (None, None)."""
        key = self.IMAGE_KEY_FMT.format(camera=self._camera)
        if key not in self._f:
            return (None, None)
        shape = self._f[key].shape  # (N, H, W, C)
        return (shape[2], shape[1])

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "H5Reader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Stand-alone helpers ──────────────────────────────────────────────────────
def open_h5_images(path: str | Path, camera: str = H5Reader.DEFAULT_CAMERA):
    """Backwards-compat helper for ``utils/capture.py``.

    Returns ``(h5py.File, h5py.Dataset)`` for random-access frame reads, or
    ``(None, None)`` if the camera dataset isn't present. The caller is
    responsible for closing the returned ``h5py.File``.
    """
    f = h5py.File(path, "r")
    key = H5Reader.IMAGE_KEY_FMT.format(camera=camera)
    if key not in f:
        f.close()
        return None, None
    return f, f[key]


def peek_schema(path: str | Path, max_depth: int = 4) -> None:
    """Print the dataset hierarchy of an H5 file. Useful for adapting
    :class:`H5Reader`'s schema constants to a new dataset format.
    """
    path = Path(path)
    print(f"=== {path} ===")

    def _visit(name, obj):
        depth = name.count("/") + 1
        if depth > max_depth:
            return
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"  {name}/")

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
