"""H5 trajectory loading — single-arm adapter for the new dataset format.

The concrete dataset key constants (``ARM_KEY``, ``HAND_KEY``, etc.) are the
*only* place that needs updating once the new H5 schema is finalised. The
adapter contract — what fields the kinematic-replay loop reads, with what
shapes — is documented in NEW_BRANCH_PLAN.md §6.7 and reflected here.

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

        with H5Reader(path) as h5:
            traj = h5.load_trajectories()
            for i in range(traj["n_frames"]):
                arm_pose  = traj["arm_xyz_quat"][i]   # (7,) [x y z qw qx qy qz]
                hand_q    = traj["hand_q"][i]         # (17,)
                rgb_frame = h5.image(i)               # (H, W, 3) uint8

    The schema constants below are the contract we expect from the new
    dataset format. Once you supply a sample H5, verify (and adjust if
    needed) ``ARM_KEY``, ``HAND_KEY``, and ``IMAGE_KEY_FMT`` against
    :func:`peek_schema` output.
    """

    # ── Schema constants (TBD — adjust once sample H5 lands) ────────────────
    DEFAULT_CAMERA = "oakd_front_view"
    ARM_KEY  = "observations/qpos_arm_right"   # (N, 7)  [x y z qw qx qy qz]
    HAND_KEY = "observations/qpos_hand_right"  # (N, 17)
    IMAGE_KEY_FMT = "observations/images/{camera}/color"   # (N, H, W, 3) uint8
    DEPTH_KEY_FMT = "observations/images/{camera}/depth"   # (N, H, W) float32, optional

    # ── Construction ───────────────────────────────────────────────────────
    def __init__(self, h5_path: str | Path, camera: str = DEFAULT_CAMERA):
        self._path = Path(h5_path)
        self._camera = camera
        self._f: h5py.File | None = h5py.File(self._path, "r")

    @property
    def n_frames(self) -> int:
        return self._f[self.ARM_KEY].shape[0]

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
        if self.ARM_KEY not in self._f:
            raise KeyError(f"H5 file lacks {self.ARM_KEY!r}; run peek_schema()")
        if self.HAND_KEY not in self._f:
            raise KeyError(f"H5 file lacks {self.HAND_KEY!r}; run peek_schema()")

        arm  = self._f[self.ARM_KEY][()]
        hand = self._f[self.HAND_KEY][()]

        if arm.shape[1] != N_ARM_POSE_DIMS:
            raise ValueError(
                f"{self.ARM_KEY} has {arm.shape[1]} cols; expected {N_ARM_POSE_DIMS}"
            )
        if hand.shape[1] != N_HAND_DOFS:
            raise ValueError(
                f"{self.HAND_KEY} has {hand.shape[1]} cols; expected {N_HAND_DOFS}"
            )
        if arm.shape[0] != hand.shape[0]:
            raise ValueError(
                f"frame mismatch: arm has {arm.shape[0]}, hand has {hand.shape[0]}"
            )

        n = arm.shape[0]
        print(f"[h5] {self._path.name}: {n} frames, arm{arm.shape}, hand{hand.shape}")
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


def read_h5_intrinsic(
    path: str | Path,
    camera: str = H5Reader.DEFAULT_CAMERA,
) -> np.ndarray:
    """Return the 3×3 camera intrinsic matrix stored in an H5, or raise.

    Looks under ``observations/images/{camera}/intrinsics`` (shape (9,)).
    """
    key = f"observations/images/{camera}/intrinsics"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"H5 file lacks {key!r}")
        return np.array(f[key]).reshape(3, 3).astype(float)


def read_h5_extrinsic(
    path: str | Path,
    camera: str = H5Reader.DEFAULT_CAMERA,
    *,
    mount_xyz: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    mount_rpy: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the camera's 4×4 T_world_cam given the recorded T_robot_cam.

    The H5 stores ``observations/images/{camera}/extrinsics`` (shape (16,))
    as a flattened row-major 4×4. Per inspection of the dataset format this
    is ``T_robot_cam`` — the camera's pose in the panda_link0 frame.

    To convert to world frame we compose it with ``T_world_robot``, the
    wrapper transform that places ``panda_link0`` at ``mount_xyz`` (with
    optional XYZ-extrinsic Euler rotation ``mount_rpy``)::

        T_world_cam = T_world_robot @ T_robot_cam

    Both ``mount_xyz`` and ``mount_rpy`` should match
    ``SceneConfig.robot.mount_xyz`` / ``mount_rpy`` for the converted pose
    to align with the simulated robot.
    """
    key = f"observations/images/{camera}/extrinsics"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"H5 file lacks {key!r}")
        T_robot_cam = np.array(f[key]).reshape(4, 4).astype(float)

    T_world_robot = np.eye(4)
    T_world_robot[:3, 3] = np.asarray(mount_xyz, dtype=float)
    if any(r != 0.0 for r in mount_rpy):
        from scipy.spatial.transform import Rotation
        T_world_robot[:3, :3] = Rotation.from_euler(
            "XYZ", mount_rpy
        ).as_matrix()

    return T_world_robot @ T_robot_cam


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
