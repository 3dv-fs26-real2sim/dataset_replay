"""H5 trajectory loading — single-arm adapter for both Maple and Egoverse.

Schema dispatched on the ``dataset`` keyword:

  * **maple**: per-frame arm key ``observations/qpos_arm_right``, hand
    ``observations/qpos_hand_right``, OakD images under
    ``observations/images/oakd_front_view/color``, per-frame ``intrinsics``
    and ``extrinsics`` arrays alongside. No top-level ``actions_*``.
  * **egoverse**: per-frame arm key ``observations/qpos_arm``, hand
    ``observations/qpos_hand``, Aria images under
    ``observations/images/aria_rgb_cam/color``. Top-level ``actions_arm`` /
    ``actions_hand`` available for action-driven replay.

Only depends on h5py and numpy — no Isaac Sim imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .constants import N_ARM_POSE_DIMS, N_HAND_DOFS


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class H5Schema:
    """Per-dataset H5 layout description."""
    default_camera:    str
    arm_key:           str
    hand_key:          str
    image_key_fmt:     str
    depth_key_fmt:     str
    has_actions:       bool
    action_arm_key:    str | None = None
    action_hand_key:   str | None = None


SCHEMA: dict[str, H5Schema] = {
    "maple": H5Schema(
        default_camera="oakd_front_view",
        arm_key="observations/qpos_arm_right",
        hand_key="observations/qpos_hand_right",
        image_key_fmt="observations/images/{camera}/color",
        depth_key_fmt="observations/images/{camera}/depth",
        has_actions=False,
    ),
    "egoverse": H5Schema(
        default_camera="aria_rgb_cam",
        arm_key="observations/qpos_arm",
        hand_key="observations/qpos_hand",
        image_key_fmt="observations/images/{camera}/color",
        depth_key_fmt="observations/images/{camera}/depth",
        has_actions=True,
        action_arm_key="actions_arm",
        action_hand_key="actions_hand",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
class H5Reader:
    """Adapter between an H5 file and the kinematic replay loop.

    Usage::

        with H5Reader(path, dataset="maple") as h5:
            traj = h5.load_trajectories()
        with H5Reader(path, dataset="egoverse", use_actions=True) as h5:
            traj = h5.load_trajectories()

        for i in range(traj["n_frames"]):
            arm_pose  = traj["arm_xyz_quat"][i]   # (7,) [x y z qw qx qy qz]
            hand_q    = traj["hand_q"][i]         # (17,)
            rgb_frame = h5.image(i)               # (H, W, 3) uint8
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        dataset: str,
        camera: str | None = None,
        use_actions: bool = False,
    ):
        if dataset not in SCHEMA:
            raise ValueError(
                f"unknown dataset {dataset!r}; expected one of {list(SCHEMA)}"
            )
        self.schema   = SCHEMA[dataset]
        self._dataset = dataset
        self._path    = Path(h5_path)
        self._camera  = camera or self.schema.default_camera
        self._use_actions = bool(use_actions)
        if use_actions and not self.schema.has_actions:
            raise ValueError(
                f"{dataset!r} H5s do not store top-level actions_*; "
                f"use_actions=True is unsupported."
            )
        self._f: h5py.File | None = h5py.File(self._path, "r")

    # ── Resolved keys ──────────────────────────────────────────────────────
    @property
    def arm_key(self) -> str:
        if self._use_actions:
            return self.schema.action_arm_key  # type: ignore[return-value]
        return self.schema.arm_key

    @property
    def hand_key(self) -> str:
        if self._use_actions:
            return self.schema.action_hand_key  # type: ignore[return-value]
        return self.schema.hand_key

    @property
    def n_frames(self) -> int:
        return self._f[self.arm_key].shape[0]

    @property
    def camera(self) -> str:
        return self._camera

    @property
    def dataset(self) -> str:
        return self._dataset

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
        key = self.schema.image_key_fmt.format(camera=self._camera)
        return self._f[key][frame_idx]

    def image_dataset(self):
        """Return the open ``h5py.Dataset`` for random-access frame reads.

        The capture pipeline (``utils.capture``) prefers this over per-frame
        ``image()`` calls when overlaying live sim onto recorded H5 frames.
        """
        key = self.schema.image_key_fmt.format(camera=self._camera)
        return self._f[key] if key in self._f else None

    def depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return depth image for one frame if present, else ``None``."""
        key = self.schema.depth_key_fmt.format(camera=self._camera)
        return self._f[key][frame_idx] if key in self._f else None

    def image_dims(self) -> tuple[int, int] | tuple[None, None]:
        """Return ``(width, height)`` of the camera images, or (None, None)."""
        key = self.schema.image_key_fmt.format(camera=self._camera)
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


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone helpers
# ─────────────────────────────────────────────────────────────────────────────
def open_h5_images(
    path: str | Path,
    *,
    dataset: str,
    camera: str | None = None,
):
    """Backwards-compat helper for ``utils/capture.py``.

    Returns ``(h5py.File, h5py.Dataset)`` for random-access frame reads, or
    ``(None, None)`` if the camera dataset isn't present. The caller is
    responsible for closing the returned ``h5py.File``.
    """
    if dataset not in SCHEMA:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {list(SCHEMA)}"
        )
    schema = SCHEMA[dataset]
    cam = camera or schema.default_camera
    f = h5py.File(path, "r")
    key = schema.image_key_fmt.format(camera=cam)
    if key not in f:
        f.close()
        return None, None
    return f, f[key]


def read_h5_intrinsic(
    path: str | Path,
    *,
    camera: str,
) -> np.ndarray:
    """Return the 3×3 camera intrinsic matrix stored in an H5, or raise.

    Looks under ``observations/images/{camera}/intrinsics`` (shape (9,)).
    Maple records this; egoverse does not. Egoverse intrinsics are static
    and live in :data:`utils.constants.ARIA_INTRINSICS`.
    """
    key = f"observations/images/{camera}/intrinsics"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"H5 file lacks {key!r}")
        return np.array(f[key]).reshape(3, 3).astype(float)


def read_h5_extrinsic(
    path: str | Path,
    *,
    camera: str,
    mount_xyz: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    mount_rpy: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the camera's 4×4 T_world_cam given the recorded T_robot_cam.

    The H5 stores ``observations/images/{camera}/extrinsics`` (shape (16,))
    as a flattened row-major 4×4 ``T_robot_cam`` — the camera's pose in the
    panda_link0 frame. To convert to world frame we compose with
    ``T_world_robot``::

        T_world_cam = T_world_robot @ T_robot_cam

    Both ``mount_xyz`` and ``mount_rpy`` should match
    ``cfg.robot.mount_xyz`` / ``mount_rpy`` for the converted pose to align
    with the simulated robot.

    Maple-only — egoverse H5s do not record per-frame extrinsics.
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
        T_world_robot[:3, :3] = Rotation.from_euler("XYZ", mount_rpy).as_matrix()

    return T_world_robot @ T_robot_cam


def peek_schema(path: str | Path, max_depth: int = 4) -> None:
    """Print the dataset hierarchy of an H5 file. Useful for adapting
    :class:`H5Schema` constants to a new dataset format.
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
