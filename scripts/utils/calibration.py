"""Camera-extrinsic calibration helpers — AprilTag + PnP.

The detector is ``pupil-apriltags`` (pip install pupil-apriltags), no ROS
dependency. The PnP solver is ``cv2.solvePnP`` with the iterative method,
seeded from the nominal camera pose so it converges quickly even with
weak corners.

Pure CV / numpy — no Isaac Sim imports. Safe to call from a top-level CLI
script that does not boot Kit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .config import SceneConfig


# ── AprilTag corner geometry ────────────────────────────────────────────────
def tag_corners_world(cfg: SceneConfig) -> np.ndarray:
    """Return the AprilTag's four corners in world frame, ordered
    (TL, TR, BR, BL) to match ``pupil_apriltags.Detection.corners``.

    Tag lies flat on the table top with normal +Z. In the tag-local frame
    the +Y direction is "up in tag image", +X is "right in tag image".
    """
    e = cfg.apriltag.edge_size / 2.0
    local = np.array([
        [-e, +e, 0.0],   # TL
        [+e, +e, 0.0],   # TR
        [+e, -e, 0.0],   # BR
        [-e, -e, 0.0],   # BL
    ], dtype=float)
    T = cfg.apriltag_world_pose()
    return (T[:3, :3] @ local.T + T[:3, 3:4]).T   # (4, 3)


# ── AprilTag detection ──────────────────────────────────────────────────────
def detect_apriltag(
    image_gray: np.ndarray, family: str, tag_id: int,
) -> Optional[np.ndarray]:
    """Detect a single AprilTag in an 8-bit grayscale image.

    Returns a (4, 2) array of pixel-space corner coordinates in the order
    ``pupil_apriltags`` returns them (which matches our :func:`tag_corners_world`
    ordering). Returns ``None`` if the requested tag id was not detected.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise ImportError(
            "pupil-apriltags is required for AprilTag calibration. "
            "Install with: pip install pupil-apriltags"
        ) from exc

    if image_gray.ndim != 2:
        raise ValueError(f"image_gray must be 2-D; got shape {image_gray.shape}")

    det = Detector(families=family)
    detections = det.detect(image_gray, estimate_tag_pose=False)
    for d in detections:
        if d.tag_id == tag_id:
            return np.asarray(d.corners, dtype=float)   # (4, 2)
    return None


# ── PnP ─────────────────────────────────────────────────────────────────────
def solve_extrinsic_pnp(
    corners_image: np.ndarray,
    corners_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    nominal_T_world_cam: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Solve PnP and return ``(T_world_cam, mean_reprojection_error_px)``.

    Uses ``cv2.SOLVEPNP_ITERATIVE`` seeded from ``nominal_T_world_cam`` when
    supplied — improves convergence with only 4 corners.
    """
    import cv2

    object_points = corners_world.astype(np.float32).reshape(-1, 1, 3)
    image_points  = corners_image.astype(np.float32).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float32)
    dist = (np.asarray(dist, dtype=np.float32)
            if dist is not None else np.zeros((5,), dtype=np.float32))

    use_extrinsic_guess = nominal_T_world_cam is not None
    if use_extrinsic_guess:
        T_cw0 = np.linalg.inv(nominal_T_world_cam)
        rvec0, _ = cv2.Rodrigues(T_cw0[:3, :3].astype(np.float32))
        tvec0 = T_cw0[:3, 3].astype(np.float32).reshape(3, 1)
    else:
        rvec0 = np.zeros((3, 1), dtype=np.float32)
        tvec0 = np.zeros((3, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist,
        rvec=rvec0, tvec=tvec0,
        useExtrinsicGuess=use_extrinsic_guess,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed to converge")

    R_cw, _ = cv2.Rodrigues(rvec)             # world → camera
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3]  = tvec.ravel()
    T_world_cam = np.linalg.inv(T_cw)

    # Reprojection error.
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    return T_world_cam, float(err.mean())


# ── Persistence ─────────────────────────────────────────────────────────────
def save_extrinsic(
    T_world_cam: np.ndarray,
    path: Path | str,
    *,
    intrinsics: dict | None = None,
    distortion: tuple | list | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a calibrated ``T_world_cam`` (and optional metadata) to ``.npz``.

    Replay reads only the ``T_world_cam`` field; the rest are kept for
    auditability.
    """
    payload = {"T_world_cam": np.asarray(T_world_cam, dtype=float)}
    if intrinsics is not None:
        payload["intrinsics"] = np.array(
            [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
            dtype=float)
    if distortion is not None:
        payload["distortion"] = np.asarray(distortion, dtype=float)
    if metadata is not None:
        for k, v in metadata.items():
            payload[f"meta_{k}"] = v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    print(f"[calibration] Saved {path}")


def load_extrinsic(path: Path | str) -> np.ndarray:
    """Load ``T_world_cam`` from an extrinsic ``.npz``."""
    return np.load(Path(path))["T_world_cam"]
