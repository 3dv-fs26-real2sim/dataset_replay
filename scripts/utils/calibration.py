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
    """Return the AprilTag's four corners in world frame, in the order
    ``pupil_apriltags.Detection.corners`` returns them (the canonical TL,
    TR, BR, BL order of the tag's intrinsic image), applied to the **rotated**
    tag pose (``T_world_tag`` includes the printed quad's ``rotation_z_deg``).

    NOTE on planar-PnP ambiguity: 4 coplanar points admit two camera-pose
    solutions related by a flip about an axis in the tag plane. The wrapper
    in ``calibrate_extrinsic.py`` uses ``solve_pose_square_planar`` +
    ``select_correct_branch`` to pick the geometrically valid one and seeds
    the iterative multi-view refinement with it.

    **Caveat — sim/real mismatch:** the reported PnP camera pose is only
    physically meaningful if the tag's *world* pose (``cfg.apriltag``) matches
    where the printed tag actually sits in the room. If the SceneConfig has
    the wrong tag pose, PnP will still find a self-consistent fit but rotate
    the camera to compensate. ``calibrate_extrinsic.py`` emits a diagnostic
    when the recovered pose looks geometrically off so you know to fix the
    SceneConfig.
    """
    e = cfg.apriltag.edge_size / 2.0
    # Order = (BL, BR, TR, TL) of the tag's intrinsic image, traversed
    # counter-clockwise — this is the order pupil-apriltags actually returns
    # in (verified empirically against the bundled H5 dataset by minimising
    # reprojection through the H5-stored extrinsic over all 24 corner
    # permutations × 4 rotation_z choices: the unique 6.3-px-error optimum
    # was [BL,BR,TR,TL] + rotation_z=-90°). The library's docstring text
    # "(-1,1),(1,1),(1,-1),(-1,-1)" describes the homography source, not
    # the corners[] array order.
    local = np.array([
        [-e, -e, 0.0],   # BL of intrinsic
        [+e, -e, 0.0],   # BR
        [+e, +e, 0.0],   # TR
        [-e, +e, 0.0],   # TL
    ], dtype=float)
    T = _tag_world_pose_with_rotation(cfg)
    return (T[:3, :3] @ local.T + T[:3, 3:4]).T   # (4, 3)


# ── AprilTag detection ──────────────────────────────────────────────────────
def _make_detector(family: str):
    """Return a configured pupil-apriltags Detector (single import gate)."""
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise ImportError(
            "pupil-apriltags is required for AprilTag calibration. "
            "Install with: pip install pupil-apriltags"
        ) from exc
    return Detector(families=family)


def detect_apriltag(
    image_gray: np.ndarray, family: str, tag_id: int,
) -> Optional[np.ndarray]:
    """Detect a single AprilTag in an 8-bit grayscale image.

    Returns a (4, 2) array of pixel-space corner coordinates in the order
    ``pupil_apriltags`` returns them (which matches our :func:`tag_corners_world`
    ordering). Returns ``None`` if the requested tag id was not detected.
    """
    if image_gray.ndim != 2:
        raise ValueError(f"image_gray must be 2-D; got shape {image_gray.shape}")
    det = _make_detector(family)
    for d in det.detect(image_gray, estimate_tag_pose=False):
        if d.tag_id == tag_id:
            return np.asarray(d.corners, dtype=float)   # (4, 2)
    return None


def detect_apriltag_with_quality(
    image_gray: np.ndarray, family: str, tag_id: int,
    *, detector=None,
) -> Optional[tuple[np.ndarray, float, int]]:
    """Detect a single AprilTag and return ``(corners, decision_margin, hamming)``.

    ``decision_margin`` is pupil-apriltags' confidence score (higher = better);
    ``hamming`` is the bit-error count of the decoded tag (0 = perfect).
    Pass ``detector`` to reuse a Detector across calls (for video scans).
    """
    if image_gray.ndim != 2:
        raise ValueError(f"image_gray must be 2-D; got shape {image_gray.shape}")
    det = detector if detector is not None else _make_detector(family)
    for d in det.detect(image_gray, estimate_tag_pose=False):
        if d.tag_id == tag_id:
            return (np.asarray(d.corners, dtype=float),
                    float(d.decision_margin), int(d.hamming))
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


# ── Planar-square PnP with branch selection ─────────────────────────────────
def _tag_world_pose_with_rotation(cfg: SceneConfig) -> np.ndarray:
    """Return T_world_tag including the printed-tag's ``rotation_z_deg``.

    ``cfg.apriltag_world_pose()`` is identity-rotation; the visual quad has
    an extra Z rotation that ``utils.apriltag.add_apriltag_plane`` applies in
    USD. For IPPE_SQUARE we need the *visible* tag's frame, so include it.
    """
    T = cfg.apriltag_world_pose().copy()
    theta = np.deg2rad(cfg.apriltag.rotation_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    T[:3, :3] = T[:3, :3] @ Rz
    return T


def solve_pose_square_planar(
    corners_image: np.ndarray,
    edge_size: float,
    T_world_tag: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """IPPE_SQUARE pose estimation with both ambiguity branches.

    Returns ``(T_world_cam_a, T_world_cam_b, (err_a_px, err_b_px))`` — the two
    planar PnP solutions and their per-point reprojection errors.

    ``corners_image`` MUST be in the canonical (TL, TR, BR, BL) order that
    IPPE_SQUARE expects (which matches the existing ``tag_corners_world``
    ordering and pupil-apriltags' output for this dataset's tag rotation).
    """
    import cv2

    e = edge_size / 2.0
    # IPPE_SQUARE requires this specific (TL, TR, BR, BL) ordering of the
    # marker-frame object points per OpenCV docs. pupil-apriltags returns
    # corners in (BL, BR, TR, TL) order — re-pair the image points to
    # match the IPPE-required object order.
    obj_pts_marker = np.array([
        [-e, +e, 0.0],   # TL
        [+e, +e, 0.0],   # TR
        [+e, -e, 0.0],   # BR
        [-e, -e, 0.0],   # BL
    ], dtype=np.float64).reshape(-1, 1, 3)
    # corners_image is in (BL, BR, TR, TL); re-order to (TL, TR, BR, BL).
    reorder_to_ippe = np.asarray(corners_image, dtype=np.float64)[[3, 2, 1, 0]]
    img_pts = reorder_to_ippe.reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = (np.asarray(dist, dtype=np.float64)
            if dist is not None else np.zeros((5,), dtype=np.float64))

    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj_pts_marker, img_pts, K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n_sol < 1:
        raise RuntimeError("IPPE_SQUARE returned no solutions")

    T_world_cams = []
    err_pts = []
    for rvec, tvec, err in zip(rvecs, tvecs, errs):
        R_mc, _ = cv2.Rodrigues(rvec)              # marker -> camera
        T_mc = np.eye(4); T_mc[:3, :3] = R_mc; T_mc[:3, 3] = tvec.ravel()
        T_cm = np.linalg.inv(T_mc)                 # camera -> marker
        T_wc = T_world_tag @ T_cm                  # camera -> world
        T_world_cams.append(T_wc)
        err_pts.append(float(np.asarray(err).ravel()[0]))

    if n_sol == 1:
        T_world_cams.append(T_world_cams[0])
        err_pts.append(err_pts[0])
    return T_world_cams[0], T_world_cams[1], (err_pts[0], err_pts[1])


def select_correct_branch(
    T_world_cam_a: np.ndarray, T_world_cam_b: np.ndarray,
    T_world_tag: np.ndarray,
    *, seed_T_world_cam: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Pick the geometrically-valid branch from a planar-PnP pair.

    Selection rule (in order):
      1. If both branches put the camera on the +Z side of the tag plane
         (camera_pos minus tag_origin dotted with tag_normal > 0), prefer
         the one closest to ``seed_T_world_cam`` (if supplied) — this
         resolves residual ambiguity using the nominal pose.
      2. Otherwise, the branch with the camera on the +Z side of the tag is
         the physically valid one (camera looks DOWN at a flat-on-table tag).

    Returns ``(T_world_cam_chosen, branch_idx)`` with branch_idx in {0, 1}.
    """
    tag_origin = T_world_tag[:3, 3]
    tag_normal = T_world_tag[:3, 2]

    def score_in_front(T_wc):
        cam_pos = T_wc[:3, 3]
        return float((cam_pos - tag_origin) @ tag_normal)

    in_front_a = score_in_front(T_world_cam_a) > 0
    in_front_b = score_in_front(T_world_cam_b) > 0

    if in_front_a and not in_front_b:
        return T_world_cam_a, 0
    if in_front_b and not in_front_a:
        return T_world_cam_b, 1

    # Both (or neither) are in front of the tag — fall back to seed proximity.
    def seed_dist(T_wc):
        if seed_T_world_cam is None:
            return 0.0
        return float(np.linalg.norm(T_wc[:3, 3] - seed_T_world_cam[:3, 3]))

    if seed_dist(T_world_cam_a) <= seed_dist(T_world_cam_b):
        return T_world_cam_a, 0
    return T_world_cam_b, 1


# ── Multi-view PnP (joint over many frames of the same static tag) ─────────
def solve_extrinsic_pnp_multiview(
    corners_image_per_frame: list[np.ndarray],
    corners_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    nominal_T_world_cam: np.ndarray | None = None,
    ransac_reproj_threshold_px: float = 3.0,
    ransac_iterations: int = 500,
) -> tuple[np.ndarray, float, dict]:
    """Estimate one ``T_world_cam`` from many detections of the same static tag.

    The camera is rigidly mounted and the AprilTag is at a fixed world pose, so
    every per-frame detection of the four corners corresponds to the *same*
    four 3-D world points. Stacking N detections gives us 4N point pairs for
    one PnP — much more robust than using any single frame, and tolerates a
    handful of outlier detections via RANSAC.

    Args:
        corners_image_per_frame: list of (4, 2) pixel corners, one per detected
            frame. Order must match ``corners_world`` (TL, TR, BR, BL).
        corners_world: (4, 3) world-frame tag corner coordinates.
        K, dist: camera intrinsics and distortion (OpenCV convention).
        nominal_T_world_cam: optional initial guess to seed the iterative refine.
        ransac_reproj_threshold_px: max reprojection error (px) for an inlier.
        ransac_iterations: max RANSAC iterations.

    Returns:
        ``(T_world_cam, mean_inlier_reproj_err_px, info)`` where ``info``
        carries ``n_frames``, ``n_inlier_points``, ``per_frame_err`` (px,
        one per input frame), and ``inlier_frame_mask`` (bool array, length
        N, True iff *all four* corners of that frame were RANSAC inliers).
    """
    import cv2

    if len(corners_image_per_frame) == 0:
        raise ValueError("Need at least one frame's corner detection.")

    K = np.asarray(K, dtype=np.float32)
    dist = (np.asarray(dist, dtype=np.float32)
            if dist is not None else np.zeros((5,), dtype=np.float32))

    n_frames = len(corners_image_per_frame)
    object_points = np.tile(
        corners_world.astype(np.float32), (n_frames, 1)
    ).reshape(-1, 1, 3)                                   # (4N, 1, 3)
    image_points = np.concatenate(
        [c.astype(np.float32).reshape(-1, 1, 2) for c in corners_image_per_frame],
        axis=0,
    )                                                     # (4N, 1, 2)

    # Seed with the nominal pose so the iterative step inside RANSAC converges.
    use_extrinsic_guess = nominal_T_world_cam is not None
    if use_extrinsic_guess:
        T_cw0 = np.linalg.inv(nominal_T_world_cam)
        rvec0, _ = cv2.Rodrigues(T_cw0[:3, :3].astype(np.float32))
        tvec0 = T_cw0[:3, 3].astype(np.float32).reshape(3, 1)
    else:
        rvec0 = np.zeros((3, 1), dtype=np.float32)
        tvec0 = np.zeros((3, 1), dtype=np.float32)

    # Strategy: when we have a strong seed (IPPE_SQUARE), prefer plain iterative
    # PnP over RANSAC. RANSAC's per-iteration sampling discards the seed and
    # can converge on a degenerate subset that fits 4 points perfectly but
    # doesn't generalise. Iterative PnP starting from the IPPE_SQUARE pose
    # sits in the right basin and uses all 4N point pairs at once.
    #
    # If the seed is supplied (which is the recommended path from
    # ``calibrate_extrinsic.py``), do iterative PnP over all points then
    # rerun RANSAC ONLY for outlier-frame detection (informational).
    if use_extrinsic_guess:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist,
            rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP failed to converge")
        # Mark inliers post-hoc using the same threshold for the report.
        proj_check, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        per_pt_err = np.linalg.norm(
            proj_check.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
        inliers_idx = np.where(per_pt_err <= ransac_reproj_threshold_px)[0]
    elif n_frames == 1:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist,
            rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP failed to converge")
        inliers_idx = np.arange(4, dtype=np.int32)
    else:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, dist,
            iterationsCount=ransac_iterations,
            reprojectionError=ransac_reproj_threshold_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 4:
            raise RuntimeError(
                f"cv2.solvePnPRansac failed (ok={ok}, "
                f"inliers={None if inliers is None else len(inliers)})"
            )
        inliers_idx = inliers.ravel()

        # Refine on inliers only for a tighter solution.
        ok, rvec, tvec = cv2.solvePnP(
            object_points[inliers_idx], image_points[inliers_idx], K, dist,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP refinement on inliers failed")

    # Convert to T_world_cam and compute reprojection stats.
    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec.ravel()
    T_world_cam = np.linalg.inv(T_cw)

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err_per_pt = np.linalg.norm(
        proj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
    )                                                     # (4N,)
    inlier_pt_mask = np.zeros(len(err_per_pt), dtype=bool)
    inlier_pt_mask[inliers_idx] = True

    per_frame_err = err_per_pt.reshape(n_frames, 4).mean(axis=1)
    inlier_frame_mask = inlier_pt_mask.reshape(n_frames, 4).all(axis=1)
    mean_inlier_err = float(err_per_pt[inlier_pt_mask].mean())

    info = {
        "n_frames":            n_frames,
        "n_inlier_points":     int(inlier_pt_mask.sum()),
        "n_inlier_frames":     int(inlier_frame_mask.sum()),
        "per_frame_err":       per_frame_err,
        "inlier_frame_mask":   inlier_frame_mask,
    }
    return T_world_cam, mean_inlier_err, info


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
