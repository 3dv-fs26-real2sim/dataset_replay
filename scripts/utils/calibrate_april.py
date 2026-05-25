"""AprilTag-based camera-extrinsic calibration — startup library.

Public entry points (most callers only need these two):

  * ``nominal_oakd_pose(cfg)`` — pure math from ``MapleSceneConfig``;
    returns ``T_world_cam`` from the configured (position, lookat, up).
  * ``calibrate_from_h5(h5_path, cfg)`` — scan every H5 frame, detect the
    AprilTag, run joint PnP+RANSAC, return the refined ``T_world_cam``
    plus diagnostics. The replay script auto-calls this at startup.

Internals (kept module-public so ``scripts/calibrate/calibrate_april.py``
can reuse them):

  * ``tag_corners_world``, ``_tag_world_pose_with_rotation``
  * ``detect_apriltag``, ``detect_apriltag_with_quality``
  * ``solve_extrinsic_pnp``, ``solve_extrinsic_pnp_multiview``
  * ``solve_pose_square_planar``, ``select_correct_branch``
  * ``lookat_to_T_world_cam``

Pure CV / numpy — no Isaac Sim imports. Safe to call before SimulationApp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ─────────────────────────────────────────────────────────────────────────────
def nominal_oakd_pose(cfg) -> np.ndarray:
    """Return ``T_world_cam`` from ``cfg.camera``'s nominal (position, lookat, up).

    Used as PnP seed inside :func:`calibrate_from_h5` and as the fallback
    when ``kinematic_replay_maple.py`` runs without ``--h5`` or with
    ``--no-calibrate``, or when the AprilTag isn't detectable.
    """
    return lookat_to_T_world_cam(
        cfg.camera.nominal_position,
        cfg.camera.nominal_lookat,
        cfg.camera.nominal_up,
    )


def calibrate_from_h5(
    h5_path: Path | str,
    cfg,
    *,
    camera: str | None = None,
    min_decision_margin: float = 30.0,
    max_hamming: int = 0,
    ransac_threshold_px: float = 3.0,
) -> dict:
    """Detect the AprilTag across every H5 frame, run joint PnP+RANSAC,
    return refined pose and diagnostics.

    Parameters
    ----------
    h5_path
        Path to a maple-format H5 with images stored under
        ``observations/images/<camera>/color``. The per-frame K stored under
        ``observations/images/<camera>/intrinsics`` is intentionally NOT
        used — PnP runs against ``cfg.camera.intrinsics`` (the doc K), which
        overlay comparisons showed aligns sim to real best.
    cfg
        ``MapleSceneConfig`` (anything with the right ``camera``/``apriltag``
        attributes works).
    camera
        H5 camera key. Defaults to ``cfg.camera.name``.
    min_decision_margin, max_hamming
        Quality filters applied per-frame to drop weak/ambiguous detections.
    ransac_threshold_px
        Multi-view RANSAC reprojection inlier threshold.

    Returns
    -------
    dict with keys:
        ``T_world_cam``        : (4, 4) refined pose
        ``T_world_cam_nominal``: (4, 4) starting pose (nominal lookat)
        ``n_frames_scanned``   : int — total H5 frames inspected
        ``n_frames_detected``  : int — frames where the tag was detected after filters
        ``n_inlier_frames``    : int — frames whose all-4 corners survived RANSAC
        ``residual_rms_px``    : float — mean inlier reprojection error
        ``delta_pos_mm``       : (3,)   refined - nominal translation, in mm
        ``delta_rot_deg``      : float  rotation magnitude (deg) of refined vs nominal
        ``frame_indices``      : list[int] detected-frame indices
        ``per_frame_err``      : (n_det,) float per-frame reprojection error
        ``inlier_frame_mask``  : (n_det,) bool RANSAC inlier mask
        ``K``, ``dist``        : intrinsics used (SceneConfig doc K)

    Raises
    ------
    RuntimeError if no frames yield a usable detection.

    Time
    ----
    ~4–6 s on the bundled ``20250922_143954.h5`` (300 frames, 480×270);
    ~8–12 s on a 600-frame H5. Full-scan is always on — we revisit if
    the wait gets disruptive in practice.
    """
    import cv2

    from .h5_loader import H5Reader

    h5_path = Path(h5_path)
    cam_name = camera or cfg.camera.name
    nominal = nominal_oakd_pose(cfg)

    # Use the SceneConfig "doc" K (spec-sheet) for PnP — NOT the H5-stored K.
    # Overlay comparisons showed doc K aligns the sim to the real recording
    # best; the H5-stored K is mis-scaled. The sim render FOV uses this same
    # doc K (see config_maple.OakDCameraConfig), so PnP and render share one
    # K — a mismatch is what makes the sim feed look zoomed in the overlay.
    Kd = cfg.camera.intrinsics
    K = np.array([
        [Kd["fx"], 0.0, Kd["cx"]],
        [0.0, Kd["fy"], Kd["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    dist = np.asarray(cfg.camera.distortion, dtype=float)

    # ── Frame-by-frame detection ───────────────────────────────────────────
    detector = _make_detector(cfg.apriltag.family)
    frame_indices: list[int] = []
    corners_list: list[np.ndarray] = []
    margins: list[float] = []
    rejected = 0

    with H5Reader(h5_path, dataset="maple", camera=cam_name) as h5:
        n_total = h5.n_frames
        for i in range(n_total):
            rgb = h5.image(i)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            res = detect_apriltag_with_quality(
                gray, cfg.apriltag.family, cfg.apriltag.tag_id, detector=detector,
            )
            if res is None:
                continue
            corners, margin, hamming = res
            if margin < min_decision_margin or hamming > max_hamming:
                rejected += 1
                continue
            frame_indices.append(i)
            corners_list.append(corners)
            margins.append(margin)

    if not frame_indices:
        raise RuntimeError(
            f"No usable AprilTag detections in {h5_path.name} "
            f"(scanned {n_total} frames, {rejected} rejected by quality filters)."
        )

    # ── Disambiguate planar-PnP via IPPE_SQUARE on the highest-margin frame ─
    best_idx = int(np.argmax(margins))
    T_world_tag = _tag_world_pose_with_rotation(cfg)
    T_a, T_b, _ = solve_pose_square_planar(
        corners_list[best_idx], cfg.apriltag.edge_size, T_world_tag, K, dist,
    )
    T_seed, _ = select_correct_branch(T_a, T_b, T_world_tag,
                                      seed_T_world_cam=nominal)

    # ── Joint PnP+RANSAC over all detections ───────────────────────────────
    corners_world = tag_corners_world(cfg)
    T_world_cam, err, info = solve_extrinsic_pnp_multiview(
        corners_list, corners_world, K, dist,
        nominal_T_world_cam=T_seed,
        ransac_reproj_threshold_px=ransac_threshold_px,
    )

    # Delta stats refined vs nominal.
    dp = T_world_cam[:3, 3] - nominal[:3, 3]
    R_delta = T_world_cam[:3, :3] @ nominal[:3, :3].T
    rot_deg = float(np.rad2deg(np.arccos(
        np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0))))

    return {
        "T_world_cam":         T_world_cam,
        "T_world_cam_nominal": nominal,
        "n_frames_scanned":    int(n_total),
        "n_frames_detected":   int(info["n_frames"]),
        "n_inlier_frames":     int(info["n_inlier_frames"]),
        "residual_rms_px":     float(err),
        "delta_pos_mm":        (dp * 1000.0).astype(float),
        "delta_rot_deg":       rot_deg,
        "frame_indices":       frame_indices,
        "per_frame_err":       info["per_frame_err"],
        "inlier_frame_mask":   info["inlier_frame_mask"],
        "margins":             margins,
        "K":                   K,
        "dist":                dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# lookat → T_world_cam (used by nominal_oakd_pose and PnP seeding)
# ─────────────────────────────────────────────────────────────────────────────
def lookat_to_T_world_cam(
    position: tuple[float, float, float] | np.ndarray,
    lookat:   tuple[float, float, float] | np.ndarray,
    up:       tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Build a 4×4 OpenCV-convention camera-in-world transform from a
    ``(position, lookat, up)`` triple.

    Camera axes (in world frame): +Z = forward (toward ``lookat``),
    +X = right, +Y = down. With world-up = +Z, "down in image" is roughly
    -world-up, so the y-axis is the down-cross-forward direction.
    """
    pos    = np.asarray(position, dtype=float)
    lookat = np.asarray(lookat,   dtype=float)
    up     = np.asarray(up,       dtype=float)

    forward = lookat - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        raise ValueError("lookat direction is parallel to 'up'; pick a different up vector.")
    right /= n
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = pos
    return T


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag corner geometry
# ─────────────────────────────────────────────────────────────────────────────
def tag_corners_world(cfg) -> np.ndarray:
    """Return the AprilTag's four corners in world frame, in the order
    ``pupil_apriltags.Detection.corners`` returns them (BL, BR, TR, TL of
    the tag's intrinsic image), applied to the **rotated** tag pose
    (``T_world_tag`` includes the printed quad's ``rotation_z_deg``).

    NOTE on planar-PnP ambiguity: 4 coplanar points admit two camera-pose
    solutions related by a flip about an axis in the tag plane. The
    wrapper in :func:`calibrate_from_h5` uses :func:`solve_pose_square_planar`
    + :func:`select_correct_branch` to pick the geometrically valid one and
    seeds the iterative multi-view refinement with it.

    **Caveat — sim/real mismatch:** the reported PnP camera pose is only
    physically meaningful if the tag's *world* pose (``cfg.apriltag``) matches
    where the printed tag actually sits in the room. If the SceneConfig has
    the wrong tag pose, PnP will still find a self-consistent fit but rotate
    the camera to compensate.
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


def _tag_world_pose_with_rotation(cfg) -> np.ndarray:
    """Return T_world_tag including the printed-tag's ``rotation_z_deg``.

    ``cfg.apriltag_world_pose()`` is identity-rotation; the visual quad has
    an extra Z rotation that :func:`utils.apriltag.add_apriltag_plane` applies
    in USD. For IPPE_SQUARE we need the *visible* tag's frame, so include it.
    """
    T = cfg.apriltag_world_pose().copy()
    theta = np.deg2rad(cfg.apriltag.rotation_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    T[:3, :3] = T[:3, :3] @ Rz
    return T


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag detection
# ─────────────────────────────────────────────────────────────────────────────
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
    ``pupil_apriltags`` returns them (which matches :func:`tag_corners_world`).
    Returns ``None`` if the requested tag id was not detected.
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


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame PnP (kept module-public for diagnostic tooling)
# ─────────────────────────────────────────────────────────────────────────────
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

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    return T_world_cam, float(err.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Planar-square PnP with branch selection
# ─────────────────────────────────────────────────────────────────────────────
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

    ``corners_image`` MUST be in the canonical (BL, BR, TR, TL) order that
    pupil-apriltags returns (the function re-orders internally to match
    OpenCV's IPPE_SQUARE convention).
    """
    import cv2

    e = edge_size / 2.0
    # IPPE_SQUARE requires (TL, TR, BR, BL) ordering of the marker-frame
    # object points per OpenCV docs. pupil-apriltags returns corners in
    # (BL, BR, TR, TL) order — re-pair the image points to match.
    obj_pts_marker = np.array([
        [-e, +e, 0.0],   # TL
        [+e, +e, 0.0],   # TR
        [+e, -e, 0.0],   # BR
        [-e, -e, 0.0],   # BL
    ], dtype=np.float64).reshape(-1, 1, 3)
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
      1. Prefer the branch with the camera on the +Z side of the tag plane
         (camera looks DOWN at a flat-on-table tag).
      2. If both (or neither) pass that test, fall back to the branch
         closest to ``seed_T_world_cam`` in translation.

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

    def seed_dist(T_wc):
        if seed_T_world_cam is None:
            return 0.0
        return float(np.linalg.norm(T_wc[:3, 3] - seed_T_world_cam[:3, 3]))

    if seed_dist(T_world_cam_a) <= seed_dist(T_world_cam_b):
        return T_world_cam_a, 0
    return T_world_cam_b, 1


# ─────────────────────────────────────────────────────────────────────────────
# Multi-view PnP (joint over many frames of the same static tag)
# ─────────────────────────────────────────────────────────────────────────────
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

    The camera is rigidly mounted and the AprilTag is at a fixed world pose,
    so every per-frame detection of the four corners corresponds to the
    *same* four 3-D world points. Stacking N detections gives us 4N point
    pairs for one PnP — much more robust than using any single frame, and
    tolerates a handful of outlier detections via RANSAC.

    Returns
    -------
    ``(T_world_cam, mean_inlier_reproj_err_px, info)`` where ``info`` carries:
      - ``n_frames``           : int
      - ``n_inlier_points``    : int
      - ``n_inlier_frames``    : int
      - ``per_frame_err``      : (N,) px, one per input frame
      - ``inlier_frame_mask``  : (N,) bool — True iff all four corners
                                  of that frame were RANSAC inliers.
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

    use_extrinsic_guess = nominal_T_world_cam is not None
    if use_extrinsic_guess:
        T_cw0 = np.linalg.inv(nominal_T_world_cam)
        rvec0, _ = cv2.Rodrigues(T_cw0[:3, :3].astype(np.float32))
        tvec0 = T_cw0[:3, 3].astype(np.float32).reshape(3, 1)
    else:
        rvec0 = np.zeros((3, 1), dtype=np.float32)
        tvec0 = np.zeros((3, 1), dtype=np.float32)

    # Strategy: when we have a strong seed (IPPE_SQUARE), prefer plain
    # iterative PnP over RANSAC. RANSAC's per-iteration sampling discards
    # the seed and can converge on a degenerate subset that fits 4 points
    # perfectly but doesn't generalise. Iterative PnP starting from the
    # IPPE_SQUARE pose sits in the right basin and uses all 4N point pairs
    # at once.
    if use_extrinsic_guess:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, dist,
            rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP failed to converge")
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

        ok, rvec, tvec = cv2.solvePnP(
            object_points[inliers_idx], image_points[inliers_idx], K, dist,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP refinement on inliers failed")

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
