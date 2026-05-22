"""Desk-based Aria extrinsic calibration — library + one-call entry point.

Public entry points (most callers only need these two):

  * ``compute_nominal_aria_pose(cfg)`` — pure math from ``SceneConfig``;
    returns ``T_world_cam = T_world_base @ ARIA_EXTRINSICS_RIGHT``.
  * ``refine_aria_extrinsic(sam_mask_path, cfg)`` — loads the SAM table-mask
    NPZ, extracts three feature lines, runs LM refinement off the nominal
    pose, returns the refined ``T_world_cam`` and delta stats.

Internals (kept module-public so the experiments folder can reuse them):

  * ``extract_feature_lines`` — per-pixel SAM frequency → top/left/seam 2D
    lines via RANSAC.
  * ``refine_world_pose`` — generic LM over a 6-DOF twist correction aligning
    sampled 3D-line points to 2D image lines.
  * ``project_world_points``, ``se3_from_twist``, ``intrinsics_matrix`` —
    helpers reused by diagnostics in ``3dv/experiments/table_align_egoverse/``.

Pure numpy + scipy. No Isaac Sim, no pxr.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ─────────────────────────────────────────────────────────────────────────────
def intrinsics_matrix(intrinsics: dict) -> np.ndarray:
    """Build the 3×3 K from a ``CameraConfig.intrinsics`` dict."""
    return np.array([
        [intrinsics["fx"],            0.0, intrinsics["cx"]],
        [           0.0, intrinsics["fy"], intrinsics["cy"]],
        [           0.0,            0.0,             1.0],
    ])


def compute_nominal_aria_pose(cfg) -> np.ndarray:
    """Return ``T_world_cam`` from ``cfg`` alone — no live USD stage needed.

    Computed as ``T_world_base @ ARIA_EXTRINSICS_RIGHT`` where
    ``T_world_base`` is the wrapper's effective pose: translation
    ``cfg.robot.mount_xyz`` and rotation from ``cfg.robot.mount_rpy``
    (identity by default). Matches what ``scene._add_robot`` places the
    kinematic ``panda_link0`` at, so the result is identical to reading
    the prim's world transform off a built stage.
    """
    from .constants import ARIA_EXTRINSICS_RIGHT     # local import: pure-py

    T_world_base = np.eye(4)
    T_world_base[:3, 3] = np.asarray(cfg.robot.mount_xyz, dtype=float)
    if any(r != 0.0 for r in cfg.robot.mount_rpy):
        from scipy.spatial.transform import Rotation
        T_world_base[:3, :3] = Rotation.from_euler(
            "XYZ", cfg.robot.mount_rpy
        ).as_matrix()
    return T_world_base @ np.asarray(ARIA_EXTRINSICS_RIGHT, dtype=float)


def refine_aria_extrinsic(sam_mask_path: Path, cfg) -> dict:
    """Run desk-based extrinsic refinement on a SAM table-mask NPZ.

    Loads ``sam_mask_path`` (must contain key ``mask`` of shape ``(N, H, W)``
    in ``{0, 1}``), aggregates into a per-pixel frequency map, extracts the
    three table-edge lines, and refines the nominal Aria pose (from
    :func:`compute_nominal_aria_pose`) by LM-fitting against those lines.

    Returns a dict::

        {
            'T_world_cam':           (4, 4) refined pose,
            'T_world_cam_nominal':   (4, 4) starting pose,
            'xi':                    (6,)  LM twist correction,
            'residual_rms_px':       float scalar,
            'delta_pos_mm':          (3,)  refined - nominal translation in mm,
            'delta_rot_deg':         float rotation angle in degrees,
            'message':               scipy LM termination message,
        }

    Raises ``KeyError`` if the NPZ is missing ``mask`` and ``RuntimeError`` if
    any of the three feature lines couldn't be extracted from the mask.
    """
    from .constants import (                          # local import: pure-py
        TABLE_LEFT_EDGE_WORLD,
        TABLE_SEAM_WORLD,
        TABLE_TOP_EDGE_WORLD,
    )

    sam_mask_path = Path(sam_mask_path)
    with np.load(sam_mask_path) as d:
        if "mask" not in d.files:
            raise KeyError(
                f"{sam_mask_path} missing required key 'mask' "
                f"(found: {d.files})"
            )
        masks = d["mask"]
    if masks.ndim != 3:
        raise ValueError(f"Expected SAM masks of shape (N, H, W); got {masks.shape}")

    sam_freq = masks.astype(np.float32).mean(axis=0)
    T_nominal = compute_nominal_aria_pose(cfg)
    K = intrinsics_matrix(cfg.camera.intrinsics)

    # Predict the seam search band from the nominal pose so we adapt when
    # the nominal drifts between sessions; ±100 px around the projected
    # seam midpoint is the standard window from the dev/experiments work.
    pix, _ = project_world_points(T_nominal, TABLE_SEAM_WORLD, K)
    u_mid = float(0.5 * (pix[0, 0] + pix[1, 0]))
    img_w = int(K[0, 2] * 2)
    seam_u_range = (
        max(0, int(u_mid) - 100),
        min(img_w, int(u_mid) + 100),
    )

    feat = extract_feature_lines(sam_freq, seam_u_range=seam_u_range)
    for key in ("top", "left", "seam"):
        if feat[key] is None:
            raise RuntimeError(
                f"Could not extract '{key}' line from SAM mask "
                f"{sam_mask_path.name}; check the mask coverage or pass a "
                f"hand-tuned seam_u_range."
            )

    result = refine_world_pose(
        T_nominal, K,
        [
            (TABLE_TOP_EDGE_WORLD,  feat["top"]),
            (TABLE_LEFT_EDGE_WORLD, feat["left"]),
            (TABLE_SEAM_WORLD,      feat["seam"]),
        ],
    )

    # Convenience stats for the caller's log line.
    T = result["T_world_cam"]
    dp = T[:3, 3] - T_nominal[:3, 3]
    R_delta = T[:3, :3] @ T_nominal[:3, :3].T
    rot_deg = float(np.rad2deg(np.arccos(
        np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0))))

    result["delta_pos_mm"] = dp * 1000.0
    result["delta_rot_deg"] = rot_deg
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) refinement
# ─────────────────────────────────────────────────────────────────────────────
def se3_from_twist(xi: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from ``(tx, ty, tz, rx, ry, rz)``.

    Rotation is axis-angle (magnitude in radians); translation is applied in
    the frame of the matrix this transform is left-multiplied onto.
    """
    xi = np.asarray(xi, dtype=float)
    t = xi[:3]
    r = xi[3:]
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        R = np.eye(3)
    else:
        k = r / theta
        Kx = np.array([
            [    0, -k[2],  k[1]],
            [ k[2],     0, -k[0]],
            [-k[1],  k[0],     0],
        ])
        R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def project_world_points(
    T_world_cam: np.ndarray, pts_world: np.ndarray, K: np.ndarray,
) -> tuple:
    """Project ``(N, 3)`` world points through a column-vector-convention
    ``T_world_cam`` (camera-in-world) with pinhole intrinsics ``K``.

    Returns
    -------
    pixels : ``(N, 2)`` image pixel coordinates.
    depth  : ``(N,)``  camera-frame Z of each point.
    """
    T_cam_world = np.linalg.inv(T_world_cam)
    pts = np.asarray(pts_world, dtype=float)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1).T
    pts_cam = (T_cam_world @ pts_h)[:3].T
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def _sample_line(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a * (1 - ts) + b * ts


def refine_world_pose(
    T_world_cam_nominal: np.ndarray,
    K: np.ndarray,
    correspondences: Iterable,
    *,
    n_samples: int = 40,
    max_nfev: int = 200,
) -> dict:
    """Refine ``T_world_cam`` to align projected 3D lines with 2D image lines.

    Parameters
    ----------
    T_world_cam_nominal
        4×4 column-vector camera-in-world pose used as the starting point.
    K
        3×3 pinhole intrinsics matrix.
    correspondences
        Iterable of ``(endpoints_3d, line_2d)`` where
        ``endpoints_3d`` is ``(2, 3)`` world-frame line endpoints and
        ``line_2d`` is a ``(nx, ny, c)`` implicit line (``None`` entries are
        skipped). At least 3 valid correspondences are required.
    n_samples
        Points sampled uniformly along each 3D segment.
    max_nfev
        Maximum Levenberg–Marquardt function evaluations.

    Returns
    -------
    dict with keys:
        ``T_world_cam``           refined 4×4 pose
        ``T_world_cam_nominal``   echoed input
        ``xi``                    6-vector twist correction
        ``residuals``             final residuals (one per sample)
        ``residual_rms_px``       root-mean-squared residual magnitude
        ``status``, ``message``   scipy solver outputs
    """
    corr = [(np.asarray(e, float), np.asarray(l, float))
            for e, l in correspondences if l is not None]
    if len(corr) < 3:
        raise ValueError(f"Need at least 3 line correspondences, got {len(corr)}")

    samples = [_sample_line(e[0], e[1], n_samples) for e, _ in corr]

    def residuals(xi):
        T = se3_from_twist(xi) @ T_world_cam_nominal
        out = []
        for s, (_, line) in zip(samples, corr):
            pix, z = project_world_points(T, s, K)
            if np.any(z <= 0):
                out.extend([1e3] * len(s))
                continue
            nx, ny, c = line
            out.extend(nx * pix[:, 0] + ny * pix[:, 1] + c)
        return np.asarray(out)

    sol = least_squares(
        residuals, np.zeros(6), method="lm", max_nfev=max_nfev,
        x_scale=np.full(6, 0.01),
    )
    T_refined = se3_from_twist(sol.x) @ T_world_cam_nominal

    return {
        "T_world_cam": T_refined,
        "T_world_cam_nominal": T_world_cam_nominal,
        "xi": sol.x,
        "residuals": sol.fun,
        "residual_rms_px": float(np.sqrt(np.mean(sol.fun ** 2))),
        "status": int(sol.status),
        "message": str(sol.message),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAM table-mask feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _fit_line_ransac(us, vs, *, n_iter=400, thresh=2.0, rng=None):
    """RANSAC fit of ``nx * u + ny * v + c = 0`` (|(nx, ny)| = 1).

    Returns ``(nx, ny, c)`` refit by PCA on the inliers, or ``None`` if fewer
    than two inliers were found.
    """
    if rng is None:
        rng = np.random.default_rng()
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    n = len(us)
    if n < 2:
        return None

    best_count = 0
    best_inl = np.zeros(n, dtype=bool)
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        dx, dy = us[j] - us[i], vs[j] - vs[i]
        L = np.hypot(dx, dy)
        if L < 1e-6:
            continue
        nx, ny = -dy / L, dx / L
        c = -(nx * us[i] + ny * vs[i])
        d = np.abs(nx * us + ny * vs + c)
        inl = d < thresh
        if inl.sum() > best_count:
            best_count = int(inl.sum())
            best_inl = inl

    if best_count < 2:
        return None

    pts = np.stack([us[best_inl], vs[best_inl]], axis=1)
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    _, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, -1]                          # largest eigenvalue
    normal = np.array([-direction[1], direction[0]])
    c = -float(normal @ mu)
    return np.array([float(normal[0]), float(normal[1]), c])


def _per_column_top(mask):
    """First row index per column where ``mask`` is 1 (or -1 if empty)."""
    W = mask.shape[1]
    out = np.full(W, -1, dtype=np.int32)
    cols_any = mask.any(axis=0)
    if cols_any.any():
        out[cols_any] = np.argmax(mask[:, cols_any], axis=0)
    return out


def _per_row_left(mask):
    """First col index per row where ``mask`` is 1 (or -1 if empty)."""
    H = mask.shape[0]
    out = np.full(H, -1, dtype=np.int32)
    rows_any = mask.any(axis=1)
    if rows_any.any():
        out[rows_any] = np.argmax(mask[rows_any, :], axis=1)
    return out


def extract_feature_lines(
    sam_freq: np.ndarray,
    *,
    mask_threshold: float = 0.5,
    seam_u_range: tuple = (200, 450),
    rng_seed: int = 0,
) -> dict:
    """Extract top/left/seam line features from a SAM per-pixel frequency map.

    Parameters
    ----------
    sam_freq
        ``(H, W)`` array in ``[0, 1]`` — typically ``sam_masks.mean(axis=0)``.
    mask_threshold
        Binary mask threshold used for the edge fits (default ``0.5``).
    seam_u_range
        Inclusive-exclusive column range searched for the seam. Default
        ``(200, 450)`` covers the centre band of a 640-wide Aria image;
        pass a window centred on the predicted seam location for best results.
    rng_seed
        RANSAC seed for reproducibility.

    Returns
    -------
    dict with keys ``top``, ``left``, ``seam`` (each a ``(3,)`` implicit line
    array or ``None`` if extraction failed), plus diagnostic data
    ``sam_med`` (uint8 binary mask) and ``seam_pts`` (candidate seam pixels).
    """
    if sam_freq.ndim != 2:
        raise ValueError(f"sam_freq must be 2D, got shape {sam_freq.shape}")

    H, W = sam_freq.shape
    sam_med = (sam_freq > mask_threshold).astype(np.uint8)

    rng = np.random.default_rng(rng_seed)

    # Top edge: per-column first-white row, restricted to the flat central band.
    top_row = _per_column_top(sam_med)
    xs = np.arange(W)
    top_smooth = np.convolve(top_row.astype(float), np.ones(15) / 15, mode="same")
    top_slope = np.gradient(top_smooth)
    flat = (np.abs(top_slope) < 0.3) & (top_row > 10) & (top_row < H - 10)
    top_line = _fit_line_ransac(xs[flat], top_row[flat], rng=rng) if flat.any() else None

    # Left edge: per-row first-white column, restricted to the upper slanted band.
    left_col = _per_row_left(sam_med)
    ys = np.arange(H)
    left_valid = (left_col > 5) & (left_col < W - 5) & (ys < int(H * 0.8))
    left_smooth = np.convolve(left_col.astype(float), np.ones(15) / 15, mode="same")
    left_slope = np.gradient(left_smooth)
    slanted = left_valid & (np.abs(left_slope) > 0.3)
    left_line = _fit_line_ransac(left_col[slanted], ys[slanted], rng=rng) if slanted.any() else None

    # Seam: per-row minimum of ``sam_freq`` in the search band — a thin vertical
    # region of "mostly-not-table" flanked by "mostly-table".
    u0, u1 = int(seam_u_range[0]), int(seam_u_range[1])
    u0, u1 = max(0, u0), min(W, u1)
    seam_pts = []
    for r in range(H):
        if sam_freq[r].max() < mask_threshold:
            continue
        roi = sam_freq[r] > 0.3
        if roi.sum() < 30:
            continue
        row_vals = sam_freq[r].copy()
        row_vals[~roi] = 1.0
        sub = row_vals[u0:u1]
        if sub.size == 0 or sub.min() > mask_threshold:
            continue
        c_min = int(np.argmin(sub)) + u0
        seam_pts.append((c_min, r))
    seam_pts = np.array(seam_pts, dtype=np.float64) if seam_pts else np.zeros((0, 2))

    seam_line: Optional[np.ndarray] = None
    if len(seam_pts) >= 10:
        seam_line = _fit_line_ransac(seam_pts[:, 0], seam_pts[:, 1],
                                     n_iter=500, rng=rng)

    return {
        "top": top_line,
        "left": left_line,
        "seam": seam_line,
        "sam_med": sam_med,
        "seam_pts": seam_pts,
    }
