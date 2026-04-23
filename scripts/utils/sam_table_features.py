"""Extract dominant 2D line features from aggregated SAM table masks.

Given a per-pixel table-probability map ``sam_freq`` (the mean of N binary SAM
masks over a sequence), we locate three lines used by the camera extrinsic
refiner:

    top   — the far edge of the table (near-horizontal, image upper band)
    left  — the slanted left edge of the table (diagonal)
    seam  — the physical seam running down the centre of the table top

Each is returned as an implicit line ``(nx, ny, c)`` with unit normal such
that ``nx * u + ny * v + c == 0`` on the line.

Pure numpy — no Isaac Sim or pxr dependencies.
"""

from typing import Optional

import numpy as np


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
        Inclusive-exclusive column range searched for the seam.  Default
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
