"""6-DOF camera-pose refinement by aligning projected 3D lines to 2D image lines.

Each correspondence is a 3D line segment on the table (endpoints in world
frame) paired with a 2D implicit line ``(nx, ny, c)`` fitted to the SAM mask.
The residual for a correspondence is the signed perpendicular distance from
each projected 3D-line sample point to the 2D line; Levenberg–Marquardt over
six twist parameters drives the residuals to zero.

Pure numpy + scipy.  No Isaac Sim.
"""

from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


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
        skipped).  At least 3 valid correspondences are required.
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
