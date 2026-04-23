"""Refine the Aria camera's world-frame pose from a SAM table-mask NPZ.

Aligns the projected 3D table edges (top, left, seam — all on the table-top
plane) with dominant 2D lines extracted from an aggregated SAM mask.  The
refined ``T_world_cam`` is saved as an NPZ that ``kinematic_replay.py``
consumes via ``--refined-extrinsic``.

Runs without Isaac Sim — only ``pxr`` (standalone) is needed to read the
nominal robot-base transform out of the USD scene.

Examples
--------
    # Single-arm scene, writes data/sam_masks_aria_extrinsic.npz
    python scripts/refine_camera_extrinsic.py --sam-masks data/sam_masks.npz

    # Dual-arm scene with diagnostic image
    python scripts/refine_camera_extrinsic.py \\
        --sam-masks data/sam_masks_20250829.npz \\
        --mode dual --diagnostic-png
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from utils.constants import (
    CAMERA_CONFIGS, FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    TABLE_LEFT_EDGE_WORLD, TABLE_SEAM_WORLD, TABLE_TOP_EDGE_WORLD,
    USD_PATH_DUAL, USD_PATH_SINGLE,
)
from utils.poses import average_poses
from utils.pose_refine import project_world_points, refine_world_pose
from utils.sam_table_features import extract_feature_lines


def _intrinsics_matrix(intr: dict) -> np.ndarray:
    return np.array([
        [intr["fx"],        0.0, intr["cx"]],
        [       0.0, intr["fy"], intr["cy"]],
        [       0.0,        0.0,        1.0],
    ])


def _prim_world_transform(stage, prim_path: str) -> np.ndarray:
    """Column-vector-convention world transform of a USD prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T


def compute_nominal_world_pose(mode: str, camera_name: str) -> np.ndarray:
    """Reconstruct the nominal ``T_world_cam`` for the given scene mode.

    Matches ``utils.camera.compute_camera_world_pose`` but works from a
    stand-alone pxr stage (no Isaac Sim needed).
    """
    extr = CAMERA_CONFIGS[camera_name]["extrinsics"]
    usd_path = USD_PATH_SINGLE if mode == "single" else USD_PATH_DUAL
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    poses = [_prim_world_transform(stage, FRANKA_RIGHT_PATH) @ extr["right"]]
    if mode == "dual":
        poses.append(_prim_world_transform(stage, FRANKA_LEFT_PATH) @ extr["left"])
    return average_poses(poses) if len(poses) > 1 else poses[0]


def _predict_seam_u_range(T_world_cam, K, half_window: int = 100) -> tuple:
    """Predict the seam's image-column range from the current camera pose,
    so that feature extraction adapts when the true pose drifts.
    """
    pix, _ = project_world_points(T_world_cam, TABLE_SEAM_WORLD, K)
    u_mid = float(0.5 * (pix[0, 0] + pix[1, 0]))
    width = int(K[0, 2] * 2)        # cx * 2 ≈ image width
    u0 = max(0, int(u_mid) - half_window)
    u1 = min(width, int(u_mid) + half_window)
    return (u0, u1)


def _save_diagnostic_png(out_path: Path, sam_freq, feat, T_nominal, T_refined, K):
    """Before/after overlay showing projected edges on the SAM frequency map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = sam_freq.shape
    correspondences = [
        ("top",  TABLE_TOP_EDGE_WORLD,  (1.0, 1.0, 0.0)),
        ("left", TABLE_LEFT_EDGE_WORLD, (1.0, 0.6, 0.0)),
        ("seam", TABLE_SEAM_WORLD,      (1.0, 0.0, 1.0)),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, T) in zip(axs, [("nominal", T_nominal), ("refined", T_refined)]):
        ax.imshow(sam_freq, cmap="magma", vmin=0, vmax=1)
        for label, pts3d, color in correspondences:
            samples = np.linspace(pts3d[0], pts3d[1], 60)
            pix, z = project_world_points(T, samples, K)
            if np.any(z <= 0):
                continue
            ax.plot(pix[:, 0], pix[:, 1], color=color, lw=1.2, label=f"proj {label}")
        for label, color in [("top", "cyan"), ("left", "orange"), ("seam", "magenta")]:
            line = feat[label]
            if line is None:
                continue
            nx, ny, c = line
            direction = np.array([-ny, nx])
            mid = -c * np.array([nx, ny])
            ts = np.linspace(-1500, 1500, 3000)
            uu = mid[0] + ts * direction[0]
            vv = mid[1] + ts * direction[1]
            keep = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            ax.plot(uu[keep], vv[keep], color=color, lw=0.5, ls="--",
                    label=f"SAM {label}", alpha=0.7)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sam-masks", type=str, required=True,
                        help="NPZ with key 'mask' of shape (N, H, W) in {0, 1}")
    parser.add_argument("--camera", type=str, default="aria",
                        choices=list(CAMERA_CONFIGS.keys()),
                        help="Camera whose extrinsic is being refined (default: aria)")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "dual"],
                        help="Scene mode used to compute nominal T_world_cam (default: single)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path "
                             "(default: <sam_masks_dir>/<sam_stem>_<camera>_extrinsic.npz)")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="SAM frequency threshold for the binary mask (default: 0.5)")
    parser.add_argument("--seam-u-range", type=int, nargs=2, default=None,
                        metavar=("U0", "U1"),
                        help="Override the seam search band; default is predicted "
                             "from the nominal pose ±100 px.")
    parser.add_argument("--diagnostic-png", action="store_true",
                        help="Write a <output>.png showing nominal vs refined overlay")
    args = parser.parse_args()

    sam_path = Path(args.sam_masks)
    with np.load(sam_path) as d:
        if "mask" not in d.files:
            raise SystemExit(f"{sam_path} missing required key 'mask' "
                             f"(found: {d.files})")
        sam = d["mask"]
    if sam.ndim != 3:
        raise SystemExit(f"Expected SAM masks of shape (N, H, W), got {sam.shape}")
    print(f"[refine] SAM masks: {sam.shape} from {sam_path}")

    sam_freq = sam.astype(np.float32).mean(axis=0)

    T_nominal = compute_nominal_world_pose(args.mode, args.camera)
    K = _intrinsics_matrix(CAMERA_CONFIGS[args.camera]["intrinsics"])
    pos_nom = T_nominal[:3, 3]
    print(f"[refine] Nominal T_world_cam position: "
          f"[{pos_nom[0]:.4f}, {pos_nom[1]:.4f}, {pos_nom[2]:.4f}]")

    if args.seam_u_range is not None:
        seam_u_range = tuple(args.seam_u_range)
    else:
        seam_u_range = _predict_seam_u_range(T_nominal, K)
    print(f"[refine] Seam search band: u ∈ [{seam_u_range[0]}, {seam_u_range[1]})")

    feat = extract_feature_lines(
        sam_freq,
        mask_threshold=args.mask_threshold,
        seam_u_range=seam_u_range,
    )
    for key in ("top", "left", "seam"):
        if feat[key] is None:
            raise SystemExit(
                f"[refine] Could not extract '{key}' line from SAM mask — "
                "check --seam-u-range / --mask-threshold."
            )
        nx, ny, c = feat[key]
        print(f"[refine] SAM {key:<4} line:  nx={nx:+.4f}  ny={ny:+.4f}  c={c:+.3f}")

    result = refine_world_pose(
        T_nominal, K,
        [
            (TABLE_TOP_EDGE_WORLD,  feat["top"]),
            (TABLE_LEFT_EDGE_WORLD, feat["left"]),
            (TABLE_SEAM_WORLD,      feat["seam"]),
        ],
    )

    T_new = result["T_world_cam"]
    dp = T_new[:3, 3] - T_nominal[:3, 3]
    R_delta = T_new[:3, :3] @ T_nominal[:3, :3].T
    rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))))

    print(f"[refine] ξ = {np.array2string(result['xi'], precision=4)}")
    print(f"[refine] Δpos = [{dp[0]*1000:+.1f}, {dp[1]*1000:+.1f}, {dp[2]*1000:+.1f}] mm  "
          f"(|Δ|={np.linalg.norm(dp)*1000:.1f} mm)")
    print(f"[refine] Δrotation = {rot_deg:.3f} deg")
    print(f"[refine] residual rms = {result['residual_rms_px']:.3f} px  "
          f"({result['message']})")

    if args.output is None:
        out_path = sam_path.with_name(f"{sam_path.stem}_{args.camera}_extrinsic.npz")
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        T_world_cam=T_new,
        T_world_cam_nominal=T_nominal,
        xi=result["xi"],
        residual_rms_px=result["residual_rms_px"],
        camera=args.camera,
        mode=args.mode,
        sam_masks_path=str(sam_path),
    )
    print(f"[refine] saved → {out_path}")

    if args.diagnostic_png:
        png_path = out_path.with_suffix(".png")
        _save_diagnostic_png(png_path, sam_freq, feat, T_nominal, T_new, K)
        print(f"[refine] saved → {png_path}")


if __name__ == "__main__":
    main()
