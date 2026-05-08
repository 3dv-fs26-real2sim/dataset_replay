"""Render a depth image of the table from the Aria camera viewpoint in Isaac Sim.

Sets up the scene (table + ground) with the robot hidden, captures depth
from the calibrated Aria camera, and saves as a colorized PNG and raw NPZ.

Output resolution matches Aria / H5 convention: 480x640 (HxW).

With --mask, only table-surface pixels are kept (depth below a threshold);
all other pixels are set to 0.  The default threshold (1.0 m) sits in the
natural gap between the table surface (~0.33-0.77 m) and the ground plane
(~1.81 m+).

Usage:
    python dataset_replay/scripts/render_table_depth.py --headless
    python dataset_replay/scripts/render_table_depth.py --mask --headless
    python dataset_replay/scripts/render_table_depth.py --mask --mask-threshold 0.9 --headless
"""

import argparse

import numpy as np

from utils.app import add_common_args, create_app
from utils.constants import (
    CAMERA_CONFIGS, FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    FRANKA_LEFT_BASE_PATH, FRANKA_RIGHT_BASE_PATH, OUTPUT_DIR,
)

NUM_WARMUP_FRAMES = 10
DEFAULT_MASK_THRESHOLD = 1.0   # metres — in the gap between table and ground


parser = argparse.ArgumentParser(
    description="Render depth image of table from calibrated camera viewpoint",
)
add_common_args(parser)
parser.set_defaults(mode="single")
parser.add_argument(
    "--camera", type=str, default="aria", choices=list(CAMERA_CONFIGS.keys()),
    help="Camera calibration to use (default: aria)",
)
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="Output directory (default: dataset_replay/outputs)",
)
parser.add_argument(
    "--mask", action="store_true",
    help="Keep only table-surface pixels; set everything else to 0",
)
parser.add_argument(
    "--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD,
    help=f"Depth cutoff in metres for --mask (default: {DEFAULT_MASK_THRESHOLD})",
)
args = parser.parse_args()

# Resolve output directory.
output_dir = OUTPUT_DIR if args.output_dir is None else __import__("pathlib").Path(args.output_dir)

# Intrinsics determine the render resolution.
intrinsics = CAMERA_CONFIGS[args.camera]["intrinsics"]
render_width = intrinsics["width"]    # 640
render_height = intrinsics["height"]  # 480

simulation_app = create_app(args, width=render_width, height=render_height)

# ── Isaac Sim imports (must come after SimulationApp creation) ───────────────
import omni.replicator.core as rep                   # noqa: E402
from pxr import UsdGeom                              # noqa: E402

from utils.camera import setup_camera                # noqa: E402
from utils.scene import build_scene                  # noqa: E402


def main():
    # ── 1. Build scene programmatically ──────────────────────────────────────
    stage = build_scene(args.mode)

    # ── 2. Setup calibrated camera ───────────────────────────────────────────
    camera_prim_path = setup_camera(
        stage, args.camera, args.mode,
        FRANKA_LEFT_BASE_PATH, FRANKA_RIGHT_BASE_PATH,
    )
    print(f"[camera] Viewport set to {camera_prim_path}")

    # ── 3. Hide robot prims (keep transforms for camera calibration) ─────────
    robot_paths = [FRANKA_RIGHT_PATH]
    if args.mode == "dual":
        robot_paths.append(FRANKA_LEFT_PATH)

    for prim_path in robot_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            print(f"[scene] Hidden {prim_path}")

    # ── 4. Create render product + depth annotator ───────────────────────────
    rp = rep.create.render_product(camera_prim_path, (render_width, render_height))

    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([rp])

    # ── 5. Warm up renderer ──────────────────────────────────────────────────
    print(f"[render] Warming up ({NUM_WARMUP_FRAMES} frames at {render_width}x{render_height})...")
    for _ in range(NUM_WARMUP_FRAMES):
        rep.orchestrator.step(rt_subframes=8, pause_timeline=False)

    # ── 6. Read depth ────────────────────────────────────────────────────────
    depth = depth_annot.get_data()
    if isinstance(depth, dict):
        depth = depth["data"]
    depth = np.asarray(depth, dtype=np.float32)

    # Ensure correct shape (H, W).
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    print(f"[depth] Shape: {depth.shape}, dtype: {depth.dtype}")

    # Clamp infinite values for statistics.
    finite_mask = np.isfinite(depth)
    if finite_mask.any():
        d_min = float(depth[finite_mask].min())
        d_max = float(depth[finite_mask].max())
        d_mean = float(depth[finite_mask].mean())
        print(f"[depth] Range: [{d_min:.4f}, {d_max:.4f}] m, mean: {d_mean:.4f} m")
    else:
        print("[depth] Warning: no finite depth values")

    # ── 7. Apply table mask (optional) ──────────────────────────────────────
    if args.mask:
        table_mask = depth < args.mask_threshold
        n_table = int(table_mask.sum())
        print(f"[mask] threshold={args.mask_threshold:.2f} m  "
              f"table pixels={n_table} ({100 * n_table / depth.size:.1f}%)")
        depth = np.where(table_mask, depth, 0.0)

    # ── 8. Save outputs ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_masked" if args.mask else ""

    npz_path = output_dir / f"table_depth{suffix}.npz"
    np.savez_compressed(str(npz_path), depth=depth)
    print(f"[save] {npz_path}")

    # Colorized visualization.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = output_dir / f"table_depth{suffix}.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    plot_depth = depth.copy()
    # Show non-finite or masked-out (0) pixels as gray.
    plot_depth[(~np.isfinite(plot_depth)) | (plot_depth == 0)] = np.nan
    im = ax.imshow(plot_depth, cmap="turbo")
    plt.colorbar(im, ax=ax, label="Depth (m)")
    title = f"Table depth — {args.camera} camera ({args.mode} mode)"
    if args.mask:
        title += f"  [masked < {args.mask_threshold:.1f} m]"
    ax.set_title(title)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    fig.savefig(str(png_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {png_path}")

    # ── 9. Done ──────────────────────────────────────────────────────────────
    simulation_app.close()
    print("[done]")


if __name__ == "__main__":
    main()
