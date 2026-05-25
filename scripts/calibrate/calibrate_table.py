"""Sanity-check tool for SAM-table-based egoverse Aria refinement.

Runs the same ``utils.calibrate_table.refine_aria_extrinsic`` that
``kinematic_replay_egoverse.py`` calls at startup, then writes:

  * an ``.npz`` with the refined ``T_world_cam`` + diagnostics
  * a PNG overlay showing the detected SAM edge fits vs the projected
    sim table edges (top, left, seam)
  * (optional) an MP4 overlay sweeping the same projection over the H5

These outputs are **never read back by the replay scripts** — they exist
purely so you can spot-check that the auto-refinement is producing a
sensible pose.

Pure CV / numpy — no Isaac Sim.

Usage::

    python scripts/calibrate/calibrate_table.py \\
        --h5 data/egoverse/h5/<session>.h5
    python scripts/calibrate/calibrate_table.py --h5 ... \\
        --sam-mask data/egoverse/desk/<stem>_desk.npz --viz-mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.calibrate_table import (
    intrinsics_matrix,
    project_world_points,
    refine_aria_extrinsic,
)
from utils.config import PROJECT_ROOT
from utils.config_egoverse import EgoverseSceneConfig
from utils.constants import (
    TABLE_LEFT_EDGE_WORLD,
    TABLE_SEAM_WORLD,
    TABLE_TOP_EDGE_WORLD,
)


OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "outputs" / "calibration"


def _default_sam_mask_path(h5_path: Path) -> Path:
    return h5_path.parent.parent / "desk" / f"{h5_path.stem}_desk.npz"


def _draw_overlay(frame_rgb: np.ndarray, T_world_cam: np.ndarray,
                  K: np.ndarray, sam_freq: np.ndarray | None) -> np.ndarray:
    """Render the overlay: projected sim table edges + (optionally) the
    SAM frequency map as a translucent green layer."""
    import cv2

    img = frame_rgb.copy()
    H, W = img.shape[:2]

    if sam_freq is not None:
        mask = (sam_freq > 0.5).astype(np.uint8) * 255
        green = np.zeros_like(img)
        green[:, :, 1] = mask
        img = cv2.addWeighted(img, 1.0, green, 0.4, 0.0)

    # Project the three reference edges and draw them.
    edges = [
        (TABLE_TOP_EDGE_WORLD,  (255, 0, 0),   "top"),    # red
        (TABLE_LEFT_EDGE_WORLD, (0, 0, 255),   "left"),   # blue
        (TABLE_SEAM_WORLD,      (255, 255, 0), "seam"),   # cyan-ish
    ]
    for endpoints, color, label in edges:
        pix, z = project_world_points(T_world_cam, endpoints, K)
        if np.any(z <= 0):
            continue
        p0 = tuple(pix[0].astype(int))
        p1 = tuple(pix[1].astype(int))
        cv2.line(img, p0, p1, color, 2)
        cv2.putText(img, label, p0, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)

    return img


def _write_mp4_overlay(h5_path: Path, T_world_cam: np.ndarray, K: np.ndarray,
                       sam_freq: np.ndarray | None, camera: str,
                       output_path: Path, fps: float = 50.0) -> None:
    import imageio.v2 as imageio

    from utils.h5_loader import H5Reader

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path),
                                fps=max(1, int(round(fps))), codec="libx264")
    with H5Reader(h5_path, dataset="egoverse", camera=camera) as h5:
        n = h5.n_frames
        for i in range(n):
            rgb = h5.image(i)
            overlay = _draw_overlay(rgb, T_world_cam, K, sam_freq)
            writer.append_data(overlay)
    writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, required=True,
                        help="Path to the egoverse H5 file.")
    parser.add_argument("--sam-mask", type=Path, default=None,
                        help="SAM table-mask NPZ (default: "
                             "data/egoverse/desk/<stem>_desk.npz)")
    parser.add_argument("--camera", type=str, default="aria_rgb_cam",
                        help="H5 camera key (default: aria_rgb_cam)")
    parser.add_argument("--output", type=Path, default=None,
                        help="NPZ output (default: "
                             "outputs/calibration/<stem>_egoverse.npz)")
    parser.add_argument("--viz-png", type=Path, default=None,
                        help="Overlay PNG output (default: same stem as "
                             "--output, _overlay.png suffix)")
    parser.add_argument("--viz-mp4", type=Path, nargs="?", const="auto",
                        default=None,
                        help="Sweep overlay MP4. Pass without value to "
                             "default to <stem>_overlay.mp4.")
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"[error] H5 file not found: {args.h5}")
        return 1
    sam_mask_path = args.sam_mask or _default_sam_mask_path(args.h5)
    if not sam_mask_path.exists():
        print(f"[error] SAM mask not found: {sam_mask_path}")
        return 1

    cfg = EgoverseSceneConfig()
    print(f"[calibrate-table] Refining Aria pose against {sam_mask_path.name}…")
    try:
        result = refine_aria_extrinsic(sam_mask_path, cfg)
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"[calibrate-table] FAILED: {e}")
        return 1

    T = result["T_world_cam"]
    print(f"[calibrate-table] rms residual: {result['residual_rms_px']:.2f} px")
    print(f"[calibrate-table] Δpos vs nominal: "
          f"{np.linalg.norm(result['delta_pos_mm']):.1f} mm, "
          f"Δrot: {result['delta_rot_deg']:.2f}°")
    print(f"[calibrate-table] camera world pos: {T[:3, 3]}")

    # ── Outputs ──────────────────────────────────────────────────────────
    out = args.output or OUTPUT_DIR_DEFAULT / f"{args.h5.stem}_egoverse.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             T_world_cam=T,
             T_world_cam_nominal=result["T_world_cam_nominal"],
             xi=result["xi"],
             residual_rms_px=result["residual_rms_px"],
             delta_pos_mm=result["delta_pos_mm"],
             delta_rot_deg=result["delta_rot_deg"])
    print(f"[calibrate-table] wrote {out}")

    # PNG overlay — first H5 frame.
    K = intrinsics_matrix(cfg.camera.intrinsics)
    with np.load(sam_mask_path) as d:
        sam_freq = d["mask"].astype(np.float32).mean(axis=0)

    from utils.h5_loader import H5Reader
    with H5Reader(args.h5, dataset="egoverse", camera=args.camera) as h5:
        rgb = h5.image(0)
    overlay = _draw_overlay(rgb, T, K, sam_freq)
    viz_png = args.viz_png or out.with_name(f"{out.stem}_overlay.png")
    viz_png.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(viz_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[calibrate-table] wrote {viz_png}")

    # MP4 overlay (optional).
    if args.viz_mp4 is not None:
        mp4 = (out.with_name(f"{out.stem}_overlay.mp4")
               if args.viz_mp4 == "auto" else Path(args.viz_mp4))
        _write_mp4_overlay(args.h5, T, K, sam_freq, args.camera, mp4)
        print(f"[calibrate-table] wrote {mp4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
