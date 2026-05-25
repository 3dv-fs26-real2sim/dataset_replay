"""Sanity-check tool for AprilTag-based maple calibration.

Runs the same ``utils.calibrate_april.calibrate_from_h5`` that
``kinematic_replay_maple.py`` calls at startup, then writes:

  * an ``.npz`` with the refined ``T_world_cam`` + diagnostics
  * a PNG overlay on a representative frame (detected tag corners vs the
    projected sim tag corners)
  * (optional) an MP4 overlay sweeping the same projection over the whole H5

These outputs are **never read back by the replay scripts** — they exist
purely so you can spot-check that the auto-calibration is producing a
sensible pose.

Pure CV / numpy — no Isaac Sim. Safe to run from any environment with the
project requirements installed.

Usage::

    python scripts/calibrate/calibrate_april.py --h5 data/maple/h5/<session>.h5
    python scripts/calibrate/calibrate_april.py --h5 ... --viz-mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.calibrate_april import (
    calibrate_from_h5,
    tag_corners_world,
)
from utils.config import PROJECT_ROOT
from utils.config_maple import MapleSceneConfig


OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "outputs" / "calibration"


def _project(T_world_cam: np.ndarray, K: np.ndarray, dist: np.ndarray,
             pts_world: np.ndarray) -> np.ndarray:
    """Project ``pts_world`` (N, 3) through a column-vector T_world_cam."""
    import cv2
    T_cw = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    tvec = T_cw[:3, 3].reshape(3, 1)
    proj, _ = cv2.projectPoints(
        pts_world.astype(np.float64).reshape(-1, 1, 3),
        rvec, tvec, K.astype(np.float64), dist.astype(np.float64),
    )
    return proj.reshape(-1, 2)


def _draw_overlay(frame_rgb: np.ndarray, cfg, T_world_cam: np.ndarray,
                  K: np.ndarray, dist: np.ndarray,
                  detected_corners: np.ndarray | None) -> np.ndarray:
    """Render the overlay: detected tag corners (green) vs projected sim
    tag corners (orange) + world axes at the tag origin."""
    import cv2

    img = frame_rgb.copy()

    # Projected sim tag corners.
    corners_world = tag_corners_world(cfg)
    proj_corners = _project(T_world_cam, K, dist, corners_world)
    pts = proj_corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 165, 0), thickness=2)
    for px in proj_corners:
        cv2.circle(img, (int(px[0]), int(px[1])), 4, (255, 165, 0), -1)

    # Detected corners (if any).
    if detected_corners is not None:
        for px in detected_corners:
            cv2.circle(img, (int(px[0]), int(px[1])), 5, (0, 255, 0), 2)

    # World axes at the tag origin.
    origin = cfg.apriltag_world_pose()[:3, 3]
    axes = origin[None, :] + np.eye(3) * 0.05            # 5 cm axis arrows
    axis_pts = _project(T_world_cam, K, dist,
                        np.concatenate([origin[None, :], axes], axis=0))
    o = tuple(axis_pts[0].astype(int))
    for end, color in zip(axis_pts[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.arrowedLine(img, o, tuple(end.astype(int)), color, 2)

    return img


def _write_mp4_overlay(h5_path: Path, cfg, T_world_cam: np.ndarray,
                       K: np.ndarray, dist: np.ndarray, camera: str,
                       output_path: Path, fps: float = 10.0) -> None:
    import cv2
    import imageio.v2 as imageio

    from utils.calibrate_april import (
        _make_detector, detect_apriltag_with_quality,
    )
    from utils.h5_loader import H5Reader

    detector = _make_detector(cfg.apriltag.family)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path),
                                fps=max(1, int(round(fps))), codec="libx264")
    with H5Reader(h5_path, dataset="maple", camera=camera) as h5:
        n = h5.n_frames
        for i in range(n):
            rgb = h5.image(i)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            res = detect_apriltag_with_quality(
                gray, cfg.apriltag.family, cfg.apriltag.tag_id,
                detector=detector,
            )
            corners = res[0] if res is not None else None
            overlay = _draw_overlay(rgb, cfg, T_world_cam, K, dist, corners)
            writer.append_data(overlay)
    writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, required=True,
                        help="Path to the maple H5 file.")
    parser.add_argument("--camera", type=str, default="oakd_front_view",
                        help="H5 camera key (default: oakd_front_view)")
    parser.add_argument("--output", type=Path, default=None,
                        help="NPZ output (default: "
                             "outputs/calibration/<stem>_maple.npz)")
    parser.add_argument("--viz-png", type=Path, default=None,
                        help="Overlay PNG output (default: same stem as --output, "
                             "_overlay.png suffix)")
    parser.add_argument("--viz-mp4", type=Path, nargs="?", const="auto",
                        default=None,
                        help="Sweep overlay MP4. Pass without value to default "
                             "to <stem>_overlay.mp4 next to the NPZ.")
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"[error] H5 file not found: {args.h5}")
        return 1

    cfg = MapleSceneConfig()
    print(f"[calibrate-april] Scanning {args.h5.name} for tag {cfg.apriltag.tag_id} "
          f"({cfg.apriltag.family})…")
    try:
        result = calibrate_from_h5(args.h5, cfg, camera=args.camera)
    except (KeyError, ValueError, RuntimeError, ImportError) as e:
        print(f"[calibrate-april] FAILED: {e}")
        return 1

    T = result["T_world_cam"]
    print(f"[calibrate-april] inliers: "
          f"{result['n_inlier_frames']}/{result['n_frames_detected']} "
          f"(of {result['n_frames_scanned']} scanned)")
    print(f"[calibrate-april] rms reprojection err: "
          f"{result['residual_rms_px']:.2f} px")
    print(f"[calibrate-april] camera world pos: {T[:3, 3]}")
    print(f"[calibrate-april] Δpos vs nominal: "
          f"{np.linalg.norm(result['delta_pos_mm']):.1f} mm, "
          f"Δrot: {result['delta_rot_deg']:.2f}°")

    # ── Outputs ──────────────────────────────────────────────────────────
    out = args.output or OUTPUT_DIR_DEFAULT / f"{args.h5.stem}_maple.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             T_world_cam=T,
             T_world_cam_nominal=result["T_world_cam_nominal"],
             K=result["K"], dist=result["dist"],
             n_frames_scanned=result["n_frames_scanned"],
             n_frames_detected=result["n_frames_detected"],
             n_inlier_frames=result["n_inlier_frames"],
             residual_rms_px=result["residual_rms_px"],
             delta_pos_mm=result["delta_pos_mm"],
             delta_rot_deg=result["delta_rot_deg"])
    print(f"[calibrate-april] wrote {out}")

    # PNG overlay — best-margin frame.
    viz_png = args.viz_png or out.with_name(f"{out.stem}_overlay.png")
    import cv2
    from utils.calibrate_april import (
        _make_detector, detect_apriltag_with_quality,
    )
    from utils.h5_loader import H5Reader
    detector = _make_detector(cfg.apriltag.family)
    best_frame = int(result["frame_indices"][np.argmax(result["margins"])])
    with H5Reader(args.h5, dataset="maple", camera=args.camera) as h5:
        rgb = h5.image(best_frame)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    res = detect_apriltag_with_quality(
        gray, cfg.apriltag.family, cfg.apriltag.tag_id, detector=detector,
    )
    detected = res[0] if res is not None else None
    overlay = _draw_overlay(rgb, cfg, T, result["K"], result["dist"], detected)
    viz_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(viz_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[calibrate-april] wrote {viz_png}")

    # MP4 overlay (optional).
    if args.viz_mp4 is not None:
        mp4 = (out.with_name(f"{out.stem}_overlay.mp4")
               if args.viz_mp4 == "auto" else Path(args.viz_mp4))
        _write_mp4_overlay(args.h5, cfg, T, result["K"], result["dist"],
                           args.camera, mp4)
        print(f"[calibrate-april] wrote {mp4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
