"""Calibrate the OAK-D world extrinsic via a single AprilTag detection + PnP.

The AprilTag is assumed to lie on the table top at the pose described by the
``SceneConfig`` (default: 7.5 cm tag36h11 id=5 in the top-right corner). One
RGB frame is enough: detect the four tag corners, then solve PnP against the
known world-frame corners to recover ``T_world_cam``. The result is written
to ``assets/calibration/oakd_extrinsic.npz``.

The intrinsics MUST be supplied via either ``--intrinsics-json`` (with keys
fx, fy, cx, cy and optional width/height) OR by editing
``SceneConfig.camera.intrinsics`` and setting ``--use-config-intrinsics``.

Usage:
    # From an image file (factory cal supplied via JSON):
    python scripts/calibrate_extrinsic.py \\
        --from-image first_frame.png \\
        --intrinsics-json oakd_factory.json

    # From an H5 frame (e.g. frame 0 of a recorded session):
    python scripts/calibrate_extrinsic.py \\
        --from-h5 data/h5/session.h5 --frame 0 \\
        --intrinsics-json oakd_factory.json

No Isaac Sim. Pure CV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from project root: `python dataset_replay/scripts/calibrate_extrinsic.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.calibration import (
    detect_apriltag, save_extrinsic, solve_extrinsic_pnp, tag_corners_world,
)
from utils.camera import lookat_to_T_world_cam
from utils.config import SceneConfig


# ── Image loading ────────────────────────────────────────────────────────────
def _load_image(args) -> np.ndarray:
    """Return an 8-bit grayscale image from --from-image or --from-h5."""
    import cv2

    if args.from_image is not None:
        img = cv2.imread(str(args.from_image), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {args.from_image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.from_h5 is not None:
        from utils.h5_loader import H5Reader
        with H5Reader(args.from_h5, camera=args.camera) as h5:
            rgb = h5.image(args.frame)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    raise SystemExit("Pass --from-image or --from-h5")


# ── Intrinsics loading ───────────────────────────────────────────────────────
def _load_intrinsics(args, cfg: SceneConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, dist)`` for solvePnP.

    Priority: --intrinsics-json > cfg.camera.intrinsics (only when --use-config-intrinsics).
    """
    if args.intrinsics_json is not None:
        with open(args.intrinsics_json) as f:
            data = json.load(f)
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = data.get("distortion") or [0.0] * 5
    elif args.use_config_intrinsics:
        K = cfg.camera.intrinsics
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        dist = list(cfg.camera.distortion)
    else:
        raise SystemExit(
            "Pass --intrinsics-json or --use-config-intrinsics. The factory "
            "calibration must be provided; we cannot guess fx/fy/cx/cy."
        )

    if any(v == 0.0 for v in (fx, fy, cx, cy)):
        raise SystemExit("Intrinsics fx/fy/cx/cy contain zeros; cannot proceed.")

    K_mat = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return K_mat, np.asarray(dist, dtype=float)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-image", type=Path, help="RGB or grayscale image file")
    src.add_argument("--from-h5", type=Path, help="H5 file path (use with --frame)")

    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index when reading from H5 (default: 0)")
    parser.add_argument("--camera", type=str, default="oakd_front_view",
                        help="Camera name when reading from H5")

    parser.add_argument("--intrinsics-json", type=Path, default=None,
                        help='JSON with keys {fx, fy, cx, cy, distortion?}')
    parser.add_argument("--use-config-intrinsics", action="store_true",
                        help="Use SceneConfig.camera.intrinsics (must be edited first)")

    parser.add_argument("--out", type=Path, default=None,
                        help="Output .npz path (default: SceneConfig.camera.extrinsic_path)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print result but do not write the file")
    args = parser.parse_args()

    cfg = SceneConfig()

    image_gray = _load_image(args)
    print(f"[calib] Loaded image: shape={image_gray.shape}, dtype={image_gray.dtype}")

    corners_image = detect_apriltag(
        image_gray, cfg.apriltag.family, cfg.apriltag.tag_id,
    )
    if corners_image is None:
        raise SystemExit(
            f"AprilTag id={cfg.apriltag.tag_id} (family {cfg.apriltag.family}) "
            f"not found in image. Check the tag is visible and unobstructed."
        )
    print(f"[calib] Detected tag corners (px):\n{corners_image}")

    corners_world = tag_corners_world(cfg)
    print(f"[calib] Expected tag corners (world m):\n{corners_world}")

    K, dist = _load_intrinsics(args, cfg)
    nominal = lookat_to_T_world_cam(
        cfg.camera.nominal_position, cfg.camera.nominal_lookat, cfg.camera.nominal_up,
    )

    T_world_cam, err = solve_extrinsic_pnp(
        corners_image, corners_world, K, dist, nominal_T_world_cam=nominal,
    )
    print(f"[calib] PnP mean reprojection error: {err:.3f} px")
    print(f"[calib] T_world_cam =\n{np.array2string(T_world_cam, precision=5)}")
    print(f"[calib] camera xyz in world: {T_world_cam[:3, 3]}")

    if args.no_save:
        print("[calib] --no-save set; skipping write.")
        return 0

    out_path = args.out or cfg.camera.extrinsic_path
    save_extrinsic(
        T_world_cam, out_path,
        intrinsics={"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
        distortion=dist,
        metadata={"reproj_error_px": err,
                  "tag_id":          cfg.apriltag.tag_id,
                  "tag_family":      cfg.apriltag.family,
                  "tag_edge_size":   cfg.apriltag.edge_size},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
