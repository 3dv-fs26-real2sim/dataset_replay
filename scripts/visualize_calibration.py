"""Overlay sim AprilTag vs H5-detected AprilTag from the calibrated camera view.

Given a calibrated ``T_world_cam`` (saved by ``calibrate_extrinsic.py``) and
the H5 recording, this script:

  * Detects the AprilTag in every frame of the H5 video.
  * Projects the **sim** AprilTag corners (read from ``SceneConfig`` —
    where the simulation thinks the tag is in the world) into image
    space using the calibrated camera.
  * Optionally also projects sim *table edges* and the *world XYZ axes*
    so you can visually check whether the sim scene lines up with the
    real scene (not just the tag itself).
  * Overlays everything on each frame and writes:
      - a single PNG of the best-detection frame for inspection, and
      - an MP4 sweep across the whole video.

Detected tag corners (real)        →  GREEN
Projected sim tag corners          →  ORANGE
Centroid offset arrow              →  YELLOW (when both visible)
Projected sim table top edges      →  CYAN   (--show-table)
Projected sim world XYZ axes       →  R/G/B  (--show-axes)

A near-zero tag-centroid offset means the calibrated extrinsic is
self-consistent with the sim's tag-world-pose (PnP minimised this
explicitly). Real misalignment between sim and reality shows up in the
*other* overlays — e.g. if the sim's cyan table edge cuts across the
real table by 5 cm, the SceneConfig dimensions or the AprilTag pose
needs adjustment.

Usage:
    python scripts/visualize_calibration.py \\
        --from-h5 data/h5/session.h5 \\
        --extrinsic assets/calibration/oakd_extrinsic.npz

    # Or with explicit intrinsics override (otherwise read from the .npz
    # if present, else from the H5 stored K, else from SceneConfig):
    python scripts/visualize_calibration.py \\
        --from-h5 data/h5/session.h5 \\
        --extrinsic assets/calibration/oakd_extrinsic.npz \\
        --intrinsics-source h5

No Isaac Sim. Pure CV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root: `python dataset_replay/scripts/visualize_calibration.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.calibration import (
    detect_apriltag_with_quality, tag_corners_world, _make_detector,
)
from utils.config import PROJECT_ROOT, SceneConfig

# Colours (BGR — OpenCV).
COLOR_REAL    = (60, 220, 60)      # green   — H5-detected corners
COLOR_SIM     = (40, 140, 255)     # orange  — projected sim tag corners
COLOR_ARROW   = (0, 220, 255)      # yellow  — centroid offset arrow
COLOR_TABLE   = (255, 255, 0)      # cyan    — sim table top edges
COLOR_AXIS_X  = (0,   0, 220)      # red
COLOR_AXIS_Y  = (0, 220,   0)      # green (axis)
COLOR_AXIS_Z  = (220, 0,   0)      # blue
COLOR_TEXT    = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)


# ── Intrinsics resolution ────────────────────────────────────────────────────
def _resolve_intrinsics(args, cfg, npz) -> tuple[np.ndarray, np.ndarray]:
    """Pick K, dist from the requested source."""
    import json

    # An explicit JSON path beats everything else.
    if args.intrinsics_json is not None:
        with open(args.intrinsics_json) as f:
            data = json.load(f)
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = data.get("distortion") or [0.0] * 5
        print(f"[viz] intrinsics source: json={args.intrinsics_json}")
        K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        return K_mat, np.asarray(dist, dtype=float)

    src = args.intrinsics_source
    if src == "auto":
        if "intrinsics" in npz.files:
            src = "extrinsic-npz"
        elif args.from_h5 is not None:
            src = "h5"
        else:
            src = "config"
    print(f"[viz] intrinsics source: {src}")

    if src == "extrinsic-npz":
        if "intrinsics" not in npz.files:
            raise SystemExit("Extrinsic .npz lacks 'intrinsics' field; pick another source.")
        fx, fy, cx, cy = npz["intrinsics"].tolist()
        dist = (npz["distortion"].tolist()
                if "distortion" in npz.files else list(cfg.camera.distortion))
    elif src == "h5":
        if args.from_h5 is None:
            raise SystemExit("--intrinsics-source h5 requires --from-h5")
        from utils.h5_loader import read_h5_intrinsic
        K_h5 = read_h5_intrinsic(args.from_h5, camera=args.camera)
        fx, fy = float(K_h5[0, 0]), float(K_h5[1, 1])
        cx, cy = float(K_h5[0, 2]), float(K_h5[1, 2])
        dist = list(cfg.camera.distortion)
    elif src == "config":
        K = cfg.camera.intrinsics
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        dist = list(cfg.camera.distortion)
    else:
        raise SystemExit(f"Unknown --intrinsics-source: {src}")

    K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K_mat, np.asarray(dist, dtype=float)


# ── Projection ───────────────────────────────────────────────────────────────
def _project_world_to_pixel(
    points_world: np.ndarray, T_world_cam: np.ndarray,
    K: np.ndarray, dist: np.ndarray,
) -> np.ndarray:
    """Project ``(N, 3)`` world points to ``(N, 2)`` pixels via ``T_world_cam``."""
    import cv2

    T_cam_world = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3].astype(np.float64))
    tvec = T_cam_world[:3, 3].astype(np.float64).reshape(3, 1)
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float64),
                                dist.astype(np.float64))
    return proj.reshape(-1, 2)


# ── Drawing ──────────────────────────────────────────────────────────────────
def _draw_quad(img, pts2d: np.ndarray, color, *, label: str | None = None,
               thickness: int = 2):
    import cv2
    pts = pts2d.reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p.tolist()), 3, color, -1, lineType=cv2.LINE_AA)
    if label is not None:
        # Label near the first corner.
        org = tuple((pts[0] + np.array([4, -6])).tolist())
        _text(img, label, org, color)


def _text(img, txt: str, org: tuple[int, int], color):
    import cv2
    # Draw black background for legibility.
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 2),
                  COLOR_TEXT_BG, -1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)


def _clip_segment(p1, p2, w, h):
    """cv2.clipLine wrapper that returns ints or None if both off-screen."""
    import cv2
    rect = (0, 0, w, h)
    p1i = tuple(int(round(v)) for v in p1)
    p2i = tuple(int(round(v)) for v in p2)
    inside, q1, q2 = cv2.clipLine(rect, p1i, p2i)
    return (q1, q2) if inside else None


def _draw_polyline_clipped(img, pts_px: np.ndarray, color, *, thickness=1,
                           closed: bool = True):
    """Draw a polyline edge-by-edge with clipLine — robust to vertices that
    project far off-screen (otherwise OpenCV draws garbage stripes)."""
    import cv2
    h, w = img.shape[:2]
    pts = pts_px.reshape(-1, 2)
    n = len(pts)
    upper = n if closed else n - 1
    for i in range(upper):
        seg = _clip_segment(pts[i], pts[(i + 1) % n], w, h)
        if seg is None:
            continue
        cv2.line(img, seg[0], seg[1], color, thickness=thickness,
                 lineType=cv2.LINE_AA)


def _draw_axes(img, origin_px: np.ndarray, axes_tip_px: np.ndarray):
    """Draw an XYZ tripod given pre-projected origin + 3 axis tip pixels.

    Each arrow is clipped to the image so an off-frame origin doesn't blow up.
    """
    import cv2
    h, w = img.shape[:2]
    for tip, col in zip(axes_tip_px, (COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z)):
        seg = _clip_segment(origin_px, tip, w, h)
        if seg is None:
            continue
        cv2.arrowedLine(img, seg[0], seg[1], col, 2,
                        line_type=cv2.LINE_AA, tipLength=0.2)


def _annotate_frame(
    rgb: np.ndarray,
    detected_corners: np.ndarray | None,
    sim_corners: np.ndarray,
    *,
    frame_idx: int,
    n_total: int,
    centroid_px_offset: float | None,
    detection_margin: float | None,
    sim_table_loop_px: np.ndarray | None = None,
    sim_axes: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return a BGR-annotated copy of an RGB frame ready for writing."""
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]

    # Optional sim-scene overlays first (so the tag quads sit on top).
    if sim_table_loop_px is not None:
        _draw_polyline_clipped(img, sim_table_loop_px, COLOR_TABLE,
                               thickness=1, closed=True)
    if sim_axes is not None:
        origin_px, axes_tip_px = sim_axes
        _draw_axes(img, origin_px, axes_tip_px)

    # Sim AprilTag is always projectable.
    _draw_quad(img, sim_corners, COLOR_SIM, label="sim")

    if detected_corners is not None:
        _draw_quad(img, detected_corners, COLOR_REAL, label="real")
        c_real = detected_corners.mean(axis=0)
        c_sim  = sim_corners.mean(axis=0)
        cv2.arrowedLine(img, tuple(c_real.astype(int).tolist()),
                        tuple(c_sim.astype(int).tolist()),
                        COLOR_ARROW, 2, line_type=cv2.LINE_AA, tipLength=0.2)

    # Header bar with metadata.
    header_h = 56
    header = np.full((header_h, W, 3), 30, dtype=np.uint8)
    img = np.vstack([header, img])

    _text(img, f"frame {frame_idx}/{n_total - 1}", (8, 18), (220, 220, 220))
    if detection_margin is not None:
        _text(img, f"tag margin: {detection_margin:.1f}", (8, 38),
              COLOR_REAL)
    else:
        _text(img, "tag NOT detected (occluded)", (8, 38), (120, 120, 120))
    if centroid_px_offset is not None:
        col = COLOR_ARROW if centroid_px_offset < 5.0 else (60, 80, 255)
        _text(img, f"sim<->real centroid: {centroid_px_offset:.1f} px",
              (180, 38), col)

    # Legend in the corner.
    _text(img, "GREEN = real (detected)", (W - 240, 18), COLOR_REAL)
    _text(img, "ORANGE = sim (projected)", (W - 240, 38), COLOR_SIM)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-h5", type=Path, required=True,
                        help="H5 file path")
    parser.add_argument("--camera", type=str, default="oakd_front_view",
                        help="Camera name inside the H5 (default: oakd_front_view)")
    parser.add_argument("--extrinsic", type=Path, default=None,
                        help="Calibrated extrinsic .npz "
                             "(default: SceneConfig.camera.extrinsic_path)")
    parser.add_argument("--use-h5-extrinsic", action="store_true",
                        help="Bypass --extrinsic and use the H5-stored "
                             "T_robot_cam composed with mount_xyz instead. "
                             "Useful as a validation reference: if sim "
                             "landmarks align under this extrinsic but not "
                             "under PnP, the SceneConfig sim model is the "
                             "thing that's wrong, not the PnP code.")
    parser.add_argument("--mount-xyz", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), default=None,
                        help="Override cfg.robot.mount_xyz when composing "
                             "the H5 extrinsic. Use this to test what a "
                             "corrected arm placement would look like "
                             "without editing SceneConfig.")
    parser.add_argument("--intrinsics-source", type=str, default="auto",
                        choices=["auto", "extrinsic-npz", "h5", "config"],
                        help="Where to read K/distortion from. "
                             "'auto' = .npz if present, else h5.")
    parser.add_argument("--intrinsics-json", type=Path, default=None,
                        help="Explicit K override (overrides --intrinsics-source). "
                             "JSON with keys fx, fy, cx, cy, optional 'distortion' list.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "calibration",
                        help="Output directory")
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Filename prefix (default: derived from H5 stem)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Output MP4 fps (default: 10)")
    parser.add_argument("--no-mp4", action="store_true",
                        help="Skip the MP4 sweep; only write the PNG")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip the best-frame PNG; only write the MP4")
    parser.add_argument("--show-table", action="store_true",
                        help="Also project sim table top edges (cyan loop). "
                             "Misalignment between cyan and the visible real "
                             "table edge is a true sim/real geometry mismatch.")
    parser.add_argument("--show-axes", action="store_true",
                        help="Also draw a sim world XYZ tripod (R=X, G=Y, B=Z) "
                             "at the world origin, projected through the "
                             "calibrated camera.")
    parser.add_argument("--axis-length", type=float, default=0.20,
                        help="Length of the XYZ axis arrows in metres (default 0.20)")
    parser.add_argument("--axis-origin", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), default=None,
                        help="World-frame origin of the XYZ tripod. Default: "
                             "table-top centre (visible in frame). Pass "
                             "'0 0 0' to anchor at the world origin instead.")
    args = parser.parse_args()

    import cv2  # late import, only needed here

    cfg = SceneConfig()

    extrinsic_source_label: str
    if args.use_h5_extrinsic:
        from utils.h5_loader import read_h5_extrinsic
        mount_xyz = (tuple(args.mount_xyz) if args.mount_xyz is not None
                     else cfg.robot.mount_xyz)
        T_world_cam = read_h5_extrinsic(
            args.from_h5, camera=args.camera,
            mount_xyz=mount_xyz, mount_rpy=cfg.robot.mount_rpy,
        )
        npz = type("EmptyNpz", (), {"files": []})()  # duck-type for _resolve_intrinsics
        npz.files = []
        extrinsic_source_label = (
            f"H5-stored T_robot_cam @ mount_xyz={mount_xyz}"
        )
        # H5-stored case: K must come from H5 (default) or explicit override.
        if args.intrinsics_source == "auto":
            args.intrinsics_source = "h5"
    else:
        extrinsic_path = args.extrinsic or cfg.camera.extrinsic_path
        if not extrinsic_path.exists():
            raise SystemExit(
                f"Extrinsic file not found: {extrinsic_path}\n"
                f"Run `python scripts/calibrate_extrinsic.py --from-h5 "
                f"{args.from_h5} --scan-h5 --use-h5-intrinsics` first, or "
                f"pass --use-h5-extrinsic to use the H5-recorded pose."
            )
        npz = np.load(extrinsic_path, allow_pickle=False)
        T_world_cam = npz["T_world_cam"]
        extrinsic_source_label = str(extrinsic_path)

    print(f"[viz] extrinsic source: {extrinsic_source_label}")
    print(f"[viz] camera xyz in world: {T_world_cam[:3, 3]}")
    print(f"[viz] camera +Z (forward) : {T_world_cam[:3, 2]}")

    K, dist = _resolve_intrinsics(args, cfg, npz)

    # Sim AprilTag corners (where the simulation thinks the tag lives), and
    # their projection through the calibrated camera. The sim corners are
    # constant for every frame because the camera is static in world.
    sim_corners_world = tag_corners_world(cfg)
    sim_corners_px = _project_world_to_pixel(
        sim_corners_world, T_world_cam, K, dist,
    )
    print(f"[viz] Sim AprilTag corners (px):\n{sim_corners_px}")

    # Optional sim-scene overlays — these are the *real* sim/real diagnostics
    # (the tag itself overlays by construction since PnP minimises that exact
    # residual; the table edges and world axes are independent).
    sim_table_loop_px: np.ndarray | None = None
    if args.show_table:
        x_lo, x_hi = cfg.table.x_extent
        y_lo, y_hi = cfg.table.y_extent
        z = cfg.table.top_z
        # Closed loop of the table-top rectangle (CCW from -X-Y corner).
        loop_world = np.array([
            [x_lo, y_lo, z],
            [x_hi, y_lo, z],
            [x_hi, y_hi, z],
            [x_lo, y_hi, z],
        ], dtype=float)
        sim_table_loop_px = _project_world_to_pixel(
            loop_world, T_world_cam, K, dist,
        )

    sim_axes: tuple[np.ndarray, np.ndarray] | None = None
    if args.show_axes:
        L = float(args.axis_length)
        if args.axis_origin is None:
            # Table-top centre: visible by default through the OAK-D's view.
            cx_w, cy_w = cfg.table.centre_xy
            origin_w = np.array([cx_w, cy_w, cfg.table.top_z], dtype=float)
        else:
            origin_w = np.asarray(args.axis_origin, dtype=float)
        axis_pts_world = np.array([
            origin_w,
            origin_w + np.array([L,   0.0, 0.0]),
            origin_w + np.array([0.0, L,   0.0]),
            origin_w + np.array([0.0, 0.0, L  ]),
        ], dtype=float)
        axis_pts_px = _project_world_to_pixel(
            axis_pts_world, T_world_cam, K, dist,
        )
        print(f"[viz] Axis origin (world): {tuple(origin_w.tolist())}")
        sim_axes = (axis_pts_px[0], axis_pts_px[1:])

    # Open H5 video.
    from utils.h5_loader import H5Reader

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_prefix or args.from_h5.stem
    if args.use_h5_extrinsic:
        stem = f"{stem}_h5ext"   # so PnP and H5 outputs don't clobber

    detector = _make_detector(cfg.apriltag.family)

    with H5Reader(args.from_h5, camera=args.camera) as h5:
        n_total = h5.n_frames
        W, H = h5.image_dims()
        if W is None:
            raise SystemExit(f"Camera {args.camera!r} not found in {args.from_h5}")

        # First pass: scan detections, collect stats.
        per_frame_detections: list[np.ndarray | None] = []
        per_frame_margin: list[float | None] = []
        per_frame_offset_px: list[float | None] = []

        for i in range(n_total):
            rgb = h5.image(i)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            res = detect_apriltag_with_quality(
                gray, cfg.apriltag.family, cfg.apriltag.tag_id, detector=detector,
            )
            if res is None:
                per_frame_detections.append(None)
                per_frame_margin.append(None)
                per_frame_offset_px.append(None)
                continue
            corners, margin, _ = res
            offset = float(np.linalg.norm(
                corners.mean(axis=0) - sim_corners_px.mean(axis=0)
            ))
            per_frame_detections.append(corners)
            per_frame_margin.append(margin)
            per_frame_offset_px.append(offset)

        n_detected = sum(c is not None for c in per_frame_detections)
        offsets = np.array([o for o in per_frame_offset_px if o is not None])
        if n_detected > 0:
            print(f"[viz] Detected in {n_detected}/{n_total} frames. "
                  f"sim<->real centroid offset: "
                  f"min={offsets.min():.1f}px, "
                  f"median={np.median(offsets):.1f}px, "
                  f"max={offsets.max():.1f}px")
        else:
            print(f"[viz] WARNING: tag never detected in any of {n_total} frames.")

        # Pick best frame (highest decision margin among detections).
        best_idx = None
        if n_detected > 0:
            margins = [m if m is not None else -1 for m in per_frame_margin]
            best_idx = int(np.argmax(margins))
            print(f"[viz] best detection frame: {best_idx} "
                  f"(margin={per_frame_margin[best_idx]:.1f}, "
                  f"offset={per_frame_offset_px[best_idx]:.1f}px)")

        # Second pass: render outputs.
        png_path = args.out_dir / f"{stem}_calib_overlay_best.png"
        mp4_path = args.out_dir / f"{stem}_calib_overlay.mp4"

        if not args.no_png and best_idx is not None:
            rgb = h5.image(best_idx)
            ann = _annotate_frame(
                rgb, per_frame_detections[best_idx], sim_corners_px,
                frame_idx=best_idx, n_total=n_total,
                centroid_px_offset=per_frame_offset_px[best_idx],
                detection_margin=per_frame_margin[best_idx],
                sim_table_loop_px=sim_table_loop_px,
                sim_axes=sim_axes,
            )
            cv2.imwrite(str(png_path), ann)
            print(f"[viz] wrote {png_path}")

        if not args.no_mp4:
            # Output dims include the 56-px header bar we add per frame.
            out_w, out_h = W, H + 56
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(mp4_path), fourcc, args.fps, (out_w, out_h),
            )
            if not writer.isOpened():
                raise SystemExit(f"Failed to open video writer for {mp4_path}")
            try:
                for i in range(n_total):
                    rgb = h5.image(i)
                    ann = _annotate_frame(
                        rgb, per_frame_detections[i], sim_corners_px,
                        frame_idx=i, n_total=n_total,
                        centroid_px_offset=per_frame_offset_px[i],
                        detection_margin=per_frame_margin[i],
                        sim_table_loop_px=sim_table_loop_px,
                        sim_axes=sim_axes,
                    )
                    writer.write(ann)
            finally:
                writer.release()
            print(f"[viz] wrote {mp4_path}")

    # Summary
    print()
    print("=== Calibration summary ===")
    print(f"  H5:           {args.from_h5}")
    print(f"  extrinsic:    {extrinsic_path}")
    print(f"  detected:     {n_detected}/{n_total} frames")
    if n_detected > 0:
        print(f"  median sim<->real centroid offset: {np.median(offsets):.2f} px")
        print(f"  max    sim<->real centroid offset: {offsets.max():.2f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
