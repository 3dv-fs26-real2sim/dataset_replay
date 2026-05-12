"""Calibrate the OAK-D world extrinsic via AprilTag detection + PnP.

The AprilTag is assumed to lie on the table top at the pose described by the
``SceneConfig`` (default: 7.5 cm tag36h11 id=5 in the back-right corner).
Three input modes are supported:

  * ``--from-image``: a single still image.
  * ``--from-h5 ... --frame N``: a single frame from an H5 recording.
  * ``--from-h5 ... --scan-h5``: iterate **every** frame in the H5,
    detect the tag where it's visible (skipping occluded frames), filter
    poor detections, and run a single joint PnP+RANSAC over all good
    detections. Robust to the tag being occluded most of the time.

The intrinsics MUST be supplied via ``--intrinsics-json`` (with keys
fx, fy, cx, cy and optional ``distortion``), ``--use-config-intrinsics``
(reads ``SceneConfig.camera.intrinsics``), or ``--use-h5-intrinsics``
(reads the K stored in the H5 alongside the video — recommended for
``--scan-h5``).

Usage:
    # Single frame from an image file
    python scripts/calibrate_extrinsic.py \\
        --from-image first_frame.png \\
        --intrinsics-json oakd_factory.json

    # Single frame from H5 (frame 0 by default)
    python scripts/calibrate_extrinsic.py \\
        --from-h5 data/h5/session.h5 --frame 0 \\
        --use-h5-intrinsics

    # Robust full-video scan (joint PnP+RANSAC across all detected frames)
    python scripts/calibrate_extrinsic.py \\
        --from-h5 data/h5/session.h5 --scan-h5 \\
        --use-h5-intrinsics

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
    _make_detector, _tag_world_pose_with_rotation,
    detect_apriltag, detect_apriltag_with_quality,
    save_extrinsic, select_correct_branch,
    solve_extrinsic_pnp, solve_extrinsic_pnp_multiview,
    solve_pose_square_planar, tag_corners_world,
)
from utils.camera import lookat_to_T_world_cam
from utils.config import SceneConfig


# ── Image loading (single-frame paths) ───────────────────────────────────────
def _load_single_image_gray(args) -> np.ndarray:
    """Return an 8-bit grayscale image from --from-image or --from-h5 --frame."""
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

    Priority: --intrinsics-json > --use-h5-intrinsics > --use-config-intrinsics.
    """
    if args.intrinsics_json is not None:
        with open(args.intrinsics_json) as f:
            data = json.load(f)
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = data.get("distortion") or [0.0] * 5
    elif args.use_h5_intrinsics:
        if args.from_h5 is None:
            raise SystemExit("--use-h5-intrinsics requires --from-h5")
        from utils.h5_loader import read_h5_intrinsic
        K_h5 = read_h5_intrinsic(args.from_h5, camera=args.camera)
        fx, fy = float(K_h5[0, 0]), float(K_h5[1, 1])
        cx, cy = float(K_h5[0, 2]), float(K_h5[1, 2])
        dist = list(cfg.camera.distortion)
    elif args.use_config_intrinsics:
        K = cfg.camera.intrinsics
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        dist = list(cfg.camera.distortion)
    else:
        raise SystemExit(
            "Pass --intrinsics-json, --use-h5-intrinsics, or "
            "--use-config-intrinsics. The factory calibration must be "
            "provided; we cannot guess fx/fy/cx/cy."
        )

    if any(v == 0.0 for v in (fx, fy, cx, cy)):
        raise SystemExit("Intrinsics fx/fy/cx/cy contain zeros; cannot proceed.")

    K_mat = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return K_mat, np.asarray(dist, dtype=float)


# ── Single-frame path ────────────────────────────────────────────────────────
def _calibrate_single_frame(args, cfg, K, dist, nominal):
    image_gray = _load_single_image_gray(args)
    print(f"[calib] Loaded image: shape={image_gray.shape}, dtype={image_gray.dtype}")

    corners_image = detect_apriltag(image_gray, cfg.apriltag.family, cfg.apriltag.tag_id)
    if corners_image is None:
        raise SystemExit(
            f"AprilTag id={cfg.apriltag.tag_id} (family {cfg.apriltag.family}) "
            f"not found in image. Try --scan-h5 to scan all frames, or pick a "
            f"different --frame where the tag is visible."
        )
    print(f"[calib] Detected tag corners (px):\n{corners_image}")

    corners_world = tag_corners_world(cfg)
    T_world_cam, err = solve_extrinsic_pnp(
        corners_image, corners_world, K, dist, nominal_T_world_cam=nominal,
    )
    print(f"[calib] PnP mean reprojection error: {err:.3f} px")
    return T_world_cam, err, {"n_frames": 1, "n_inlier_frames": 1}


# ── Sanity check + sim/real diagnostic ──────────────────────────────────────
def _sanity_check_extrinsic(
    T_world_cam: np.ndarray, cfg: SceneConfig,
    h5_path: Path | None, camera: str,
    frame_indices: list[int], corners_list: list[np.ndarray],
) -> dict:
    """Detect physically-implausible PnP results and back-project actual tag.

    A "good" calibration has the camera above the table with its forward
    axis pointing roughly downward. We flag deviations as warnings so the
    user knows to inspect the SceneConfig if the calibration looks off.

    Returns a dict with diagnostic flags and the back-projected actual
    tag center (computed by intersecting the principal-corner ray of the
    best detection with the table-top plane, given the JUST-RECOVERED
    extrinsic — cross-checked by repeating it for the H5-stored extrinsic
    when available).
    """
    issues = []
    cam_pos = T_world_cam[:3, 3]
    cam_fwd = T_world_cam[:3, 2]   # OpenCV +Z forward
    table_top_z = cfg.table.top_z

    if cam_pos[2] < table_top_z:
        issues.append(
            f"camera Z={cam_pos[2]:.3f} m is BELOW table top "
            f"({table_top_z} m) — sim tag pose is likely wrong."
        )
    if cam_fwd[2] > 0.05:
        issues.append(
            f"camera forward +Z component = {cam_fwd[2]:+.3f} (>0); "
            f"the camera appears to look UP, not down at the workspace."
        )
    x_lo, x_hi = cfg.table.x_extent
    if cam_pos[0] > x_hi:
        issues.append(
            f"camera X={cam_pos[0]:+.3f} is past the back wall "
            f"(x_extent={cfg.table.x_extent}) — likely wrong PnP branch "
            f"or wrong sim tag pose."
        )

    # Back-project from the recovered extrinsic to find where the SCENE-IMPLIED
    # tag lives. By construction this should match cfg.apriltag_world_pose()
    # since PnP minimised that residual. The interesting comparison is with
    # the H5-stored extrinsic if available — that gives the *actual* tag
    # world position independent of SceneConfig.
    backproj_via_pnp  = _backproject_tag_centre(
        T_world_cam, cfg, h5_path, camera, frame_indices, corners_list,
    ) if h5_path is not None else None

    backproj_via_h5 = None
    if h5_path is not None:
        try:
            from utils.h5_loader import read_h5_extrinsic
            T_h5 = read_h5_extrinsic(
                h5_path, camera=camera,
                mount_xyz=cfg.robot.mount_xyz, mount_rpy=cfg.robot.mount_rpy,
            )
            backproj_via_h5 = _backproject_tag_centre(
                T_h5, cfg, h5_path, camera, frame_indices, corners_list,
            )
        except (KeyError, FileNotFoundError):
            pass

    print()
    print("=== Calibration sanity check ===")
    if not issues:
        print("  ✓ camera pose passes basic plausibility checks.")
    else:
        print("  ⚠ Potential issues:")
        for line in issues:
            print(f"    - {line}")

    print()
    print("=== Sim-vs-real AprilTag position ===")
    print(f"  SceneConfig tag world pos : {cfg.apriltag_world_pose()[:3, 3]}")
    if backproj_via_h5 is not None:
        delta = backproj_via_h5 - cfg.apriltag_world_pose()[:3, 3]
        print(f"  H5-extrinsic back-proj    : {backproj_via_h5}")
        print(f"  delta (H5_actual - sim)   : {delta}  ||{np.linalg.norm(delta):.3f} m||")
        if np.linalg.norm(delta) > 0.05:
            print(
                f"  NOTE: H5-stored extrinsic disagrees with SceneConfig by "
                f"{np.linalg.norm(delta):.3f} m. Two ways to interpret:\n"
                f"        (a) If the AprilTag world pose in SceneConfig is "
                f"correct (this is the recommended source of truth, since "
                f"the AprilTag is bolted to the table and its position is "
                f"measured directly with a ruler), then the H5-stored "
                f"extrinsic is wrong — likely because its `T_robot_cam` "
                f"composes with `cfg.robot.mount_xyz` and one of them is "
                f"slightly off. The PnP-derived camera above is the canonical "
                f"extrinsic; treat the `h5ext_*` overlays as a sim/real "
                f"diagnostic, not ground truth.\n"
                f"        (b) If the H5-extrinsic is the source of truth, "
                f"update `apriltag.back_right_corner_to_back_wall` / "
                f"`back_right_corner_to_right_edge` in utils/config.py so "
                f"that `apriltag_world_pose()` matches the back-projected "
                f"position."
            )
    else:
        print(f"  (H5 back-projection unavailable — no stored extrinsic)")

    return {
        "issues": issues,
        "backproj_via_pnp": backproj_via_pnp,
        "backproj_via_h5":  backproj_via_h5,
    }


def _backproject_tag_centre(
    T_world_cam: np.ndarray, cfg: SceneConfig,
    h5_path: Path, camera: str,
    frame_indices: list[int], corners_list: list[np.ndarray],
) -> np.ndarray:
    """Back-project the median detected tag-centre pixel to world via the
    given extrinsic, intersecting the table-top plane."""
    from utils.h5_loader import read_h5_intrinsic
    K = read_h5_intrinsic(h5_path, camera=camera)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    centres = np.array([c.mean(axis=0) for c in corners_list])
    centre = np.median(centres, axis=0)

    ray_cam = np.array([(centre[0] - cx) / fx,
                        (centre[1] - cy) / fy, 1.0])
    ray_world = T_world_cam[:3, :3] @ ray_cam
    origin = T_world_cam[:3, 3]
    z_plane = cfg.table.top_z + cfg.apriltag.z_offset_above_table
    if abs(ray_world[2]) < 1e-9:
        return np.array([np.nan, np.nan, np.nan])
    t = (z_plane - origin[2]) / ray_world[2]
    return origin + t * ray_world


# ── Multi-frame scan path ────────────────────────────────────────────────────
def _scan_h5_for_detections(
    h5_path: Path, camera: str, family: str, tag_id: int,
    *, min_decision_margin: float, max_hamming: int,
) -> tuple[list[int], list[np.ndarray], list[float], int]:
    """Run AprilTag detection on every frame and filter by quality.

    Returns ``(frame_indices, corners_per_frame, margins, n_total_frames)``.
    """
    import cv2

    from utils.h5_loader import H5Reader

    detector = _make_detector(family)
    frame_indices: list[int] = []
    corners_list: list[np.ndarray] = []
    margins: list[float] = []
    rejected = 0

    with H5Reader(h5_path, camera=camera) as h5:
        n_total = h5.n_frames
        for i in range(n_total):
            rgb = h5.image(i)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            res = detect_apriltag_with_quality(
                gray, family, tag_id, detector=detector,
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

    print(
        f"[calib] Scanned {n_total} frames: "
        f"{len(frame_indices)} good detections, "
        f"{rejected} rejected (margin<{min_decision_margin} or hamming>{max_hamming}), "
        f"{n_total - len(frame_indices) - rejected} not detected."
    )
    if not frame_indices:
        raise SystemExit(
            f"No usable AprilTag detections in {h5_path.name}. Lower "
            f"--min-decision-margin or check the tag is actually visible "
            f"somewhere in the recording."
        )
    return frame_indices, corners_list, margins, n_total


def _calibrate_scan_h5(args, cfg, K, dist, nominal):
    if args.from_h5 is None:
        raise SystemExit("--scan-h5 requires --from-h5")

    frame_indices, corners_list, margins, n_total = _scan_h5_for_detections(
        args.from_h5, args.camera,
        cfg.apriltag.family, cfg.apriltag.tag_id,
        min_decision_margin=args.min_decision_margin,
        max_hamming=args.max_hamming,
    )

    # Disambiguate the planar-PnP twofold using IPPE_SQUARE on the
    # highest-margin frame. solvePnP's iterative method seeded with the
    # nominal pose can land in the wrong branch (camera reflected through
    # the tag plane) for planar configurations; IPPE_SQUARE returns BOTH
    # branches explicitly so we can pick the geometrically valid one.
    best_idx = int(np.argmax(margins))
    T_world_tag = _tag_world_pose_with_rotation(cfg)
    T_a, T_b, (err_a, err_b) = solve_pose_square_planar(
        corners_list[best_idx], cfg.apriltag.edge_size, T_world_tag, K, dist,
    )
    T_seed, branch = select_correct_branch(T_a, T_b, T_world_tag,
                                           seed_T_world_cam=nominal)
    print(f"[calib] IPPE_SQUARE on best-margin frame "
          f"{frame_indices[best_idx]} (margin={margins[best_idx]:.1f}): "
          f"chose branch {branch} (errs=[{err_a:.3f}, {err_b:.3f}] px)")
    print(f"[calib]   seed cam pos = {T_seed[:3, 3]}")
    print(f"[calib]   seed cam fwd = {T_seed[:3, 2]}")

    corners_world = tag_corners_world(cfg)
    T_world_cam, err, info = solve_extrinsic_pnp_multiview(
        corners_list, corners_world, K, dist,
        nominal_T_world_cam=T_seed,
        ransac_reproj_threshold_px=args.ransac_threshold_px,
    )
    print(
        f"[calib] Joint PnP+RANSAC: {info['n_inlier_frames']}/{info['n_frames']} "
        f"frames inlier, mean inlier reproj err = {err:.3f} px"
    )

    # Print a per-frame breakdown so the user can see which frames were noisy.
    per_err = info["per_frame_err"]
    inlier_mask = info["inlier_frame_mask"]
    if len(frame_indices) <= 30:
        print("[calib] per-frame errors (px):")
        for fi, e, m, ok in zip(frame_indices, per_err, margins, inlier_mask):
            tag = "[I]" if ok else "[X]"
            print(f"    {tag} frame {fi:4d}  err={e:6.2f}  margin={m:6.1f}")
    else:
        print(f"[calib] per-frame errors: min={per_err.min():.2f}, "
              f"median={np.median(per_err):.2f}, max={per_err.max():.2f} px")

    # Sanity check: report when the recovered pose looks unphysical and
    # back-project the actual tag position from the H5 video so the user
    # can correct SceneConfig if needed.
    info["sanity"] = _sanity_check_extrinsic(
        T_world_cam, cfg, args.from_h5, args.camera, frame_indices, corners_list,
    )

    info["frame_indices"] = frame_indices
    info["margins"] = margins
    info["n_total_frames"] = n_total
    return T_world_cam, err, info


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-image", type=Path, help="RGB or grayscale image file")
    src.add_argument("--from-h5", type=Path, help="H5 file path")

    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index when reading a single H5 frame "
                             "(ignored if --scan-h5; default: 0)")
    parser.add_argument("--scan-h5", action="store_true",
                        help="Scan ALL frames of the H5, detect the tag in "
                             "every frame where it's visible, and run joint "
                             "PnP+RANSAC. Robust to occlusion.")
    parser.add_argument("--camera", type=str, default="oakd_front_view",
                        help="Camera name when reading from H5")

    parser.add_argument("--intrinsics-json", type=Path, default=None,
                        help='JSON with keys {fx, fy, cx, cy, distortion?}')
    parser.add_argument("--use-config-intrinsics", action="store_true",
                        help="Use SceneConfig.camera.intrinsics")
    parser.add_argument("--use-h5-intrinsics", action="store_true",
                        help="Use the K stored alongside the H5 video "
                             "(observations/images/{camera}/intrinsics)")

    # Quality filters for --scan-h5
    parser.add_argument("--min-decision-margin", type=float, default=30.0,
                        help="Reject pupil-apriltags detections with "
                             "decision_margin below this (default: 30)")
    parser.add_argument("--max-hamming", type=int, default=0,
                        help="Reject detections with hamming distance above "
                             "this (default: 0 = perfect decode only)")
    parser.add_argument("--ransac-threshold-px", type=float, default=3.0,
                        help="Multi-view RANSAC reprojection inlier threshold "
                             "(default: 3.0 px)")

    parser.add_argument("--out", type=Path, default=None,
                        help="Output .npz path. Default depends on input mode: "
                             "for --from-h5 we write "
                             "assets/calibration/<h5_stem>_oakd_front_extrinsic.npz "
                             "AND copy that to SceneConfig.camera.extrinsic_path "
                             "so test_setup.py / replay scripts pick it up "
                             "automatically. For --from-image we write to "
                             "SceneConfig.camera.extrinsic_path only.")
    parser.add_argument("--no-link", action="store_true",
                        help="When in --from-h5 mode, skip the copy to "
                             "SceneConfig.camera.extrinsic_path; only the "
                             "H5-stem-named .npz is written.")
    parser.add_argument("--no-save", action="store_true",
                        help="Print result but do not write the file")
    args = parser.parse_args()

    cfg = SceneConfig()
    K, dist = _load_intrinsics(args, cfg)
    nominal = lookat_to_T_world_cam(
        cfg.camera.nominal_position, cfg.camera.nominal_lookat, cfg.camera.nominal_up,
    )
    corners_world = tag_corners_world(cfg)
    print(f"[calib] Expected tag corners (world m):\n{corners_world}")

    if args.scan_h5:
        T_world_cam, err, info = _calibrate_scan_h5(args, cfg, K, dist, nominal)
    else:
        T_world_cam, err, info = _calibrate_single_frame(args, cfg, K, dist, nominal)

    print(f"[calib] T_world_cam =\n{np.array2string(T_world_cam, precision=5)}")
    print(f"[calib] camera xyz in world: {T_world_cam[:3, 3]}")

    if args.no_save:
        print("[calib] --no-save set; skipping write.")
        return 0

    metadata = {
        "reproj_error_px":  err,
        "tag_id":           cfg.apriltag.tag_id,
        "tag_family":       cfg.apriltag.family,
        "tag_black_edge_size":   cfg.apriltag.edge_size,
        "tag_printed_edge_size": cfg.apriltag.printed_edge_size,
        "n_frames_used":    info.get("n_frames", 1),
        "n_inlier_frames":  info.get("n_inlier_frames", 1),
    }
    if "n_total_frames" in info:
        metadata["n_total_frames"] = info["n_total_frames"]
    if args.from_h5 is not None:
        metadata["h5_source"] = str(args.from_h5)
        metadata["h5_camera"] = args.camera

    # Decide output path. With --from-h5 we ALSO write a per-H5-named copy so
    # multiple datasets can each carry their own calibration without clobbering
    # the canonical one. The canonical extrinsic_path is what test_setup.py
    # and the replay loop read by default.
    intrinsics_dict = {"fx": K[0, 0], "fy": K[1, 1],
                       "cx": K[0, 2], "cy": K[1, 2]}

    if args.out is not None:
        primary_out = args.out
    elif args.from_h5 is not None:
        primary_out = (
            cfg.camera.extrinsic_path.parent
            / f"{args.from_h5.stem}_{args.camera}_extrinsic.npz"
        )
    else:
        primary_out = cfg.camera.extrinsic_path

    save_extrinsic(T_world_cam, primary_out,
                   intrinsics=intrinsics_dict, distortion=dist,
                   metadata=metadata)

    # Mirror to the canonical path if it isn't already that file, so existing
    # consumers (test_setup, kinematic_replay) pick up the latest calibration.
    canonical = cfg.camera.extrinsic_path
    if (args.from_h5 is not None and not args.no_link
            and primary_out.resolve() != canonical.resolve()):
        save_extrinsic(T_world_cam, canonical,
                       intrinsics=intrinsics_dict, distortion=dist,
                       metadata=metadata)
        print(f"[calibration] Mirrored to canonical {canonical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
