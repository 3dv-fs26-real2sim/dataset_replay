"""Export a lab-ready ``<stem>_kinematic_replay.npz`` from a recorded H5.

This is the **collaboration bridge** between ``dataset_replay`` (which owns
the calibration / quaternion / object-composition maths) and ``pandaorca_lab``
(which trains a residual-PPO tracker and should not have to import any of
that). Running this once per demo produces a single self-describing NPZ that
the lab consumes directly — no ``sys.path`` injection, no Isaac Sim.

It is **CPU-only**: it never boots SimulationApp. The same camera-pose maths
the replay scripts run at startup (Aria SAM refinement / OakD AprilTag PnP,
or the nominal fallback) is reused here, then the result is baked into the
NPZ.

Output contract (see pandaorca_lab/README.md for the authoritative spec)::

    arm_ee_pose    (T, 7)   EE/wrist pose in robot-base (panda_link0) frame,
                            [x, y, z, qw, qx, qy, qz]; quaternion already in
                            URDF panda_link8 convention (post tool_quat_to_urdf).
    hand_joint_pos (T, 17)  absolute hand joint targets (HAND_JOINT_NAMES order).
    object_pose    (T, 7)   object pose in WORLD frame, [x,y,z, qw,qx,qy,qz]
                            (T_world_cam @ T_cam_obj). Omitted if no --object-traj.
    meta           JSON str  schema_version, rig, fps, n_frames, mount_xyz,
                            ee_offset, T_world_cam (4x4, row-major), quat
                            convention, joint_names_hand, object_name, source_h5,
                            arm_source, calibration {...}, created_utc.

Usage::

    conda activate 3dv   # only needs numpy/scipy/h5py (+cv2 for maple PnP)
    python scripts/export_kinematic_replay.py \\
        --h5 data/egoverse/h5/20250804_104715.h5 \\
        --object-traj data/egoverse/pose/20250804_104715_duck_vdahand.npz \\
        --object-name duck
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import select_config
from utils.constants import (
    EE_WRIST_OFFSET_IN_LINK8,
    H5_DEFAULT_FPS,
    HAND_JOINT_NAMES,
    OUTPUT_DIR,
)
from utils.h5_loader import H5Reader, SCHEMA
from utils.poses import load_pose_trajectory
from utils.rotation import detect_quaternion_order, tool_quat_to_urdf

SCHEMA_VERSION = "1.0"


def _infer_dataset(h5_path: Path) -> str:
    """Guess 'egoverse' / 'maple' from the path; fall back to egoverse."""
    parts = {p.lower() for p in h5_path.parts}
    for name in ("egoverse", "maple"):
        if name in parts:
            return name
    return "egoverse"


def _to_jsonable(obj):
    """Recursively convert numpy types to plain Python for json.dumps."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _resolve_t_world_cam(args, dataset: str, cfg, h5_path: Path) -> tuple[np.ndarray, dict]:
    """Return (T_world_cam, calibration_info) for the chosen rig.

    egoverse: nominal Aria pose, optionally SAM-mask-refined.
    maple:    nominal OakD lookat pose, optionally AprilTag-PnP-refined.
    """
    if dataset == "egoverse":
        from utils.calibrate_table import (
            compute_nominal_aria_pose,
            refine_aria_extrinsic,
        )

        sam_mask = (
            Path(args.sam_mask).resolve() if args.sam_mask else
            h5_path.parent.parent / "desk" / f"{h5_path.stem}_desk.npz"
        )
        if not args.no_refine and sam_mask.is_file():
            res = refine_aria_extrinsic(sam_mask, cfg)
            T = np.asarray(res["T_world_cam"], dtype=np.float64)
            dp = res["delta_pos_mm"]
            print(
                f"[export] T_world_cam refined (SAM {sam_mask.name}): "
                f"Δpos {dp[0]:+.1f}/{dp[1]:+.1f}/{dp[2]:+.1f} mm, "
                f"Δrot {res['delta_rot_deg']:.2f}°, "
                f"residual RMS {res['residual_rms_px']:.2f} px"
            )
            return T, {
                "method": "aria_sam_refined",
                "sam_mask": str(sam_mask),
                "delta_pos_mm": dp,
                "delta_rot_deg": res["delta_rot_deg"],
                "residual_rms_px": res["residual_rms_px"],
            }
        T = np.asarray(compute_nominal_aria_pose(cfg), dtype=np.float64)
        print("[export] T_world_cam = nominal (T_world_base @ ARIA_EXTRINSICS_RIGHT)")
        return T, {"method": "aria_nominal"}

    # maple
    from utils.calibrate_april import calibrate_from_h5, nominal_oakd_pose

    if not args.no_calibrate:
        res = calibrate_from_h5(h5_path, cfg)
        T = np.asarray(res["T_world_cam"], dtype=np.float64)
        dp = res["delta_pos_mm"]
        print(
            f"[export] T_world_cam refined (AprilTag PnP): "
            f"{res['n_inlier_frames']}/{res['n_frames_detected']} inlier frames, "
            f"Δpos {dp[0]:+.1f}/{dp[1]:+.1f}/{dp[2]:+.1f} mm, "
            f"Δrot {res['delta_rot_deg']:.2f}°, "
            f"residual RMS {res['residual_rms_px']:.2f} px"
        )
        return T, {
            "method": "oakd_apriltag_pnp",
            "n_inlier_frames": res["n_inlier_frames"],
            "n_frames_detected": res["n_frames_detected"],
            "delta_pos_mm": dp,
            "delta_rot_deg": res["delta_rot_deg"],
            "residual_rms_px": res["residual_rms_px"],
        }
    T = np.asarray(nominal_oakd_pose(cfg), dtype=np.float64)
    print("[export] T_world_cam = nominal OakD lookat pose")
    return T, {"method": "oakd_nominal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="recorded demonstration H5")
    parser.add_argument("--dataset", choices=list(SCHEMA), default=None,
                        help="rig schema; inferred from the path if omitted")
    parser.add_argument("--object-traj", default="",
                        help="(T,4,4) object pose .npz (camera frame by default)")
    parser.add_argument("--object-traj-frame", choices=["camera", "world"],
                        default="camera",
                        help="frame of --object-traj (default camera)")
    parser.add_argument("--object-name", default="duck",
                        help="object label stored in metadata")
    parser.add_argument("--out", default="",
                        help="output .npz path (default outputs/<rig>/replay/"
                             "<stem>_kinematic_replay.npz)")
    # egoverse extrinsic controls
    parser.add_argument("--use-actions", dest="use_actions", action="store_true",
                        default=None,
                        help="egoverse: read top-level actions_* (default for "
                             "egoverse — anchors the residual to commanded teleop)")
    parser.add_argument("--use-observations", dest="use_actions",
                        action="store_false",
                        help="egoverse: read observations/qpos_* instead of actions")
    parser.add_argument("--sam-mask", default="",
                        help="egoverse: SAM desk-mask .npz for extrinsic refinement "
                             "(auto-discovered alongside the H5 if omitted)")
    parser.add_argument("--no-refine", action="store_true",
                        help="egoverse: skip SAM refinement, use nominal Aria pose")
    # maple extrinsic controls
    parser.add_argument("--no-calibrate", action="store_true",
                        help="maple: skip AprilTag PnP, use nominal OakD pose")
    args = parser.parse_args()

    h5_path = Path(args.h5).resolve()
    if not h5_path.is_file():
        raise SystemExit(f"[export] H5 not found: {h5_path}")
    dataset = args.dataset or _infer_dataset(h5_path)
    cfg = select_config(dataset)

    # Default egoverse to the actions stream (matches the lab's residual prior);
    # maple has no actions, so always read observations.
    use_actions = args.use_actions
    if use_actions is None:
        use_actions = dataset == "egoverse"
    if dataset == "maple":
        use_actions = False

    # ── Robot streams ────────────────────────────────────────────────────────
    with H5Reader(h5_path, dataset=dataset, use_actions=use_actions) as h5:
        traj = h5.load_trajectories()
    arm_raw = np.asarray(traj["arm_xyz_quat"], dtype=np.float64)   # (T, 7)
    hand = np.asarray(traj["hand_q"], dtype=np.float64)            # (T, 17)
    n = int(traj["n_frames"])

    # Canonicalise quaternion order to wxyz, then tool → URDF panda_link8 frame.
    arm = detect_quaternion_order(arm_raw, "arm").copy()
    arm_ee = arm.copy()
    for i in range(n):
        arm_ee[i, 3:7] = tool_quat_to_urdf(arm[i, 3:7])

    # ── Camera pose ───────────────────────────────────────────────────────────
    T_world_cam, calib = _resolve_t_world_cam(args, dataset, cfg, h5_path)

    # ── Object trajectory → world frame ───────────────────────────────────────
    object_pose = None
    if args.object_traj:
        T_obj = load_pose_trajectory(Path(args.object_traj).resolve())   # (T,4,4)
        if T_obj.shape[0] != n:
            raise SystemExit(
                f"[export] object traj length {T_obj.shape[0]} != H5 length {n}"
            )
        if args.object_traj_frame == "camera":
            T_world_obj = np.einsum("ij,njk->nik", T_world_cam, T_obj)
        else:
            T_world_obj = T_obj
        obj_p = T_world_obj[:, :3, 3]
        obj_q_xyzw = R.from_matrix(T_world_obj[:, :3, :3]).as_quat()      # xyzw
        obj_q = np.concatenate([obj_q_xyzw[:, 3:4], obj_q_xyzw[:, :3]], axis=1)
        object_pose = np.concatenate([obj_p, obj_q], axis=1).astype(np.float32)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "schema_version": SCHEMA_VERSION,
        "rig": dataset,
        "fps": float(H5_DEFAULT_FPS[dataset]),
        "n_frames": n,
        "mount_xyz": list(cfg.robot.mount_xyz),
        "ee_offset": EE_WRIST_OFFSET_IN_LINK8.tolist(),
        "T_world_cam": T_world_cam.tolist(),
        "quat_convention": "wxyz",
        "arm_frame": "panda_link0 (robot base); quaternion URDF panda_link8",
        "object_frame": "world",
        "joint_names_hand": list(HAND_JOINT_NAMES),
        "object_name": args.object_name if object_pose is not None else None,
        "arm_source": "actions" if use_actions else "observations/qpos",
        "source_h5": str(h5_path),
        "object_traj": str(Path(args.object_traj).resolve()) if args.object_traj else None,
        "calibration": _to_jsonable(calib),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    fname = f"{h5_path.stem}_kinematic_replay.npz"
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = OUTPUT_DIR / dataset / "replay" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "arm_ee_pose": arm_ee.astype(np.float32),
        "hand_joint_pos": hand.astype(np.float32),
        "meta": json.dumps(meta),
    }
    if object_pose is not None:
        arrays["object_pose"] = object_pose

    np.savez(out_path, **arrays)
    print(f"[export] wrote {out_path}")
    print(f"[export]   rig={dataset} fps={meta['fps']} frames={n} "
          f"arm_source={meta['arm_source']} "
          f"object={'yes' if object_pose is not None else 'no'}")


if __name__ == "__main__":
    main()
