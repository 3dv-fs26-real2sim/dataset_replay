"""Build a residual-RL demo npz from a raw H5 recording (dataset_replay style).

Reuses the *same* Lula IK the kinematic-replay scripts use (:mod:`utils.ik`),
so the training baseline and the replay rig share one kinematic convention. The
output npz is what :func:`lab.envs.demo_loader.load_demo` consumes.

Pipeline
--------
1. Read the wrist (xyz + quat) + hand (17 qpos) trajectory from the H5
   (:class:`utils.h5_loader.H5Reader`).
2. Retarget the wrist pose to the 7 Panda arm joints, per frame, with a
   warm-started Lula IK on the ``ee_target`` frame — exactly as
   ``utils.robot.make_ik_position_setter`` does during replay.
3. Compose the object trajectory into the panda_link0 frame:
   * EgoVerse: ``T_link0_obj = ARIA_EXTRINSICS_RIGHT @ T_cam_obj`` (the object
     npz is in the Aria camera frame; the base-relative extrinsic is constant).
   * Generic: pass ``--object-traj`` already in ``panda_link0`` or ``world``.
4. Write ``obj_trajectory / wrist_pos / wrist_rot_aa / hand_qpos /
   hand_joint_names / arm_qpos / frame / dataset``.

Boots ``SimulationApp`` first (Lula needs it). Run from the repo root::

    python lab/scripts/make_demo.py --dataset egoverse \
        --h5 data/egoverse/h5/20250804_104715.h5 \
        --object duck --out data/egoverse/demos/egoverse_duck_104715.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Repo root → make ``lab`` (and the bundled ``utils``) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401

parser = argparse.ArgumentParser(description="Build a residual-RL demo npz from an H5 recording.")
parser.add_argument("--dataset", choices=("egoverse", "maple"), required=True)
parser.add_argument("--h5", type=Path, required=True, help="Raw H5 recording.")
parser.add_argument("--out", type=Path, required=True, help="Output demo npz path.")
parser.add_argument("--object", type=str, default="duck",
                    help="Object name (for auto-finding the pose npz under data/<dataset>/pose/).")
parser.add_argument("--object-traj", type=Path, default=None,
                    help="(N,4,4) object pose npz. Default: data/<dataset>/pose/<h5_stem>_<object>_vdahand.npz.")
parser.add_argument("--object-traj-frame", choices=("aria_camera", "oakd_camera", "world", "panda_link0"), default=None,
                    help="Frame of the object npz. Default: aria_camera (egoverse) / oakd_camera (maple).")
parser.add_argument("--desk-mask", type=Path, default=None,
                    help="EgoVerse: SAM table-mask npz to refine the Aria extrinsic (calibrate_table). "
                         "Default: data/egoverse/desk/<h5_stem>_desk.npz if present.")
parser.add_argument("--no-desk-refine", action="store_true",
                    help="EgoVerse: skip desk-SAM refinement, use the nominal Aria extrinsic.")
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

# ── Boot the simulator (Lula needs it) ────────────────────────────────────────
from utils.app import create_app  # noqa: E402

simulation_app = create_app(argparse.Namespace(headless=args.headless))

# ── Post-boot imports ─────────────────────────────────────────────────────────
from scipy.spatial.transform import Rotation  # noqa: E402

from utils.config import select_config  # noqa: E402
from utils.constants import (  # noqa: E402
    ARIA_EXTRINSICS_RIGHT, EE_FRAME_NAME, HAND_JOINT_NAMES,
    WRIST_HOME_POSITION, WRIST_HOME_ROTATION,
)
from utils.h5_loader import H5Reader  # noqa: E402
from utils.ik import create_ik_solver, solve_ik_for_pose  # noqa: E402
from utils.poses import load_pose_trajectory  # noqa: E402
from utils.rotation import detect_quaternion_order, rotation_matrix_to_wxyz, tool_quat_to_urdf  # noqa: E402


def _load_object_traj(path: Path) -> np.ndarray:
    """Load a (N,4,4) object trajectory from an npz OR a dir of NNNNNN.txt files
    (FoundationPose per-frame 4x4)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise SystemExit(f"[make_demo] no *.txt pose files in {path}")
        return np.stack([np.loadtxt(f) for f in files]).astype(np.float64)
    return load_pose_trajectory(path)


def _aria_T_link0_cam(cfg) -> np.ndarray:
    """EgoVerse camera-in-base transform (T_link0_cam).

    Uses the desk-SAM-refined Aria extrinsic when a table mask is available
    (``calibrate_table.refine_aria_extrinsic`` → world-fixed ``T_world_cam``,
    converted to base frame via the mount), matching kinematic_replay_egoverse;
    falls back to the nominal ``ARIA_EXTRINSICS_RIGHT`` otherwise.
    """
    if args.no_desk_refine:
        return np.asarray(ARIA_EXTRINSICS_RIGHT, dtype=float)
    mask = args.desk_mask or (lab.DATA_DIR / "egoverse" / "desk" / f"{args.h5.stem}_desk.npz")
    if not Path(mask).exists():
        print(f"[make_demo] no desk mask ({mask}); using nominal Aria extrinsic.")
        return np.asarray(ARIA_EXTRINSICS_RIGHT, dtype=float)
    from utils.calibrate_table import refine_aria_extrinsic
    T_world_cam = refine_aria_extrinsic(Path(mask), cfg)["T_world_cam"]
    T_world_link0 = np.eye(4)
    T_world_link0[:3, 3] = np.asarray(cfg.robot.mount_xyz)
    print(f"[make_demo] desk-refined Aria extrinsic from {Path(mask).name}")
    return np.linalg.inv(T_world_link0) @ T_world_cam        # T_link0_cam


def _resolve_object_traj(cfg) -> tuple[np.ndarray | None, str]:
    """Load and frame-resolve the object trajectory into the panda_link0 frame.

    Frames: ``aria_camera`` (egoverse, base-relative ARIA extrinsic), ``oakd_camera``
    (maple, world-fixed AprilTag-PnP extrinsic from the H5), or ``panda_link0``.
    """
    # Auto-find object poses (per-frame T_cam_obj), consistent across datasets:
    #   data/<ds>/pose/<h5_stem>/         — a dir of FoundationPose NNNNNN.txt, or
    #   data/<ds>/pose/<h5_stem>_<obj>.npz — a packed (N,4,4) npz.
    traj_path = args.object_traj
    pose_dir = lab.DATA_DIR / args.dataset / "pose"
    for cand in (pose_dir / args.h5.stem, pose_dir / f"{args.h5.stem}_{args.object}.npz"):
        if traj_path is None and cand.exists():
            traj_path = cand
    if traj_path is None:
        print(f"[make_demo] no object trajectory found for {args.object!r} under {pose_dir} — "
              f"writing a static frame-0 object pose at the table centre.")
        return None, "none"

    traj = _load_object_traj(traj_path)  # (N, 4, 4)
    frame = args.object_traj_frame or ("aria_camera" if args.dataset == "egoverse" else "oakd_camera")
    if frame == "aria_camera":
        # T_link0_obj = T_link0_cam @ T_cam_obj (desk-refined extrinsic if available)
        traj = np.einsum("ij,njk->nik", _aria_T_link0_cam(cfg), traj)
    elif frame == "oakd_camera":
        # FoundationPose poses are T_cam_obj. Compose with the AprilTag-PnP
        # T_world_cam (from the H5, the SAME calibration kinematic_replay_maple
        # uses), then world → panda_link0 by subtracting the mount.
        from utils.calibrate_april import calibrate_from_h5
        T_world_cam = calibrate_from_h5(str(args.h5), cfg, camera="oakd_front_view")["T_world_cam"]
        traj = np.einsum("ij,njk->nik", T_world_cam, traj)        # → world
        traj[:, :3, 3] -= np.asarray(cfg.robot.mount_xyz)         # → panda_link0
    elif frame == "world":
        traj[:, :3, 3] -= np.asarray(cfg.robot.mount_xyz)
    print(f"[make_demo] object trajectory: {Path(traj_path).name} "
          f"({traj.shape[0]} frames, {frame} → panda_link0)")
    return traj.astype(np.float32), f"{frame}:{Path(traj_path).name}"


def main() -> None:
    cfg = select_config(args.dataset)

    with H5Reader(args.h5, dataset=args.dataset) as h5:
        t = h5.load_trajectories()
    arm_xyz_quat = detect_quaternion_order(t["arm_xyz_quat"], "arm")  # (N, 7) [x y z qw qx qy qz]
    hand_q = t["hand_q"].astype(np.float32)                           # (N, 17)
    n = t["n_frames"]
    if args.max_frames:
        n = min(n, args.max_frames)

    # ── Arm IK retarget (warm-started Lula, ee_target frame) ──────────────────
    solver = create_ik_solver(cfg.urdf_path, cfg.lula_descriptor, "right")
    seed, ok = solve_ik_for_pose(solver, EE_FRAME_NAME, WRIST_HOME_POSITION,
                                 rotation_matrix_to_wxyz(WRIST_HOME_ROTATION))
    prev = seed if ok else np.zeros(7, dtype=float)
    arm_qpos = np.zeros((n, 7), dtype=np.float32)
    fails = 0
    for i in range(n):
        q_urdf = tool_quat_to_urdf(arm_xyz_quat[i, 3:7])
        sol, ok = solve_ik_for_pose(solver, EE_FRAME_NAME, arm_xyz_quat[i, :3], q_urdf, warm_start=prev)
        if ok and sol is not None:
            prev = sol
        else:
            fails += 1
        arm_qpos[i] = prev
    print(f"[make_demo] arm IK: {n} frames, {fails} failures (held previous on failure)")

    # ── Object trajectory (panda_link0 frame) ─────────────────────────────────
    obj_traj, obj_src = _resolve_object_traj(cfg)
    if obj_traj is None:
        T0 = np.eye(4, dtype=np.float32)
        T0[:3, 3] = (cfg.table.x_extent[0] + 0.3, cfg.table.right_table_centre_y, 0.02)
        obj_traj = np.repeat(T0[None], n, axis=0)
    else:
        n = min(n, obj_traj.shape[0])
        arm_qpos, hand_q, arm_xyz_quat = arm_qpos[:n], hand_q[:n], arm_xyz_quat[:n]
        obj_traj = obj_traj[:n]

    # ── Wrist (informational) → axis-angle ────────────────────────────────────
    wrist_pos = arm_xyz_quat[:n, :3].astype(np.float32)
    q = arm_xyz_quat[:n, 3:7]                                  # wxyz
    wrist_rot_aa = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_rotvec().astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        obj_trajectory=obj_traj,
        wrist_pos=wrist_pos,
        wrist_rot_aa=wrist_rot_aa,
        hand_qpos=hand_q,
        hand_joint_names=np.array(list(HAND_JOINT_NAMES)),
        arm_qpos=arm_qpos,
        frame="panda_link0",
        dataset=args.dataset,
        arm_qpos_source=f"lula_ik_from_h5:{EE_FRAME_NAME}",
        object_source=obj_src,
    )
    print(f"[make_demo] wrote {args.out}  ({n} frames, dataset={args.dataset})")


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
