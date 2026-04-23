"""Compute a palm-point pose trajectory from an H5 wrist trajectory.

Reads ``arm_<side>`` / ``hand_<side>`` rows from an H5 file (the same format
``kinematic_replay.py`` consumes), computes a 6D palm pose per frame, and
saves them as an ``(N, 4, 4)`` NPZ.

Two knobs control what "palm" means and which frame the output lives in:

    --from-link {wrist, palm}   (default: palm)
        wrist: build T_base_wrist straight from the H5 row. Fast, but the
            revolute <side>_wrist joint lives downstream of this frame, so
            an offset applied here drifts relative to the real palm as the
            hand flexes.
        palm:  forward-kinematic along the URDF chain
            fer_link8 → connector_mount → (orcahand-internal) world
            → <side>_tower → [revolute <side>_wrist, angle = hand_<side>[:, 0]]
            → <side>_palm.
            The offset is applied in the actual palm link frame, so it
            tracks the palm regardless of wrist flexion.

    --out-frame {camera, world}   (default: camera)
        camera: save T_cam_palm using --camera's calibrated extrinsics
            (consistent with the object-pose NPZs in data/poses_*.npz that
            utils/object.py consumes).
        world:  save T_world_palm in the USD world frame.

``--offset-cm`` / ``--rotation-deg`` apply a local offset in the source link
frame (wrist tool-convention frame or palm link frame, per --from-link).

With ``--visualize``: open Isaac Sim like ``kinematic_replay.py`` and render
a sphere tracking the palm point each frame. The sphere is always placed in
world space; --out-frame only affects the NPZ.

See docs/get_palm_pose.md for the URDF chain sources, verification, and NPZ
schema.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Ensure stdout is unbuffered so status lines appear in real time even when
# Isaac Sim's kit reconfigures stdio midway through the run.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Pure-math / h5py modules — safe to import without starting Isaac Sim.
from utils.constants import (
    H5_PATH_SINGLE, H5_PATH_DUAL, USD_PATH_SINGLE, USD_PATH_DUAL,
    CAMERA_CONFIGS, FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    EE_WRIST_OFFSET_IN_LINK8,
    HAND_LEFT_JOINT_NAMES, HAND_RIGHT_JOINT_NAMES,
)
from utils.h5_loader import load_h5
from utils.rotation import (
    detect_quaternion_order, wxyz_to_rotation_matrix, tool_quat_to_urdf,
)

# Sanity: the H5 hand array column ordering is assumed to match
# HAND_<SIDE>_JOINT_NAMES (same list that utils/robot.py uses to map names to
# articulation DOF indices during kinematic_replay). Column 0 is the wrist.
assert HAND_LEFT_JOINT_NAMES[0] == "left_wrist", HAND_LEFT_JOINT_NAMES[0]
assert HAND_RIGHT_JOINT_NAMES[0] == "right_wrist", HAND_RIGHT_JOINT_NAMES[0]
_WRIST_HAND_INDEX = 0


# ── Local helpers (inlined to avoid importing utils/app.py, which pulls in
#    isaacsim at import time and would fire up the extension system even
#    for the no-sim code path). ────────────────────────────────────────────


def resolve_h5_path(mode: str) -> Path:
    return H5_PATH_SINGLE if mode == "single" else H5_PATH_DUAL


def resolve_usd_path(mode: str) -> Path:
    return USD_PATH_SINGLE if mode == "single" else USD_PATH_DUAL


def build_offset_transform(offset_cm, rotation_deg) -> np.ndarray:
    """Build a local-frame offset transform from translation (cm) + rotation (deg).

    Rotation is interpreted as extrinsic XYZ Euler (scipy "XYZ"). The result
    is applied in the source-link frame — see --from-link.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("XYZ", rotation_deg, degrees=True).as_matrix()
    T[:3, 3] = np.asarray(offset_cm, dtype=float) / 100.0
    return T


def wrist_row_to_T(row: np.ndarray) -> np.ndarray:
    """H5 row [x y z qw qx qy qz] → 4x4 transform in the robot base frame.

    The quaternion is in tool convention (identity = hand pointing down), so
    the returned transform represents the EE-wrist frame in that convention.
    """
    T = np.eye(4)
    T[:3, :3] = wxyz_to_rotation_matrix(row[3:])
    T[:3, 3] = row[:3]
    return T


# ── URDF forward-kinematics chain: fer_link8 → <side>_palm ──────────────────
# Hard-coded from pandaorca_description/urdf/fer_orcahand_<side>_extended.urdf
# (see docs/get_palm_pose.md for the joint origins and axes this was
# transcribed from). The chain has 5 fixed joints and 1 revolute joint
# (<side>_wrist), so forward kinematics only needs the wrist DOF value.


def _rt(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a 4x4 transform from a URDF <origin xyz rpy> element.

    URDF RPY is "fixed-axis roll, pitch, yaw" which equals scipy's extrinsic
    "xyz" Euler convention.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()
    T[:3, 3] = xyz
    return T


# fer_link8 → connector_mount (identical on both arms).
_FER_TO_CONNECTOR = _rt(
    [0.033733, 0.080000, 0.002642],
    [2.389137, -0.000003, -1.570796],
)

_SIDE_CHAIN = {
    "left": {
        # connector_mount → (orcahand-internal) world
        "connector_to_orcaworld":  _rt(
            [-0.000644, 0.138410, 0.036434],
            [-2.301578,  0.006410,  3.136221],
        ),
        # world → world2<side>_tower_fixed_jointbody → <side>_tower
        # (second hop has xyz=rpy=0 so we skip it)
        "orcaworld_to_tower":      _rt([-0.04, 0.0, 0.04575]),
        # <side>_tower → <side>_wrist_jointbody at joint value 0 (xyz only, rpy=0)
        "tower_to_wrist_origin":   _rt([-0.04063, 0.0, 0.10487]),
        # Revolute axis xyz="-1 0 0" → rotation about -x by joint value.
        "wrist_axis_sign":         -1.0,
        # <side>_wrist_jointbody → <side>_palm (fixed, rpy=0)
        "wrist_jointbody_to_palm": _rt([0.002, 0.00144, 0.03872]),
    },
    "right": {
        "connector_to_orcaworld":  _rt(
            [0.160644, 0.138410, 0.036434],
            [-2.301578, -0.006405, -3.136215],
        ),
        "orcaworld_to_tower":      _rt([0.04, 0.0, 0.04575]),
        "tower_to_wrist_origin":   _rt([0.04063, 0.0, 0.10487]),
        # Revolute axis xyz="1 0 0" → rotation about +x by joint value.
        "wrist_axis_sign":         +1.0,
        "wrist_jointbody_to_palm": _rt([-0.002, 0.00144, 0.03872]),
    },
}


def T_fer_link8_to_palm(side: str, wrist_angle_rad: float) -> np.ndarray:
    """Forward-kinematic transform from fer_link8 to the <side>_palm link."""
    c = _SIDE_CHAIN[side]
    Rx = np.eye(4)
    Rx[:3, :3] = Rotation.from_rotvec(
        [c["wrist_axis_sign"] * float(wrist_angle_rad), 0.0, 0.0]
    ).as_matrix()
    return (
        _FER_TO_CONNECTOR
        @ c["connector_to_orcaworld"]
        @ c["orcaworld_to_tower"]
        @ c["tower_to_wrist_origin"]
        @ Rx
        @ c["wrist_jointbody_to_palm"]
    )


def h5_row_to_T_base_fer_link8(row: np.ndarray) -> np.ndarray:
    """Convert an H5 wrist row to T_base_fer_link8.

    Mirrors the inverse of the shift that utils/ik.py applies: the H5 wrist
    position is fer_link8 + R_urdf @ EE_WRIST_OFFSET_IN_LINK8, and the
    orientation passed to IK is tool_quat_to_urdf(row[3:]).
    """
    wrist_pos = row[:3]
    q_urdf_wxyz = tool_quat_to_urdf(row[3:])
    R_urdf = wxyz_to_rotation_matrix(q_urdf_wxyz)
    fer_pos = wrist_pos - R_urdf @ EE_WRIST_OFFSET_IN_LINK8
    T = np.eye(4)
    T[:3, :3] = R_urdf
    T[:3, 3] = fer_pos
    return T


def compute_palm_trajectory(
    arm_rows: np.ndarray,
    T_local_offset: np.ndarray,
    T_world_base: np.ndarray,
    *,
    from_link: str = "wrist",
    side: str = "right",
    hand_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Per frame, compute the world-frame palm pose.

    - ``from_link="wrist"``: T_world_palm = T_world_base @ T_base_wrist @ T_local_offset.
      T_base_wrist comes straight from the H5 row, in tool convention.
    - ``from_link="palm"``: T_world_palm = T_world_base @ T_base_fer_link8
      @ T_fer_link8_to_palm(side, q_wrist) @ T_local_offset.
      T_fer_link8_to_palm uses the URDF chain + the current revolute
      wrist joint value q_wrist = hand_rows[i, 0].
    """
    n = arm_rows.shape[0]
    T_out = np.empty((n, 4, 4), dtype=np.float64)

    if from_link == "wrist":
        for i in range(n):
            T_out[i] = T_world_base @ wrist_row_to_T(arm_rows[i]) @ T_local_offset
        return T_out

    if from_link == "palm":
        if hand_rows is None:
            raise ValueError(
                "--from-link palm requires hand_rows (for the revolute wrist DOF)."
            )
        for i in range(n):
            T_base_fer = h5_row_to_T_base_fer_link8(arm_rows[i])
            q_wrist = float(hand_rows[i, _WRIST_HAND_INDEX])
            T_fer_palm = T_fer_link8_to_palm(side, q_wrist)
            T_out[i] = T_world_base @ T_base_fer @ T_fer_palm @ T_local_offset
        return T_out

    raise ValueError(f"Unknown from_link: {from_link!r}")


def get_robot_base_world_transform(mode: str, side: str) -> np.ndarray:
    """Return T_world_base for the chosen arm by reading the scene USD.

    Uses only ``pxr`` (no Isaac Sim startup) so this works on the no-sim code
    path too. The base pose is static in our scenes, so reading it from the
    authored USD matches what the live stage reports at runtime.
    """
    import os
    from pxr import Usd, UsdGeom

    usd_path = resolve_usd_path(mode)
    prim_path = FRANKA_RIGHT_PATH if side == "right" else FRANKA_LEFT_PATH

    # Suppress USD's "Unresolved reference prim path" warnings emitted at the
    # C++ level during Stage.Open — they stem from physics sub-layers that
    # Isaac Sim resolves at runtime but plain pxr does not. They don't affect
    # the xform cache computation we need here.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        stage = Usd.Stage.Open(str(usd_path))
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Could not find {prim_path} in {usd_path}. "
            f"(mode='{mode}' side='{side}')"
        )
    # UsdGeom.XformCache returns a row-vector matrix; transpose to column.
    gf_mat = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.array(gf_mat, dtype=float).T


def default_output_path(
    h5_path: Path, side: str, from_link: str, frame_tag: str,
    offset_cm, rotation_deg,
) -> Path:
    def _fmt(v):
        return f"{v:g}".replace("-", "m").replace(".", "p")
    tag = f"{side}_from-{from_link}_frame-{frame_tag}"
    tag += f"_o{_fmt(offset_cm[0])}_{_fmt(offset_cm[1])}_{_fmt(offset_cm[2])}"
    if any(abs(r) > 1e-9 for r in rotation_deg):
        tag += f"_r{_fmt(rotation_deg[0])}_{_fmt(rotation_deg[1])}_{_fmt(rotation_deg[2])}"
    return h5_path.parent / f"palm_poses_{h5_path.stem}_{tag}.npz"


def compute_camera_world_pose_standalone(mode: str, camera_name: str) -> np.ndarray:
    """Compute T_world_cam from the authored USD without starting Isaac Sim.

    Mirrors utils/camera.py::compute_camera_world_pose but reads the static
    base transforms via pxr (same path as get_robot_base_world_transform).
    """
    from utils.poses import average_poses

    extrinsics = CAMERA_CONFIGS[camera_name]["extrinsics"]

    # Right arm is always present in both single and dual scenes.
    T_world_base_right = get_robot_base_world_transform(mode, "right")
    poses = [T_world_base_right @ extrinsics["right"]]

    if mode == "dual":
        T_world_base_left = get_robot_base_world_transform(mode, "left")
        poses.append(T_world_base_left @ extrinsics["left"])

    return poses[0] if len(poses) == 1 else average_poses(poses)


def transform_trajectory_to_frame(
    palm_traj_world: np.ndarray,
    out_frame: str,
    T_world_cam: np.ndarray | None,
) -> np.ndarray:
    """Optionally re-express a (N,4,4) world-frame trajectory in camera frame."""
    if out_frame == "world":
        return palm_traj_world
    if out_frame == "camera":
        if T_world_cam is None:
            raise ValueError("T_world_cam required for out_frame='camera'")
        T_cam_world = np.linalg.inv(T_world_cam)
        return np.einsum("ij,njk->nik", T_cam_world, palm_traj_world)
    raise ValueError(f"Unknown out_frame: {out_frame!r}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["single", "dual"], default="single",
                        help="Which scene / H5 file to use (default: single)")
    parser.add_argument("--side", choices=["left", "right"], default="right",
                        help="Which arm's wrist to offset from (default: right)")
    parser.add_argument("--use-actions", action="store_true",
                        help="Use actions_arm_* instead of observations/qpos_arm_*")

    parser.add_argument("--offset-cm", type=float, nargs=3, default=[0.0, 3.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Translation applied in the source link frame "
                             "(see --from-link), in cm. Default '0 3 0' = 3 cm "
                             "along +y in the palm link frame.")
    parser.add_argument("--rotation-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("RX", "RY", "RZ"),
                        help="Extrinsic XYZ Euler rotation applied in the source link "
                             "frame (deg). Default '0 0 0' = no rotation.")
    parser.add_argument("--from-link", choices=["wrist", "palm"], default="palm",
                        help="Which link to treat as the source frame for --offset-cm / "
                             "--rotation-deg. 'palm' (default) walks the URDF chain "
                             "fer_link8 → ... → <side>_palm — offset applies in the "
                             "actual palm link frame and tracks the palm through wrist "
                             "flexion. 'wrist' uses the H5 wrist frame (tool convention, "
                             "identity = hand pointing down) — faster but does not "
                             "account for the revolute <side>_wrist joint. See "
                             "docs/get_palm_pose.md.")

    parser.add_argument("--out-frame", choices=["camera", "world"], default="camera",
                        help="Reference frame used in the saved NPZ 'poses' array. "
                             "'camera' (default) expresses each pose as T_cam_palm "
                             "using --camera's calibrated extrinsics — consistent with "
                             "the object-pose NPZs consumed by utils/object.py. "
                             "'world' expresses each pose as T_world_palm in the USD "
                             "world frame.")
    parser.add_argument("--output", type=str, default=None,
                        help="NPZ output path (default: "
                             "data/palm_poses_<h5-stem>_<side>_from-<wrist|palm>_"
                             "frame-<camera|world>_o<X>_<Y>_<Z>[_r<RX>_<RY>_<RZ>].npz). "
                             "Negative numbers are encoded with 'm' (e.g. -5 → m5) "
                             "and decimals with 'p' (e.g. 0.5 → 0p5). The rotation "
                             "suffix is omitted when --rotation-deg is 0 0 0.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing the NPZ file.")

    # Visualization (Isaac Sim) — isaacsim is only imported inside run_visualization.
    parser.add_argument("--visualize", action="store_true",
                        help="Open Isaac Sim and render a sphere tracking the palm pose.")
    parser.add_argument("--camera", type=str, default="aria",
                        choices=list(CAMERA_CONFIGS.keys()),
                        help="Calibrated camera (default: aria). Used as the viewport "
                             "camera when --visualize, and as the reference frame for "
                             "the NPZ when --out-frame=camera.")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Sim without GUI (only with --visualize).")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--sphere-radius", type=float, default=0.03,
                        help="Tracking sphere radius in metres (default: 0.03 = 3 cm).")
    parser.add_argument("--sphere-color", type=float, nargs=3, default=[1.0, 0.2, 0.2],
                        metavar=("R", "G", "B"),
                        help="Tracking sphere RGB color in [0, 1].")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable the always-on-top debug-draw overlay point.")
    parser.add_argument("--overlay-size", type=float, default=25.0,
                        help="Debug-draw overlay point size in pixels (default: 25).")
    return parser


def main():
    args = build_parser().parse_args()
    h5_path = resolve_h5_path(args.mode)
    cam_tag = f" camera={args.camera}" if args.out_frame == "camera" else ""
    print(f"[palm] H5: {h5_path}")
    print(f"[palm] config: side={args.side} from_link={args.from_link} "
          f"offset_cm={args.offset_cm} rotation_deg={args.rotation_deg} "
          f"out_frame={args.out_frame}{cam_tag}")

    # ── Load H5 trajectory ──────────────────────────────────────────────────
    data = load_h5(h5_path, args.use_actions, args.mode)
    # Normalize quaternion order for every available arm (needed for both
    # palm-trajectory computation and sim replay).
    for side in ("left", "right"):
        key = f"arm_{side}"
        if data.get(key) is not None:
            data[key] = detect_quaternion_order(data[key], key)

    arm_rows = data.get(f"arm_{args.side}")
    hand_rows = data.get(f"hand_{args.side}")
    if arm_rows is None:
        other = "right" if args.side == "left" else "left"
        raise RuntimeError(
            f"H5 file has no arm_{args.side} data. Try --side {other}."
        )
    if args.from_link == "palm" and hand_rows is None:
        raise RuntimeError(
            f"--from-link palm needs hand_{args.side} data (for the revolute "
            f"wrist DOF), but it's missing from {h5_path}."
        )

    T_local_offset = build_offset_transform(args.offset_cm, args.rotation_deg)

    if args.visualize:
        # Defer T_world_base lookup + palm-trajectory computation + NPZ save
        # to inside run_visualization, which reads T_world_base from the
        # live Isaac Sim stage. Opening a USD stage via plain pxr *before*
        # SimulationApp starts corrupts Isaac Sim's USD plugin state and
        # crashes on app startup — so we avoid that path here.
        run_visualization(args, h5_path, data, arm_rows, hand_rows, T_local_offset)
    else:
        # No-sim path: read the base transform from the authored USD with
        # pure pxr (no Isaac Sim extensions loaded).
        T_world_base = get_robot_base_world_transform(args.mode, args.side)
        print(f"[palm] T_world_base (from USD) translation = {T_world_base[:3, 3]}")
        palm_traj_world = compute_palm_trajectory(
            arm_rows, T_local_offset, T_world_base,
            from_link=args.from_link, side=args.side, hand_rows=hand_rows,
        )
        print(f"[palm] Computed {palm_traj_world.shape[0]} poses (world frame). "
              f"First xyz = {palm_traj_world[0, :3, 3]}")

        T_world_cam = None
        if args.out_frame == "camera":
            T_world_cam = compute_camera_world_pose_standalone(args.mode, args.camera)
            print(f"[palm] T_world_cam ({args.camera}) translation = "
                  f"{T_world_cam[:3, 3]}")

        palm_traj_out = transform_trajectory_to_frame(
            palm_traj_world, args.out_frame, T_world_cam,
        )
        print(f"[palm] Output frame = {args.out_frame}"
              + (f" ({args.camera})" if args.out_frame == "camera" else "")
              + f"; first xyz = {palm_traj_out[0, :3, 3]}")
        save_palm_npz(args, h5_path, palm_traj_out)


def save_palm_npz(args, h5_path: Path, palm_traj: np.ndarray) -> None:
    if args.no_save:
        return
    # Filename encodes the output frame so camera-frame and world-frame
    # dumps for the same (side, link, offset) don't collide.
    frame_tag = args.camera if args.out_frame == "camera" else "world"
    out_path = (
        Path(args.output) if args.output
        else default_output_path(
            h5_path, args.side, args.from_link, frame_tag,
            args.offset_cm, args.rotation_deg,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        poses=palm_traj,
        offset_cm=np.asarray(args.offset_cm, dtype=float),
        rotation_deg=np.asarray(args.rotation_deg, dtype=float),
        side=args.side,
        from_link=args.from_link,
        out_frame=args.out_frame,
        camera=args.camera if args.out_frame == "camera" else "",
        h5_path=str(h5_path),
    )
    print(f"[palm] Saved → {out_path}")


def run_visualization(
    args, h5_path: Path, data: dict,
    arm_rows: np.ndarray, hand_rows: np.ndarray | None,
    T_local_offset: np.ndarray,
) -> None:
    """Open Isaac Sim, replay the H5 trajectory, and render a tracking sphere."""
    # SimulationApp must be created before any other Isaac Sim imports.
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": args.headless,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
    })

    import omni.usd
    import omni.kit.app
    from isaacsim.core.api import World
    from pxr import Gf, UsdGeom

    from utils.robot import add_articulations, setup_arms_ik
    from utils.camera import setup_camera, get_prim_world_transform, compute_camera_world_pose

    # Enable the debug-draw extension before we try to acquire its interface.
    # Without this, acquire_debug_draw_interface() silently returns a no-op
    # handle on many Isaac Sim builds and the overlay is invisible.
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.util.debug_draw", "omni.isaac.debug_draw"):
        try:
            if not ext_manager.is_extension_enabled(ext_name):
                ext_manager.set_extension_enabled_immediate(ext_name, True)
            print(f"[palm] Extension enabled: {ext_name}")
        except Exception as exc:
            print(f"[palm] Could not enable {ext_name}: {exc}")

    omni.usd.get_context().open_stage(str(resolve_usd_path(args.mode)))
    world = World()
    arms = add_articulations(world, args.mode)
    world.reset()
    setup_arms_ik(arms)
    stage = omni.usd.get_context().get_stage()

    # Set the viewport to the calibrated camera (matches kinematic_replay.py).
    camera_prim_path = setup_camera(
        stage, args.camera, args.mode,
        FRANKA_LEFT_PATH, FRANKA_RIGHT_PATH,
    )
    print(f"[camera] Viewport set to {camera_prim_path}")

    # Now that the stage is loaded, look up T_world_base for the chosen arm
    # and build the world-frame palm trajectory.
    base_prim_path = FRANKA_RIGHT_PATH if args.side == "right" else FRANKA_LEFT_PATH
    T_world_base = get_prim_world_transform(stage, base_prim_path)
    print(f"[palm] T_world_base (from stage) translation = {T_world_base[:3, 3]}")
    palm_traj = compute_palm_trajectory(
        arm_rows, T_local_offset, T_world_base,
        from_link=args.from_link, side=args.side, hand_rows=hand_rows,
    )
    print(f"[palm] Computed {palm_traj.shape[0]} poses (world frame). "
          f"First xyz = {palm_traj[0, :3, 3]}")

    # Save NPZ in the requested output frame. The sphere keeps using the
    # world-frame palm_traj below (it's a USD prim in world space).
    T_world_cam = None
    if args.out_frame == "camera":
        T_world_cam = compute_camera_world_pose(
            stage, CAMERA_CONFIGS[args.camera]["extrinsics"],
            left_prim_path=FRANKA_LEFT_PATH if args.mode == "dual" else None,
            right_prim_path=FRANKA_RIGHT_PATH,
        )
        print(f"[palm] T_world_cam ({args.camera}) translation = "
              f"{T_world_cam[:3, 3]}")
    palm_traj_out = transform_trajectory_to_frame(palm_traj, args.out_frame, T_world_cam)
    print(f"[palm] Output frame = {args.out_frame}"
          + (f" ({args.camera})" if args.out_frame == "camera" else "")
          + f"; first xyz = {palm_traj_out[0, :3, 3]}")
    save_palm_npz(args, h5_path, palm_traj_out)

    # ── Spawn tracking sphere ──────────────────────────────────────────────
    # Prefer Isaac Sim's VisualSphere (creates a proper OmniPBR material so
    # the colour actually renders); fall back to a raw UsdGeom.Sphere.
    sphere_root_path = "/World/palm_marker"
    sphere_obj = _spawn_tracking_sphere(
        sphere_root_path, palm_traj[0],
        radius=float(args.sphere_radius),
        color=np.asarray(args.sphere_color, dtype=float),
    )

    pos0 = palm_traj[0, :3, 3]
    print(f"[palm] Sphere at world xyz = ({pos0[0]:.4f}, {pos0[1]:.4f}, {pos0[2]:.4f}), "
          f"radius = {args.sphere_radius} m, color = {args.sphere_color}")

    # Acquire the Isaac Sim debug-draw interface for an always-on-top overlay
    # (depth-test disabled — visible even when the hand occludes the sphere).
    draw = None
    if not args.no_overlay:
        for mod_name in ("isaacsim.util.debug_draw", "omni.isaac.debug_draw"):
            try:
                mod = __import__(mod_name, fromlist=["_debug_draw"])
                draw = mod._debug_draw.acquire_debug_draw_interface()
                print(f"[palm] Debug-draw overlay acquired via {mod_name}.")
                break
            except Exception as exc:
                print(f"[palm]   {mod_name} unavailable: {exc}")
        if draw is None:
            print("[palm] Debug-draw unavailable; overlay disabled (sphere only).")

    n_frames = data["n_frames"]
    n_palm = palm_traj.shape[0]
    print(f"\n[palm] Visualizing {n_frames} frames...  (Ctrl-C to stop)\n")
    try:
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                break
            for side, arm in arms.items():
                arm["set_positions"](
                    data[f"arm_{side}"][frame_idx],
                    data[f"hand_{side}"][frame_idx],
                )
            if frame_idx < n_palm:
                T = palm_traj[frame_idx]
                _update_tracking_sphere(sphere_obj, stage, sphere_root_path, T)
                if draw is not None:
                    px, py, pz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
                    draw.clear_points()
                    draw.draw_points(
                        [(px, py, pz)],
                        [(
                            float(args.sphere_color[0]),
                            float(args.sphere_color[1]),
                            float(args.sphere_color[2]),
                            1.0,
                        )],
                        [float(args.overlay_size)],
                    )
            world.step(render=True)
            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / n_frames
                print(f"  frame {frame_idx:5d}/{n_frames}  ({pct:.1f}%)")
    except KeyboardInterrupt:
        print("\n[palm] Interrupted by user.")

    print("[palm] Done.")
    simulation_app.close()


def _spawn_tracking_sphere(sphere_root_path: str, initial_pose: np.ndarray,
                           radius: float, color: np.ndarray):
    """Create a visual-only sphere at ``initial_pose``.

    Returns the Isaac Sim VisualSphere object, or ``None`` if that path is
    unavailable (in which case a raw UsdGeom.Sphere is created at the same
    prim path as a fallback).
    """
    from pxr import Gf, UsdGeom
    import omni.usd

    pos0 = initial_pose[:3, 3]

    try:
        from isaacsim.core.api.objects import VisualSphere
        sphere_obj = VisualSphere(
            prim_path=sphere_root_path,
            name="palm_marker",
            position=np.asarray(pos0, dtype=float),
            radius=float(radius),
            color=color.astype(float),
        )
        print(f"[palm] Spawned VisualSphere at {sphere_root_path} "
              f"(radius={radius} m).")
        return sphere_obj
    except Exception as exc:
        print(f"[palm] VisualSphere unavailable ({exc}); falling back to "
              f"UsdGeom.Sphere + displayColor.")
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, sphere_root_path)
        sphere = UsdGeom.Sphere.Define(stage, f"{sphere_root_path}/geom")
        sphere.GetRadiusAttr().Set(float(radius))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color.tolist())])
        from utils.object import set_object_world_pose
        set_object_world_pose(stage, sphere_root_path, initial_pose)
        return None


def _update_tracking_sphere(sphere_obj, stage, sphere_root_path: str,
                            T_world_palm: np.ndarray) -> None:
    """Drive the tracking sphere to a new pose each frame."""
    if sphere_obj is not None:
        from scipy.spatial.transform import Rotation as _R
        pos = T_world_palm[:3, 3]
        q_xyzw = _R.from_matrix(T_world_palm[:3, :3]).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        sphere_obj.set_world_pose(
            position=np.asarray(pos, dtype=float),
            orientation=q_wxyz,
        )
    else:
        from utils.object import set_object_world_pose
        set_object_world_pose(stage, sphere_root_path, T_world_palm)


if __name__ == "__main__":
    main()
