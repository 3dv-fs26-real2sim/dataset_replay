"""Headless diagnostic script for dual-mode replay discontinuity.

Runs the same IK + simulation loop as test_replay.py --mode dual, but logs
per-frame joint values, IK targets, IK results, and detects discontinuities.

Usage (from Isaac Sim python):
    python diagnose_dual_replay.py [--use-actions] [--max-frames N]

Output:
    dataset_replay/outputs/dual_replay_diagnosis.npz   — raw arrays
    dataset_replay/outputs/dual_replay_diagnosis.txt   — human-readable summary
"""

import argparse
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Diagnose dual replay discontinuity")
parser.add_argument("--use-actions", action="store_true",
                    help="Use actions_* instead of observations/qpos_*")
parser.add_argument("--max-frames", type=int, default=0,
                    help="Limit to first N frames (0 = all)")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                                 "width": 640, "height": 480})

import numpy as np
import h5py
import omni
import omni.usd
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

SCRIPT_DIR = Path(__file__).parent
USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_dual.usd"
H5_PATH = SCRIPT_DIR / "../data/20250829_180500.h5"
OUTPUT_DIR = SCRIPT_DIR / "../outputs"

FRANKA_LEFT_PATH  = "/World/fer_orcahand_left_extended"
FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"

N_ARM_POSE_DIMS = 7
N_HAND_DOFS = 17

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

HAND_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_thumb_mcp",  "left_thumb_abd",  "left_thumb_pip",  "left_thumb_dip",
    "left_index_abd",  "left_index_mcp",  "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd",   "left_ring_mcp",   "left_ring_pip",
    "left_pinky_abd",  "left_pinky_mcp",  "left_pinky_pip",
]

HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

EE_FRAME_NAME_LEFT  = "fer_link8"
EE_FRAME_NAME_RIGHT = "fer_link8"

LULA_DESCRIPTOR_PATH = SCRIPT_DIR / "../../pandaorca_description/lula/fer_robot_descriptor.yaml"
URDF_PATH_LEFT  = SCRIPT_DIR / "../../pandaorca_description/urdf/fer_orcahand_left_extended.urdf"
URDF_PATH_RIGHT = SCRIPT_DIR / "../../pandaorca_description/urdf/fer_orcahand_right_extended.urdf"

WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

HAND_HOME_JOINT_VALUES = np.array([0.0] * N_HAND_DOFS)
HOME_HOLD_SECONDS = 1.0  # shorter hold for diagnosis
FPS = 50.0

# ---------- reused helpers from test_replay.py ----------

def rotation_matrix_to_wxyz(rot_matrix):
    q_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

Q_TOOL_TO_URDF = np.array([0.0, 1.0, 0.0, 0.0])

def tool_quat_to_urdf(q_tool_wxyz):
    return quat_multiply(Q_TOOL_TO_URDF, q_tool_wxyz)

def detect_quaternion_order(arm_data, label):
    w_if_wxyz = np.mean(np.abs(arm_data[:, 3]))
    w_if_xyzw = np.mean(np.abs(arm_data[:, 6]))
    print(f"[quat] {label}: mean|col3|={w_if_wxyz:.4f}  mean|col6|={w_if_xyzw:.4f}")
    if w_if_xyzw > w_if_wxyz:
        print(f"[quat] {label}: detected xyzw → reordering to wxyz")
        reordered = arm_data.copy()
        reordered[:, 3] = arm_data[:, 6]
        reordered[:, 4] = arm_data[:, 3]
        reordered[:, 5] = arm_data[:, 4]
        reordered[:, 6] = arm_data[:, 5]
        return reordered
    print(f"[quat] {label}: wxyz ordering, using as-is")
    return arm_data

def solve_ik_for_pose(solver, ee_frame_name, position, orientation_wxyz, warm_start=None):
    joint_positions, success = solver.compute_inverse_kinematics(
        frame_name=ee_frame_name,
        target_position=position,
        target_orientation=orientation_wxyz,
        warm_start=warm_start,
    )
    return joint_positions if success else None, success

def create_ik_solver(urdf_path, label):
    solver = LulaKinematicsSolver(
        robot_description_path=str(LULA_DESCRIPTOR_PATH.resolve()),
        urdf_path=str(urdf_path.resolve()),
    )
    print(f"[IK] Solver ({label}) created. Joints: {solver.get_joint_names()}")
    return solver

def resolve_dof_indices(art, names, label):
    dof_names = list(art.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    def candidate_names(name):
        if name.startswith("panda_joint"):
            return [name, name.replace("panda_joint", "fer_joint", 1)]
        if name.startswith("fer_joint"):
            return [name, name.replace("fer_joint", "panda_joint", 1)]
        return [name]
    indices = []
    for name in names:
        resolved = False
        for cand in candidate_names(name):
            if cand in name_to_idx:
                indices.append(name_to_idx[cand])
                resolved = True
                break
        if resolved:
            continue
        matches = [d for d in dof_names if d.endswith(name)]
        if len(matches) == 1:
            indices.append(name_to_idx[matches[0]])
        else:
            raise RuntimeError(f"Cannot find '{name}' in {label} DOFs: {dof_names}")
    return np.array(indices, dtype=int)

# ---------- data loading ----------

def load_h5_dual(path, use_actions):
    prefix = "actions" if use_actions else "observations/qpos"
    with h5py.File(path, "r") as f:
        arm_left = f[f"{prefix}_arm_left"][()]
        arm_right = f[f"{prefix}_arm_right"][()]
        hand_left = f[f"{prefix}_hand_left"][()]
        hand_right = f[f"{prefix}_hand_right"][()]
    n = arm_left.shape[0]
    print(f"[h5] Loaded {n} frames from {path}")
    return {
        "arm_left": arm_left, "arm_right": arm_right,
        "hand_left": hand_left, "hand_right": hand_right,
        "n_frames": n,
    }

# ---------- main ----------

def main():
    data = load_h5_dual(H5_PATH, args.use_actions)
    n_frames = data["n_frames"]
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
        print(f"[diag] Limiting to {n_frames} frames")

    data["arm_right"] = detect_quaternion_order(data["arm_right"], "arm_right")
    data["arm_left"]  = detect_quaternion_order(data["arm_left"],  "arm_left")

    # Load scene
    omni.usd.get_context().open_stage(str(USD_PATH))
    world = World()
    franka_right = SingleArticulation(prim_path=FRANKA_RIGHT_PATH,
                                       name="franka_right")
    franka_left  = SingleArticulation(prim_path=FRANKA_LEFT_PATH,
                                       name="franka_left")
    world.scene.add(franka_right)
    world.scene.add(franka_left)
    world.reset()

    # Create IK solvers
    ik_solver_right = create_ik_solver(URDF_PATH_RIGHT, "right")
    ik_solver_left  = create_ik_solver(URDF_PATH_LEFT,  "left")

    # Compute home IK
    home_wrist_quat = rotation_matrix_to_wxyz(WRIST_HOME_ROTATION)
    home_arm_joints, _ = solve_ik_for_pose(
        ik_solver_right, EE_FRAME_NAME_RIGHT,
        WRIST_HOME_POSITION, home_wrist_quat,
    )
    if home_arm_joints is None:
        raise RuntimeError("Right arm home IK failed")
    print(f"[IK] Home arm joints RIGHT: {home_arm_joints}")

    # NOTE: test_replay.py uses `home_arm_joints` (right) for both arms.
    # We replicate the SAME behavior here to reproduce the bug.
    home_arm_joints_for_left = home_arm_joints  # <-- same as test_replay.py

    # Resolve DOF indices
    arm_idx_right  = resolve_dof_indices(franka_right, ARM_JOINT_NAMES, "right")
    hand_idx_right = resolve_dof_indices(franka_right, HAND_RIGHT_JOINT_NAMES, "right")
    arm_idx_left   = resolve_dof_indices(franka_left,  ARM_JOINT_NAMES, "left")
    hand_idx_left  = resolve_dof_indices(franka_left,  HAND_LEFT_JOINT_NAMES, "left")

    # Set home positions (same as test_replay.py)
    q_home_right = franka_right.get_joint_positions().copy()
    q_home_right[arm_idx_right] = home_arm_joints
    q_home_right[hand_idx_right] = HAND_HOME_JOINT_VALUES
    franka_right.set_joint_positions(q_home_right)

    q_home_left = franka_left.get_joint_positions().copy()
    q_home_left[arm_idx_left] = home_arm_joints_for_left  # right arm's joints!
    q_home_left[hand_idx_left] = HAND_HOME_JOINT_VALUES
    franka_left.set_joint_positions(q_home_left)

    # ----- Logging arrays -----
    # Per frame, per arm: IK target (pos+quat), IK result joints, IK success,
    #                     actual joints read back from simulation, joint delta
    log_dtype = np.float64
    ik_target_right  = np.zeros((n_frames, 7), dtype=log_dtype)  # pos(3) + urdf_quat(4)
    ik_target_left   = np.zeros((n_frames, 7), dtype=log_dtype)
    ik_result_right  = np.zeros((n_frames, 7), dtype=log_dtype)  # 7 arm joint values
    ik_result_left   = np.zeros((n_frames, 7), dtype=log_dtype)
    ik_success_right = np.zeros(n_frames, dtype=bool)
    ik_success_left  = np.zeros(n_frames, dtype=bool)
    actual_joints_right = np.zeros((n_frames, 7), dtype=log_dtype)  # read-back from sim
    actual_joints_left  = np.zeros((n_frames, 7), dtype=log_dtype)
    h5_raw_right     = np.zeros((n_frames, 7), dtype=log_dtype)  # raw H5 wrist pose
    h5_raw_left      = np.zeros((n_frames, 7), dtype=log_dtype)

    # Prepare IK state (mirroring make_ik_position_setter)
    base_right = franka_right.get_joint_positions().copy()
    buf_right  = base_right.copy()
    prev_right = home_arm_joints.copy()

    base_left = franka_left.get_joint_positions().copy()
    buf_left  = base_left.copy()
    prev_left = home_arm_joints_for_left.copy()

    # Hold home briefly
    hold_frames = max(1, int(round(HOME_HOLD_SECONDS * FPS)))
    print(f"[diag] Holding home for {hold_frames} frames...")
    for _ in range(hold_frames):
        world.step(render=False)

    # ----- Replay loop with full logging -----
    print(f"\n[diag] Replaying {n_frames} frames with full logging...\n")
    ik_fail_right = 0
    ik_fail_left  = 0

    for i in range(n_frames):
        if not simulation_app.is_running():
            n_frames = i
            break

        # --- LEFT ARM ---
        wrist_left = data["arm_left"][i]
        h5_raw_left[i] = wrist_left
        pos_left = wrist_left[:3]
        quat_urdf_left = tool_quat_to_urdf(wrist_left[3:])
        ik_target_left[i, :3] = pos_left
        ik_target_left[i, 3:] = quat_urdf_left

        joints_left, success_left = solve_ik_for_pose(
            ik_solver_left, EE_FRAME_NAME_LEFT,
            pos_left, quat_urdf_left, warm_start=prev_left,
        )
        ik_success_left[i] = success_left
        if joints_left is not None:
            ik_result_left[i] = joints_left
            prev_left = joints_left.copy()
        else:
            ik_fail_left += 1
            ik_result_left[i] = prev_left

        buf_left[:] = base_left
        buf_left[arm_idx_left] = ik_result_left[i]
        buf_left[hand_idx_left] = HAND_HOME_JOINT_VALUES + data["hand_left"][i]
        franka_left.set_joint_positions(buf_left)

        # --- RIGHT ARM ---
        wrist_right = data["arm_right"][i]
        h5_raw_right[i] = wrist_right
        pos_right = wrist_right[:3]
        quat_urdf_right = tool_quat_to_urdf(wrist_right[3:])
        ik_target_right[i, :3] = pos_right
        ik_target_right[i, 3:] = quat_urdf_right

        joints_right, success_right = solve_ik_for_pose(
            ik_solver_right, EE_FRAME_NAME_RIGHT,
            pos_right, quat_urdf_right, warm_start=prev_right,
        )
        ik_success_right[i] = success_right
        if joints_right is not None:
            ik_result_right[i] = joints_right
            prev_right = joints_right.copy()
        else:
            ik_fail_right += 1
            ik_result_right[i] = prev_right

        buf_right[:] = base_right
        buf_right[arm_idx_right] = ik_result_right[i]
        buf_right[hand_idx_right] = HAND_HOME_JOINT_VALUES + data["hand_right"][i]
        franka_right.set_joint_positions(buf_right)

        # --- STEP SIM ---
        world.step(render=False)

        # --- READ BACK actual joints from simulation ---
        actual_joints_right[i] = franka_right.get_joint_positions()[arm_idx_right]
        actual_joints_left[i]  = franka_left.get_joint_positions()[arm_idx_left]

        if i % 200 == 0:
            pct = 100.0 * i / n_frames
            print(f"  frame {i:5d}/{n_frames}  ({pct:.1f}%)  "
                  f"IK fail L={ik_fail_left} R={ik_fail_right}")

    # ----- Save raw data -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "dual_replay_diagnosis.npz"
    np.savez(
        npz_path,
        h5_raw_right=h5_raw_right[:n_frames],
        h5_raw_left=h5_raw_left[:n_frames],
        ik_target_right=ik_target_right[:n_frames],
        ik_target_left=ik_target_left[:n_frames],
        ik_result_right=ik_result_right[:n_frames],
        ik_result_left=ik_result_left[:n_frames],
        ik_success_right=ik_success_right[:n_frames],
        ik_success_left=ik_success_left[:n_frames],
        actual_joints_right=actual_joints_right[:n_frames],
        actual_joints_left=actual_joints_left[:n_frames],
    )
    print(f"\n[diag] Saved raw data to {npz_path}")

    # ----- Analysis -----
    txt_path = OUTPUT_DIR / "dual_replay_diagnosis.txt"
    lines = []
    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 72)
    log("DUAL REPLAY DIAGNOSIS REPORT")
    log("=" * 72)

    log(f"\nTotal frames: {n_frames}")
    log(f"IK failures:  left={ik_fail_left}/{n_frames}  right={ik_fail_right}/{n_frames}")

    for side, ik_res, ik_ok, actual, h5_raw, label in [
        ("RIGHT", ik_result_right[:n_frames], ik_success_right[:n_frames],
         actual_joints_right[:n_frames], h5_raw_right[:n_frames], "right"),
        ("LEFT",  ik_result_left[:n_frames],  ik_success_left[:n_frames],
         actual_joints_left[:n_frames],  h5_raw_left[:n_frames],  "left"),
    ]:
        log(f"\n{'─' * 72}")
        log(f"  {side} ARM ANALYSIS")
        log(f"{'─' * 72}")

        # IK result joint deltas (frame-to-frame)
        ik_delta = np.diff(ik_res, axis=0)
        ik_delta_norm = np.linalg.norm(ik_delta, axis=1)
        max_delta_idx = np.argmax(ik_delta_norm)

        log(f"\n  IK joint delta (frame-to-frame L2 norm):")
        log(f"    mean: {ik_delta_norm.mean():.6f} rad")
        log(f"    max:  {ik_delta_norm.max():.6f} rad  at frame {max_delta_idx}→{max_delta_idx+1}")
        log(f"    p95:  {np.percentile(ik_delta_norm, 95):.6f} rad")
        log(f"    p99:  {np.percentile(ik_delta_norm, 99):.6f} rad")

        # Detect discontinuities: joint delta > threshold
        JUMP_THRESHOLD = 0.3  # rad, ~17 degrees — already large for single frame
        jumps = np.where(ik_delta_norm > JUMP_THRESHOLD)[0]
        log(f"\n  Frames with joint jump > {JUMP_THRESHOLD} rad: {len(jumps)}")
        if len(jumps) > 0:
            log(f"    First 20 jump frames:")
            for j in jumps[:20]:
                log(f"      frame {j}→{j+1}: delta={ik_delta_norm[j]:.4f} rad  "
                    f"ik_ok[{j}]={ik_ok[j]}  ik_ok[{j+1}]={ik_ok[j+1]}")
                log(f"        joints[{j}]:   {ik_res[j]}")
                log(f"        joints[{j+1}]: {ik_res[j+1]}")
                log(f"        h5 pos[{j}]:   {h5_raw[j, :3]}  "
                    f"h5 pos[{j+1}]: {h5_raw[j+1, :3]}")
                reach_j   = np.linalg.norm(h5_raw[j, :3])
                reach_jp1 = np.linalg.norm(h5_raw[j+1, :3])
                log(f"        reach[{j}]: {reach_j:.4f}m  reach[{j+1}]: {reach_jp1:.4f}m")

        # IK vs actual (sim read-back) discrepancy
        sim_diff = np.abs(ik_res - actual)
        sim_diff_max = sim_diff.max(axis=1)
        log(f"\n  IK result vs sim read-back max discrepancy per frame:")
        log(f"    mean: {sim_diff_max.mean():.6f} rad")
        log(f"    max:  {sim_diff_max.max():.6f} rad")
        if sim_diff_max.max() > 0.01:
            bad_frames = np.where(sim_diff_max > 0.01)[0]
            log(f"    Frames with discrepancy > 0.01 rad: {len(bad_frames)}")
            for bf in bad_frames[:10]:
                log(f"      frame {bf}: max_diff={sim_diff_max[bf]:.4f}")
                log(f"        IK result:  {ik_res[bf]}")
                log(f"        Sim actual: {actual[bf]}")

        # Per-joint statistics
        log(f"\n  Per-joint IK result range:")
        for j in range(7):
            lo = ik_res[:, j].min()
            hi = ik_res[:, j].max()
            log(f"    joint {j}: [{lo:+.4f}, {hi:+.4f}]  range={hi-lo:.4f}")

        # IK failure distribution
        fail_frames = np.where(~ik_ok)[0]
        if len(fail_frames) > 0:
            log(f"\n  IK failure frames (first 30): {fail_frames[:30]}")
            log(f"  IK failure positions:")
            for ff in fail_frames[:10]:
                log(f"    frame {ff}: pos={h5_raw[ff, :3]}  "
                    f"reach={np.linalg.norm(h5_raw[ff, :3]):.4f}m")

    log(f"\n{'=' * 72}")
    log("END OF REPORT")
    log(f"{'=' * 72}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[diag] Report saved to {txt_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
