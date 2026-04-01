"""Quick headless single-mode diagnostic — compare sim read-back vs IK set."""

import argparse
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--use-actions", action="store_true")
parser.add_argument("--max-frames", type=int, default=300)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                                 "width": 640, "height": 480})

import numpy as np
import h5py
import omni, omni.usd
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

SCRIPT_DIR = Path(__file__).parent
USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_single.usd"
H5_PATH = SCRIPT_DIR / "../data/20250804_104715.h5"
OUTPUT_DIR = SCRIPT_DIR / "../outputs"

FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

EE_FRAME_NAME = "fer_link8"
LULA_DESCRIPTOR_PATH = SCRIPT_DIR / "../../pandaorca_description/lula/fer_robot_descriptor.yaml"
URDF_PATH_RIGHT = SCRIPT_DIR / "../../pandaorca_description/urdf/fer_orcahand_right_extended.urdf"

WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
HAND_HOME = np.array([0.0] * 17)


def rotation_matrix_to_wxyz(m):
    q = Rotation.from_matrix(m).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


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


def tool_quat_to_urdf(q):
    return quat_multiply(Q_TOOL_TO_URDF, q)


def detect_quaternion_order(arm_data, label):
    if np.mean(np.abs(arm_data[:, 6])) > np.mean(np.abs(arm_data[:, 3])):
        r = arm_data.copy()
        r[:, 3] = arm_data[:, 6]
        r[:, 4] = arm_data[:, 3]
        r[:, 5] = arm_data[:, 4]
        r[:, 6] = arm_data[:, 5]
        print(f"[quat] {label}: xyzw -> wxyz")
        return r
    print(f"[quat] {label}: wxyz as-is")
    return arm_data


def resolve_dof_indices(art, names, label):
    dof_names = list(art.dof_names)
    name_map = {n: i for i, n in enumerate(dof_names)}
    indices = []
    for name in names:
        candidates = [name]
        if "panda_joint" in name:
            candidates.append(name.replace("panda_joint", "fer_joint", 1))
        found = False
        for c in candidates:
            if c in name_map:
                indices.append(name_map[c])
                found = True
                break
        if not found:
            matches = [d for d in dof_names if d.endswith(name)]
            if len(matches) == 1:
                indices.append(name_map[matches[0]])
            else:
                raise RuntimeError(f"Cannot find '{name}' in {label} DOFs: {dof_names}")
    return np.array(indices, dtype=int)


def main():
    # Write output to file so it survives Isaac Sim stdout redirect
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "single_replay_diagnosis.txt"
    out = open(out_path, "w")

    def log(s=""):
        print(s)
        out.write(s + "\n")
        out.flush()

    try:
        log("[diag] Starting single-mode diagnosis...")

        prefix = "actions" if args.use_actions else "observations/qpos"
        with h5py.File(H5_PATH, "r") as f:
            key_arm = f"{prefix}_arm"
            key_hand = f"{prefix}_hand"
            arm = f[key_arm][()] if key_arm in f else f[f"{prefix}_arm_right"][()]
            hand = f[key_hand][()] if key_hand in f else f[f"{prefix}_hand_right"][()]
        n = min(arm.shape[0], args.max_frames)
        log(f"[diag] Loaded {arm.shape[0]} frames, using {n}")

        arm = detect_quaternion_order(arm[:n], "single")

        log("[diag] Loading USD stage...")
        omni.usd.get_context().open_stage(str(USD_PATH))
        world = World()
        fr = SingleArticulation(prim_path=FRANKA_RIGHT_PATH, name="fr")
        world.scene.add(fr)
        world.reset()
        log("[diag] World reset done")

        solver = LulaKinematicsSolver(
            robot_description_path=str(LULA_DESCRIPTOR_PATH.resolve()),
            urdf_path=str(URDF_PATH_RIGHT.resolve()),
        )
        log("[diag] IK solver created")

        hq = rotation_matrix_to_wxyz(WRIST_HOME_ROTATION)
        home_j, ok = solver.compute_inverse_kinematics(
            frame_name=EE_FRAME_NAME,
            target_position=WRIST_HOME_POSITION,
            target_orientation=hq,
        )
        if not ok:
            log("FATAL: Home IK failed!")
            simulation_app.close()
            return
        log(f"[IK] Home: {home_j}")

        arm_idx = resolve_dof_indices(fr, ARM_JOINT_NAMES, "fr")
        hand_idx = resolve_dof_indices(fr, HAND_RIGHT_JOINT_NAMES, "fr")
        base = fr.get_joint_positions().copy()
        base[arm_idx] = home_j
        base[hand_idx] = HAND_HOME
        fr.set_joint_positions(base)

        log("[diag] Holding home for 50 frames...")
        for _ in range(50):
            world.step(render=False)

        ik_res = np.zeros((n, 7))
        actual = np.zeros((n, 7))
        prev = home_j.copy()
        fails = 0
        buf = base.copy()

        log(f"[diag] Replaying {n} frames...")
        for i in range(n):
            if not simulation_app.is_running():
                log(f"[diag] Sim stopped at frame {i}")
                n = i
                break
            p = arm[i, :3]
            q = tool_quat_to_urdf(arm[i, 3:])
            j, ok = solver.compute_inverse_kinematics(
                frame_name=EE_FRAME_NAME,
                target_position=p,
                target_orientation=q,
                warm_start=prev,
            )
            if ok:
                ik_res[i] = j
                prev = j.copy()
            else:
                fails += 1
                ik_res[i] = prev

            buf[:] = base
            buf[arm_idx] = ik_res[i]
            buf[hand_idx] = HAND_HOME + hand[i]
            fr.set_joint_positions(buf)
            world.step(render=False)
            actual[i] = fr.get_joint_positions()[arm_idx]

            if i % 200 == 0:
                log(f"  frame {i}/{n}  fails={fails}")

        diff = np.abs(ik_res[:n] - actual[:n])
        maxd = diff.max(axis=1)
        log(f"\n=== SINGLE MODE DIAGNOSIS ===")
        log(f"Frames: {n}, IK failures: {fails}")
        log(f"Sim read-back discrepancy:")
        log(f"  mean: {maxd.mean():.6f} rad")
        log(f"  max:  {maxd.max():.6f} rad")
        log(f"  frames > 0.01 rad: {(maxd > 0.01).sum()}")
        log(f"  frames > 0.05 rad: {(maxd > 0.05).sum()}")

        ik_delta = np.linalg.norm(np.diff(ik_res[:n], axis=0), axis=1)
        log(f"IK joint delta: mean={ik_delta.mean():.6f}  max={ik_delta.max():.6f}")
        actual_delta = np.linalg.norm(np.diff(actual[:n], axis=0), axis=1)
        log(f"Sim joint delta: mean={actual_delta.mean():.6f}  max={actual_delta.max():.6f}")

        log(f"\nReport saved to {out_path}")
        out.close()
        simulation_app.close()

    except Exception:
        traceback.print_exc()
        traceback.print_exc(file=out)
        out.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
