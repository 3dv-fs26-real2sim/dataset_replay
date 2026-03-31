import argparse
import asyncio
import ctypes
import importlib
import inspect
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Test replay script")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--fps", type=float, default=60.0, help="Playback frame rate (default: 60)")
parser.add_argument(
    "--use-actions",
    action="store_true",
    help="Use actions_* instead of observations/qpos_* for replay",
)
parser.add_argument(
    "--mode",
    type=str,
    default="dual",
    choices=["single", "dual"],
    help="Choose between single arm (right) or dual arm setup (default: dual)",
)
#* Recording works, but VERY slow.
parser.add_argument(
    "--record",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Record replay video to outputs/H5NAME_replay.mp4 (default: False)",
)
args = parser.parse_args()

APP_WIDTH = 1280
APP_HEIGHT = 720

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RayTracedLighting",
        "width": APP_WIDTH,
        "height": APP_HEIGHT,
    }
)

import numpy as np
import h5py
import imageio.v2 as imageio
import omni
import omni.usd
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

SCRIPT_DIR = Path(__file__).parent
if args.mode == "single":
    USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_single.usd"
    H5_PATH = SCRIPT_DIR / "../data/20250804_104715.h5"
elif args.mode == "dual":
    USD_PATH = SCRIPT_DIR / "../../pandaorca_description/usd/pandaorca_dual.usd"
    H5_PATH = SCRIPT_DIR / "../data/20250829_180500.h5"
else:
    raise ValueError(f"Invalid mode: {args.mode}")

OUTPUT_DIR = SCRIPT_DIR / "../outputs"
REPLAY_VIDEO_NAME = f"{H5_PATH.stem}_replay"

# Prim paths in the stage
FRANKA_LEFT_PATH  = "/World/fer_orcahand_left_extended"
FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"

# Number of dimensions expected from the h5 data
N_ARM_POSE_DIMS = 7   # 3 position (xyz) + 4 quaternion (wxyz)
N_HAND_DOFS     = 17

# Joint names and order. May need fix.
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

# --- IK Solver Configuration ---
# End-effector frame names in the URDF kinematic tree.
# Change these to target a different frame for testing.
EE_FRAME_NAME_LEFT  = "fer_link8" 
EE_FRAME_NAME_RIGHT = "fer_link8"

# Paths for the Lula IK solver.
# The descriptor defines the active cspace (fer_joint1-7).
# Per-side URDFs are needed because the EE frames live in the hand portion.
LULA_DESCRIPTOR_PATH = SCRIPT_DIR / "../../pandaorca_description/lula/fer_robot_descriptor.yaml"
URDF_PATH_LEFT  = SCRIPT_DIR / "../../pandaorca_description/urdf/fer_orcahand_left_extended.urdf"
URDF_PATH_RIGHT = SCRIPT_DIR / "../../pandaorca_description/urdf/fer_orcahand_right_extended.urdf"

# Initial wrist target pose for IK (replaces hardcoded joint angles).
# In each arm's coordinate system:
#   position  = [0.40, 0.0, 0.3]
#   rotation  = x=[1,0,0], z=[0,0,-1]  →  R = [[1,0,0],[0,-1,0],[0,0,-1]]
INITIAL_WRIST_POSITION = np.array([0.40, 0.0, 0.3])
INITIAL_WRIST_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

HAND_JOINT_VALUES_INITIAL = np.array([0.0] * N_HAND_DOFS)

# Show initial pose before replay starts.
INITIAL_HOLD_SECONDS = 3.0


def create_ik_solver(urdf_path: Path, label: str) -> LulaKinematicsSolver:
    """Create a Lula IK solver for the FER arm using the given URDF."""
    solver = LulaKinematicsSolver(
        robot_description_path=str(LULA_DESCRIPTOR_PATH.resolve()),
        urdf_path=str(urdf_path.resolve()),
    )
    print(f"[IK] Solver ({label}) created. Active joints: {solver.get_joint_names()}")
    print(f"[IK] Available frames: {solver.get_all_frame_names()}")
    return solver


def rotation_matrix_to_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    q_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # scipy returns xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])



def detect_quaternion_order(arm_data: np.ndarray, label: str) -> np.ndarray:
    """Check whether columns 3:7 are wxyz or xyzw quaternions, reorder to wxyz if needed.

    Returns the (possibly reordered) arm data array.
    """
    quat_wxyz = arm_data[:, 3:7]
    norms_wxyz = np.linalg.norm(quat_wxyz, axis=1)
    mean_norm_wxyz = np.mean(norms_wxyz)

    # Both orderings have the same norm (they're the same 4 values), so norm alone
    # can't distinguish them. Instead, check if w (scalar part) is typically the
    # largest component — for small rotations from identity, w is close to 1.
    # With wxyz: column 3 is w.  With xyzw: column 6 is w.
    w_if_wxyz = np.mean(np.abs(arm_data[:, 3]))
    w_if_xyzw = np.mean(np.abs(arm_data[:, 6]))

    print(f"[quat] {label}: norm={mean_norm_wxyz:.4f}  "
          f"mean|col3|={w_if_wxyz:.4f}  mean|col6|={w_if_xyzw:.4f}")

    if abs(mean_norm_wxyz - 1.0) > 0.1:
        print(f"[quat] WARNING: quaternion norm is {mean_norm_wxyz:.4f}, expected ~1.0. "
              "The arm data may not be in the expected [xyz, quat] format.")

    if w_if_xyzw > w_if_wxyz:
        print(f"[quat] {label}: detected xyzw ordering (col6 looks like scalar part). "
              "Reordering to wxyz.")
        reordered = arm_data.copy()
        reordered[:, 3] = arm_data[:, 6]  # w
        reordered[:, 4] = arm_data[:, 3]  # x
        reordered[:, 5] = arm_data[:, 4]  # y
        reordered[:, 6] = arm_data[:, 5]  # z
        return reordered
    else:
        print(f"[quat] {label}: appears to be wxyz ordering. Using as-is.")
        return arm_data


def solve_ik_for_pose(
    solver: LulaKinematicsSolver,
    ee_frame_name: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    warm_start: np.ndarray = None,
) -> tuple[np.ndarray | None, bool]:
    """Solve IK for a target wrist pose.

    Args:
        solver: LulaKinematicsSolver instance
        ee_frame_name: target end-effector frame in the URDF
        position: target xyz position
        orientation_wxyz: target orientation as wxyz quaternion
        warm_start: previous joint solution for faster convergence

    Returns:
        (joint_positions, success)
    """
    joint_positions, success = solver.compute_inverse_kinematics(
        frame_name=ee_frame_name,
        target_position=position,
        target_orientation=orientation_wxyz,
        warm_start=warm_start,
    )
    return joint_positions if success else None, success


def setup_articulation(prim_path: str, world: World) -> SingleArticulation:
    name = prim_path.lstrip("/").replace("/", "_")
    art = SingleArticulation(prim_path=prim_path, name=name)
    world.scene.add(art)
    return art


def print_dof_info(label: str, art: SingleArticulation):
    print(f"\n[DOF] {label}: {art.num_dof} DOFs")
    for i, name in enumerate(art.dof_names):
        print(f"      [{i:2d}] {name}")


def resolve_dof_indices(art: SingleArticulation, names: list[str], label: str) -> np.ndarray:
    """Map canonical joint names to articulation DOF indices."""
    dof_names = list(art.dof_names)
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    def candidate_names(name: str) -> list[str]:
        # Support both Franka naming (panda_joint*) and FER naming (fer_joint*).
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
                if cand != name:
                    print(f"[DOF] '{name}' matched via alias '{cand}'")
                indices.append(name_to_idx[cand])
                resolved = True
                break
        if resolved:
            continue

        matches = [dof for dof in dof_names if dof.endswith(name)]
        if len(matches) == 1:
            print(f"[DOF] '{name}' matched via suffix to '{matches[0]}'")
            indices.append(name_to_idx[matches[0]])
        elif len(matches) > 1:
            raise RuntimeError(
                f"[DOF] Ambiguous suffix match for '{name}' in {label}: {matches}"
            )
        else:
            raise RuntimeError(f"[DOF] Cannot find '{name}' in {label} DOFs: {dof_names}")

    return np.array(indices, dtype=int)


def _read_dataset(f: h5py.File, candidates: list[str], required: bool = True):
    for key in candidates:
        if key in f:
            return f[key][()], key
    if required:
        raise KeyError(f"None of these datasets exist: {candidates}")
    return None, None


def load_h5(path: Path, use_actions: bool, mode: str):
    """Load joint trajectories from either side-specific or generic h5 schemas."""
    if use_actions:
        arm_left_keys = ["actions_arm_left"]
        arm_right_keys = ["actions_arm_right"]
        hand_left_keys = ["actions_hand_left"]
        hand_right_keys = ["actions_hand_right"]
        arm_single_keys = ["actions_arm"]
        hand_single_keys = ["actions_hand"]
    else:
        arm_left_keys = ["observations/qpos_arm_left"]
        arm_right_keys = ["observations/qpos_arm_right"]
        hand_left_keys = ["observations/qpos_hand_left"]
        hand_right_keys = ["observations/qpos_hand_right"]
        arm_single_keys = ["observations/qpos_arm"]
        hand_single_keys = ["observations/qpos_hand"]

    with h5py.File(path, "r") as f:
        if mode == "dual":
            arm_left, arm_left_key = _read_dataset(f, arm_left_keys)
            arm_right, arm_right_key = _read_dataset(f, arm_right_keys)
            hand_left, hand_left_key = _read_dataset(f, hand_left_keys)
            hand_right, hand_right_key = _read_dataset(f, hand_right_keys)

            n = arm_left.shape[0]
            assert arm_left.shape   == (n, N_ARM_POSE_DIMS),  f"arm_left shape:  {arm_left.shape}"
            assert arm_right.shape  == (n, N_ARM_POSE_DIMS),  f"arm_right shape: {arm_right.shape}"
            assert hand_left.shape  == (n, N_HAND_DOFS), f"hand_left shape:  {hand_left.shape}"
            assert hand_right.shape == (n, N_HAND_DOFS), f"hand_right shape: {hand_right.shape}"

            print(f"[h5] Loaded {n} frames from {path}")
            print(f"     arm_left:  {arm_left.shape}  ({arm_left_key})")
            print(f"     arm_right: {arm_right.shape} ({arm_right_key})")
            print(f"     hand_left: {hand_left.shape} ({hand_left_key})")
            print(f"     hand_right:{hand_right.shape} ({hand_right_key})")

            return {
                "arm_left": arm_left,
                "arm_right": arm_right,
                "hand_left": hand_left,
                "hand_right": hand_right,
                "n_frames": n,
            }

        arm_right, arm_right_key = _read_dataset(f, arm_right_keys + arm_single_keys)
        hand_right, hand_right_key = _read_dataset(f, hand_right_keys + hand_single_keys)
        arm_left, arm_left_key = _read_dataset(f, arm_left_keys, required=False)
        hand_left, hand_left_key = _read_dataset(f, hand_left_keys, required=False)

    n = arm_right.shape[0]
    assert arm_right.shape == (n, N_ARM_POSE_DIMS), f"arm_right shape: {arm_right.shape}"
    assert hand_right.shape == (n, N_HAND_DOFS), f"hand_right shape: {hand_right.shape}"

    if arm_left is not None and hand_left is not None:
        assert arm_left.shape == (n, N_ARM_POSE_DIMS), f"arm_left shape: {arm_left.shape}"
        assert hand_left.shape == (n, N_HAND_DOFS), f"hand_left shape: {hand_left.shape}"

    print(f"[h5] Loaded {n} frames from {path}")
    print(f"     arm_right:  {arm_right.shape} ({arm_right_key})")
    print(f"     hand_right: {hand_right.shape} ({hand_right_key})")
    if arm_left is not None and hand_left is not None:
        print(f"     arm_left:   {arm_left.shape} ({arm_left_key})")
        print(f"     hand_left:  {hand_left.shape} ({hand_left_key})")

    return {
        "arm_left": arm_left,
        "arm_right": arm_right,
        "hand_left": hand_left,
        "hand_right": hand_right,
        "n_frames": n,
    }


def make_ik_position_setter(
    art: SingleArticulation,
    arm_idx: np.ndarray,
    hand_idx: np.ndarray,
    ik_solver: LulaKinematicsSolver,
    ee_frame_name: str,
    hand_initial: np.ndarray,
    initial_arm_joints: np.ndarray,
):
    """Return a callable that solves IK for each wrist pose and sets joint positions.

    Args:
        art: The robot articulation.
        arm_idx: DOF indices for the arm joints.
        hand_idx: DOF indices for the hand joints.
        ik_solver: Lula IK solver instance.
        ee_frame_name: target end-effector frame for IK.
        hand_initial: Initial hand joint values (offsets are added to these).
        initial_arm_joints: IK solution for the initial pose, used as first warm-start.
    """
    base = art.get_joint_positions().copy()
    buf = base.copy()
    state = {
        "prev_arm_joints": initial_arm_joints.copy(),
        "ik_failures": 0,
    }

    def set_positions(wrist_pose: np.ndarray, q_hand: np.ndarray):
        """Set joint positions from a wrist pose and hand joint angles.

        Args:
            wrist_pose: [x, y, z, qw, qx, qy, qz] — absolute wrist pose (wxyz quat).
            q_hand: hand joint angle offsets (added to hand_initial).
        """
        position = wrist_pose[:3]
        orientation_wxyz = wrist_pose[3:]

        arm_joints, _ = solve_ik_for_pose(
            ik_solver, ee_frame_name, position, orientation_wxyz,
            warm_start=state["prev_arm_joints"],
        )

        buf[:] = base
        if arm_joints is not None:
            buf[arm_idx] = arm_joints
            state["prev_arm_joints"] = arm_joints.copy()
        else:
            state["ik_failures"] += 1
            buf[arm_idx] = state["prev_arm_joints"]

        buf[hand_idx] = hand_initial + q_hand
        art.set_joint_positions(buf)

    def get_ik_failure_count() -> int:
        return state["ik_failures"]

    set_positions.get_ik_failure_count = get_ik_failure_count
    return set_positions


def setup_capture(total_frames: int):
    """Configure imageio writer and viewport access for in-memory video capture."""
    viewport_utility = _import_viewport_utility_module()
    if viewport_utility is None:
        print("[capture] Recording disabled: 'omni.kit.viewport.utility' is unavailable.")
        return None, None

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        print("[capture] Recording disabled: no active viewport found.")
        return None, None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{REPLAY_VIDEO_NAME}.mp4"
    writer = imageio.get_writer(str(output_path), fps=max(1, int(round(args.fps))), codec="libx264")
    print(f"[capture] Recording replay to {output_path} via imageio ({total_frames} frames target)")

    return {
        "writer": writer,
        "output_path": output_path,
        "viewport_utility": viewport_utility,
        "viewport": viewport,
        "width": APP_WIDTH,
        "height": APP_HEIGHT,
        "frames_written": 0,
        "failed_frames": 0,
        "logged_meta": False,
    }, output_path


def _import_viewport_utility_module():
    """Import viewport utility module, enabling extension first if needed."""
    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        pass

    try:
        app = importlib.import_module("omni.kit.app")
        ext_manager = app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("omni.kit.viewport.utility", True)
    except Exception as exc:
        print(f"[capture] Could not enable omni.kit.viewport.utility extension: {exc}")
        return None

    try:
        return importlib.import_module("omni.kit.viewport.utility")
    except ModuleNotFoundError:
        return None


def _decode_capture_frame(payload) -> np.ndarray | None:
    """Best-effort conversion of viewport capture callback payload to HxWx3 uint8."""
    if payload is None:
        return None

    if isinstance(payload, np.ndarray):
        frame = payload
    elif isinstance(payload, memoryview):
        frame = np.frombuffer(payload, dtype=np.uint8)
    elif isinstance(payload, (bytes, bytearray)):
        frame = np.frombuffer(payload, dtype=np.uint8)
    else:
        return None

    if frame.ndim == 3 and frame.shape[-1] in (3, 4):
        return frame[..., :3]
    return None


def _find_dimensions(values, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    """Infer image width/height from callback args with a safe fallback."""
    for value in values:
        if hasattr(value, "width") and hasattr(value, "height"):
            try:
                w = int(getattr(value, "width"))
                h = int(getattr(value, "height"))
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

    ints = [int(v) for v in values if isinstance(v, (int, np.integer)) and int(v) > 0]
    for i in range(len(ints) - 1):
        w, h = ints[i], ints[i + 1]
        if w >= 64 and h >= 64:
            return w, h

    return fallback_w, fallback_h


def _decode_raw_with_dims(raw: np.ndarray, width: int, height: int) -> np.ndarray | None:
    if raw.ndim != 1:
        return None

    pixel_count = width * height
    if pixel_count <= 0:
        return None

    if raw.size == pixel_count * 4:
        return raw.reshape(height, width, 4)[..., :3]
    if raw.size == pixel_count * 3:
        return raw.reshape(height, width, 3)

    # Fallback: infer channel count from length if exact.
    if raw.size % pixel_count == 0:
        channels = raw.size // pixel_count
        if channels >= 3:
            return raw.reshape(height, width, channels)[..., :3]

    return None


def _buffer_to_numpy_1d(buffer_obj, buffer_size: int) -> np.ndarray | None:
    """Convert callback buffer object to a uint8 1D numpy view."""
    if buffer_size <= 0:
        return None

    if isinstance(buffer_obj, np.ndarray):
        return buffer_obj.reshape(-1).astype(np.uint8, copy=False)
    try:
        mv = memoryview(buffer_obj)
        return np.frombuffer(mv, dtype=np.uint8, count=buffer_size)
    except Exception:
        pass
    if isinstance(buffer_obj, memoryview):
        return np.frombuffer(buffer_obj, dtype=np.uint8, count=buffer_size)
    if isinstance(buffer_obj, (bytes, bytearray)):
        return np.frombuffer(buffer_obj, dtype=np.uint8, count=buffer_size)

    try:
        return np.frombuffer(buffer_obj, dtype=np.uint8, count=buffer_size)
    except Exception:
        pass

    # Pointer-like callback payload (common in ByteCapture): read bytes via ctypes.
    try:
        ptr = int(buffer_obj)
    except Exception:
        try:
            ptr = int(getattr(buffer_obj, "value"))
        except Exception:
            ptr = None

    if ptr is None:
        # PyCapsule callback payload (common in Kit C++ callback bridges).
        try:
            pycapsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
            pycapsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            pycapsule_get_pointer.restype = ctypes.c_void_p
            ptr = int(pycapsule_get_pointer(buffer_obj, None))
        except Exception:
            # Fallback: common pointer-like attributes.
            for attr in ("ptr", "data", "address"):
                try:
                    ptr = int(getattr(buffer_obj, attr))
                    break
                except Exception:
                    continue

    if ptr is None:
        return None

    if ptr == 0:
        return None

    raw_bytes = ctypes.string_at(ptr, buffer_size)
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def _wait_capture_completion(helper):
    """Handle both sync and async wait_for_result implementations."""
    if not hasattr(helper, "wait_for_result"):
        return

    try:
        result = helper.wait_for_result(completion_frames=30)
    except TypeError:
        result = helper.wait_for_result()

    if not inspect.isawaitable(result):
        return

    task = asyncio.ensure_future(result)
    for _ in range(180):
        if task.done():
            break
        simulation_app.update()

    if task.done():
        exc = task.exception()
        if exc is not None:
            raise exc


def capture_frame_to_writer(recorder) -> bool:
    """Capture one viewport frame to memory and append it to the imageio writer."""
    holder = {"frame": None}

    def on_capture(*cb_args, **cb_kwargs):
        # ByteCapture callback signature: (buffer, buffer_size, width, height, byte_format)
        if len(cb_args) >= 5:
            if not recorder["logged_meta"]:
                recorder["logged_meta"] = True
                print(
                    f"[capture] Callback meta: size={int(cb_args[1])} "
                    f"w={int(cb_args[2])} h={int(cb_args[3])} format={cb_args[4]}"
                )
            raw = _buffer_to_numpy_1d(cb_args[0], int(cb_args[1]))
            if raw is not None:
                decoded = _decode_raw_with_dims(raw, int(cb_args[2]), int(cb_args[3]))
                if decoded is not None:
                    holder["frame"] = decoded
                    return

        all_values = list(cb_args) + list(cb_kwargs.values())
        width, height = _find_dimensions(all_values, recorder["width"], recorder["height"])

        for value in cb_args:
            decoded = _decode_capture_frame(value)
            if decoded is not None:
                holder["frame"] = decoded
                return
        for value in cb_kwargs.values():
            decoded = _decode_capture_frame(value)
            if decoded is not None:
                holder["frame"] = decoded
                return

        for value in all_values:
            if isinstance(value, memoryview):
                raw = np.frombuffer(value, dtype=np.uint8)
            elif isinstance(value, (bytes, bytearray)):
                raw = np.frombuffer(value, dtype=np.uint8)
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                raw = value.astype(np.uint8, copy=False)
            else:
                try:
                    raw = np.frombuffer(value, dtype=np.uint8)
                except Exception:
                    continue

            decoded = _decode_raw_with_dims(raw, width, height)
            if decoded is not None:
                holder["frame"] = decoded
                return

    helper = recorder["viewport_utility"].capture_viewport_to_buffer(recorder["viewport"], on_capture)

    _wait_capture_completion(helper)

    if holder["frame"] is None:
        # Some Kit versions deliver callback data on a subsequent app update.
        for _ in range(6):
            simulation_app.update()
            if holder["frame"] is not None:
                break

    frame = holder["frame"]
    if frame is None:
        recorder["failed_frames"] += 1
        if recorder["failed_frames"] <= 3:
            print(
                "[capture] Warning: frame capture callback returned no decodable image "
                f"(buffer_type={type(cb_args[0]).__name__ if cb_args else 'unknown'})."
            )
        return False

    recorder["writer"].append_data(frame)
    recorder["frames_written"] += 1
    return True


def close_recorder(recorder):
    if recorder is None:
        return
    recorder["writer"].close()


def main():
    data = load_h5(H5_PATH, args.use_actions, args.mode)
    n_frames = data["n_frames"]

    # Detect and fix quaternion ordering in arm data (ensure wxyz).
    data["arm_right"] = detect_quaternion_order(data["arm_right"], "arm_right")
    if data["arm_left"] is not None:
        data["arm_left"] = detect_quaternion_order(data["arm_left"], "arm_left")

    # Print first frame diagnostics.
    sample = data["arm_right"][0]
    print(f"[debug] First frame arm_right: {sample}")
    print(f"[debug]   Position (xyz): {sample[:3]}")
    print(f"[debug]   Quaternion (wxyz): {sample[3:]}")
    print(f"[debug]   Quaternion norm: {np.linalg.norm(sample[3:]):.4f}")

    # Load USD scene
    omni.usd.get_context().open_stage(str(USD_PATH))

    # Create world and add articulations
    world = World()
    franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
    if args.mode == "dual":
        franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
    world.reset()

    # Create per-side IK solvers (different URDFs for left/right hand frames).
    ik_solver_right = create_ik_solver(URDF_PATH_RIGHT, "right")
    if args.mode == "dual":
        ik_solver_left = create_ik_solver(URDF_PATH_LEFT, "left")

    # Compute initial arm joint values via IK.
    initial_wrist_quat = rotation_matrix_to_wxyz(INITIAL_WRIST_ROTATION)
    initial_arm_joints, _ = solve_ik_for_pose(
        ik_solver_right, EE_FRAME_NAME_RIGHT,
        INITIAL_WRIST_POSITION, initial_wrist_quat,
    )
    if initial_arm_joints is None:
        raise RuntimeError(
            f"IK failed for initial wrist pose "
            f"(pos={INITIAL_WRIST_POSITION}, quat={initial_wrist_quat}). "
            f"Check EE_FRAME_NAME_RIGHT='{EE_FRAME_NAME_RIGHT}' and the Lula descriptor."
        )
    print(f"[IK] Initial arm joints (rad): {initial_arm_joints}")

    print_dof_info("franka_right", franka_right)
    arm_idx_right = resolve_dof_indices(franka_right, ARM_JOINT_NAMES, "franka_right")
    hand_idx_right = resolve_dof_indices(franka_right, HAND_RIGHT_JOINT_NAMES, "franka_right")
    if args.mode == "dual":
        print_dof_info("franka_left", franka_left)
        arm_idx_left = resolve_dof_indices(franka_left, ARM_JOINT_NAMES, "franka_left")
        hand_idx_left = resolve_dof_indices(franka_left, HAND_LEFT_JOINT_NAMES, "franka_left")

    # Set initial joint values through resolved DOF indices.
    q_init_right = franka_right.get_joint_positions().copy()
    q_init_right[arm_idx_right] = initial_arm_joints
    q_init_right[hand_idx_right] = HAND_JOINT_VALUES_INITIAL
    franka_right.set_joint_positions(q_init_right)
    if args.mode == "dual":
        q_init_left = franka_left.get_joint_positions().copy()
        q_init_left[arm_idx_left] = initial_arm_joints
        q_init_left[hand_idx_left] = HAND_JOINT_VALUES_INITIAL
        franka_left.set_joint_positions(q_init_left)

    set_right = make_ik_position_setter(
        franka_right, arm_idx_right, hand_idx_right,
        ik_solver_right, EE_FRAME_NAME_RIGHT,
        HAND_JOINT_VALUES_INITIAL, initial_arm_joints,
    )
    if args.mode == "dual":
        set_left = make_ik_position_setter(
            franka_left, arm_idx_left, hand_idx_left,
            ik_solver_left, EE_FRAME_NAME_LEFT,
            HAND_JOINT_VALUES_INITIAL, initial_arm_joints,
        )

    hold_frames = max(1, int(round(INITIAL_HOLD_SECONDS * args.fps)))
    total_capture_frames = hold_frames + n_frames
    recorder, output_path = (None, None)
    if args.record:
        recorder, output_path = setup_capture(total_capture_frames)
    else:
        print("[capture] Recording disabled by --no-record")
    captured_frames = 0

    print(f"[replay] Holding initial pose for {INITIAL_HOLD_SECONDS:.1f}s ({hold_frames} frames)")
    for _ in range(hold_frames):
        if not simulation_app.is_running():
            close_recorder(recorder)
            simulation_app.close()
            return
        world.step(render=True)
        if recorder is not None and capture_frame_to_writer(recorder):
            captured_frames += 1

    print(f"\n[replay] Starting {n_frames} frames at {args.fps} fps...  (Ctrl-C to stop)\n")
    try:
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                break

            if args.mode == "dual":
                set_left(data["arm_left"][frame_idx], data["hand_left"][frame_idx])
            set_right(data["arm_right"][frame_idx], data["hand_right"][frame_idx])

            world.step(render=True)
            if recorder is not None and capture_frame_to_writer(recorder):
                captured_frames += 1

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / n_frames
                print(f"  frame {frame_idx:5d}/{n_frames}  ({pct:.1f}%)")
    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.")

    # Report IK failure stats
    ik_failures_right = set_right.get_ik_failure_count()
    if args.mode == "dual":
        ik_failures_left = set_left.get_ik_failure_count()
        print(f"[IK] Failures: right={ik_failures_right}/{n_frames}, "
              f"left={ik_failures_left}/{n_frames}")
    else:
        print(f"[IK] Failures: {ik_failures_right}/{n_frames}")

    close_recorder(recorder)
    if recorder is not None:
        print(f"[capture] Saved replay video to {output_path} ({captured_frames} frames)")

    print("[replay] Done.")

    simulation_app.close()


if __name__ == "__main__":
    main()