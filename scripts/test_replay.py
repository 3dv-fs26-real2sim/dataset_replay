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
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

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

# Number of DOFs expected from the h5 data
N_ARM_DOFS  = 7
N_HAND_DOFS = 17

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

# From invkin_pose.py
ARM_JOINT_VALUES_INITIAL = np.array([0.184280, -0.225290, -0.193213, -2.701355, -0.069782, 2.478938, 0.050881])
HAND_JOINT_VALUES_INITIAL = np.array([0.0] * N_HAND_DOFS)

# Show initial pose before replay starts.
INITIAL_HOLD_SECONDS = 3.0


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
            assert arm_left.shape   == (n, N_ARM_DOFS),  f"arm_left shape:  {arm_left.shape}"
            assert arm_right.shape  == (n, N_ARM_DOFS),  f"arm_right shape: {arm_right.shape}"
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
    assert arm_right.shape == (n, N_ARM_DOFS), f"arm_right shape: {arm_right.shape}"
    assert hand_right.shape == (n, N_HAND_DOFS), f"hand_right shape: {hand_right.shape}"

    if arm_left is not None and hand_left is not None:
        assert arm_left.shape == (n, N_ARM_DOFS), f"arm_left shape: {arm_left.shape}"
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


def make_position_setter(
    art: SingleArticulation,
    arm_idx: np.ndarray,
    hand_idx: np.ndarray,
    arm_initial: np.ndarray,
    hand_initial: np.ndarray,
):
    """Return a callable that applies h5 deltas relative to initial joint values."""
    base = art.get_joint_positions().copy()
    buf = base.copy()

    def set_positions(q_arm: np.ndarray, q_hand: np.ndarray):
        # h5 values are replayed as offsets from the configured initial pose.
        buf[:] = base
        buf[arm_idx] = arm_initial + q_arm
        buf[hand_idx] = hand_initial + q_hand
        art.set_joint_positions(buf)

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

    # Load USD scene
    omni.usd.get_context().open_stage(str(USD_PATH))
    
    # Create world and add articulations
    world = World()
    franka_right = setup_articulation(FRANKA_RIGHT_PATH, world)
    if args.mode == "dual":
        franka_left = setup_articulation(FRANKA_LEFT_PATH, world)
    world.reset()

    print_dof_info("franka_right", franka_right)
    arm_idx_right = resolve_dof_indices(franka_right, ARM_JOINT_NAMES, "franka_right")
    hand_idx_right = resolve_dof_indices(franka_right, HAND_RIGHT_JOINT_NAMES, "franka_right")
    if args.mode == "dual":
        print_dof_info("franka_left", franka_left)
        arm_idx_left = resolve_dof_indices(franka_left, ARM_JOINT_NAMES, "franka_left")
        hand_idx_left = resolve_dof_indices(franka_left, HAND_LEFT_JOINT_NAMES, "franka_left")

    # Set initial joint values through resolved DOF indices.
    q_init_right = franka_right.get_joint_positions().copy()
    q_init_right[arm_idx_right] = ARM_JOINT_VALUES_INITIAL
    q_init_right[hand_idx_right] = HAND_JOINT_VALUES_INITIAL
    franka_right.set_joint_positions(q_init_right)
    if args.mode == "dual":
        q_init_left = franka_left.get_joint_positions().copy()
        q_init_left[arm_idx_left] = ARM_JOINT_VALUES_INITIAL
        q_init_left[hand_idx_left] = HAND_JOINT_VALUES_INITIAL
        franka_left.set_joint_positions(q_init_left)

    set_right = make_position_setter(
        franka_right,
        arm_idx_right,
        hand_idx_right,
        ARM_JOINT_VALUES_INITIAL,
        HAND_JOINT_VALUES_INITIAL,
    )
    if args.mode == "dual":
        set_left = make_position_setter(
            franka_left,
            arm_idx_left,
            hand_idx_left,
            ARM_JOINT_VALUES_INITIAL,
            HAND_JOINT_VALUES_INITIAL,
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

    close_recorder(recorder)
    if recorder is not None:
        print(f"[capture] Saved replay video to {output_path} ({captured_frames} frames)")

    print("[replay] Done.")
    
    simulation_app.close()


if __name__ == "__main__":
    main()