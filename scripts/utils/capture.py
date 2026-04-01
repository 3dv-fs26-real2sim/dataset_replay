"""Video capture pipeline for Isaac Sim viewport.

Depends on imageio, numpy, and Isaac Sim viewport utilities (deferred import).
Must be imported after SimulationApp is created.
"""

import asyncio
import ctypes
import importlib
import inspect
from pathlib import Path

import numpy as np
import imageio.v2 as imageio


def setup_capture(
    total_frames: int,
    output_path: Path,
    fps: float,
    width: int = 1280,
    height: int = 720,
):
    """Configure imageio writer and viewport access for in-memory video capture.

    Returns:
        (recorder_dict, output_path) or (None, None) if capture is unavailable.
    """
    viewport_utility = _import_viewport_utility_module()
    if viewport_utility is None:
        print("[capture] Recording disabled: 'omni.kit.viewport.utility' is unavailable.")
        return None, None

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        print("[capture] Recording disabled: no active viewport found.")
        return None, None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(output_path), fps=max(1, int(round(fps))), codec="libx264")
    print(f"[capture] Recording replay to {output_path} via imageio ({total_frames} frames target)")

    return {
        "writer": writer,
        "output_path": output_path,
        "viewport_utility": viewport_utility,
        "viewport": viewport,
        "width": width,
        "height": height,
        "frames_written": 0,
        "failed_frames": 0,
        "logged_meta": False,
    }, output_path


def capture_frame_to_writer(recorder, simulation_app) -> bool:
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

    _wait_capture_completion(helper, simulation_app)

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


def close_recorder(recorder) -> None:
    """Close the video writer."""
    if recorder is None:
        return
    recorder["writer"].close()


# ── Private helpers ───────────────────────────────────────────────────────────


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


def _wait_capture_completion(helper, simulation_app):
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
