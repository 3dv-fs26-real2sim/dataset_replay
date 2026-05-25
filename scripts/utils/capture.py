"""Video capture pipeline for Isaac Sim viewport.

Depends on imageio, numpy, and Isaac Sim viewport utilities (deferred import).
Must be imported after SimulationApp is created.
"""
# TODO: Recording is very slow and laggy. Investigate better methods if available.

import asyncio
import ctypes
import inspect
from pathlib import Path

import numpy as np
import imageio.v2 as imageio


def setup_recording(
    args,
    h5_path,
    n_frames: int,
    video_suffix: str,
    app_width: int,
    app_height: int,
    *,
    dataset: str,
):
    """Configure video recorders based on CLI flags.

    This is a high-level helper that wires up ``setup_capture``,
    ``setup_sidebyside``, and ``setup_overlay`` according to the recording-
    related argparse flags (``--record-sim``, ``--record-sidebyside``,
    ``--record-overlay``, ``--no-fast-record``, ``--h5-camera``).

    Args:
        args: Parsed CLI args (must have ``record_sim``, ``record_sidebyside``,
            ``no_fast_record``, ``h5_camera``, ``fps`` attributes; may also
            provide ``record_overlay`` as a comma-separated string of alphas).
        h5_path: ``Path`` to the H5 file (used for output naming and sbs source).
        n_frames: Total frame count.
        video_suffix: Descriptive suffix appended to output filenames
            (e.g. ``"qpos_duck"``).
        app_width: Render width in pixels.
        app_height: Render height in pixels.
        dataset: ``"maple"`` or ``"egoverse"`` — picks the right H5 image-key
            template when opening the file for side-by-side / overlay.

    Returns:
        ``(recorder, sim_output_path, sbs_recorder, sbs_output_path, overlay_recorders)``
        where ``overlay_recorders`` is a list of recorder dicts (one per alpha).
    """
    from .constants import OUTPUT_DIR

    overlay_alphas = _parse_overlay_alphas(getattr(args, "record_overlay", None))
    needs_viewport_capture = (
        args.record_sim or args.record_sidebyside or bool(overlay_alphas)
    )
    deferred = not args.no_fast_record

    recorder, sim_output_path = (None, None)
    if needs_viewport_capture:
        if args.record_sim:
            video_name = f"{h5_path.stem}_replay_{video_suffix}.mp4"
            sim_output_path = OUTPUT_DIR / video_name
        else:
            sim_output_path = None  # memory-only capture for sbs / overlay
        recorder, sim_output_path = setup_capture(
            n_frames, sim_output_path,
            args.fps, app_width, app_height, deferred=deferred,
        )

    sbs_recorder, sbs_output_path = (None, None)
    if args.record_sidebyside:
        sbs_name = f"{h5_path.stem}_sidebyside_{video_suffix}.mp4"
        sbs_recorder, sbs_output_path = setup_sidebyside(
            n_frames, OUTPUT_DIR / sbs_name,
            args.fps, app_width, app_height, h5_path, args.h5_camera,
            dataset=dataset,
        )

    overlay_recorders = []
    for alpha in overlay_alphas:
        ov_name = f"{h5_path.stem}_overlay_a{alpha:.2f}_{video_suffix}.mp4"
        ov_recorder, _ = setup_overlay(
            n_frames, OUTPUT_DIR / ov_name,
            args.fps, app_width, app_height, h5_path, args.h5_camera, alpha,
            dataset=dataset,
        )
        if ov_recorder is not None:
            overlay_recorders.append(ov_recorder)

    if not needs_viewport_capture:
        print("[capture] No recording flags set.")

    return recorder, sim_output_path, sbs_recorder, sbs_output_path, overlay_recorders


def _parse_overlay_alphas(spec) -> list[float]:
    """Parse the comma-separated --record-overlay string into a list of floats.

    Values outside [0, 1] or unparsable tokens are rejected with ValueError.
    Returns an empty list when ``spec`` is None or empty.
    """
    if spec is None:
        return []
    alphas = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--record-overlay alpha must be in [0, 1]; got {value}"
            )
        alphas.append(value)
    return alphas


def setup_capture(
    total_frames: int,
    output_path: Path | None,
    fps: float,
    width: int = 1280,
    height: int = 720,
    deferred: bool = True,
):
    """Configure viewport access for frame capture, optionally writing to MP4.

    Args:
        output_path: Path for the MP4 file.  Pass ``None`` to capture frames
            into memory only (useful when frames are needed for side-by-side
            composition but no standalone video is desired).
        deferred: If True (default), buffer frames in memory and encode after the
            loop finishes.  Faster replay but uses more RAM.  Set to False for
            inline per-frame encoding (slower but low memory).

    Returns:
        (recorder_dict, output_path) or (None, None) if capture is unavailable.
    """
    from .viewport import get_viewport_utility
    viewport_utility = get_viewport_utility()
    if viewport_utility is None:
        print("[capture] Recording disabled: 'omni.kit.viewport.utility' is unavailable.")
        return None, None

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        print("[capture] Recording disabled: no active viewport found.")
        return None, None

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output_path), fps=max(1, int(round(fps))), codec="libx264")
        mode = "deferred" if deferred else "inline"
        print(f"[capture] Recording replay to {output_path} via imageio ({total_frames} frames target, {mode} encoding)")
    else:
        print(f"[capture] Viewport capture enabled (memory only, {total_frames} frames target)")

    return {
        "writer": writer,
        "output_path": output_path,
        "viewport_utility": viewport_utility,
        "viewport": viewport,
        "width": width,
        "height": height,
        "frames_written": 0,
        "failed_frames": 0,
        "deferred": deferred,
        "frames": [],
        "last_frame": None,
    }, output_path


def capture_frame_to_writer(recorder, simulation_app) -> bool:
    """Capture one viewport frame to memory and append it to the imageio writer."""
    holder = {"frame": None}

    def on_capture(*cb_args, **cb_kwargs):
        # ByteCapture callback signature: (buffer, buffer_size, width, height, byte_format)
        if len(cb_args) >= 5:
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
                "[capture] Warning: frame capture callback returned no decodable image."
            )
        return False

    recorder["last_frame"] = frame
    if recorder["writer"] is not None:
        if recorder["deferred"]:
            recorder["frames"].append(frame)
        else:
            recorder["writer"].append_data(frame)
    recorder["frames_written"] += 1
    return True


def close_recorder(recorder) -> None:
    """Encode any deferred frames and close the video writer."""
    if recorder is None:
        return
    if recorder["writer"] is not None:
        _encode_deferred_frames(recorder)
        recorder["writer"].close()



# ── Side-by-side (comparison) capture ────────────────────────────────────────


def setup_sidebyside(total_frames, output_path, fps, sim_width, sim_height,
                     h5_path, h5_camera, *, dataset: str):
    """Set up side-by-side capture: Isaac Sim viewport (left) + H5 original (right).

    Returns:
        (recorder_dict, output_path) or (None, None) if H5 camera data is missing.
    """
    from .h5_loader import open_h5_images

    h5_file, h5_dataset = open_h5_images(h5_path, dataset=dataset, camera=h5_camera)
    if h5_file is None:
        print(f"[sidebyside] Cannot set up comparison: camera '{h5_camera}' not in {h5_path}")
        return None, None

    h5_h, h5_w = h5_dataset.shape[1], h5_dataset.shape[2]
    target_h = max(sim_height, h5_h)

    # Compute scaled widths preserving aspect ratio.
    sim_w_scaled = round(sim_width * target_h / sim_height) if sim_height != target_h else sim_width
    h5_w_scaled = round(h5_w * target_h / h5_h) if h5_h != target_h else h5_w
    out_w = sim_w_scaled + h5_w_scaled
    # libx264 requires even dimensions.
    out_w += out_w % 2
    target_h += target_h % 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=max(1, int(round(fps))), codec="libx264")
    print(f"[sidebyside] Recording comparison to {output_path} "
          f"({out_w}x{target_h}, {total_frames} frames target)")

    return {
        "writer": writer,
        "output_path": output_path,
        "h5_file": h5_file,
        "h5_dataset": h5_dataset,
        "target_h": target_h,
        "sim_w_scaled": sim_w_scaled,
        "h5_w_scaled": h5_w_scaled,
        "out_w": out_w,
        "frames": [],
        "frames_written": 0,
    }, output_path


def capture_sidebyside_frame(recorder, sim_frame, frame_idx: int) -> bool:
    """Compose one side-by-side frame and buffer it for deferred encoding."""
    if recorder is None or sim_frame is None:
        return False

    ds = recorder["h5_dataset"]
    idx = min(frame_idx, ds.shape[0] - 1)
    h5_frame = ds[idx]

    target_h = recorder["target_h"]
    left = _resize_to_height(sim_frame, target_h, recorder["sim_w_scaled"])
    right = _resize_to_height(h5_frame, target_h, recorder["h5_w_scaled"])

    combined = np.concatenate([left, right], axis=1)
    # Ensure width matches expected (pad if rounding caused 1px difference).
    if combined.shape[1] < recorder["out_w"]:
        pad = np.zeros((target_h, recorder["out_w"] - combined.shape[1], 3), dtype=np.uint8)
        combined = np.concatenate([combined, pad], axis=1)
    elif combined.shape[1] > recorder["out_w"]:
        combined = combined[:, :recorder["out_w"], :]

    recorder["frames"].append(combined)
    recorder["frames_written"] += 1
    return True


def close_sidebyside(recorder) -> None:
    """Encode deferred frames, close the writer and H5 file handle."""
    if recorder is None:
        return
    _encode_deferred_frames(recorder)
    recorder["writer"].close()
    recorder["h5_file"].close()


# ── Overlay (alpha-blended sim + H5) capture ─────────────────────────────────


def setup_overlay(
    total_frames, output_path, fps,
    sim_width, sim_height, h5_path, h5_camera, alpha: float,
    *, dataset: str,
):
    """Set up overlay capture: alpha-blend Isaac Sim viewport with H5 frame.

    The output is written at simulation resolution; the H5 frame is resized to
    match before blending. ``alpha`` is the sim's opacity in [0, 1]:
    ``alpha=1`` keeps the sim only, ``alpha=0`` keeps the H5 frame only.

    Returns:
        (recorder_dict, output_path) or (None, None) if H5 camera data is missing.
    """
    from .h5_loader import open_h5_images

    h5_file, h5_dataset = open_h5_images(h5_path, dataset=dataset, camera=h5_camera)
    if h5_file is None:
        print(f"[overlay] Cannot set up overlay: camera '{h5_camera}' not in {h5_path}")
        return None, None

    out_w = sim_width + sim_width % 2
    out_h = sim_height + sim_height % 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=max(1, int(round(fps))), codec="libx264")
    print(f"[overlay] Recording overlay to {output_path} "
          f"(alpha={alpha:.2f}, {out_w}x{out_h}, {total_frames} frames target)")

    return {
        "writer": writer,
        "output_path": output_path,
        "h5_file": h5_file,
        "h5_dataset": h5_dataset,
        "alpha": float(alpha),
        "out_w": out_w,
        "out_h": out_h,
        "frames": [],
        "frames_written": 0,
    }, output_path


def capture_overlay_frame(recorder, sim_frame, frame_idx: int) -> bool:
    """Blend ``sim_frame`` with the matching H5 frame and buffer for encoding."""
    if recorder is None or sim_frame is None:
        return False

    ds = recorder["h5_dataset"]
    idx = min(frame_idx, ds.shape[0] - 1)
    h5_frame = ds[idx]

    out_h, out_w = recorder["out_h"], recorder["out_w"]
    sim_r = _resize_to_height(sim_frame, out_h, out_w)
    h5_r = _resize_to_height(h5_frame, out_h, out_w)

    alpha = recorder["alpha"]
    blended = (
        alpha * sim_r.astype(np.float32) + (1.0 - alpha) * h5_r.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    recorder["frames"].append(blended)
    recorder["frames_written"] += 1
    return True


def close_overlay(recorder) -> None:
    """Encode deferred frames, close the writer and H5 file handle."""
    if recorder is None:
        return
    _encode_deferred_frames(recorder)
    recorder["writer"].close()
    recorder["h5_file"].close()


# ── Private helpers ───────────────────────────────────────────────────────────


def _encode_deferred_frames(recorder) -> int:
    """Encode all buffered frames to the video writer. Returns frame count."""
    frames = recorder.get("frames")
    if not frames:
        return 0
    n = len(frames)
    print(f"[capture] Encoding {n} buffered frames...")
    for i, frame in enumerate(frames):
        recorder["writer"].append_data(frame)
        if (i + 1) % 200 == 0:
            print(f"  [capture] encoded {i + 1}/{n}")
    recorder["frames"] = []
    print(f"[capture] Encoding complete ({n} frames).")
    return n


def _resize_to_height(frame, target_height: int, target_width: int = 0):
    """Resize a frame to target_height (and optionally target_width)."""
    h, w = frame.shape[:2]
    if h == target_height and (target_width == 0 or w == target_width):
        return frame
    from scipy.ndimage import zoom
    if target_width == 0:
        scale = target_height / h
        target_width = round(w * scale)
    sy = target_height / h
    sx = target_width / w
    return zoom(frame, (sy, sx, 1), order=1).astype(np.uint8)



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
