"""Extract camera images from an H5 dataset file to MP4 video.

No Isaac Sim or GPU required — this is a pure H5-read + imageio-encode pipeline.

Usage:
    python dataset_replay/scripts/record_h5.py
    python dataset_replay/scripts/record_h5.py --mode single --h5-camera aria
    python dataset_replay/scripts/record_h5.py --h5-camera oakd --fps 30
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root: `python dataset_replay/scripts/record_h5.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.constants import H5_IMAGE_PATHS, H5_DEFAULT_CAMERA, OUTPUT_DIR
from utils.h5_loader import get_available_cameras

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Extract camera images from H5 dataset to MP4 video",
)
parser.add_argument(
    "--mode", type=str, default="dual", choices=["single", "dual"],
    help="Which default H5 file to use (default: dual)",
)
parser.add_argument(
    "--h5-camera", type=str, default=H5_DEFAULT_CAMERA,
    choices=list(H5_IMAGE_PATHS.keys()),
    help=f"Camera to extract (default: {H5_DEFAULT_CAMERA})",
)
parser.add_argument(
    "--fps", type=float, default=50.0,
    help="Output video frame rate (default: 50.0)",
)
parser.add_argument(
    "--h5-path", type=str, default=None,
    help="Path to a specific H5 file (overrides --mode)",
)
args = parser.parse_args()

# ── Resolve H5 path ─────────────────────────────────────────────────────────

if args.h5_path is not None:
    h5_path = Path(args.h5_path)
else:
    from utils.app import resolve_h5_path
    h5_path = resolve_h5_path(args.mode)

if not h5_path.exists():
    print(f"[error] H5 file not found: {h5_path}")
    sys.exit(1)

# ── Check available cameras ─────────────────────────────────────────────────

cameras = get_available_cameras(h5_path)
if not cameras:
    print(f"[error] No image datasets found in {h5_path}")
    sys.exit(1)

print(f"[h5-video] Available cameras in {h5_path.name}:")
for name, shape in cameras.items():
    n, h, w, c = shape
    print(f"  {name}: {n} frames, {w}x{h}")

if args.h5_camera not in cameras:
    print(f"\n[error] Camera '{args.h5_camera}' not found. "
          f"Available: {list(cameras.keys())}")
    sys.exit(1)

# ── Export ───────────────────────────────────────────────────────────────────

# Lazy-import so the script stays lightweight (no Isaac Sim deps).
import imageio.v2 as imageio
import h5py

ds_path = H5_IMAGE_PATHS[args.h5_camera]
output_name = f"{h5_path.stem}_{args.h5_camera}.mp4"
output_path = OUTPUT_DIR / output_name

with h5py.File(h5_path, "r") as f:
    ds = f[ds_path]
    n_frames, h, w = ds.shape[0], ds.shape[1], ds.shape[2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path), fps=max(1, int(round(args.fps))), codec="libx264",
    )

    print(f"\n[h5-video] Exporting {n_frames} frames ({w}x{h}) "
          f"@ {args.fps} fps to {output_path}")
    for i in range(n_frames):
        writer.append_data(ds[i])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_frames}")

    writer.close()

print(f"[h5-video] Done: {output_path}")
