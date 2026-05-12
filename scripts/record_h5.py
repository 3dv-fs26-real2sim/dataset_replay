"""Extract camera images from an H5 dataset file to MP4 video.

No Isaac Sim or GPU required — pure H5 read + imageio encode.

Usage:
    python dataset_replay/scripts/record_h5.py --h5 path/to/file.h5
    # If you have an older 50 Hz dataset (e.g. `*_50hz`), override:
    python dataset_replay/scripts/record_h5.py --h5 old.h5 --fps 50
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root: `python dataset_replay/scripts/record_h5.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import PROJECT_ROOT  # type: ignore[import-not-found]
from utils.constants import H5_DEFAULT_FPS
from utils.h5_loader import H5Reader

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True,
                        help="Path to the H5 file")
    parser.add_argument("--camera", type=str, default=H5Reader.DEFAULT_CAMERA,
                        help=f"Camera name in H5 (default: {H5Reader.DEFAULT_CAMERA})")
    parser.add_argument("--fps", type=float, default=H5_DEFAULT_FPS,
                        help=f"Output video frame rate (default: {H5_DEFAULT_FPS}, "
                             f"matching H5_DEFAULT_FPS in utils/constants.py)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output MP4 path (default: outputs/<stem>_<camera>.mp4)")
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"[error] H5 file not found: {args.h5}")
        return 1

    output = args.output or OUTPUT_DIR / f"{args.h5.stem}_{args.camera}.mp4"

    # Lazy import to keep the script lightweight.
    import imageio.v2 as imageio

    with H5Reader(args.h5, camera=args.camera) as h5:
        ds = h5.image_dataset()
        if ds is None:
            print(f"[error] Camera {args.camera!r} not present in {args.h5.name}")
            return 1
        n, h, w = ds.shape[0], ds.shape[1], ds.shape[2]

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output), fps=max(1, int(round(args.fps))), codec="libx264",
        )
        print(f"[h5-video] {args.h5.name} → {output}  ({n} frames, {w}x{h} @ {args.fps} fps)")
        for i in range(n):
            writer.append_data(ds[i])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{n}")
        writer.close()

    print(f"[h5-video] Done: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
