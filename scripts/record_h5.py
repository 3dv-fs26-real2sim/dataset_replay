"""Extract camera images from an H5 dataset file to MP4 video.

No Isaac Sim or GPU required — pure H5 read + imageio encode.

Usage:
    # Camera name disambiguates rig + default fps:
    #   oakd_*  → maple, 10 Hz
    #   aria_*  → egoverse, 50 Hz
    python dataset_replay/scripts/record_h5.py --h5 data/maple/h5/<session>.h5
    python dataset_replay/scripts/record_h5.py --h5 data/egoverse/h5/<session>.h5

    # Explicit overrides:
    python dataset_replay/scripts/record_h5.py --h5 old.h5 \\
        --camera oakd_front_view --fps 30
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root: `python dataset_replay/scripts/record_h5.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import PROJECT_ROOT  # type: ignore[import-not-found]
from utils.constants import H5_DEFAULT_FPS
from utils.h5_loader import H5Reader, SCHEMA

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Camera-name prefix → dataset/rig. Drives default --fps so the MP4 plays back
# at the same rate as the source recording without the user passing --fps.
_CAMERA_PREFIX_TO_DATASET = {
    "oakd_":  "maple",
    "aria_":  "egoverse",
}


def _infer_dataset(camera_name: str) -> str:
    """Map a camera name (e.g. ``oakd_front_view``) to its rig key.

    Raises ``ValueError`` if the camera name doesn't match any known prefix.
    """
    for prefix, dataset in _CAMERA_PREFIX_TO_DATASET.items():
        if camera_name.startswith(prefix):
            return dataset
    raise ValueError(
        f"unknown camera name prefix in {camera_name!r}; "
        f"expected one of {list(_CAMERA_PREFIX_TO_DATASET)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, required=True,
                        help="Path to the H5 file")
    parser.add_argument("--camera", type=str, default="aria_rgb_cam",
                        help="Camera name in H5. Prefix determines default "
                             "fps and H5 schema dispatch (oakd_* → maple, "
                             "aria_* → egoverse). Default: aria_rgb_cam.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output video frame rate. Default: per-rig "
                             "value picked from --camera (see "
                             "H5_DEFAULT_FPS in utils/constants.py).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output MP4 path (default: outputs/<stem>_<camera>.mp4)")
    args = parser.parse_args()

    if not args.h5.exists():
        print(f"[error] H5 file not found: {args.h5}")
        return 1

    try:
        dataset = _infer_dataset(args.camera)
    except ValueError as e:
        print(f"[error] {e}")
        return 1

    fps = args.fps if args.fps is not None else H5_DEFAULT_FPS[dataset]

    output = args.output or OUTPUT_DIR / f"{args.h5.stem}_{args.camera}.mp4"

    # Lazy import to keep the script lightweight.
    import imageio.v2 as imageio

    with H5Reader(args.h5, dataset=dataset, camera=args.camera) as h5:
        ds = h5.image_dataset()
        if ds is None:
            print(f"[error] Camera {args.camera!r} not present in {args.h5.name}")
            return 1
        n, h, w = ds.shape[0], ds.shape[1], ds.shape[2]

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output), fps=max(1, int(round(fps))), codec="libx264",
        )
        print(f"[h5-video] {args.h5.name} ({dataset}) → {output}  "
              f"({n} frames, {w}x{h} @ {fps} fps)")
        for i in range(n):
            writer.append_data(ds[i])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{n}")
        writer.close()

    print(f"[h5-video] Done: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
