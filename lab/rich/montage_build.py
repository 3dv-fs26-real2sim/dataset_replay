"""Tile the per-stream videos (+ the contact-force hand) into montages.

Reads the individual MVR stream videos and the contact-hand video, all time-synced
(same rollout, same fps), and writes:
  * mvr_views.mp4       2x3: the 5 viewpoints (aria / front / hero / side / top)
  * mvr_modalities.mp4  2x2: Aria POV as rgb / depth / normals / segmentation
  * master_montage.mp4  3x3: front | force | side / top | aria | hero / depth | normals | seg
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

from viz_common import compose_grid


def _build(out_path, panels, rows, cols, fps, tile_h, tile_w):
    """Tile the (video, label) panels frame-by-frame into one montage."""
    panels = [(p, lab) for p, lab in panels if Path(p).exists()]
    if not panels:
        print(f"[montage] skip {Path(out_path).name}: no inputs found")
        return
    readers = [imageio.get_reader(str(p)) for p, _ in panels]
    labels = [lab for _, lab in panels]
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    n = 0
    for frames in zip(*readers):                      # stops at the shortest stream
        writer.append_data(compose_grid(list(frames), rows, cols, labels, tile_h=tile_h, tile_w=tile_w))
        n += 1
    writer.close()
    for r in readers:
        r.close()
    print(f"[montage] wrote {out_path}  ({n} frames @ {fps}fps, {rows}x{cols})")


def build_montages(mvr_dir, hand_path, montage_dir, fps=50, tile_h=360, tile_w=480):
    """Build the three montages from the per-stream videos in ``mvr_dir`` + ``hand_path``."""
    m = Path(mvr_dir)
    out = Path(montage_dir); out.mkdir(parents=True, exist_ok=True)
    hand = Path(hand_path)

    _build(out / "mvr_views.mp4",
           [(m / "aria_rgb.mp4", "aria"), (m / "front_rgb.mp4", "front"),
            (m / "hero_rgb.mp4", "hero"), (m / "side_rgb.mp4", "side"),
            (m / "top_rgb.mp4", "top")],
           2, 3, fps, tile_h, tile_w)

    _build(out / "mvr_modalities.mp4",
           [(m / "aria_rgb.mp4", "aria rgb"), (m / "aria_depth.mp4", "depth"),
            (m / "aria_normals.mp4", "normals"), (m / "aria_seg.mp4", "segmentation")],
           2, 2, fps, tile_h, tile_w)

    # front | FORCE  | side
    # top   | ARIA   | hero
    # depth | normals| segmentation
    _build(out / "master_montage.mp4",
           [(m / "front_rgb.mp4", "front"), (hand, "contact force"), (m / "side_rgb.mp4", "side"),
            (m / "top_rgb.mp4", "top"), (m / "aria_rgb.mp4", "aria"), (m / "hero_rgb.mp4", "hero"),
            (m / "aria_depth.mp4", "depth"), (m / "aria_normals.mp4", "normals"),
            (m / "aria_seg.mp4", "segmentation")],
           3, 3, fps, tile_h, tile_w)
