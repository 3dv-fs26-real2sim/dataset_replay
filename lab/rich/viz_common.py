"""CPU-only montage + modality-colorization helpers (numpy + matplotlib colormaps)."""
from __future__ import annotations

import numpy as np
import matplotlib.cm as cm


def colorize_depth(depth: np.ndarray, cmap: str = "turbo",
                   dmin: float | None = None, dmax: float | None = None) -> np.ndarray:
    """float depth (H,W, meters; inf/0 = no hit) -> (H,W,3) uint8 via a colormap.

    Range defaults to the 2nd/98th percentile of valid depths so the scene spans
    the full colormap regardless of background.
    """
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 1e-6)
    if not valid.any():
        return np.zeros((*d.shape, 3), np.uint8)
    dmin = float(np.percentile(d[valid], 2)) if dmin is None else dmin
    dmax = float(np.percentile(d[valid], 98)) if dmax is None else dmax
    n = np.clip((d - dmin) / max(1e-6, dmax - dmin), 0, 1)
    rgb = (cm.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 30
    return rgb


def colorize_normals(normals: np.ndarray) -> np.ndarray:
    """normals (H,W,3 or 4) in [-1,1] -> (H,W,3) uint8 ([-1,1]->[0,255])."""
    n = np.asarray(normals, np.float32)[..., :3]
    return (((n + 1.0) * 0.5).clip(0, 1) * 255).astype(np.uint8)


def fit_into(frame: np.ndarray, h: int, w: int, bg: int = 20) -> np.ndarray:
    """Scale (H,W,3) into an h*w cell preserving aspect ratio, letterbox-padded."""
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    frame = frame[..., :3].astype(np.uint8)
    fh, fw = frame.shape[:2]
    s = min(h / fh, w / fw)
    nh, nw = max(1, int(round(fh * s))), max(1, int(round(fw * s)))
    yi = (np.arange(nh) / s).astype(np.int32).clip(0, fh - 1)
    xi = (np.arange(nw) / s).astype(np.int32).clip(0, fw - 1)
    cell = np.full((h, w, 3), bg, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = frame[yi[:, None], xi[None, :]]
    return cell


def compose_grid(tiles: list[np.ndarray], rows: int, cols: int,
                 labels: list[str] | None = None, tile_h: int = 360, tile_w: int = 480,
                 gap: int = 4) -> np.ndarray:
    """Tile up to rows*cols frames into a labeled grid (aspect-preserving cells)."""
    cells = []
    for i in range(rows * cols):
        if i < len(tiles) and tiles[i] is not None:
            c = fit_into(tiles[i], tile_h, tile_w)
            if labels and i < len(labels) and labels[i]:
                c = label_tile(c, labels[i])
        else:
            c = np.full((tile_h, tile_w, 3), 20, np.uint8)
        cells.append(c)
    sep_v = np.full((tile_h, gap, 3), 30, np.uint8)
    rows_img = []
    for r in range(rows):
        row = []
        for cc in range(cols):
            row.append(cells[r * cols + cc])
            if cc < cols - 1:
                row.append(sep_v)
        rows_img.append(np.concatenate(row, axis=1))
    sep_h = np.full((gap, rows_img[0].shape[1], 3), 30, np.uint8)
    out = []
    for r, rim in enumerate(rows_img):
        out.append(rim)
        if r < rows - 1:
            out.append(sep_h)
    out = np.concatenate(out, axis=0)
    ph, pw = out.shape[0] % 2, out.shape[1] % 2
    if ph or pw:
        out = np.pad(out, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return out


def label_tile(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a small caption into the top-left via a tiny built-in 5x7 bitmap font."""
    out = frame.copy()
    _draw_text(out, text, 6, 6, scale=2)
    return out


# 5x7 bitmap font: uppercase letters, digits, and a few symbols.
_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
    "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
}


def _draw_text(img: np.ndarray, text: str, x0: int, y0: int, scale: int = 2,
               color=(255, 255, 0)) -> None:
    """Draw `text` (uppercased) into img at (x0,y0) with a dark backing box."""
    text = text.upper()
    gw = sum((len(_FONT.get(ch, _FONT[" "])[0]) + 1) for ch in text) * scale
    gh = 7 * scale
    box = img[max(0, y0 - 2):y0 + gh + 2, max(0, x0 - 2):x0 + gw + 2]
    box[:] = (0.35 * box).astype(np.uint8)
    cx = x0
    for ch in text:
        glyph = _FONT.get(ch, _FONT[" "])
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    img[y0 + r * scale:y0 + (r + 1) * scale, cx + c * scale:cx + (c + 1) * scale] = color
        cx += (len(glyph[0]) + 1) * scale
