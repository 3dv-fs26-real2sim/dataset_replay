"""Semantic labeling + grouped-segmentation helpers (adapted from
dataset_replay/scripts/utils/segmentation.py).

Lets the MVR capture collapse the whole robot (panda + orcahand) into ONE
segmentation class instead of per-link instance ids. Tag the robot/object/table
subtrees, render Replicator ``semantic_segmentation`` (colorize=False → uint32 id
buffer + idToLabels), resolve class→ids once after warmup, then paint per class.

Isaac-Sim-dependent functions import omni/pxr/isaacsim lazily.
"""
from __future__ import annotations

import numpy as np

# dataset_replay's class palette (robot=green, desk=blue, duck=magenta).
CLASS_COLORS = {"robot": (40, 200, 70), "desk": (50, 130, 255), "duck": (230, 40, 200)}


def _add_label(prim, label: str) -> str:
    """Apply semantic ``class=label`` to a prim (Isaac 5.1 add_labels, else legacy)."""
    from isaacsim.core.utils import semantics as _sem
    if hasattr(_sem, "add_labels"):
        for call in (
            lambda: _sem.add_labels(prim, labels=[label], instance_name="class"),
            lambda: _sem.add_labels(prim, [label], "class"),
            lambda: _sem.add_labels(prim, [label]),
        ):
            try:
                call(); return "add_labels"
            except TypeError:
                continue
            except Exception:
                break
    _sem.add_update_semantics(prim, label)
    return "add_update_semantics"


def label_subtree(stage, root_path: str, label: str) -> int:
    """Tag ``root_path`` + every editable descendant Gprim with ``class=label``.

    Instance-proxy meshes (the flattened robot USD) are read-only and inherit the
    label from the root via UsdSemantics inheritance.
    """
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[semantics] WARN: prim not found at {root_path} (skipped)")
        return 0
    used = _add_label(root, label)
    n = 1
    for prim in Usd.PrimRange(root):
        if prim == root or prim.IsInstanceProxy():
            continue
        if prim.IsA(UsdGeom.Gprim):
            _add_label(prim, label); n += 1
    print(f"[semantics] labeled '{label}' subtree {root_path} ({n} prims, via {used})")
    return n


def seg_buffer(annot):
    """(seg uint32 (H,W), info dict) from a semantic_segmentation annotator."""
    data = annot.get_data()
    if isinstance(data, dict):
        return np.asarray(data.get("data")), data.get("info", {})
    return np.asarray(data), {}


def resolve_class_ids(info: dict, labels) -> dict:
    """{class: [semantic id,...]} from annotator info['idToLabels']."""
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    out = {lab: [] for lab in labels}
    for key, value in id_to_labels.items():
        raw = value.get("class", "") if isinstance(value, dict) else value
        names = {c.strip().lower() for c in str(raw).replace(",", " ").split()}
        for lab in labels:
            if lab.lower() in names:
                try:
                    out[lab].append(int(key))
                except (TypeError, ValueError):
                    pass
    return out


def build_id_color_map(class_ids: dict, class_colors: dict = None) -> dict:
    """{class:[ids]} + {class:(r,g,b)} → {id:(r,g,b)} for fast painting."""
    class_colors = class_colors or CLASS_COLORS
    id_color = {}
    for cls, ids in class_ids.items():
        color = class_colors.get(cls)
        if color is None:
            continue
        for sid in ids:
            id_color[int(sid)] = tuple(int(c) for c in color)
    return id_color


def colorize_seg_ids(seg: np.ndarray, id_color: dict, background=(20, 20, 20)) -> np.ndarray:
    """uint32 id buffer → (H,W,3) uint8: each class id painted its color."""
    out = np.empty((*seg.shape[:2], 3), dtype=np.uint8)
    out[:] = np.asarray(background, np.uint8)
    for sid, color in id_color.items():
        out[seg == sid] = np.asarray(color, np.uint8)
    return out
