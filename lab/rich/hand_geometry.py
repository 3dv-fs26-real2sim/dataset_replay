"""Load the right OrcaHand visual meshes, run zero-config forward kinematics,
and split them into skin pads vs structure.

Pure trimesh + yourdfpy (no Isaac Sim). Link names are ``right_*`` so they map
directly onto the per-pad contact data. Meshes resolve via the URDF's relative
``../meshes/orcahand/...`` paths from ``assets/urdf/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

_REPO = Path(__file__).resolve().parents[2]   # dataset_replay/
_URDF = _REPO / "assets" / "urdf" / "orcahand_right_extended.urdf"

SKIN_LABEL = {
    "right_index_ip": "index tip", "right_middle_ip": "middle tip", "right_ring_ip": "ring tip",
    "right_pinky_ip": "pinky tip", "right_thumb_dp": "thumb tip",
    "right_index_pp": "index mid", "right_middle_pp": "middle mid", "right_ring_pp": "ring mid",
    "right_pinky_pp": "pinky mid", "right_thumb_ip": "thumb mid", "right_palm": "palm",
}


def load_hand(cfg=None):
    """Return (skins, structure): lists of {name, link, mesh} (trimesh, world frame)."""
    urdf_dir = _URDF.parent
    robot = yourdfpy.URDF.load(str(_URDF), build_scene_graph=True, load_meshes=False,
                               filename_handler=lambda f: f)
    robot.update_cfg(np.zeros(len(robot.actuated_joints)) if cfg is None else cfg)
    skins, structure = [], []
    for link in robot.robot.links:
        if "tower" in link.name:
            continue
        T_link = robot.get_transform(link.name)
        for v in link.visuals:
            g = getattr(v, "geometry", None)
            if g is None or g.mesh is None:
                continue
            path = (urdf_dir / g.mesh.filename).resolve()
            if not path.exists():
                continue
            mesh = trimesh.load(str(path), process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            mesh.apply_transform(T_link @ (v.origin if v.origin is not None else np.eye(4)))
            rec = {"name": v.name or link.name, "link": link.name, "mesh": mesh}
            (skins if "skin" in (g.mesh.filename or "") else structure).append(rec)
    return skins, structure
