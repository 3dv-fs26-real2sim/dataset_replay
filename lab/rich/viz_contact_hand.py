"""Animated contact-force hand render.

``render_contact_hand`` draws the OrcaHand (light grey) with its 11 skin pads
tinted on a monochrome-red ramp driven per-frame by the measured per-pad contact
force, and writes an mp4. Offline (pyrender + EGL, no Isaac Sim).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import pyrender
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_geometry import load_hand

REDS = LinearSegmentedColormap.from_list("reds", ["#ffe2da", "#ff8a6b", "#e8331b", "#7a0a00"])
STRUCT_RGB = (0.84, 0.84, 0.87)


def _look_at(eye, center, up):
    z = eye - center; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _palm_normal(skins):
    m = next(s for s in skins if s["link"] == "right_palm")["mesh"]
    n = (m.face_normals * m.area_faces[:, None]).sum(0)
    return n / np.linalg.norm(n)


def _mat(rgb, rough=0.5):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*rgb, 1.0], metallicFactor=0.0, roughnessFactor=rough, alphaMode="OPAQUE")


def render_contact_hand(contacts_npz, out_path, fmax=None, gamma=0.5, fps=50, w=720, h=840):
    """Render the per-pad contact force on the hand to ``out_path`` (mp4).

    ``gamma`` is a perceptual ramp on (force/fmax): 0.5 (sqrt) expands the low end
    so the gentle sustained grasp stays visible while the closing transient saturates.
    """
    d = np.load(contacts_npz, allow_pickle=True)
    names = [str(b) for b in d["body_names"]]
    fmag = d["force_mag"]
    n_frames = len(fmag)
    if fmax is None:
        fmax = float(np.percentile(fmag[fmag > 1e-6], 98)) if (fmag > 1e-6).any() else 1.0
    by_link = {nm: fmag[:, i] for i, nm in enumerate(names)}

    skins, structure = load_hand()
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.16, 0.16, 0.18])
    pts = []
    for s in structure:
        scene.add(pyrender.Mesh.from_trimesh(s["mesh"], material=_mat(STRUCT_RGB, 0.65), smooth=False))
        pts.append(s["mesh"].vertices)
    skin_mats = {}
    for s in skins:
        pm = pyrender.Mesh.from_trimesh(s["mesh"], material=_mat(REDS(0.0)[:3]), smooth=True)
        scene.add(pm)
        skin_mats[s["link"]] = pm.primitives[0].material   # from_trimesh copies the material
        pts.append(s["mesh"].vertices)

    pts = np.vstack(pts)
    center = (pts.min(0) + pts.max(0)) / 2
    radius = np.linalg.norm(pts.max(0) - pts.min(0)) / 2
    view_d = -_palm_normal(skins)
    up = np.array([0, 0, 1.0])
    cam_pose = _look_at(center + view_d * radius * 4.0, center, up)
    scene.add(pyrender.OrthographicCamera(xmag=radius * 1.05 * w / h, ymag=radius * 1.05), pose=cam_pose)
    right, cam_up = cam_pose[:3, 0], cam_pose[:3, 1]
    for dirv, inten, col in [(view_d + 0.85 * cam_up - 0.75 * right, 4.2, (1, 1, 1)),
                             (view_d - 0.5 * cam_up + 0.8 * right, 1.1, (0.85, 0.88, 1.0)),
                             (-view_d + 1.1 * cam_up - 0.3 * right, 2.0, (1, 1, 1))]:
        dv = dirv / np.linalg.norm(dirv)
        scene.add(pyrender.DirectionalLight(color=list(col), intensity=inten),
                  pose=_look_at(center + dv * radius * 5, center, up))

    r = pyrender.OffscreenRenderer(w, h)
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for t in range(n_frames):
        for link, mat in skin_mats.items():
            v = float(np.clip(by_link.get(link, np.zeros(n_frames))[t] / fmax, 0.0, 1.0)) ** gamma
            mat.baseColorFactor = [*REDS(v)[:3], 1.0]
        color, _ = r.render(scene, flags=flags)
        rgb = color[..., :3].astype(np.float32)
        a = color[..., 3:4].astype(np.float32) / 255.0
        frame = (rgb * a + 255.0 * (1 - a)).clip(0, 255).astype(np.uint8)
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            frame = np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)), mode="edge")
        writer.append_data(frame)
    writer.close()
    r.delete()
    return out_path
