"""Multi-viewpoint render helpers.

Free USD cameras + Replicator render products around the workspace, plus the
calibrated Aria camera (desk-refined extrinsic, the real demo POV). The Aria view
also carries depth / world-normals / robot-grouped semantic segmentation. Read the
annotators AFTER env.step()+uenv.render() so the render products have ticked.
"""
from __future__ import annotations

import numpy as np

from viz_common import colorize_depth, colorize_normals
from semantics import seg_buffer, resolve_class_ids, build_id_color_map, colorize_seg_ids, label_subtree

# Workspace center (duck pickup -> lift region) the free cams look at.
LOOK = (0.18, -0.21, 0.84)
# free views: (name, eye offset from LOOK, hfov_deg, modalities)
FREE_VIEWS = [
    ("front", (0.50, 0.00, 0.14), 55.0, ["rgb"]),
    ("side",  (0.02, 0.52, 0.16), 55.0, ["rgb"]),
    ("top",   (0.04, 0.02, 0.62), 55.0, ["rgb"]),
    ("hero",  (0.46, 0.42, 0.36), 55.0, ["rgb"]),
]
SEG_CLASSES = ["robot", "desk", "object"]
# Per-stream videos written by the capture (one .mp4 each).
STREAMS = ["aria_rgb", "front_rgb", "side_rgb", "top_rgb", "hero_rgb",
           "aria_depth", "aria_normals", "aria_seg"]


def lookat_T(eye, look):
    """World->cam transform (OpenCV/ros basis: cols = right, down, forward)."""
    pos = np.asarray(eye, float); tgt = np.asarray(look, float); up = np.array([0, 0, 1.0])
    fwd = tgt - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, pos
    return T


def read_annot(getter, dtype):
    """Replicator annotator get_data() -> ndarray (H,W[,C]) or None (buffer-aware)."""
    data = getter()
    if isinstance(data, dict):
        data = data.get("data", None)
    if data is None or not hasattr(data, "shape") or len(data.shape) < 2:
        return None
    try:
        arr = np.frombuffer(data, dtype=dtype).reshape(*data.shape)
    except Exception:
        arr = np.asarray(data)
        if arr.ndim < 2:
            return None
    return arr if arr.size else None


def setup_rich_camera(T_world_cam, fx, w, h, prim_path, modalities):
    """USD camera at world transform ``T_world_cam`` (cv basis) + annotators.

    Returns ({modality: annotator}, {T_world_cam, fx, w, h}).
    """
    import omni.usd
    from pxr import Gf, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, prim_path)
    pose_usd = T_world_cam @ np.diag([1.0, -1.0, -1.0, 1.0])   # cv -> USD (looks down -Z)
    xf = UsdGeom.Xformable(cam.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*pose_usd.T.flatten().tolist()))
    fmm = 24.0
    cam.GetFocalLengthAttr().Set(fmm)
    cam.GetHorizontalApertureAttr().Set(fmm * w / fx)
    cam.GetVerticalApertureAttr().Set(fmm * h / fx)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    import omni.replicator.core as rep
    rp = rep.create.render_product(prim_path, (w, h))
    name_map = {"rgb": "rgb", "depth": "distance_to_image_plane",
                "normals": "normals", "seg": "semantic_segmentation"}
    anns = {}
    for m in modalities:
        init = {"colorize": False} if m == "seg" else None
        a = rep.AnnotatorRegistry.get_annotator(name_map[m], init_params=init, device="cpu")
        a.attach([rp]); anns[m] = a
    return anns, {"T_world_cam": T_world_cam, "fx": fx, "w": w, "h": h}


def build_cameras(res=480, look=LOOK, dataset="egoverse"):
    """Create the dataset POV + 4 free cameras. Returns (cams, cam_params).

    The POV (stream key ``aria``) is the rig's real camera: EgoVerse → the Aria
    head-cam (``T_world_link0(mount) @ ARIA_EXTRINSICS_RIGHT``); MAPLE → the
    OAK-D at its nominal world lookat (``config_maple``).
    """
    from utils.config import select_config
    cfg = select_config(dataset)
    h = res; w = h * 4 // 3
    if dataset == "egoverse":
        from utils.constants import ARIA_EXTRINSICS_RIGHT, ARIA_INTRINSICS
        T_world_link0 = np.eye(4)
        T_world_link0[:3, 3] = cfg.robot.mount_xyz
        T_pov = T_world_link0 @ ARIA_EXTRINSICS_RIGHT      # world->cam (cv basis)
        pov_fx, pov_w, pov_h = ARIA_INTRINSICS["fx"], ARIA_INTRINSICS["width"], ARIA_INTRINSICS["height"]
    else:  # maple → OAK-D world-fixed nominal lookat
        oakd = cfg.camera
        T_pov = lookat_T(oakd.nominal_position, oakd.nominal_lookat)
        pov_fx, pov_w, pov_h = oakd.intrinsics["fx"], oakd.width, oakd.height
    specs = [("aria", T_pov, pov_fx, pov_w, pov_h, ["rgb", "depth", "normals", "seg"])]
    for name, off, hfov, mods in FREE_VIEWS:
        eye = tuple(np.array(look) + np.array(off))
        fx = w / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
        specs.append((name, lookat_T(eye, look), fx, w, h, mods))
    cams, params = {}, {}
    for name, T, fx, cw, ch, mods in specs:
        cams[name], params[name] = setup_rich_camera(T, fx, cw, ch, f"/World/MVR_{name}", mods)
    return cams, params


def label_scene(stage, ns="/World/envs/env_0"):
    """Group-label the scene for semantic segmentation (robot / desk / duck)."""
    label_subtree(stage, f"{ns}/Robot", "robot")
    label_subtree(stage, f"{ns}/Object", "object")
    label_subtree(stage, f"{ns}/Table", "desk")


class MVRReader:
    """Reads one frame per stream from the cameras (RGB per view; Aria depth /
    normals / grouped segmentation). The seg class->colour map resolves on the
    first frame that reports labels."""

    def __init__(self, cams, cam_params, seg_classes=SEG_CLASSES):
        self.cams = cams
        self.cam_params = cam_params
        self.seg_classes = seg_classes
        self.id_color = {}

    def read(self):
        out = {}
        for n in self.cams:
            a = read_annot(self.cams[n]["rgb"].get_data, np.uint8)
            ch, cw = self.cam_params[n]["h"], self.cam_params[n]["w"]
            out[f"{n}_rgb"] = (np.ascontiguousarray(a[..., :3], np.uint8)
                               if a is not None else np.zeros((ch, cw, 3), np.uint8))
        ah, aw = self.cam_params["aria"]["h"], self.cam_params["aria"]["w"]
        black = np.zeros((ah, aw, 3), np.uint8)
        da = read_annot(self.cams["aria"]["depth"].get_data, np.float32)
        out["aria_depth"] = colorize_depth(da) if da is not None else black
        no = read_annot(self.cams["aria"]["normals"].get_data, np.float32)
        out["aria_normals"] = colorize_normals(no) if no is not None else black
        seg, info = seg_buffer(self.cams["aria"]["seg"])
        if not self.id_color and isinstance(info, dict) and info:
            self.id_color = build_id_color_map(resolve_class_ids(info, self.seg_classes))
        out["aria_seg"] = (colorize_seg_ids(seg, self.id_color)
                           if (seg.ndim >= 2 and seg.size and self.id_color) else black)
        return out
