"""AprilTag plane spawning for camera-extrinsic calibration.

The plane carries a wood-table-relative pose computed from
``SceneConfig.apriltag_world_pose()``. The image texture is sourced from
``cfg.apriltag.image_path`` (user supplies a PNG, e.g. from
``pupil-labs/apriltag-imgs``).

Depends on ``pxr`` — must be imported after ``SimulationApp`` is created.
"""

from __future__ import annotations

import warnings

from pxr import Gf, UsdGeom

from .config import SceneConfig
from .textures import bind_image_texture, define_quad_mesh, set_quad_unit_uvs


def add_apriltag_plane(
    stage,
    cfg: SceneConfig,
    *,
    prim_path: str = "/World/AprilTag",
) -> str:
    """Spawn a textured square at ``cfg.apriltag_world_pose()``.

    Returns the prim path of the AprilTag mesh.
    """
    if not cfg.apriltag.image_path.exists():
        warnings.warn(
            f"AprilTag image not found at {cfg.apriltag.image_path}; "
            f"the tag will render as a blank quad. Drop a PNG (e.g. from "
            f"pupil-labs/apriltag-imgs) at this path."
        )

    edge = cfg.apriltag.edge_size

    # Wrap in an Xform so the world transform is on the parent and the mesh
    # itself stays in tag-local frame (clean for unit UVs).
    UsdGeom.Xform.Define(stage, prim_path)
    mesh = define_quad_mesh(stage, f"{prim_path}/quad", size_xy=(edge, edge))
    set_quad_unit_uvs(mesh)
    bind_image_texture(stage, str(mesh.GetPath()), cfg.apriltag.image_path,
                       material_name=f"apriltag_{cfg.apriltag.tag_id}_mat")

    # Apply the world pose to the parent Xform.
    T = cfg.apriltag_world_pose()
    parent = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(parent)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*T[:3, 3].tolist()))
    return prim_path
