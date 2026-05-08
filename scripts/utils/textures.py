"""Image-texture binding helpers for procedurally-built UsdGeom.Mesh prims.

Uses ``UsdPreviewSurface`` (cross-renderer, file-texture-based) so the
material renders consistently in Isaac Sim's RTX viewport without an MDL
dependency. If RTX rendering looks flat, swap the shader id for an
``OmniPBR.mdl``-backed shader; the rest of the binding flow is identical.

Depends on ``pxr`` — must be imported after ``SimulationApp`` is created.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


# ─────────────────────────────────────────────────────────────────────────────
def bind_image_texture(
    stage: Usd.Stage,
    mesh_prim_path: str,
    texture_path: Path | str,
    *,
    material_root: str = "/World/Looks",
    material_name: str | None = None,
) -> str:
    """Bind a ``UsdPreviewSurface`` material with a file-backed diffuse texture
    to the mesh at ``mesh_prim_path``. Idempotent: re-uses an existing material
    of the same name if already defined under ``material_root``.

    UVs must already be authored on the mesh as a ``primvars:st`` Float2Array
    (use :func:`set_box_planar_uvs` or :func:`set_quad_unit_uvs` before calling
    this). Returns the material prim path.
    """
    mat_name = material_name or Path(texture_path).stem + "_mat"
    mat_path = Sdf.Path(f"{material_root}/{mat_name}")

    UsdGeom.Scope.Define(stage, material_root)
    material = UsdShade.Material.Define(stage, mat_path)

    # ── Surface shader ─────────────────────────────────────────────────────
    surface = UsdShade.Shader.Define(stage, mat_path.AppendChild("surface"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    surface.CreateInput("roughness",          Sdf.ValueTypeNames.Float).Set(0.7)
    surface.CreateInput("metallic",           Sdf.ValueTypeNames.Float).Set(0.0)

    # ── UV reader ──────────────────────────────────────────────────────────
    uv_reader = UsdShade.Shader.Define(stage, mat_path.AppendChild("uvReader"))
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # ── Diffuse texture ────────────────────────────────────────────────────
    tex = UsdShade.Shader.Define(stage, mat_path.AppendChild("diffuseTex"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
    # Repeat-wrap so UVs > 1.0 tile the texture.
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)

    # ── Hook up surface output ─────────────────────────────────────────────
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))

    # ── Bind material to the mesh ──────────────────────────────────────────
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"Cannot bind texture: prim {mesh_prim_path} missing")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    return str(mat_path)


# ─────────────────────────────────────────────────────────────────────────────
def set_box_planar_uvs(
    mesh: UsdGeom.Mesh,
    *,
    extent_xyz: tuple[float, float, float],
    uv_repeat:  tuple[float, float] = (1.0, 1.0),
) -> None:
    """Author per-face planar UVs on a 6-quad-face cuboid mesh.

    Each face is projected onto its own tangent plane, scaled so the texture
    repeats ``uv_repeat`` times across the face. The mesh must follow the
    8-points / 6-quad convention used by :func:`define_box_mesh`.

    UVs are written as ``primvars:st`` with ``faceVarying`` interpolation
    (24 entries, one per face-vertex slot).
    """
    Lx, Ly, Lz = extent_xyz
    repeat_u, repeat_v = uv_repeat

    points = mesh.GetPointsAttr().Get()
    fv_indices = mesh.GetFaceVertexIndicesAttr().Get()
    fv_counts  = mesh.GetFaceVertexCountsAttr().Get()
    if list(fv_counts) != [4] * 6:
        raise RuntimeError("set_box_planar_uvs requires a 6-quad-face mesh")

    # Texture scale per metre along each axis.
    sx = repeat_u / Lx
    sy = repeat_u / Ly
    sz = repeat_v / Lz

    uvs: list[tuple[float, float]] = []
    for face_i in range(6):
        face_vert_idx = fv_indices[face_i * 4:(face_i + 1) * 4]
        face_pts = np.array([points[i] for i in face_vert_idx], dtype=float)
        # Determine dominant axis: the axis with smallest variance is the normal.
        var = face_pts.var(axis=0)
        normal_axis = int(np.argmin(var))
        for p in face_pts:
            if   normal_axis == 0:  # ±X face → UV = (Y, Z)
                uvs.append((p[1] * sy, p[2] * sz))
            elif normal_axis == 1:  # ±Y face → UV = (X, Z)
                uvs.append((p[0] * sx, p[2] * sz))
            else:                   # ±Z face → UV = (X, Y)
                uvs.append((p[0] * sx, p[1] * sy))

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set([Gf.Vec2f(*uv) for uv in uvs])


def set_quad_unit_uvs(mesh: UsdGeom.Mesh) -> None:
    """Author UVs for a single-quad mesh: one tile of the texture across the
    quad. Vertex order is assumed to be (TL, TR, BR, BL) of the quad in its
    local frame."""
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set([Gf.Vec2f(0.0, 1.0),       # TL
            Gf.Vec2f(1.0, 1.0),       # TR
            Gf.Vec2f(1.0, 0.0),       # BR
            Gf.Vec2f(0.0, 0.0)])      # BL


# ─────────────────────────────────────────────────────────────────────────────
def define_box_mesh(
    stage: Usd.Stage,
    prim_path: str,
    *,
    size_xyz: tuple[float, float, float],
    centre_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    display_color: tuple[float, float, float] = (0.7, 0.55, 0.4),
) -> UsdGeom.Mesh:
    """Create a 6-quad-face cuboid mesh centred at ``centre_xyz`` with the
    given size. Returns the ``UsdGeom.Mesh``. Compatible with
    :func:`set_box_planar_uvs` and :func:`bind_image_texture`.

    Display color is a fallback when no material is bound.
    """
    Lx, Ly, Lz = size_xyz
    cx, cy, cz = centre_xyz
    hx, hy, hz = Lx / 2, Ly / 2, Lz / 2

    points = [
        (cx - hx, cy - hy, cz - hz),  # 0  bottom corners
        (cx + hx, cy - hy, cz - hz),  # 1
        (cx + hx, cy + hy, cz - hz),  # 2
        (cx - hx, cy + hy, cz - hz),  # 3
        (cx - hx, cy - hy, cz + hz),  # 4  top corners
        (cx + hx, cy - hy, cz + hz),  # 5
        (cx + hx, cy + hy, cz + hz),  # 6
        (cx - hx, cy + hy, cz + hz),  # 7
    ]
    # Six quads, each CCW from outside.
    face_vertex_indices = [
        4, 5, 6, 7,   # +Z (top)
        0, 3, 2, 1,   # -Z (bottom)
        1, 2, 6, 5,   # +X
        0, 4, 7, 3,   # -X
        2, 3, 7, 6,   # +Y
        0, 1, 5, 4,   # -Y
    ]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateExtentAttr([
        Gf.Vec3f(cx - hx, cy - hy, cz - hz),
        Gf.Vec3f(cx + hx, cy + hy, cz + hz),
    ])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
    return mesh


def define_quad_mesh(
    stage: Usd.Stage,
    prim_path: str,
    *,
    size_xy: tuple[float, float],
    normal_axis: str = "+Z",
) -> UsdGeom.Mesh:
    """Create a single-quad ``UsdGeom.Mesh`` of ``size_xy`` lying in the plane
    perpendicular to ``normal_axis`` (only ``"+Z"`` is implemented for the
    AprilTag use case). Vertex order is (TL, TR, BR, BL) so that
    :func:`set_quad_unit_uvs` works directly.
    """
    if normal_axis != "+Z":
        raise NotImplementedError(f"normal_axis={normal_axis!r} not supported")

    sx, sy = size_xy
    hx, hy = sx / 2, sy / 2
    points = [
        (-hx, +hy, 0.0),   # TL  (in tag-local frame, +Y is "up")
        (+hx, +hy, 0.0),   # TR
        (+hx, -hy, 0.0),   # BR
        (-hx, -hy, 0.0),   # BL
    ]
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, 0.0), Gf.Vec3f(hx, hy, 0.0)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    return mesh
