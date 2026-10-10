"""Export a build123d Shape to all supported formats. Each format is independent —
a single failing export never fails the job; the error is recorded instead."""

from __future__ import annotations

import math
from pathlib import Path

import trimesh
from build123d import (
    Axis,
    ExportDXF,
    ExportSVG,
    GeomType,
    Mesher,
    Pos,
    Shape,
    Unit,
    export_gltf,
    export_step,
    export_stl,
)

FORMATS_3D = ["step", "stl", "3mf", "glb"]
FORMATS_2D = ["dxf", "svg"]


def export_step_file(shape: Shape, path: Path) -> None:
    export_step(shape, str(path))


def export_stl_file(
    shape: Shape, path: Path, tolerance: float = 0.01, angular_tolerance: float = 0.1
) -> None:
    export_stl(shape, str(path), tolerance=tolerance, angular_tolerance=angular_tolerance)


def export_3mf_file(shape: Shape, path: Path, name: str = "part") -> None:
    m = Mesher()
    m.add_shape(shape, part_number=name)
    m.write(str(path))


def export_glb_file(shape: Shape, path: Path, linear_deflection: float = 0.05) -> None:
    export_gltf(shape, str(path), binary=True, linear_deflection=linear_deflection)


def is_prismatic_plate(shape: Shape) -> bool:
    """Heuristic: part is a flat extrusion (thickness along Z and a bottom face at
    z_min covering >= 90% of the XY bounding box)."""
    try:
        bb = shape.bounding_box()
        if min(bb.size.X, bb.size.Y, bb.size.Z) != bb.size.Z:
            return False  # thickness must be along Z
        xy_area = bb.size.X * bb.size.Y
        for f in shape.faces():
            if not f.is_planar:
                continue
            n = f.normal_at()
            if n is None or n.Z > -0.99:
                continue
            if abs(float(f.center().Z) - float(bb.min.Z)) < 0.01 and f.area >= 0.9 * xy_area:
                return True
        return False
    except Exception:
        return False


def _profile_for_2d(shape: Shape, flat_brep: Path | None) -> Shape | None:
    from build123d import import_brep

    if flat_brep is not None and flat_brep.exists():
        try:
            return import_brep(str(flat_brep))
        except Exception:
            pass
    if is_prismatic_plate(shape):
        bottoms = shape.faces().sort_by(Axis.Z)
        return bottoms[0]
    return None


def export_2d(
    shape: Shape, out_dir: Path, name: str, flat_brep: Path | None
) -> dict[str, str | None]:
    """Export DXF + SVG of the flat profile. Returns {fmt: filename or None}."""
    results: dict[str, str | None] = {"dxf": None, "svg": None}
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = _profile_for_2d(shape, flat_brep)
    if profile is None:
        return results

    # move profile onto the XY plane at z=0 (bottom face already is, flat_profile may not be)
    z0 = float(profile.bounding_box().min.Z)
    if abs(z0) > 1e-9:
        profile = Pos(0, 0, -z0) * profile

    try:
        d = ExportDXF(unit=Unit.MM)
        d.add_layer("outline")
        d.add_shape(profile)
        d.write(str(out_dir / f"{name}.dxf"))
        results["dxf"] = f"{name}.dxf"
    except Exception:
        pass
    try:
        s = ExportSVG(unit=Unit.MM)
        s.add_layer("outline")
        s.add_shape(profile)
        s.write(str(out_dir / f"{name}.svg"))
        results["svg"] = f"{name}.svg"
    except Exception:
        pass
    return results


def export_all_3d(
    shape: Shape,
    out_dir: Path,
    name: str = "model",
    *,
    stl_tolerance: float = 0.01,
    stl_angular: float = 0.1,
    glb_deflection: float = 0.05,
) -> dict[str, str | None]:
    """Export step/stl/3mf/glb. Returns {fmt: filename or None}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str | None] = {}
    exporters = {
        "step": lambda p: export_step_file(shape, p),
        "stl": lambda p: export_stl_file(shape, p, stl_tolerance, stl_angular),
        "3mf": lambda p: export_3mf_file(shape, p, name),
        "glb": lambda p: export_glb_file(shape, p, glb_deflection),
    }
    for fmt, fn in exporters.items():
        path = out_dir / f"{name}.{fmt}"
        try:
            fn(path)
            results[fmt] = f"{name}.{fmt}"
        except Exception:
            results[fmt] = None
    return results


def quick_stl(shape: Shape, out_dir: Path, name: str = "mesh") -> Path:
    """STL used for renders/mesh analysis (coarser than the export STL)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.stl"
    export_stl(shape, str(path), tolerance=0.05, angular_tolerance=0.2)
    return path


# --------------------------------------------------------------------------- #
# mesh-mode exports (trimesh-based; STEP via faceted solid)                     #
# --------------------------------------------------------------------------- #


def export_mesh_all(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    name: str = "model",
    *,
    try_step: bool = True,
) -> dict[str, str | None]:
    """Export STL/3MF/GLB directly from a trimesh; STEP via mesh_to_solid.

    Returns {fmt: filename or None}. STEP is faceted/tessellated when produced
    this way — callers should mark the artifact meta accordingly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str | None] = {"step": None, "stl": None, "3mf": None, "glb": None}
    for fmt in ("stl", "3mf", "glb"):
        path = out_dir / f"{name}.{fmt}"
        try:
            mesh.export(str(path), file_type=fmt)
            results[fmt] = f"{name}.{fmt}"
        except Exception:
            results[fmt] = None
    if try_step:
        try:
            from cadagent.cad.mesh_ops import mesh_to_solid

            solid = mesh_to_solid(mesh)
            step_path = out_dir / f"{name}.step"
            export_step(solid, str(step_path))
            results["step"] = f"{name}.step"
        except Exception:
            results["step"] = None
    return results


def face_is_cylindrical_small(face: object, max_radius: float) -> bool:
    try:
        return bool(face.geom_type == GeomType.CYLINDER and face.radius < max_radius)  # type: ignore[attr-defined]
    except Exception:
        return False


def radians(deg: float) -> float:
    return deg * math.pi / 180.0
