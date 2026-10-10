"""Mesh processing for the mesh/hybrid generation modes.

Pure functions on trimesh.Trimesh — no DB, no queue. Runs inside the
cad-executor (or in-process via the CLI). The generator output is a mesh in an
arbitrary normalized space; this module cleans, orients, scales and converts
it into the agent's frame contract (mm, X=length, Y=width(+Y front), Z=up,
resting on XY, centred at origin).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import trimesh
from pydantic import BaseModel

from cadagent.cad.analysis import Analysis
from cadagent.config import get_settings
from cadagent.llm.schemas import ParamDef, PartSpec
from cadagent.logging import get_logger

log = get_logger(__name__)

AxisName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

_AXIS_VECTORS: dict[str, tuple[int, int, int]] = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}

UP_CHOICES: list[AxisName] = ["+z", "-z", "+y", "-y", "+x", "-x"]


# --------------------------------------------------------------------------- #
# post-processing parameters (LLM-free "params" for mesh versions)             #
# --------------------------------------------------------------------------- #


class MeshPostParams(BaseModel):
    """Deterministic post-processing knobs. Stored as version.params and
    re-runnable via the params job — same UX as CAD PARAMS."""

    scale: float = 1.0  # extra uniform multiplier on top of spec scaling
    up: AxisName | None = None  # generator axis that maps to +Z (None = keep/detected)
    front: AxisName | None = None  # generator axis that maps to +Y (must be ⊥ up)
    rotate_z_deg: float = 0.0  # extra yaw around +Z after orientation
    decimate_faces: int | None = None  # target triangle count (None = keep all)
    flatten_bottom_mm: float = 0.0  # cut + cap at this height above bbox min Z
    mirror_x: bool = False


def mesh_params_schema() -> list[ParamDef]:
    return [
        ParamDef(
            name="scale",
            value=1.0,
            unit="-",
            min=0.01,
            max=100.0,
            description="uniform scale multiplier (applied after spec scaling)",
        ),
        ParamDef(
            name="up",
            value="+z",
            unit="-",
            description="generator axis that becomes +Z: +x -x +y -y +z -z",
        ),
        ParamDef(
            name="front",
            value="+y",
            unit="-",
            description="generator axis that becomes +Y (front); must differ from up",
        ),
        ParamDef(
            name="rotate_z_deg",
            value=0.0,
            unit="deg",
            min=-180.0,
            max=180.0,
            description="yaw around +Z after orientation",
        ),
        ParamDef(
            name="decimate_faces",
            value=0,
            unit="faces",
            min=0,
            description="target triangle count (0 = keep generator resolution)",
        ),
        ParamDef(
            name="flatten_bottom_mm",
            value=0.0,
            unit="mm",
            min=0.0,
            max=20.0,
            description="cut and cap the mesh flat at this height above its lowest point",
        ),
        ParamDef(
            name="mirror_x",
            value=False,
            unit="-",
            description="mirror the part along X (left/right handedness)",
        ),
    ]


def merge_post_params(base: MeshPostParams, overrides: dict[str, Any]) -> MeshPostParams:
    """Apply user overrides (params job). Only known keys are honored."""
    data = base.model_dump()
    for k, v in overrides.items():
        if k not in data:
            continue
        if k in ("up", "front") and v in (None, ""):
            continue
        if k == "decimate_faces":
            v = int(v) if v else None
        data[k] = v
    if data.get("up") and data.get("front") and data["up"] == data["front"]:
        data["front"] = None
    if data.get("up") and data.get("front") and data["up"][-1] == data["front"][-1]:
        # same axis letter (x/y/z) on up and front is impossible to honour
        data["front"] = None
    return MeshPostParams(**data)


# --------------------------------------------------------------------------- #
# load / clean / decimate                                                      #
# --------------------------------------------------------------------------- #


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    """Load any mesh file; merges multi-body scenes into one Trimesh."""
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError("scene contains no triangle meshes")
        return cast(trimesh.Trimesh, trimesh.util.concatenate(parts))
    raise ValueError(f"unsupported mesh container: {type(loaded).__name__}")


def clean(mesh: trimesh.Trimesh, min_component_frac: float = 0.01) -> trimesh.Trimesh:
    """Drop floaters, fix winding/normals, fill holes — best effort."""
    try:
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            vols = np.array([max(abs(b.volume), 1e-12) for b in bodies])
            keep = vols >= vols.max() * min_component_frac
            bodies = [b for b, k in zip(bodies, keep, strict=True) if k] or [
                bodies[int(np.argmax(vols))]
            ]
            mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(bodies))
    except Exception:
        pass
    for fn in (
        mesh.merge_vertices,
        mesh.remove_unreferenced_vertices,
        mesh.remove_infinite_values,
        mesh.fix_normals,
        lambda: trimesh.repair.fill_holes(mesh),
    ):
        try:
            fn()
        except Exception:
            log.debug("mesh.clean.step_skipped", fn=getattr(fn, "__name__", "?"))
    return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric decimation to ~target_faces. No-op when already below."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import fast_simplification

        pts, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            target_reduction=1.0 - target_faces / len(mesh.faces),
        )
        return trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    except Exception:
        pass
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        log.warning("mesh.decimate.failed", exc_info=True)
        return mesh


# --------------------------------------------------------------------------- #
# orientation                                                                  #
# --------------------------------------------------------------------------- #


def _axis(axis: AxisName | None, default: AxisName) -> np.ndarray:
    return np.array(_AXIS_VECTORS[axis or default], dtype=float)


def orient(
    mesh: trimesh.Trimesh,
    up: AxisName | None,
    front: AxisName | None,
    rotate_z_deg: float = 0.0,
    mirror_x: bool = False,
) -> trimesh.Trimesh:
    """Rotate so that source `up` -> +Z and source `front` -> +Y, then yaw."""
    u = _axis(up, "+z")
    f = _axis(front, "+y")
    if np.allclose(np.abs(np.dot(u, f)), 1.0):
        # front parallel to up — pick any axis perpendicular to u
        for cand in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            if not np.allclose(np.abs(np.dot(u, cand)), 1.0):
                f = np.array(cand, dtype=float)
                break
    f = f - u * np.dot(u, f)  # orthogonalize
    f /= np.linalg.norm(f)
    n = np.cross(f, u)  # -> +X  (x = y x z in a right-handed frame)
    basis = np.stack([n, f, u], axis=1)  # columns: src axes mapped to +X,+Y,+Z
    rot = basis.T
    m4 = np.eye(4)
    m4[:3, :3] = rot
    out = mesh.copy()
    out.apply_transform(m4)
    if mirror_x:
        mirror = np.eye(4)
        mirror[0, 0] = -1.0
        out.apply_transform(mirror)
        out.invert()
    if rotate_z_deg:
        rz = trimesh.transformations.rotation_matrix(math.radians(rotate_z_deg), [0, 0, 1])
        out.apply_transform(rz)
    return out


def default_front(up: AxisName, extents: np.ndarray) -> AxisName:
    """Pick the front axis for a given up: the horizontal axis with the
    LARGER extent becomes X (length); front is the other horizontal axis."""
    up_idx = "xyz".index(up[-1])
    horiz = [i for i in range(3) if i != up_idx]
    # the wider axis becomes length/X; front (width/Y) is the other one
    other = horiz[0] if horiz[1] == sorted(horiz, key=lambda i: -extents[i])[0] else horiz[1]
    return cast(AxisName, f"+{'xyz'[other]}")  # sign convention: camera default +Y


def suggest_up(mesh: trimesh.Trimesh) -> AxisName:
    """Stability heuristic: the up axis whose bottom side has the largest flat
    contact area (lowest 2% of vertices) wins; tie-break on lower CoM height."""
    best: tuple[float, AxisName] = (-1.0, "+z")
    for up in UP_CHOICES:
        m = orient(mesh, up, None)
        z = m.vertices[:, 2]
        span = max(float(np.ptp(z)), 1e-9)
        low = z <= z.min() + 0.02 * span
        score = float(low.mean())
        # prefer smaller height ratio? keep it simple: contact fraction + margin
        if score > best[0]:
            best = (score, up)
    return best[1]


def candidate_meshes(mesh: trimesh.Trimesh) -> dict[str, trimesh.Trimesh]:
    """{up: oriented mesh} for the 6 up candidates, with default front chosen."""
    out: dict[str, trimesh.Trimesh] = {}
    for up in UP_CHOICES:
        m = orient(mesh, up, None)  # up -> +Z
        front = default_front("+z", np.asarray(m.extents))  # in the rotated frame
        out[up] = rest_on_xy_center(orient(m, "+z", front))
    return out


# --------------------------------------------------------------------------- #
# scale / rest                                                                 #
# --------------------------------------------------------------------------- #


def scale_to_spec(
    mesh: trimesh.Trimesh, spec: PartSpec | None
) -> tuple[trimesh.Trimesh, float, list[str]]:
    """Uniform scaling so the most reliable stated spec dimension matches.

    Returns (mesh, factor, issues). Aspect mismatches on other stated axes are
    reported as issues — never non-uniformly scaled.
    """
    issues: list[str] = []
    if spec is None:
        return mesh, 1.0, issues

    stated: list[tuple[float, int, str]] = []
    for dim, idx, name in (
        (spec.length_x, 0, "length/X"),
        (spec.width_y, 1, "width/Y"),
        (spec.height_z, 2, "height/Z"),
    ):
        if dim is not None and dim.source == "stated" and dim.value > 0:
            stated.append((float(dim.value), idx, name))

    target_dims = stated or [
        (float(d.value), i, n)
        for d, i, n in (
            (spec.length_x, 0, "length/X"),
            (spec.width_y, 1, "width/Y"),
            (spec.height_z, 2, "height/Z"),
        )
        if d is not None and d.value > 0
    ]
    extents = np.asarray(mesh.extents, dtype=float)
    if not target_dims:
        # nothing usable — normalize largest extent to 100 mm
        factor = 100.0 / max(float(extents.max()), 1e-9)
        issues.append("no usable spec dimensions — normalized to 100 mm largest extent")
        out = mesh.copy()
        out.apply_scale(factor)
        return out, float(factor), issues

    # scale by the axis with the largest stated value (least relative error)
    ref_value, ref_idx, _ref_name = max(target_dims, key=lambda t: t[0])
    factor = ref_value / max(extents[ref_idx], 1e-9)
    out = mesh.copy()
    out.apply_scale(factor)
    new_ext = extents * factor
    for value, idx, name in stated:
        if idx == ref_idx:
            continue
        dev = abs(new_ext[idx] - value) / value
        if dev > 0.10:
            issues.append(
                f"aspect mismatch on {name}: spec {value:g} mm vs scaled {new_ext[idx]:.1f} mm "
                f"({dev * 100:.0f}% off; generator output is not dimension-true)"
            )
    return out, float(factor), issues


def rest_on_xy_center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate so bbox min.z == 0 and XY center is at the origin."""
    out = mesh.copy()
    b = out.bounds
    out.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
    return out


def flatten_bottom(mesh: trimesh.Trimesh, height_mm: float) -> trimesh.Trimesh:
    """Cut the mesh at z = min_z + height_mm with a plane, keeping the top,
    and cap the cut. Best-effort: on failure returns the input unchanged."""
    if height_mm <= 0:
        return mesh
    z = float(mesh.bounds[0][2]) + height_mm
    try:
        sliced = mesh.slice_plane(plane_origin=[0, 0, z], plane_normal=[0, 0, 1], cap=True)
        if isinstance(sliced, trimesh.Trimesh) and len(sliced.faces):
            return sliced
    except Exception:
        log.warning("mesh.flatten_bottom.failed", exc_info=True)
    return mesh


# --------------------------------------------------------------------------- #
# mesh -> solid (hybrid bridge)                                                #
# --------------------------------------------------------------------------- #


def mesh_to_solid(mesh: trimesh.Trimesh, *, max_faces: int | None = None) -> Any:
    """Convert a (watertight) trimesh into a build123d Solid via STL + sewing.

    Returns a build123d Solid/Part. Raises on failure — caller decides the
    fallback. Decimates first when the mesh exceeds `max_faces` (default
    MESH_MAX_FACES_FOR_CAD).
    """
    from build123d import Solid

    max_faces = max_faces or get_settings().mesh_max_faces_for_cad
    if len(mesh.faces) > max_faces:
        mesh = decimate(mesh, max_faces)
    if len(mesh.faces) > max_faces * 4:
        raise ValueError(f"mesh too dense for B-rep conversion ({len(mesh.faces)} faces)")

    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "mesh.stl"
        mesh.export(str(stl), file_type="stl")

        from build123d import Shape, Shell
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.StlAPI import StlAPI_Reader
        from OCP.TopoDS import TopoDS_Shape

        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        if not reader.Read(shape, str(stl)):
            raise ValueError("StlAPI_Reader failed")

        sewing = BRepBuilderAPI_Sewing(1e-4)
        sewing.Add(shape)
        sewing.Perform()
        sewed = sewing.SewedShape()
        fixed = ShapeFix_Shape(sewed).Shape()

        wrapped = Shape.cast(fixed)
        shells = wrapped.shells()
        if not shells:
            raise ValueError("no closed shell found in mesh")
        shell = max(shells, key=lambda sh: sh.area)
        if not isinstance(shell, Shell):
            shell = Shell(shell.wrapped)
        shell = shell.fix()

    return Solid(Solid._make_solid(shell))


# --------------------------------------------------------------------------- #
# analysis                                                                     #
# --------------------------------------------------------------------------- #


def mesh_analysis(mesh: trimesh.Trimesh) -> Analysis:
    """Analysis equivalent for a trimesh (B-Rep fields get mesh equivalents)."""
    b = np.asarray(mesh.bounds)
    size = b[1] - b[0]
    warnings: list[str] = []
    watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
    if not watertight:
        warnings.append("mesh is not watertight / winding inconsistent")
    try:
        com = [round(float(c), 4) for c in mesh.center_mass]
    except Exception:
        com = []
    try:
        volume = abs(float(mesh.volume))
    except Exception:
        volume = 0.0
    sits = abs(float(b[0][2])) < 0.01
    cx, cy = float(b[:, 0].mean()), float(b[:, 1].mean())
    centered = (
        abs(cx) < max(size[0] * 0.01, 0.5) and abs(cy) < max(size[1] * 0.01, 0.5)
        if max(size[0], size[1]) > 0
        else False
    )
    if not sits:
        warnings.append("part does not rest on the XY plane (min Z != 0)")
    if not centered:
        warnings.append("part is not centred at the origin in XY")
    return Analysis(
        bbox_min=[round(float(v), 4) for v in b[0]],
        bbox_max=[round(float(v), 4) for v in b[1]],
        bbox_size=[round(float(v), 4) for v in size],
        volume=round(volume, 3),
        surface_area=round(float(mesh.area), 3),
        center_of_mass=com,
        solid_count=1,
        is_valid=watertight,
        watertight=watertight,
        face_count=len(mesh.faces),
        edge_count=0,
        cylindrical_faces=[],
        sits_on_xy=sits,
        centered_xy=centered,
        warnings=warnings,
        source="mesh",
        triangle_count=len(mesh.faces),
    )


def flat_base_ratio(mesh: trimesh.Trimesh) -> float:
    """Fraction of total surface that is downward-facing and near min Z."""
    try:
        normals = mesh.face_normals
        centers = mesh.triangles_center
        z0 = float(mesh.bounds[0][2])
        mask = (normals[:, 2] < -0.9) & (centers[:, 2] < z0 + 0.3)
        return float(mesh.area_faces[mask].sum()) / max(float(mesh.area), 1e-9)
    except Exception:
        return 0.0
