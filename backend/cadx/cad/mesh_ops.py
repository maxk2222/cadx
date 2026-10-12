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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import trimesh
from pydantic import BaseModel, Field

from cadx.cad.analysis import Analysis
from cadx.llm.schemas import ParamDef, PartSpec
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

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


class MeshOp(BaseModel):
    """One recorded mesh transformation — serialisable + replayable.

    `op` keys the OPS_REGISTRY; `args` are validated by the op's pydantic
    param model when applied (never trusted from the LLM).
    """

    op: str
    args: dict[str, Any] = Field(default_factory=dict)


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
    ops: list[MeshOp] = Field(default_factory=list)  # agent-applied transform sequence


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
    """Rebuild the solid so the -z face is a flat plane across the full footprint.

    Voxel column fill-down: every occupied XY column is extended down to the
    floor, then marching-cubes rebuilds a watertight solid. Unlike a plane cut
    this also fills under overhangs/wings whose sculpted underside would
    otherwise stay above the cap. `height_mm` then removes that much material
    from the floor (plain capped slice — safe now that every column is solid
    through the cut height). Best-effort: on failure returns the input."""
    if height_mm <= 0:
        return mesh
    out = mesh
    try:
        lo, hi = np.asarray(mesh.bounds[0], dtype=float), np.asarray(mesh.bounds[1], dtype=float)
        ext = hi - lo
        pitch = float(np.clip(min(ext[0], ext[1]) / 150.0, 0.3, 0.8))
        vg = mesh.voxelized(pitch)
        mat = np.asarray(vg.matrix)
        if mat.any():
            # fill-down: a voxel is solid if any voxel above it in its column is
            filled = np.maximum.accumulate(mat[:, :, ::-1], axis=2)[:, :, ::-1]
            rebuilt = trimesh.voxel.VoxelGrid(filled, transform=vg.transform).marching_cubes
            rebuilt.apply_transform(vg.transform)
            if len(rebuilt.faces):
                # voxel padding grows the envelope ~1 pitch on every side —
                # crop the XY silhouette back to the original bounds
                for axis in (0, 1):
                    for at, sign in ((lo[axis], 1.0), (hi[axis], -1.0)):
                        normal = np.zeros(3)
                        normal[axis] = sign
                        origin = np.zeros(3)
                        origin[axis] = at
                        cut = rebuilt.slice_plane(
                            plane_origin=origin, plane_normal=normal, cap=True
                        )
                        if isinstance(cut, trimesh.Trimesh) and len(cut.faces):
                            rebuilt = cut
                out = rebuilt
    except Exception:
        log.warning("mesh.flatten_bottom.voxel_fill.failed", exc_info=True)
        out = mesh
    if height_mm > 0:
        z = float(out.bounds[0][2]) + height_mm
        try:
            sliced = out.slice_plane(plane_origin=[0, 0, z], plane_normal=[0, 0, 1], cap=True)
            if isinstance(sliced, trimesh.Trimesh) and len(sliced.faces):
                out = sliced
        except Exception:
            log.warning("mesh.flatten_bottom.slice.failed", exc_info=True)
    return out


def smooth_preserve(
    mesh: trimesh.Trimesh,
    iterations: int = 10,
    *,
    pin_floor: bool = True,
    floor_tol: float = 0.05,
) -> trimesh.Trimesh:
    """Taubin smoothing that keeps the flat bottom and the XY envelope fixed.

    Vertices on the lowest Z layer (z <= zmin + floor_tol) are pinned so the
    manufacturing floor stays a plane; after smoothing the XY extents are
    rescaled back to the input bounds. Returns the input unchanged when the
    result is not watertight or degenerate.
    """
    out = mesh.copy()
    out.merge_vertices()
    pinned = None
    if pin_floor:
        pinned = np.flatnonzero(out.vertices[:, 2] <= out.bounds[0][2] + floor_tol)
    try:
        lap = trimesh.smoothing.laplacian_calculation(
            out, equal_weight=True, pinned_vertices=pinned
        )
        trimesh.smoothing.filter_taubin(
            out, lamb=0.5, nu=-0.53, iterations=iterations, laplacian_operator=lap
        )
    except Exception:
        log.warning("mesh.smooth_preserve.taubin.failed", exc_info=True)
        return mesh
    b0 = np.asarray(mesh.bounds, dtype=float)
    b1 = np.asarray(out.bounds, dtype=float)
    for axis in (0, 1):
        span0 = float(b0[1][axis] - b0[0][axis])
        span1 = float(b1[1][axis] - b1[0][axis])
        factor = span0 / max(span1, 1e-9)
        if abs(factor - 1.0) > 1e-4:
            c = (b0[0] + b0[1]) / 2.0
            factors = np.ones(3)
            factors[axis] = factor
            m = np.eye(4)
            m[:3, :3] = np.diag(factors)
            m[:3, 3] = c - factors * c
            out.apply_transform(m)
    if not out.is_watertight or len(out.faces) == 0:
        return mesh
    return out


def surface_roughness_deg(mesh: trimesh.Trimesh) -> float:
    """Mean dihedral angle (deg) between adjacent near-vertical faces.

    Used as a proxy for voxel striations on organic surfaces; floor and flat
    top faces (|normal.z| >= 0.95) are ignored.
    """
    try:
        angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
        if len(angles) == 0:
            return 0.0
        pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
        normals = np.asarray(mesh.face_normals, dtype=float)
        vertical = np.abs(normals[:, 2]) < 0.95
        mask = vertical[pairs[:, 0]] & vertical[pairs[:, 1]]
        sel = angles[mask]
        if len(sel) == 0:
            return 0.0
        return float(np.degrees(np.mean(sel)))
    except Exception:
        return 0.0


def is_flat_bottom(mesh: trimesh.Trimesh) -> bool:
    """True when most downward-facing area lies in a single plane at min Z."""
    try:
        normals = np.asarray(mesh.face_normals, dtype=float)
        down = normals[:, 2] < -0.9
        if not down.any():
            return False
        z0 = float(mesh.bounds[0][2])
        centers = np.asarray(mesh.triangles_center, dtype=float)
        flat = down & (np.abs(centers[:, 2] - z0) < 0.1)
        areas = np.asarray(mesh.area_faces, dtype=float)
        return float(areas[flat].sum()) / max(float(areas[down].sum()), 1e-9) >= 0.6
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# agent ops — validated, recordable mesh transformations                       #
# --------------------------------------------------------------------------- #
#
# Ops run in the final frame (mm, +Z up, resting on XY, centred) AFTER the
# deterministic baseline. Each op is validated by its own pydantic model —
# LLM-supplied args are never trusted — and applied to a copy so a failed op
# leaves the working mesh untouched.

CartesianAxis = Literal["x", "y", "z"]
_CARTESIAN: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}
BOOLEAN_MAX_FACES = 200_000
_MAX_PRIMITIVE_MM = 2000.0


class OpSetUp(BaseModel):
    """Re-orient in the current frame: the named source axes map to +Z/+Y."""

    up: AxisName = "+z"
    front: AxisName | None = None


class OpRotate(BaseModel):
    axis: CartesianAxis = "z"
    deg: float = Field(ge=-360.0, le=360.0)


class OpMirror(BaseModel):
    axis: CartesianAxis = "x"


class OpScale(BaseModel):
    """Uniform (`factor`) or per-axis (`x`/`y`/`z`) scaling about the bbox center."""

    factor: float | None = Field(default=None, ge=0.01, le=100.0)
    x: float | None = Field(default=None, ge=0.01, le=100.0)
    y: float | None = Field(default=None, ge=0.01, le=100.0)
    z: float | None = Field(default=None, ge=0.01, le=100.0)


class OpFlattenBottom(BaseModel):
    height_mm: float = Field(gt=0.0, le=50.0)


class OpSlice(BaseModel):
    """Cut + cap with an axis-aligned plane at `at_mm`; keep the chosen side."""

    axis: CartesianAxis = "z"
    at_mm: float = 0.0
    keep: Literal["low", "high"] = "high"


class OpCrop(BaseModel):
    """Keep only geometry inside the min/max box (mm)."""

    min: list[float] = Field(min_length=3, max_length=3)
    max: list[float] = Field(min_length=3, max_length=3)


class OpDecimate(BaseModel):
    faces: int = Field(ge=100, le=2_000_000)


class OpSmooth(BaseModel):
    iterations: int = Field(default=5, ge=1, le=20)
    scheme: Literal["laplacian", "taubin"] = "laplacian"


class OpSmoothPreserve(BaseModel):
    iterations: int = Field(default=10, ge=1, le=30)


class OpClipToOutline(BaseModel):
    """Clip the XY footprint to a measured plan-view outline polygon (mm)."""

    outline: list[list[float]] = Field(min_length=3, max_length=500)
    margin_mm: float = Field(default=0.3, ge=0.0, le=5.0)


class OpKeepLargest(BaseModel):
    pass


class OpDropSmall(BaseModel):
    """Drop disconnected bodies below `min_volume_frac` of the largest body."""

    min_volume_frac: float = Field(gt=0.0, lt=0.5)


class OpRepair(BaseModel):
    pass


class OpBoolean(BaseModel):
    """Union/difference with a primitive placed in mm frame coordinates."""

    op: Literal["union", "difference"] = "difference"
    shape: Literal["box", "cylinder", "sphere"] = "box"
    size: float | None = Field(default=None, gt=0.0, le=_MAX_PRIMITIVE_MM)
    size_xyz: list[float] | None = Field(default=None, min_length=3, max_length=3)
    radius: float | None = Field(default=None, gt=0.0, le=_MAX_PRIMITIVE_MM)
    height: float | None = Field(default=None, gt=0.0, le=_MAX_PRIMITIVE_MM)
    center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: CartesianAxis = "z"  # cylinder orientation
    rotate_z_deg: float = Field(default=0.0, ge=-360.0, le=360.0)


_OpFn = Callable[[trimesh.Trimesh, Any], trimesh.Trimesh]


def _bbox_center(mesh: trimesh.Trimesh) -> np.ndarray:
    b = np.asarray(mesh.bounds, dtype=float)
    return cast(np.ndarray, (b[0] + b[1]) / 2.0)


def _op_set_up(mesh: trimesh.Trimesh, p: OpSetUp) -> trimesh.Trimesh:
    return rest_on_xy_center(orient(mesh, p.up, p.front))


def _op_rotate(mesh: trimesh.Trimesh, p: OpRotate) -> trimesh.Trimesh:
    out = mesh.copy()
    m = trimesh.transformations.rotation_matrix(
        math.radians(p.deg), _CARTESIAN[p.axis], _bbox_center(mesh)
    )
    out.apply_transform(m)
    return out


def _op_mirror(mesh: trimesh.Trimesh, p: OpMirror) -> trimesh.Trimesh:
    out = mesh.copy()
    m = np.eye(4)
    idx = "xyz".index(p.axis)
    m[idx, idx] = -1.0
    out.apply_transform(m)
    out.invert()  # reflection flips winding
    return out


def _op_scale(mesh: trimesh.Trimesh, p: OpScale) -> trimesh.Trimesh:
    if p.factor is None and p.x is None and p.y is None and p.z is None:
        raise ValueError("scale needs `factor` or at least one of x/y/z")
    factors = np.array([p.x or p.factor or 1.0, p.y or p.factor or 1.0, p.z or p.factor or 1.0])
    out = mesh.copy()
    c = _bbox_center(mesh)
    m = np.eye(4)
    m[:3, :3] = np.diag(factors)
    m[:3, 3] = c - factors * c
    out.apply_transform(m)
    return out


def _op_flatten_bottom(mesh: trimesh.Trimesh, p: OpFlattenBottom) -> trimesh.Trimesh:
    return flatten_bottom(mesh, p.height_mm)


def _op_slice(mesh: trimesh.Trimesh, p: OpSlice) -> trimesh.Trimesh:
    idx = "xyz".index(p.axis)
    b = np.asarray(mesh.bounds)
    lo, hi = float(b[0][idx]), float(b[1][idx])
    if not lo + 1e-4 < p.at_mm < hi - 1e-4:
        raise ValueError(f"at_mm {p.at_mm:g} outside mesh range [{lo:.2f}, {hi:.2f}] on {p.axis}")
    normal = _CARTESIAN[p.axis].copy()
    origin = np.zeros(3)
    origin[idx] = p.at_mm
    if p.keep == "low":
        normal = -normal  # slice_plane keeps the positive-normal side
    out = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
    if out is None or not len(getattr(out, "faces", [])):
        raise ValueError("slice removed the whole mesh")
    return cast(trimesh.Trimesh, out)


def _op_crop(mesh: trimesh.Trimesh, p: OpCrop) -> trimesh.Trimesh:
    mn, mx = np.asarray(p.min, dtype=float), np.asarray(p.max, dtype=float)
    if np.any(mx <= mn):
        raise ValueError("crop max must be > min on every axis")
    out = mesh
    for i in range(3):
        for at, axis_sign in ((mn[i], 1.0), (mx[i], -1.0)):
            normal = _CARTESIAN["xyz"[i]] * axis_sign
            origin = np.zeros(3)
            origin[i] = at
            cut = out.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
            if cut is not None and len(cut.faces):
                out = cut
    if not len(out.faces):
        raise ValueError("crop removed the whole mesh")
    return out


def _op_decimate(mesh: trimesh.Trimesh, p: OpDecimate) -> trimesh.Trimesh:
    return decimate(mesh, p.faces)


def _op_smooth(mesh: trimesh.Trimesh, p: OpSmooth) -> trimesh.Trimesh:
    from trimesh.smoothing import filter_laplacian, filter_taubin

    out = mesh.copy()
    if p.scheme == "taubin":
        filter_taubin(out, iterations=p.iterations)
    else:
        filter_laplacian(out, iterations=p.iterations)
    return out


def _op_smooth_preserve(mesh: trimesh.Trimesh, p: OpSmoothPreserve) -> trimesh.Trimesh:
    return smooth_preserve(mesh, p.iterations)


def _op_clip_to_outline(mesh: trimesh.Trimesh, p: OpClipToOutline) -> trimesh.Trimesh:
    return clip_to_outline(mesh, p.outline, margin_mm=p.margin_mm)


def _body_sizes(bodies: list[trimesh.Trimesh]) -> np.ndarray:
    def size(b: trimesh.Trimesh) -> float:
        try:
            return max(abs(float(b.volume)), 1e-12)
        except Exception:
            return float(b.area)

    return np.array([size(b) for b in bodies])


def _op_keep_largest(mesh: trimesh.Trimesh, p: OpKeepLargest) -> trimesh.Trimesh:
    bodies = mesh.split(only_watertight=False)
    if len(bodies) <= 1:
        return mesh
    return bodies[int(np.argmax(_body_sizes(bodies)))]


def _op_drop_small(mesh: trimesh.Trimesh, p: OpDropSmall) -> trimesh.Trimesh:
    bodies = mesh.split(only_watertight=False)
    if len(bodies) <= 1:
        return mesh
    vols = _body_sizes(bodies)
    keep = [b for b, v in zip(bodies, vols, strict=True) if v >= vols.max() * p.min_volume_frac]
    if not keep:
        raise ValueError("drop_small would remove every body")
    return cast(trimesh.Trimesh, trimesh.util.concatenate(keep))


def _op_repair(mesh: trimesh.Trimesh, p: OpRepair) -> trimesh.Trimesh:
    return clean(mesh)


def _boolean_primitive(p: OpBoolean) -> trimesh.Trimesh:
    if p.shape == "box":
        ext = list(p.size_xyz) if p.size_xyz else [p.size] * 3 if p.size else None
        if ext is None:
            raise ValueError("box needs `size` or `size_xyz`")
        if any(float(v) <= 0 or float(v) > _MAX_PRIMITIVE_MM for v in ext):
            raise ValueError("box size out of bounds")
        prim = trimesh.creation.box(extents=ext)
        if p.rotate_z_deg:
            prim.apply_transform(
                trimesh.transformations.rotation_matrix(math.radians(p.rotate_z_deg), [0, 0, 1])
            )
    elif p.shape == "cylinder":
        if not p.radius or not p.height:
            raise ValueError("cylinder needs `radius` and `height`")
        prim = trimesh.creation.cylinder(radius=p.radius, height=p.height, sections=48)
        if p.axis == "x":
            prim.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        elif p.axis == "y":
            prim.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    else:
        if not p.radius:
            raise ValueError("sphere needs `radius`")
        prim = trimesh.creation.icosphere(subdivisions=3, radius=p.radius)
    prim.apply_translation(np.asarray(p.center, dtype=float))
    return cast(trimesh.Trimesh, prim)


def _op_boolean(mesh: trimesh.Trimesh, p: OpBoolean) -> trimesh.Trimesh:
    if len(mesh.faces) > BOOLEAN_MAX_FACES:
        raise ValueError(
            f"mesh too dense for boolean ({len(mesh.faces)} faces > {BOOLEAN_MAX_FACES}) "
            "— run mesh_decimate first"
        )
    prim = _boolean_primitive(p)
    try:
        if p.op == "union":
            out = mesh.union(prim, engine="manifold")
        else:
            out = mesh.difference(prim, engine="manifold")
    except Exception as e:
        raise ValueError(
            f"boolean {p.op} failed: {e} — mesh may be non-watertight; try mesh_repair"
        ) from e
    if out is None or not len(getattr(out, "faces", [])):
        raise ValueError(f"boolean {p.op} produced an empty mesh")
    return cast(trimesh.Trimesh, out)


OPS_REGISTRY: dict[str, tuple[type[BaseModel], _OpFn]] = {
    "set_up": (OpSetUp, _op_set_up),
    "rotate": (OpRotate, _op_rotate),
    "mirror": (OpMirror, _op_mirror),
    "scale": (OpScale, _op_scale),
    "flatten_bottom": (OpFlattenBottom, _op_flatten_bottom),
    "slice": (OpSlice, _op_slice),
    "crop": (OpCrop, _op_crop),
    "decimate": (OpDecimate, _op_decimate),
    "smooth": (OpSmooth, _op_smooth),
    "smooth_preserve": (OpSmoothPreserve, _op_smooth_preserve),
    "clip_to_outline": (OpClipToOutline, _op_clip_to_outline),
    "keep_largest": (OpKeepLargest, _op_keep_largest),
    "drop_small": (OpDropSmall, _op_drop_small),
    "repair": (OpRepair, _op_repair),
    "boolean": (OpBoolean, _op_boolean),
}


def apply_op(mesh: trimesh.Trimesh, op: MeshOp | dict[str, Any]) -> trimesh.Trimesh:
    """Validate + apply one op. Raises ValueError on bad args or op failure."""
    mo = op if isinstance(op, MeshOp) else MeshOp.model_validate(op)
    entry = OPS_REGISTRY.get(mo.op)
    if entry is None:
        raise ValueError(
            f"unknown mesh op '{mo.op}' — available: {', '.join(sorted(OPS_REGISTRY))}"
        )
    params_t, fn = entry
    params = params_t.model_validate(mo.args)
    return fn(mesh, params)


def apply_ops(
    mesh: trimesh.Trimesh, ops: Sequence[MeshOp | dict[str, Any]]
) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Apply an op list in order; returns (mesh, per-op report).

    The part is re-grounded (`rest_on_xy_center`) after every op so positional
    ops (slice/crop/boolean) always see a centred, on-XY frame — identical in
    the refine session and the executor pipeline.
    """
    report: list[dict[str, Any]] = []
    for raw in ops:
        mo = raw if isinstance(raw, MeshOp) else MeshOp.model_validate(raw)
        faces_in = len(mesh.faces)
        mesh = rest_on_xy_center(apply_op(mesh, mo))
        report.append(
            {
                "op": mo.op,
                "args": mo.args,
                "faces_in": faces_in,
                "faces_out": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
            }
        )
    return mesh, report


def baseline_mesh(
    raw: trimesh.Trimesh,
    post: MeshPostParams,
    spec: PartSpec | None,
    *,
    cad_cap_faces: int | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Deterministic baseline: clean -> decimate -> orient -> scale -> flatten -> rest.

    Shared by the executor pipeline (steps 1-5 of run_mesh_pipeline) and the
    worker-side refine session so preview and final result never drift.
    """
    mesh = clean(raw)

    target = post.decimate_faces or 0
    if cad_cap_faces:
        target = target or cad_cap_faces
    if target:
        mesh = decimate(mesh, target)

    mesh = orient(mesh, post.up, post.front, post.rotate_z_deg, post.mirror_x)

    mesh, factor, dim_issues = scale_to_spec(mesh, spec)
    if post.scale and abs(post.scale - 1.0) > 1e-9:
        mesh.apply_scale(post.scale)
        factor *= post.scale

    if post.flatten_bottom_mm > 0:
        mesh = flatten_bottom(mesh, post.flatten_bottom_mm)
    mesh = rest_on_xy_center(mesh)

    info = {
        "input_triangles": len(raw.faces),
        "triangles": len(mesh.faces),
        "orientation": {
            "up": post.up,
            "front": post.front,
            "rotate_z_deg": post.rotate_z_deg,
            "mirror_x": post.mirror_x,
        },
        "scale_factor": round(factor, 6),
        "dim_issues": dim_issues,
    }
    return mesh, info


# --------------------------------------------------------------------------- #
# mesh -> solid (hybrid bridge)                                                #
# --------------------------------------------------------------------------- #


def mesh_to_solid(mesh: trimesh.Trimesh, *, max_faces: int | None = None) -> Any:
    """Convert a (watertight) trimesh into a build123d Solid via STL + sewing.

    Returns a build123d Solid/Part. Raises on failure — caller decides the
    fallback. Decimates first when the mesh exceeds `max_faces` (default from
    runtime settings).
    """
    from build123d import Solid

    max_faces = max_faces or RuntimeSettings().mesh_max_faces_for_cad
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
        roughness_deg=surface_roughness_deg(mesh),
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


# --------------------------------------------------------------------------- #
# plan-view orientation / footprint clip                                       #
# --------------------------------------------------------------------------- #


def _footprint_poly(mesh: trimesh.Trimesh, step_frac: float = 120.0):
    """Shapely polygon of the XY footprint, centred at the origin.

    Step is derived from the mesh extent so raw (unit-scale) generator meshes
    and processed mm meshes both rasterise to a usable grid."""
    from shapely.geometry import Polygon

    from cadx.cad.mesh_features import detect_mesh_features

    extent = max(float(mesh.extents.max()), 1e-9)
    step = max(extent / step_frac, extent / 5000.0)
    feats = detect_mesh_features(mesh, step_mm=step)
    if len(feats.outline_mm) < 3:
        return None
    return Polygon(feats.outline_mm).buffer(0)


def _norm_poly(poly, target_span: float):
    """Translate centroid to origin and scale uniformly to `target_span`."""
    from shapely.affinity import scale, translate

    c = poly.centroid
    p = translate(poly, -c.x, -c.y)
    span = max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1], 1e-9)
    return scale(p, xfact=target_span / span, yfact=target_span / span, origin=(0, 0))


def outline_iou(a, b) -> float:
    """IoU of two shapely polygons (0 when degenerate)."""
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(a.intersection(b).area / union)


def orient_by_plan_view(
    mesh: trimesh.Trimesh,
    outline_mm: list[list[float]],
) -> tuple[AxisName, AxisName, bool, float] | None:
    """Find the axis orientation whose XY footprint best matches the measured
    plan-view outline. Tries all 24 (up, front) combos plus X-mirrors.

    Returns (up, front, mirror_x, iou) for the best candidate with IoU >= 0.7,
    else None — the caller falls back to the LLM orientation step."""
    from shapely.geometry import Polygon

    if len(outline_mm) < 3:
        return None
    img_poly = Polygon(outline_mm).buffer(0)
    if img_poly.is_empty or img_poly.area <= 0:
        return None
    img_normed = _norm_poly(img_poly, 100.0)

    best: tuple[float, AxisName, AxisName, bool] | None = None
    for up in UP_CHOICES:
        up_idx = "xyz".index(up[-1])
        for front_idx in range(3):
            if front_idx == up_idx:
                continue
            for front_sign in "+-":
                front = cast(AxisName, f"{front_sign}{'xyz'[front_idx]}")
                try:
                    m = orient(mesh, up, front)
                except Exception:
                    continue
                fp = _footprint_poly(m)
                if fp is None or fp.is_empty:
                    continue
                normed = _norm_poly(fp, 100.0)
                score = outline_iou(normed, img_normed)
                mirrored = _norm_poly(
                    _footprint_poly(orient(mesh, up, front, mirror_x=True)) or fp, 100.0
                )
                m_score = outline_iou(mirrored, img_normed)
                for s_, mx in ((score, False), (m_score, True)):
                    if best is None or s_ > best[0]:
                        best = (s_, up, front, mx)
    if best is None or best[0] < 0.7:
        return None
    score, up, front, mx = best
    return up, front, mx, round(score, 3)


def clip_to_outline(
    mesh: trimesh.Trimesh, outline_mm: list[list[float]], *, margin_mm: float = 0.3
) -> trimesh.Trimesh:
    """Clip the mesh to a prism extruded from the measured plan-view outline.

    Used when the generator's footprint drifts from the measured silhouette:
    the XY outline (Frame coords, mm) becomes the hard boundary. Falls back to
    the input mesh when the boolean fails or the result is not watertight."""
    try:
        from shapely.geometry import Polygon

        poly = Polygon(outline_mm)
        if poly.is_empty:
            return mesh
        # the outline lives in centred mm coords; the mesh is centred too —
        # no translation needed, just the margin
        poly = poly.buffer(margin_mm)
        z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        prism = trimesh.creation.extrude_polygon(poly, height=(z1 - z0) + 2.0)
        prism.apply_translation([0, 0, z0 - 1.0])
        out = mesh.intersection(prism, engine="manifold")
        if isinstance(out, trimesh.Trimesh) and out.is_watertight and not out.is_empty:
            return out
    except Exception:
        pass
    return mesh
