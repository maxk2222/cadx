"""Export a build123d Shape to all supported formats. Each format is independent —
a single failing export never fails the job; the error is recorded instead."""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
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
    export_step,
    export_stl,
)

FORMATS_3D = ["step", "stl", "3mf", "glb"]
FORMATS_2D = ["dxf", "svg"]

log = structlog.get_logger()


def export_step_checked(solid: Any, out_path: Path) -> None:
    """Write STEP and verify the stored shell orientation via reimport.

    OCC's STEP writer flips the shell orientation of solids sewn from
    triangle soup — a positive in-memory solid can land inside-out in the
    file (negative volume on reimport, wrong normals in some viewers).
    Reimport once; if the file reads inside-out, reverse and rewrite."""
    export_step(solid, str(out_path))
    try:
        from build123d import import_step

        if float(import_step(str(out_path)).volume) < 0:
            solid.wrapped.Reverse()
            export_step(solid, str(out_path))
    except Exception:
        pass


def _export_step_atomic(solid: Any, out_path: Path) -> None:
    """`export_step_checked` via a sibling .tmp file + os.replace.

    The analytic-STEP paths overwrite a previously-good faceted STEP in
    place — a mid-write OCC failure would truncate it. Write the candidate
    beside it first, then swap atomically on success."""
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".step.tmp")
    export_step_checked(solid, tmp)
    tmp.replace(out_path)


def export_step_file(shape: Shape, path: Path) -> None:
    export_step(shape, str(path))


def export_stl_file(
    shape: Shape, path: Path, tolerance: float = 0.01, angular_tolerance: float = 0.1
) -> None:
    export_stl(shape, str(path), tolerance=tolerance, angular_tolerance=angular_tolerance)


def export_3mf_file(
    shape: Shape,
    path: Path,
    name: str = "part",
    linear_deflection: float = 0.001,
    angular_deflection: float = 0.1,
) -> None:
    try:
        m = Mesher()
        m.add_shape(
            shape,
            part_number=name,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        m.write(str(path))
        return
    except Exception as e:
        log.warning("export.3mf_occ_failed", error=str(e)[:200])
    _export_3mf_via_trimesh(shape, path, linear_deflection, angular_deflection)


def _export_3mf_via_trimesh(
    shape: Shape, path: Path, linear_deflection: float, angular_deflection: float
) -> None:
    """trimesh fallback for shapes whose tessellation the OCC Mesher rejects
    ("3mf mesh is invalid") — helical thread seams routinely trip its strict
    manifold check while the B-Rep itself is valid. Write the mesh via
    trimesh; the pipeline's in-place repair still runs on the file after."""
    import numpy as np

    verts, tris = shape.tessellate(
        tolerance=linear_deflection, angular_tolerance=angular_deflection
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray([[v.X, v.Y, v.Z] for v in verts], dtype=float),
        faces=np.asarray(tris, dtype=np.int64),
        process=True,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.fix_normals()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="3mf")


def export_glb_file(shape: Shape, path: Path, linear_deflection: float = 0.05) -> None:
    """GLB preview in millimetres — the convention the viewer and the whole
    mesh pipeline follow.

    build123d's export_gltf converts to metres (the glTF spec unit) and writes
    no material — CAD previews came out 1000x smaller than mesh-mode GLBs with
    a default all-metallic material, i.e. a black screen in the viewer. So we
    tessellate and write the GLB via trimesh instead.
    """
    import numpy as np

    verts, tris = shape.tessellate(tolerance=linear_deflection, angular_tolerance=0.3)
    mesh = trimesh.Trimesh(
        vertices=np.asarray([[v.X, v.Y, v.Z] for v in verts], dtype=float),
        faces=np.asarray(tris, dtype=np.int64),
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=np.tile([168, 170, 178, 255], (len(mesh.faces), 1))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="glb")


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


def _premesh(shape: Shape, linear_deflection: float, angular_deflection: float) -> None:
    """Tessellate once at the finest deflection the exporters will ask for.

    Every mesher entry point (stl writer, Mesher/3mf, tessellate/glb) runs
    its own BRepMesh_IncrementalMesh — on a sewn or complex solid that is
    the dominant export cost, paid three times over. OCC keeps the finest
    triangulation per face and skips faces whose stored mesh already
    satisfies a requested deflection, so meshing once at the tightest
    tolerance lets all three consumers read the same OCC triangulation.
    Best-effort: a failed premesh just means each export meshes on its own."""
    try:
        from OCP.BRepMesh import BRepMesh_IncrementalMesh

        BRepMesh_IncrementalMesh(
            shape.wrapped,
            linear_deflection,
            True,
            angular_deflection,
            True,
        )
    except Exception:
        pass


def export_all_3d(
    shape: Shape,
    out_dir: Path,
    name: str = "model",
    *,
    stl_tolerance: float = 0.01,
    stl_angular: float = 0.1,
    glb_deflection: float = 0.05,
    parallel: int = 0,
) -> dict[str, str | None]:
    """Export step/stl/3mf/glb. Returns {fmt: filename or None}.

    ``parallel`` > 1 pre-meshes the shape once at the tightest tolerance
    (stl/3mf/glb then share OCC's cached triangulation instead of
    re-meshing three times) and runs the independent file writers in a
    thread pool — each still fails independently. 0 auto-detects."""
    from cadx.cad import resources

    workers = resources.pool_size(parallel)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 3mf defaults to a 10x finer deflection than the shipped STL — align it
    # with the STL grade so the premesh covers all three mesh consumers
    exporters: dict[str, Callable[[Path], None]] = {
        "step": lambda p: export_step_file(shape, p),
        "stl": lambda p: export_stl_file(shape, p, stl_tolerance, stl_angular),
        "3mf": lambda p: export_3mf_file(shape, p, name, stl_tolerance, stl_angular),
        "glb": lambda p: export_glb_file(shape, p, glb_deflection),
    }
    if workers > 1:
        _premesh(shape, min(stl_tolerance, glb_deflection), min(stl_angular, 0.3))
        from concurrent.futures import ThreadPoolExecutor

        results: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(exporters))) as pool:
            futures = {
                fmt: pool.submit(fn, out_dir / f"{name}.{fmt}") for fmt, fn in exporters.items()
            }
            for fmt, fut in futures.items():
                try:
                    fut.result()
                    results[fmt] = f"{name}.{fmt}"
                except Exception:
                    results[fmt] = None
        return results
    results = {}
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
    src = getattr(shape, "_cadx_mesh", None)
    if src is not None:
        # faceted hybrid: the attached mesh is already the exact surface —
        # OCC re-meshing of a sewn 40k-face solid costs minutes
        src.export(str(path))
        return path
    export_stl(shape, str(path), tolerance=0.05, angular_tolerance=0.2)
    return path


def repair_mesh_file(
    path: Path,
    *,
    max_vol_ratio: float = 1.5,
    max_bbox_growth_mm: float = 1.0,
) -> bool:
    """In-place repair of an STL/3MF/GLB export; returns watertight state
    afterwards. Best-effort: vertex merge, degenerate-face drop, hole fill,
    normal fix, then the voxel rebuild as a last resort.

    The voxel rebuild fills every enclosed void — catastrophic on topology
    optimised lattices or thin-walled parts whose interior cavities are the
    point of the design. It is only accepted when the rebuilt mesh stays
    geometrically faithful (bbox/volume drift below the limits); otherwise the
    lightly repaired mesh is written back instead and False is returned."""
    m = trimesh.load(str(path), force="mesh")
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if not m.is_watertight:
        m.fill_holes()
    m.fix_normals()
    if not m.is_watertight:
        from cadx.cad.mesh_ops import _ensure_watertight, _rebuild_drift

        rebuilt = _ensure_watertight(m)
        drift = _rebuild_drift(
            m, rebuilt, max_vol_ratio=max_vol_ratio, max_bbox_growth_mm=max_bbox_growth_mm
        )
        if drift is None:
            m = rebuilt
        else:
            log.warning("export.repair_drift", reason=drift, path=path.name)
    # repair ladders don't drop disconnected debris — a zero-volume sliver
    # sheet ships as a phantom second body and trips the bodies gate
    from cadx.cad.mesh_ops import keep_significant_components

    m, _ = keep_significant_components(m, min_fraction=1e-4)
    m.export(str(path))
    return bool(m.is_watertight and m.is_winding_consistent)


# --------------------------------------------------------------------------- #
# mesh-mode exports (trimesh-based; STEP via faceted solid)                     #
# --------------------------------------------------------------------------- #


def export_mesh_all(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    name: str = "model",
    *,
    try_step: bool = True,
    max_faces: int | None = None,
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
            from cadx.cad.mesh_ops import mesh_to_solid

            solid = mesh_to_solid(mesh, max_faces=max_faces)
            step_path = out_dir / f"{name}.step"
            export_step(solid, str(step_path))
            results["step"] = f"{name}.step"
        except Exception:
            results["step"] = None
    return results


# --------------------------------------------------------------------------- #
# parametric STEP — sewn optimized body + analytic feature cuts                 #
# --------------------------------------------------------------------------- #

# refine kinds re-cut as a plain analytic Cylinder in the STEP
_STEP_CYLINDER_KINDS = {
    "hole",
    "through_hole",
    "bore",
    "tunnel",
    "insert",
    "bearing_seat",
    "bearing",
}
_STEP_THREAD_KINDS = {"thread", "tap", "threaded_bore", "threaded_hole", "tap_hole"}
_STEP_COUNTER_KINDS = {"countersink", "counterbore"}

# RefineFeature fields accepted from dict-shaped feature records
_FEATURE_FIELDS = (
    "name",
    "kind",
    "axis",
    "u",
    "v",
    "dia",
    "depth_mm",
    "csk_dia",
    "csk_angle_deg",
    "cbore_dia",
    "cbore_depth_mm",
    "thread_spec",
    "helical",
    "dia_extra",
)


def _feat_field(feat: Any, key: str, default: Any = None) -> Any:
    if isinstance(feat, dict):
        return feat.get(key, default)
    return getattr(feat, key, default)


def _norm_refine_feature(feat: Any) -> Any | None:
    """Coerce a dict record into a RefineFeature (u/v may live in pos/x/y)."""
    from cadx.cad.fem import refine as rf

    if isinstance(feat, rf.RefineFeature):
        return feat
    if not isinstance(feat, dict):
        return feat
    kw = {k: feat[k] for k in _FEATURE_FIELDS if feat.get(k) is not None}
    axis = str(kw.get("axis") or "+z")
    if ("u" not in kw or "v" not in kw) and axis in rf._AXES:
        _ax, au, av = rf._AXES[axis]
        pos = feat.get("pos") or []
        if len(pos) > max(au, av) and pos[au] is not None and pos[av] is not None:
            kw.setdefault("u", float(pos[au]))
            kw.setdefault("v", float(pos[av]))
        elif axis.endswith("z") and feat.get("x") is not None and feat.get("y") is not None:
            kw.setdefault("u", float(feat["x"]))
            kw.setdefault("v", float(feat["y"]))
    try:
        return rf.RefineFeature(**kw)
    except Exception:
        return None


def _axis_placer(ax: int, au: int, av: int) -> Any:
    """Position helper: rotate a +Z-built shape onto axis index `ax` and
    return a placer fn(shape, lo, u, v) seating its low end at `lo`."""
    from build123d import Axis, Pos

    def place(shape: Any, lo: float, u: float, v: float) -> Any:
        if ax == 0:
            shape = shape.rotate(Axis.Y, 90)  # +Z -> +X
        elif ax == 1:
            shape = shape.rotate(Axis.X, -90)  # +Z -> +Y
        pos = [0.0, 0.0, 0.0]
        pos[ax] = lo
        pos[au], pos[av] = u, v
        return Pos(*pos) * shape

    return place


def _thread_groove_cutter(feat: Any, length: float) -> Any | None:
    """Helical internal-thread cutter (bd_warehouse IsoThread, external=False)
    built along +Z for `length` — None when no pitch is derivable."""
    from bd_warehouse.thread import IsoThread

    from cadx.cad.fem import refine as rf

    parsed = rf._parse_spec(str(_feat_field(feat, "thread_spec", "") or ""))
    d_maj, pitch = parsed if parsed else (_feat_field(feat, "dia", 0.0), rf._thread_pitch(feat))
    if not pitch:
        return None
    # 'raw' ends keep the full V profile past both ends so the groove cuts
    # cleanly through the part faces instead of fading inside the bore
    return IsoThread(
        major_diameter=float(d_maj),
        pitch=float(pitch),
        length=float(length),
        external=False,
        end_finishes=("raw", "raw"),
    )


def _analytic_cutters(feat: Any, mesh: trimesh.Trimesh) -> list[Any] | None:
    """build123d cutter solid(s) for one applied refine feature.

    Mirrors ``refine._cutter_mesh``'s pose convention: the feature opens on
    the ``axis``-signed surface at in-plane (u, v); the entry coordinate is
    the first ray hit from outside (bbox face when the channel is already
    open). Through features span the part bbox + margin; blind features cut
    ``depth_mm`` from the entry surface. Returns None when the feature has no
    usable pose/kind."""
    from build123d import Align, Cylinder

    from cadx.cad.fem import refine as rf

    axis = str(_feat_field(feat, "axis", "+z") or "+z")
    if axis not in rf._AXES:
        return None
    kind = str(_feat_field(feat, "kind", "hole") or "hole")
    if kind not in _STEP_CYLINDER_KINDS | _STEP_THREAD_KINDS | _STEP_COUNTER_KINDS:
        return None
    u, v = _feat_field(feat, "u"), _feat_field(feat, "v")
    dia = float(_feat_field(feat, "dia", 0.0) or 0.0)
    if u is None or v is None or dia <= 0:
        return None
    u, v = float(u), float(v)

    ax, au, av = rf._AXES[axis]
    neg = axis[0] == "-"
    bounds = mesh.bounds
    margin = 2.0
    span_ax = float(bounds[1][ax] - bounds[0][ax])
    depth_req = _feat_field(feat, "depth_mm")
    try:
        entry = rf._entry_point(mesh, axis, u, v)
    except Exception:
        entry = None
    if entry is None:
        entry = float(bounds[0][ax] if neg else bounds[1][ax])

    if depth_req:
        depth = float(depth_req)
        height = depth + margin
        lo = entry - margin if neg else entry - depth
    else:
        height = span_ax + 2 * margin
        lo = float(bounds[0][ax]) - margin
    place = _axis_placer(ax, au, av)

    cut_dia = float(rf._effective_dia(feat)) + float(_feat_field(feat, "dia_extra", 0.0) or 0.0)
    cutters: list[Any] = [
        place(
            Cylinder(
                radius=cut_dia / 2.0, height=height, align=(Align.CENTER, Align.CENTER, Align.MIN)
            ),
            lo,
            u,
            v,
        )
    ]
    if kind in _STEP_THREAD_KINDS:
        try:
            groove = _thread_groove_cutter(feat, height)
        except Exception:
            groove = None
        if groove is not None:
            cutters.append(place(groove, lo, u, v))
    cutters += _counter_cutters(feat, kind, cut_dia, entry, neg, margin, place, u, v)
    return cutters


def _counter_cutters(
    feat: Any,
    kind: str,
    cut_dia: float,
    entry: float,
    neg: bool,
    margin: float,
    place: Any,
    u: float,
    v: float,
) -> list[Any]:
    """Extra cutters for countersink (cone) / counterbore (second cylinder)
    kinds, seated on the same entry convention as the pilot bore."""
    from build123d import Align, Cone, Cylinder

    out: list[Any] = []
    if kind == "countersink":
        csk_dia = float(_feat_field(feat, "csk_dia", 0.0) or 0.0)
        angle = float(_feat_field(feat, "csk_angle_deg", 90.0) or 90.0)
        if csk_dia > cut_dia:
            h = (csk_dia - cut_dia) / 2.0 / math.tan(math.radians(angle) / 2.0)
            if h > 0.01:
                r_out = csk_dia / 2.0 + 0.3
                ch = h + 0.5
                if neg:
                    cone = Cone(bottom_radius=r_out, top_radius=0.05, height=ch)
                    out.append(place(cone, entry - 0.5, u, v))
                else:
                    cone = Cone(bottom_radius=0.05, top_radius=r_out, height=ch)
                    out.append(place(cone, entry - h, u, v))
    if kind == "counterbore":
        cb_dia = float(_feat_field(feat, "cbore_dia", 0.0) or 0.0)
        if cb_dia > cut_dia:
            cb_depth = float(_feat_field(feat, "cbore_depth_mm") or (cb_dia - cut_dia) / 2.0)
            cb_lo = entry - margin if neg else entry - cb_depth
            out.append(
                place(
                    Cylinder(
                        radius=cb_dia / 2.0,
                        height=cb_depth + margin,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                    ),
                    cb_lo,
                    u,
                    v,
                )
            )
    return out


def _cut_volume_ok(solid: Any, tool: Any, res: Any) -> bool:
    """Sanity gate for a difference result. A cut can only REMOVE material —
    never add — and never more than the tool's swept volume (plus slack for
    fuzzy-tolerance cleanup). Degenerate sewn-solid booleans come back as
    empty compounds or grown solids; both are refused instead of shipped."""
    try:
        tool_vol = abs(float(tool.volume))
        v = float(res.volume)
        s = float(solid.volume)
    except Exception:
        return False
    return s - tool_vol * 1.2 - 1e-3 <= v <= s * 1.005 + 1e-3


def _occ_cut(solid: Any, tool: Any, *, fuzzy_levels: tuple[float, ...] = (0.05, 0.2)) -> Any | None:
    """Boolean difference with a fuzzy-tolerance rescue for sewn solids.

    A B-Rep sewn from tessellation has facet-edge vertices at float32
    precision — plain ``solid - tool`` regularly returns an EMPTY Part on
    such bodies even though the solid reports ``is_valid`` and carries the
    right volume. ``BRepAlgoAPI_Cut`` with a fuzzy value treats
    near-coincident facet edges as shared and produces the real cut.
    Escalate the fuzzy window only while the result stays geometrically
    sane (``_cut_volume_ok``) — too-large fuzzies "heal" the cut into an
    uncut or grown solid. Every OCC boolean on a large sewn solid costs
    minutes, so secondary cutters (thread grooves, counters) pass fewer
    levels to bound worst-case runtime."""
    try:
        res = solid - tool
        if res is not None and res.solids() and _cut_volume_ok(solid, tool, res):
            return res
    except Exception:
        pass
    if not fuzzy_levels:
        return None
    try:
        from build123d import Shape
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    except Exception:
        return None
    for fuzzy in fuzzy_levels:
        try:
            op = BRepAlgoAPI_Cut(solid.wrapped, tool.wrapped)
            op.SetFuzzyValue(fuzzy)
            op.Build()
            if not op.IsDone():
                continue
            res = Shape.cast(op.Shape())
            if res is not None and res.solids() and _cut_volume_ok(solid, tool, res):
                return res
        except Exception:
            continue
    return None


def _validate_split_solid(mesh: trimesh.Trimesh) -> Any | None:
    """Sew a solid after splitting vertices at non-manifold junctions.

    ``process(validate=True)`` separates coincident-position vertices where
    faces only meet positionally (T-junctions from mesh unions). The sewed
    B-Rep then carries genuinely shared edges instead of degenerate
    coincident-face seams — OCC booleans work on it. Geometry is unchanged
    (a split move of zero distance), so no drift check is needed."""
    try:
        m = mesh.copy()
        m.process(validate=True)
        m.merge_vertices()
        from cadx.cad.mesh_boolean import sew_mesh, unify_checked

        return unify_checked(sew_mesh(m))
    except Exception:
        return None


def _cut_analytic_features(
    solid: Any,
    mesh: trimesh.Trimesh,
    normed: list[tuple[str, Any]],
    *,
    deadline: float | None = None,
) -> tuple[Any, list[dict[str, Any]], list[str], dict[str, str]]:
    """Cut every normed feature's analytic tools out of `solid` in sequence.

    Cutters are applied one at a time (pilot bore first, then thread
    groove/counters) — a single Compound tool makes the boolean far more
    fragile on sewn meshes, and a failed groove must not cost the pilot
    bore's analytic cylinder. ``deadline`` is a monotonic timestamp: once
    passed, remaining cutters are skipped and the feature is marked
    degraded — one pathological OCC boolean must not stall the export.
    Returns ``(solid, applied, failed_names, degraded)`` — degraded maps
    feature name → which sub-cutter failed; those features keep their
    analytic pilot but get an honest note."""
    applied: list[dict[str, Any]] = []
    failed: list[str] = []
    degraded: dict[str, str] = {}
    for name, feat in normed:
        cutters: list[Any] = []
        try:
            cutters = list(_analytic_cutters(feat, mesh))
        except Exception:
            failed.append(name)
            continue
        base = solid
        ok = True
        for i, tool in enumerate(cutters):
            if deadline is not None and time.monotonic() > deadline:
                degraded[name] = "time budget exceeded — cut skipped"
                if i == 0:
                    ok = False
                break
            # secondary cutters (thread groove, counters) get one fuzzy shot —
            # they are pathologically slow on large sewn solids
            levels = (0.05, 0.2) if i == 0 else (0.05,)
            t_cut = time.monotonic()
            try:
                res = _occ_cut(solid, tool, fuzzy_levels=levels)
            except Exception:
                res = None
            _cut_seconds = round(time.monotonic() - t_cut, 1)
            log.info(
                "export.analytic_step_cutter",
                feature=name,
                cutter=i,
                seconds=_cut_seconds,
                ok=bool(res is not None and res.solids()),
            )
            if res is None or not res.solids() or res.volume < solid.volume * 0.3:
                if i == 0:
                    # pilot cut failed — the feature is not analytic at all
                    solid = base
                    ok = False
                    break
                degraded[name] = f"secondary cutter {i} failed — pilot bore applied"
                break
            solid = res
        if ok:
            applied.append(
                {
                    "name": name,
                    "kind": str(_feat_field(feat, "kind", "hole")),
                    "axis": str(_feat_field(feat, "axis", "+z")),
                    "analytic": True,
                }
            )
            if name in degraded:
                applied[-1]["degraded"] = degraded[name]
        else:
            failed.append(name)
    return solid, applied, failed, degraded


# --------------------------------------------------------------------------- #
# face-swap fallback — replace a faceted bore wall with an analytic cylinder    #
# --------------------------------------------------------------------------- #
#
# When the boolean cut is dead on a B-Rep (keep-restore union seams leave
# boolean-toxic neighbourhoods: cut() spins past the per-feature timeout or
# collapses the solid to vol=0), the feature's tunnel still exists as facets.
# The swap identifies the wall's facet faces inside a cylindrical shell around
# the feature axis, drops them, and re-sews the shell with a bounded
# Geom_CylindricalSurface face bridging the remaining boundary loops. No OCC
# boolean is involved — the only risky step is sewing, already proven by
# mesh_to_solid. Every rejection returns None so the caller's honest skip —
# "faceted STEP kept" — never regresses.


def _vkey(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Vertex identity by rounded coordinate — sewn shells share vertex
    positions exactly; 0.1 µm rounding only absorbs read jitter."""
    return (round(x, 4), round(y, 4), round(z, 4))


def _face_pts_edges(
    face: Any,
) -> tuple[list[tuple[float, float, float]], list[frozenset[tuple[float, float, float]]]]:
    """Vertex coords and real edge vertex-key pairs of one face.

    ``face.vertices()`` is not wire-ordered (chords appear on merged quads),
    so edges come from a TopExp traversal instead."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    pts = [(float(v.X), float(v.Y), float(v.Z)) for v in face.vertices()]
    edges: list[frozenset[tuple[float, float, float]]] = []
    ex = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
    while ex.More():
        vex = TopExp_Explorer(ex.Current(), TopAbs_VERTEX)
        ks = []
        while vex.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(vex.Current()))
            ks.append(_vkey(p.X(), p.Y(), p.Z()))
            vex.Next()
        if len(ks) >= 2:
            edges.append(frozenset((ks[0], ks[-1])))
        ex.Next()
    return pts, edges


def _faceswap_radius_guess(feat: Any, feature: Any) -> float | None:
    """Guessed wall radius: the measured tunnel diameter when the record
    carries one (refine reams can land oversize — Ø10 spec → Ø10.4 measured),
    else the requested cut diameter (tap-drill for threads)."""
    from cadx.cad.fem import refine as rf

    for key in ("measured_dia_mm", "dia_mm", "requested_dia_mm"):
        d = _feat_field(feature, key)
        if d:
            return float(d) / 2.0
    try:
        return (
            float(rf._effective_dia(feat)) + float(_feat_field(feat, "dia_extra", 0.0) or 0.0)
        ) / 2.0
    except Exception:
        d = _feat_field(feat, "dia")
        return float(d) / 2.0 if d else None


def _faceswap_params(
    feat: Any, feature: Any, solid: Any, mesh: trimesh.Trimesh
) -> dict[str, Any] | None:
    """Axis line, guessed wall radius and axial search window."""
    from cadx.cad.fem import refine as rf

    axis = str(_feat_field(feat, "axis", "+z") or "+z")
    if axis not in rf._AXES:
        return None
    ax, au, av = rf._AXES[axis]
    u, v = _feat_field(feat, "u"), _feat_field(feat, "v")
    if u is None or v is None:
        return None
    r_guess = _faceswap_radius_guess(feat, feature)
    if r_guess is None or r_guess <= 0:
        return None
    bb = solid.bounding_box()
    lo, hi = (bb.min.X, bb.min.Y, bb.min.Z)[ax], (bb.max.X, bb.max.Y, bb.max.Z)[ax]
    if _feat_field(feat, "depth_mm"):
        # blind bore: the wall lives between the entry face and the first
        # ray hit at (u, v) — on an open blind tunnel that hit IS the cap
        try:
            entry = rf._entry_point(mesh, axis, float(u), float(v))
        except Exception:
            entry = None
        if entry is not None:
            if axis[0] == "-":
                hi = float(entry)
            else:
                lo = float(entry)
    return {
        "ax": ax,
        "au": au,
        "av": av,
        "u": float(u),
        "v": float(v),
        "r_guess": r_guess,
        "z_win": (lo - 0.6, hi + 0.6),
    }


def _grow_patch(
    pts_l: list[list[tuple[float, float, float]]], seeds: set[int], cands: set[int]
) -> set[int]:
    """Flood-fill the seed wall faces over shared vertices. Adjacency — not
    the radial band alone — so a coaxial counterbore wall or a crossing
    feature's facets can never be pulled into the patch."""
    v2f: dict[tuple[float, float, float], set[int]] = defaultdict(set)
    for i, pts in enumerate(pts_l):
        for p in pts:
            v2f[_vkey(*p)].add(i)
    patch = set(seeds)
    stack = list(seeds)
    while stack:
        i = stack.pop()
        for p in pts_l[i]:
            for j in v2f[_vkey(*p)]:
                if j in cands and j not in patch:
                    patch.add(j)
                    stack.append(j)
    return patch


def _heal_enclosed(
    patch: set[int],
    edges_l: list[list[frozenset[tuple[float, float, float]]]],
    rad_l: list[list[float]],
    ndot_l: list[float],
    r_lo: float,
    r_hi: float,
) -> set[int]:
    """Absorb faces ringed on every edge by the patch — corrupt-seam slivers
    that failed the wall test but sit enclosed inside the wall region would
    otherwise leave pinhole boundary loops. Restricted to faces inside the
    radial band AND not axis-normal: a legit flat end cap is also rimmed by
    the wall but is a valid ``Geom_Plane`` bottom the swap must keep."""
    pedge: set[frozenset[tuple[float, float, float]]] = set()
    for i in patch:
        pedge.update(edges_l[i])
    grown = set(patch)
    for i, edges in enumerate(edges_l):
        if i in grown or not edges or not rad_l[i]:
            continue
        if ndot_l[i] > 0.8:
            continue
        if all(r_lo <= r <= r_hi for r in rad_l[i]) and all(e in pedge for e in edges):
            grown.add(i)
    return grown


def _patch_boundary_loops(
    patch: set[int], edges_l: list[list[frozenset[tuple[float, float, float]]]]
) -> list[list[tuple[float, float, float]]] | None:
    """Ordered boundary vertex rings of the patch — the rims the analytic
    wall re-joins. Exactly two clean rings (one per bore end); a torn or
    partial patch is refused."""
    cnt: dict[frozenset[tuple[float, float, float]], int] = defaultdict(int)
    for i in patch:
        for e in edges_l[i]:
            cnt[e] += 1
    bnd = [e for e, c in cnt.items() if c == 1]
    if len(bnd) < 6:
        return None
    adj: dict[tuple[float, float, float], list[tuple[float, float, float]]] = defaultdict(list)
    for e in bnd:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    if any(len(nbrs) != 2 for nbrs in adj.values()):
        return None
    loops: list[list[tuple[float, float, float]]] = []
    seen: set[tuple[float, float, float]] = set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = next(n for n in adj[cur] if n != prev)
            if nxt == start:
                break
            if nxt in seen or len(loop) > 200000:
                return None
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    if len(loops) != 2 or min(len(lp) for lp in loops) < 3:
        return None
    return loops


def _fit_bore_cylinder(
    patch: set[int],
    loops: list[list[tuple[float, float, float]]],
    pts_l: list[list[tuple[float, float, float]]],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Circle-fit the patch vertices in the plane ⟂ to the axis and bound
    the cylinder by the two boundary loops. Refuses walls too irregular for
    an analytic surface (wavy organic rims, torn seams)."""
    import numpy as np

    ax, au, av = params["ax"], params["au"], params["av"]
    cu, cv, r_g = params["u"], params["v"], params["r_guess"]
    xy = np.array([[p[au], p[av]] for i in patch for p in pts_l[i]], dtype=float)
    if len(xy) < 6:
        return None
    try:
        a = np.c_[2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))]
        sol, *_ = np.linalg.lstsq(a, (xy**2).sum(axis=1), rcond=None)
        fu, fv = float(sol[0]), float(sol[1])
        if math.hypot(fu - cu, fv - cv) <= max(1.0, 0.5 * r_g):
            cu, cv = fu, fv
    except Exception:
        pass  # keep the feature's declared (u, v) as the centre
    rad = [math.hypot(p[au] - cu, p[av] - cv) for i in patch for p in pts_l[i]]
    r_cyl = float(np.median(rad))
    if not (0.5 * r_g <= r_cyl <= 1.9 * r_g):
        return None
    z_mean = [sum(p[ax] for p in loop) / len(loop) for loop in loops]
    z_lo, z_hi = min(z_mean), max(z_mean)
    if z_hi - z_lo < 0.3:
        return None
    max_dev = 0.0
    for loop, zm in zip(loops, z_mean, strict=True):
        for p in loop:
            dev = math.hypot(math.hypot(p[au] - cu, p[av] - cv) - r_cyl, p[ax] - zm)
            max_dev = max(max_dev, dev)
    # polygon-chord sagitta between the rim edges and the true circle —
    # the sew tolerance must cover it even when vertices sit exactly on
    sag = max(r_cyl * (1.0 - math.cos(math.pi / len(loop))) for loop in loops)
    sew_tol = 0.02 + 1.4 * max_dev + sag
    if max_dev > 0.8 or sew_tol > 0.9:
        return None
    return {
        **params,
        "r_cyl": r_cyl,
        "cu": cu,
        "cv": cv,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "sew_tol": max(0.04, sew_tol),
        "boundary_verts": {p for loop in loops for p in loop},
    }


def _bore_wall_patch(
    faces: list[Any], params: dict[str, Any]
) -> tuple[set[int], dict[str, Any]] | None:
    """Connected patch of facet faces forming the feature's bore wall plus
    the fitted cylinder description — None when the wall cannot be isolated
    cleanly enough to swap."""
    ax, au, av = params["ax"], params["au"], params["av"]
    u, v, r_g = params["u"], params["v"], params["r_guess"]
    z_lo, z_hi = params["z_win"]
    pts_l: list[list[tuple[float, float, float]]] = []
    edges_l: list[list[frozenset[tuple[float, float, float]]]] = []
    ndot_l: list[float] = []
    for f in faces:
        try:
            pts, edges = _face_pts_edges(f)
        except Exception:
            pts, edges = [], []
        pts_l.append(pts)
        edges_l.append(edges)
        try:
            n = f.normal_at()
            ndot_l.append(abs((float(n.X), float(n.Y), float(n.Z))[ax]))
        except Exception:
            ndot_l.append(0.0)
    rad_l = [[math.hypot(p[au] - u, p[av] - v) for p in pts] for pts in pts_l]
    win = [bool(pts) and all(z_lo <= p[ax] <= z_hi for p in pts) for pts in pts_l]
    t_seed = max(0.3, 0.12 * r_g)

    def cand(i: int, r_c: float) -> bool:
        return (
            win[i]
            and all(0.5 * r_c <= r <= 1.7 * r_c + 1.0 for r in rad_l[i])
            and ndot_l[i] <= 0.55
        )

    seeds, r_g = _wall_seed_faces(rad_l, ndot_l, win, r_g, t_seed, cand)
    if len(seeds) < 4:
        return None
    patch = _grow_patch(pts_l, seeds, {i for i in range(len(faces)) if cand(i, r_g)})
    if len(patch) < 4:
        return None
    patch = _heal_enclosed(patch, edges_l, rad_l, ndot_l, 0.5 * r_g, 1.7 * r_g + 1.0)
    loops = _patch_boundary_loops(patch, edges_l)
    if loops is None:
        return None
    meta = _fit_bore_cylinder(patch, loops, pts_l, {**params, "r_guess": r_g})
    if meta is None:
        return None
    meta["face_keys"] = [{_vkey(*p) for p in pts} for pts in pts_l]
    return patch, meta


def _wall_seed_faces(
    rad_l: list[list[float]],
    ndot_l: list[float],
    win: list[bool],
    r_g: float,
    t_seed: float,
    cand: Callable[[int, float], bool],
) -> tuple[set[int], float]:
    """Faces confidently ON the bore wall (all vertices hug the radius guess,
    normal ⊥ axis) — plus a re-centred guess when the recorded diameter is
    off the real tunnel (nominal-vs-tap mismatch, unrecorded ream)."""
    import numpy as np

    def seed_set(r_c: float) -> set[int]:
        return {
            i
            for i in range(len(rad_l))
            if win[i] and ndot_l[i] <= 0.5 and all(abs(r - r_c) <= t_seed for r in rad_l[i])
        }

    seeds = seed_set(r_g)
    if len(seeds) < 4:
        shell_r = [r for i in range(len(rad_l)) if cand(i, r_g) for r in rad_l[i]]
        if shell_r:
            r_med = float(np.median(shell_r))
            if abs(r_med - r_g) > t_seed:
                seeds = seed_set(r_med)
                if seeds:
                    r_g = r_med
    return seeds, r_g


def _bore_cylinder_face(meta: dict[str, Any]) -> Any:
    """Bounded Geom_CylindricalSurface face spanning the wall patch, reversed
    so its normal points at the axis (the bore is a void, not a boss)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Ax2, gp_Ax3, gp_Cylinder, gp_Dir, gp_Pnt

    ax, au, av = meta["ax"], meta["au"], meta["av"]
    origin = [0.0, 0.0, 0.0]
    origin[ax], origin[au], origin[av] = meta["z_lo"], meta["cu"], meta["cv"]
    direction = [0.0, 0.0, 0.0]
    direction[ax] = 1.0
    ax3 = gp_Ax3(gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction)))
    cyl = gp_Cylinder(ax3, float(meta["r_cyl"]))
    mf = BRepBuilderAPI_MakeFace(cyl, 0.0, 2.0 * math.pi, 0.0, meta["z_hi"] - meta["z_lo"])
    if not mf.IsDone():
        raise ValueError("cylinder face build failed")
    face = mf.Face()
    face.Reverse()
    return face


def _resew_swapped_shell(
    faces: list[Any], patch: set[int], cyl_face: Any, meta: dict[str, Any]
) -> Any | None:
    """Sew ``faces - patch + cyl_face`` back into a solid.

    Two passes: the analytic face and the rim-adjacent faces sew at the gap
    tolerance bridging polygon rim ↔ true circle, then the result re-joins
    the rest of the shell at the usual 1e-4 — the large tolerance never
    reaches distant lattice edges. Orientation normalised like mesh_to_solid
    (negative-volume solids break every downstream consumer)."""
    from build123d import Shape, Shell, Solid
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Shape

    bverts = meta["boundary_verts"]
    ring, rest = [], []
    for i, f in enumerate(faces):
        if i in patch:
            continue
        (ring if meta["face_keys"][i] & bverts else rest).append(f.wrapped)
    sew1 = BRepBuilderAPI_Sewing(float(meta["sew_tol"]))
    sew1.Add(cyl_face)
    for w in ring:
        sew1.Add(w)
    sew1.Perform()
    sew2 = BRepBuilderAPI_Sewing(1e-4)
    sew2.Add(sew1.SewedShape())
    for w in rest:
        sew2.Add(w)
    sew2.Perform()
    fixed = ShapeFix_Shape(sew2.SewedShape()).Shape()
    wrapped = Shape.cast(fixed)
    shells = wrapped.shells()
    if not shells:
        return None
    shell = max(shells, key=lambda sh: sh.area)
    if not isinstance(shell, Shell):
        shell = Shell(shell.wrapped)
    res = Solid(Solid._make_solid(shell.fix()))
    try:
        if float(res.volume) < 0:
            wr = res.wrapped
            wr.Reverse()
            res = Solid(wr)
    except Exception:
        pass
    return res


def _faceswap_ok(solid: Any, res: Any, meta: dict[str, Any]) -> bool:
    """Sanity gate mirroring ``_cut_volume_ok``: the swap may only remove up
    to roughly the bore's swept volume and never add material — plus the
    analytic cylinder must actually be present in the result."""
    if res is None:
        return False
    try:
        if not res.solids():
            return False
        s, v = abs(float(solid.volume)), float(res.volume)
        bore_vol = math.pi * meta["r_cyl"] ** 2 * (meta["z_hi"] - meta["z_lo"])
        if not (s - 1.3 * bore_vol - 0.5 <= v <= s * 1.02 + max(0.5, 0.05 * s)):
            return False
        r_cyl = float(meta["r_cyl"])
        for f in res.faces():
            if f.geom_type == GeomType.CYLINDER and abs(float(f.radius) - r_cyl) <= 0.3 * r_cyl:
                return True
        return False
    except Exception:
        return False


def _faceswap_bore(
    solid: Any, feature: Any, mesh: trimesh.Trimesh
) -> tuple[Any | None, str | None]:
    """Replace a feature's faceted bore wall with a true analytic cylinder.

    Fallback for solids whose OCC booleans are dead near the feature: no
    boolean is attempted — the wall's facet faces are dropped and the shell
    is re-sewn with a Geom_CylindricalSurface face spanning the boundary
    loops, ends bounded by the loops themselves (caps are never wall faces
    — their inner vertices fall inside the shell radius). Returns
    ``(solid, degraded_note)``; ``(None, None)`` on any rejection so the
    caller's honest skip is preserved. For thread/counter kinds only the
    pilot wall is swapped — the caller records the feature as degraded."""
    feat = _norm_refine_feature(feature)
    if feat is None:
        return None, None
    kind = str(_feat_field(feat, "kind", "hole") or "hole")
    if kind not in _STEP_CYLINDER_KINDS | _STEP_THREAD_KINDS | _STEP_COUNTER_KINDS:
        return None, None
    name = str(_feat_field(feature, "name", "") or kind)
    res = None
    try:
        params = _faceswap_params(feat, feature, solid, mesh)
        if params is not None:
            faces = list(solid.faces())
            found = _bore_wall_patch(faces, params)
            if found is not None:
                patch, meta = found
                res = _resew_swapped_shell(faces, patch, _bore_cylinder_face(meta), meta)
                if not _faceswap_ok(solid, res, meta):
                    res = None
    except Exception:
        res = None
    if res is None:
        log.info("export.analytic_step_faceswap_rejected", feature=name, kind=kind)
        return None, None
    note = None
    if kind in _STEP_THREAD_KINDS:
        note = "face-swap fallback — pilot bore wall analytic, thread groove not restored"
    elif kind in _STEP_COUNTER_KINDS:
        note = "face-swap fallback — pilot bore wall analytic, counter region remains faceted"
    return res, note


def _norm_for_worker(raw: Any) -> Any | None:
    """Normed RefineFeature for a worker child — also forwards measured
    tunnel diameters the normaliser drops (the face-swap fallback sizes
    its wall-search shell from them)."""
    feat = _norm_refine_feature(raw)
    if feat is not None:
        for k in ("measured_dia_mm", "dia_mm", "requested_dia_mm"):
            if _feat_field(raw, k) is not None and _feat_field(feat, k) is None:
                with contextlib.suppress(Exception):
                    setattr(feat, k, float(_feat_field(raw, k)))
    return feat


def _worker_try_faceswap(
    solid: Any, feature: Any, mesh: trimesh.Trimesh, name: str, out: dict[str, Any], why: str
) -> tuple[Any | None, str | None]:
    """Boolean-dead fallback inside a worker child: swap the feature's
    faceted bore wall for an analytic cylinder face on the SAME B-Rep
    input. Returns (solid, degraded_note); (None, None) keeps the skip."""
    try:
        swapped, swap_note = _faceswap_bore(solid, feature, mesh)
    except Exception:
        swapped, swap_note = None, None
    if swapped is None:
        return None, None
    out["warnings"].append(
        f"feature '{name}': {why} — bore wall rebuilt as an analytic face (face-swap)"
    )
    log.info("export.analytic_step_faceswap", feature=name)
    return swapped, swap_note


def _analytic_step_worker(
    stl_path: str,
    brep_in: str,
    feature: Any,
    out_brep: str,
    result_path: str,
    max_faces: int | None,
    swap_only: bool = False,
) -> None:
    """Subprocess body for ONE feature of the analytic STEP export.

    Loads the input B-Rep (empty brep_in → sew the shipped STL fresh), cuts
    the feature's analytic tools and writes the result B-Rep plus a JSON
    status the parent collects. B-Rep round-tripping keeps earlier features'
    analytic faces intact — an STL round-trip would re-facet them.
    ``swap_only`` children skip the boolean entirely (the parent only
    retries that way after a cut already spun past its timeout) and go
    straight to the face-swap — bounded work, no OCC boolean to hang."""
    import json as _json

    out: dict[str, Any] = {"applied": False, "warnings": []}
    mesh = trimesh.load(stl_path, force="mesh")
    solid = _load_solid_for_worker(brep_in, mesh, max_faces, out)
    if solid is None:
        Path(result_path).write_text(_json.dumps(out))
        return

    feat = _norm_refine_feature(feature)
    name = str(_feat_field(feature, "name", "") or _feat_field(feature, "kind", "feature"))
    try:
        cutters = _analytic_cutters(feat, mesh) or [] if feat is not None else []
    except Exception:
        cutters = []
    if not cutters:
        out["warnings"].append(f"feature '{name}': no usable refine pose — skipped")
        Path(result_path).write_text(_json.dumps(out))
        return

    ok = False
    degraded_note = None
    if swap_only:
        swapped, swap_note = _worker_try_faceswap(
            solid, feature, mesh, name, out, "boolean cut exceeded budget"
        )
        if swapped is not None:
            solid, ok, degraded_note = swapped, True, swap_note
    else:
        for i, tool in enumerate(cutters):
            try:
                res = _occ_cut(solid, tool, fuzzy_levels=(0.05,))
            except Exception:
                res = None
            if res is not None and res.solids() and res.volume >= solid.volume * 0.3:
                solid = res
                ok = True
            else:
                if i == 0:
                    # boolean cut failed or collapsed on this B-Rep — try
                    # swapping the feature's faceted bore wall for an
                    # analytic cylinder face on the SAME input first
                    swapped, swap_note = _worker_try_faceswap(
                        solid, feature, mesh, name, out, "boolean cut failed"
                    )
                    if swapped is not None:
                        solid, ok, degraded_note = swapped, True, swap_note
                        # the body is proven boolean-dead near this feature —
                        # spending the child's remaining timeout on secondary
                        # cutters risks losing the checkpoint; the swapped
                        # pilot wall is the analytic win here
                        break
                elif ok:
                    degraded_note = f"secondary cutter {i} failed — pilot bore applied"
                else:
                    ok = False
                break
    if ok:
        out["applied"] = True
        _write_worker_brep(solid, out_brep, name, out)
    else:
        out["warnings"].append(
            f"feature '{name}': analytic cut produced no valid solid — faceted STEP kept"
        )
    if degraded_note:
        out["degraded"] = degraded_note
        out["warnings"].append(f"feature '{name}': {degraded_note}")
    Path(result_path).write_text(_json.dumps(out))


def _load_solid_for_worker(
    brep_in: str, mesh: trimesh.Trimesh, max_faces: int | None, out: dict[str, Any]
) -> Any:
    """Input solid for a worker child: prior BREP checkpoint, else fresh sew."""
    if brep_in:
        try:
            from build123d import Shape
            from OCP.BRep import BRep_Builder
            from OCP.BRepTools import BRepTools
            from OCP.TopoDS import TopoDS_Shape

            sh = TopoDS_Shape()
            BRepTools.Read_s(sh, brep_in, BRep_Builder())
            return Shape.cast(sh)
        except Exception as e:
            # falling back to the raw STL would silently drop earlier analytic
            # cuts while the parent still reports them — fail instead
            out["warnings"].append(f"BREP checkpoint unreadable: {e}")
            return None
    try:
        import io

        mesh = trimesh.load(io.BytesIO(mesh.export(file_type="stl")), file_type="stl")
    except Exception:
        pass
    mesh = _decimate_for_analytic(mesh, globals()["log"])
    try:
        from cadx.cad.mesh_ops import mesh_to_solid

        return mesh_to_solid(mesh, max_faces=max_faces)
    except Exception as e:
        out["warnings"].append(f"analytic STEP skipped — mesh-to-solid failed: {e}")
        return None


def _write_worker_brep(solid: Any, out_brep: str, name: str, out: dict[str, Any]) -> None:
    """Checkpoint the post-cut solid as BREP for the next feature child."""
    try:
        from OCP.BRepTools import BRepTools

        try:
            if float(solid.volume) < 0:
                solid.wrapped.Reverse()
        except Exception:
            pass
        BRepTools.Write_s(solid.wrapped, out_brep)
        out["brep"] = out_brep
    except Exception as e:
        out["applied"] = False
        out["warnings"].append(f"feature '{name}': BREP write failed: {e}")


def _analytic_base_worker(
    stl_path: str, out_brep: str, result_path: str, max_faces: int | None
) -> None:
    """Setup child for the isolated analytic STEP export: sew the shipped
    STL into the BASE B-Rep once so the per-feature children load this
    checkpoint instead of each re-running decimate+mesh_to_solid (the sew
    is the multi-minute part and used to repeat per feature)."""
    import json as _json

    out: dict[str, Any] = {"applied": False, "warnings": []}
    try:
        mesh = trimesh.load(stl_path, force="mesh")
        solid = _load_solid_for_worker("", mesh, max_faces, out)
        if solid is not None:
            out["applied"] = True
            _write_worker_brep(solid, out_brep, "base", out)
    except Exception as e:
        out["applied"] = False
        out["warnings"].append(f"base solid sew failed: {e}")
    with contextlib.suppress(Exception):
        Path(result_path).write_text(_json.dumps(out))


def export_analytic_step_isolated(
    mesh_path: Path,
    features: list[Any],
    out_path: Path,
    *,
    timeout_s: float = 300.0,
    per_feature_s: float | None = None,
    max_faces: int | None = None,
    log: Any = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the analytic STEP export feature-by-feature in child processes.

    OCC booleans on sewn mesh solids are unbounded — a single pathological
    cut has been observed to spin >25 minutes, and a Python deadline check
    cannot interrupt a call inside OCC. Each feature therefore gets its own
    forked child with a per-feature timeout; the evolving solid round-trips
    between children as a BREP file so analytic faces survive. A feature
    that times out or collapses is skipped and the pipeline continues —
    the faceted STEP stays as the honest fallback.
    """
    import json as _json
    import multiprocessing as _mp
    import tempfile

    logger = log if log is not None else globals()["log"]
    out_path = Path(out_path)
    warnings: list[str] = []
    out: dict[str, Any] = {"step": None, "step_features": [], "warnings": warnings}
    per_feature = per_feature_s or max(30.0, timeout_s / max(len(features or []), 1))

    normed: list[tuple[str, Any]] = []
    for raw in features or []:
        feat = _norm_for_worker(raw)
        name = str(_feat_field(raw, "name", "") or _feat_field(raw, "kind", "feature"))
        if feat is None:
            warnings.append(f"feature '{name}': no usable refine pose — skipped")
            continue
        normed.append((name, feat))

    # spawn, not fork: the parent has already exercised OCC (sandbox runs),
    # and a forked child can inherit locked OCC mutexes and deadlock before
    # doing any work. Spawn costs a few seconds of interpreter startup per
    # feature but is immune to fork-safety issues.
    ctx = _mp.get_context("spawn")
    brep_in = ""
    applied: list[dict[str, Any]] = []
    t_all = time.monotonic()
    with tempfile.TemporaryDirectory(dir=out_path.parent) as td:
        tdir = Path(td)
        if normed:
            # checkpoint the sewn base solid ONCE before the feature loop —
            # brep_in="" would make every feature child re-run the
            # decimate+mesh_to_solid sew (minutes) before its own cut
            brep_in = _base_solid_checkpoint(ctx, mesh_path, per_feature, max_faces, tdir, warnings)
        for idx, (name, feat) in enumerate(normed):
            if time.monotonic() - t_all > timeout_s:
                warnings.append(
                    f"analytic STEP exceeded {timeout_s:.0f}s total budget — "
                    "remaining features skipped"
                )
                break
            if cancelled is not None and cancelled():
                warnings.append("analytic STEP cancelled — remaining features skipped")
                break
            child_brep = tdir / f"feat{idx}.brep"
            child_res = tdir / f"feat{idx}.json"
            proc = ctx.Process(
                target=_analytic_step_worker,
                args=(
                    str(mesh_path),
                    brep_in,
                    feat,
                    str(child_brep),
                    str(child_res),
                    max_faces,
                ),
                daemon=True,
            )
            proc.start()
            proc.join(per_feature)
            if proc.is_alive():
                # the boolean spun past the budget — kill it and retry
                # swap-only: the fallback does no OCC boolean and is
                # bounded by the re-sew, so it cannot hang the same way
                proc.terminate()
                proc.join(5)
                warnings.append(
                    f"feature '{name}': cut exceeded {per_feature:.0f}s — face-swap retry"
                )
                logger.warning("export.analytic_step_feature_timeout", feature=name)
                proc = ctx.Process(
                    target=_analytic_step_worker,
                    args=(
                        str(mesh_path),
                        brep_in,
                        feat,
                        str(child_brep),
                        str(child_res),
                        max_faces,
                        True,
                    ),
                    daemon=True,
                )
                proc.start()
                proc.join(per_feature)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                    continue
            try:
                cres = _json.loads(child_res.read_text())
            except Exception:
                warnings.append(f"feature '{name}': cut failed — skipped")
                continue
            warnings += [str(w) for w in cres.get("warnings", [])]
            if cres.get("applied") and cres.get("brep") and Path(cres["brep"]).exists():
                brep_in = str(cres["brep"])
                rec = {
                    "name": name,
                    "kind": str(_feat_field(feat, "kind", "hole")),
                    "axis": str(_feat_field(feat, "axis", "+z")),
                    "analytic": True,
                }
                if cres.get("degraded"):
                    rec["degraded"] = str(cres["degraded"])
                applied.append(rec)

        if brep_in and applied:
            # final solid → STEP; child wrote BREPs into tdir, so the last one
            # is the accumulated body. Without an applied feature the base
            # brep is just a re-sewn copy — the faceted STEP stays instead.
            try:
                from OCP.BRep import BRep_Builder
                from OCP.BRepTools import BRepTools
                from OCP.TopoDS import TopoDS_Shape

                sh = TopoDS_Shape()
                BRepTools.Read_s(sh, brep_in, BRep_Builder())
                _export_step_atomic(Shape.cast(sh), out_path)
                out["step"] = out_path.name
                out["step_features"] = applied
            except Exception as e:
                warnings.append(f"analytic STEP write failed: {e}")
                logger.warning("export.analytic_step_write_failed", error=str(e))
    logger.info(
        "export.analytic_step",
        features=len(features or []),
        applied=len(applied),
        step=out["step"],
        warnings=len(warnings),
    )
    return out


def _base_solid_checkpoint(
    ctx: Any,
    mesh_path: Path,
    per_feature: float,
    max_faces: int | None,
    tdir: Path,
    warnings: list[str],
) -> str:
    """Sew the base solid once in a child process and return its BREP path.

    An empty string means "no checkpoint" — feature children then re-sew
    per cut (slow, but correct)."""
    import json as _json

    base_brep = tdir / "feat_base.brep"
    base_res = tdir / "feat_base.json"
    proc = ctx.Process(
        target=_analytic_base_worker,
        args=(str(mesh_path), str(base_brep), str(base_res), max_faces),
        daemon=True,
    )
    proc.start()
    proc.join(per_feature)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        warnings.append(
            f"base solid sew exceeded {per_feature:.0f}s — feature children will re-sew per cut"
        )
        return ""
    try:
        cres = _json.loads(base_res.read_text())
    except Exception:
        cres = {}
    warnings += [str(w) for w in cres.get("warnings", [])]
    if cres.get("applied") and cres.get("brep") and Path(cres["brep"]).exists():
        return str(cres["brep"])
    warnings.append("base solid checkpoint unavailable — feature children re-sew per cut")
    return ""


def _decimate_for_analytic(mesh: trimesh.Trimesh, logger: Any) -> trimesh.Trimesh:
    """Budget the mesh for OCC sewing/booleans.

    OCC boolean cost scales with face count; quadric decimation also
    collapses the sub-tolerance coincident vertices left by keep-restore
    unions. 12k facets keep sub-0.1mm fidelity on typical lattice skins
    while cutting sew/boolean time by ~4x. Quadric collapses can reopen the
    boundary — small holes are resealed cheaply (merge/fill); the voxel
    rebuild mesh_to_solid would run instead drifts past the guard on fine
    lattice skins. Falls back to the input mesh when the decimated result
    cannot be resealed."""
    try:
        from cadx.cad.mesh_ops import decimate

        if len(mesh.faces) <= 12000:
            return mesh
        dec = decimate(mesh, 12000)
        if not dec.is_watertight:
            dec.merge_vertices()
            dec.update_faces(dec.nondegenerate_faces())
            dec.remove_unreferenced_vertices()
            dec.fill_holes()
            dec.fix_normals()
        if dec.is_watertight or not mesh.is_watertight:
            return dec
        logger.warning(
            "export.analytic_step_decimate_dropped",
            faces_in=len(mesh.faces),
            faces_out=len(dec.faces),
        )
        return mesh
    except Exception:
        return mesh


def export_analytic_step(
    mesh: trimesh.Trimesh,
    features: list[Any],
    out_path: Path,
    *,
    max_faces: int | None = None,
    log: Any = None,
    budget_s: float | None = None,
) -> dict[str, Any]:
    """STEP of the optimized body with analytic (exact) feature surfaces.

    The trimesh is sewn to a B-Rep via ``mesh_ops.mesh_to_solid`` (its drift
    guard stays in force — a divergent rebuild ships no STEP at all), then
    every applied feature is re-cut with analytic OCC solids: true
    ``Cylinder`` faces for bores, a helical ``IsoThread`` groove for threads,
    cones/counterbore cylinders for countersink/counterbore. Functional
    surfaces come out exact instead of faceted.

    `features` are the applied ``RefineFeature`` objects (or dicts carrying
    the same fields — ``u``/``v`` may also come from ``pos``/``x``/``y``).
    Never raises: a failed sew or boolean falls back to the plain faceted
    STEP (or to no STEP when sewing itself failed) and records a warning.

    Returns ``{"step": filename|None, "step_features": [...], "warnings": [...]}``
    — ``step_features`` entries are ``{"name", "kind", "axis", "analytic": True}``
    for features that were cut analytically.
    """
    logger = log if log is not None else globals()["log"]
    out_path = Path(out_path)
    warnings: list[str] = []
    out: dict[str, Any] = {"step": None, "step_features": [], "warnings": warnings}
    try:
        import io

        # Normalise topology the same way the exported STL on disk is: an
        # STL roundtrip quantizes vertices to float32 and merges them by
        # position. In-memory refine output can carry duplicate vertices /
        # non-manifold junctions (keep-restore unions) that pass
        # ``is_watertight`` but sew into boolean-dead B-Reps — the exported
        # file of the same mesh cuts cleanly.
        mesh = trimesh.load(io.BytesIO(mesh.export(file_type="stl")), file_type="stl")
    except Exception:
        pass
    mesh = _decimate_for_analytic(mesh, logger)
    try:
        from cadx.cad.mesh_ops import mesh_to_solid

        solid = mesh_to_solid(mesh, max_faces=max_faces)
    except Exception as e:
        # honest skip beats a wrong STEP — the drift guard refused or the
        # mesh simply does not sew; the caller keeps step=None
        warnings.append(f"analytic STEP skipped — mesh-to-solid failed: {e}")
        logger.warning("export.analytic_step_sew_failed", error=str(e))
        return out

    normed: list[tuple[str, Any]] = []
    for raw in features or []:
        feat = _norm_refine_feature(raw)
        name = str(_feat_field(raw, "name", "") or _feat_field(raw, "kind", "feature"))
        if feat is None:
            warnings.append(f"feature '{name}': no usable refine pose — skipped")
            logger.warning("export.analytic_step_no_pose", feature=name)
            continue
        cutters = _analytic_cutters(feat, mesh)
        if not cutters:
            warnings.append(f"feature '{name}': no analytic cutter — skipped")
            logger.warning("export.analytic_step_no_cutter", feature=name)
            continue
        normed.append((name, feat))

    # one OCC boolean on a degenerate sewn solid can spin for tens of
    # minutes — bound the whole feature loop and degrade gracefully
    deadline = time.monotonic() + budget_s if budget_s else None
    solid, applied, failed, degraded = _cut_analytic_features(
        solid, mesh, normed, deadline=deadline
    )
    if failed and not applied and (deadline is None or time.monotonic() < deadline):
        # escalation: refine keep-restore unions leave float32-coincident
        # vertices at patch seams — watertight per trimesh but the sewn
        # B-Rep is boolean-dead near them. ``process(validate=True)`` splits
        # vertices at non-manifold junctions so OCC sewing re-joins by
        # position and the solid cuts cleanly. Re-cut ALL features on the
        # escalated solid so every analytic surface ends up in one body.
        esc = _validate_split_solid(mesh)
        if esc is not None:
            solid2, applied2, failed2, degraded2 = _cut_analytic_features(
                esc, mesh, normed, deadline=deadline
            )
            if len(applied2) > len(applied):
                logger.info(
                    "export.analytic_step_escalated",
                    applied=len(applied2),
                    failed=len(failed2),
                )
                solid, applied, failed, degraded = solid2, applied2, failed2, degraded2
    for name in failed:
        warnings.append(
            f"feature '{name}': analytic cut produced no valid solid — faceted STEP kept"
        )
        logger.warning("export.analytic_step_cut_empty", feature=name)
    for name, note in degraded.items():
        warnings.append(f"feature '{name}': {note}")
        logger.warning("export.analytic_step_degraded", feature=name, note=note)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _export_step_atomic(solid, out_path)
        out["step"] = out_path.name
        out["step_features"] = applied
    except Exception as e:
        warnings.append(f"analytic STEP write failed: {e}")
        logger.warning("export.analytic_step_write_failed", error=str(e))
    logger.info(
        "export.analytic_step",
        features=len(features or []),
        applied=len(applied),
        step=out["step"],
        warnings=len(warnings),
    )
    return out


def face_is_cylindrical_small(face: object, max_radius: float) -> bool:
    try:
        return bool(face.geom_type == GeomType.CYLINDER and face.radius < max_radius)  # type: ignore[attr-defined]
    except Exception:
        return False


def radians(deg: float) -> float:
    return deg * math.pi / 180.0
