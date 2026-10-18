"""Export a build123d Shape to all supported formats. Each format is independent —
a single failing export never fails the job; the error is recorded instead."""

from __future__ import annotations

import math
import time
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
) -> tuple[Any, list[dict[str, Any]], list[str], dict[str, str]]:
    """Cut every normed feature's analytic tools out of `solid` in sequence.

    Cutters are applied one at a time (pilot bore first, then thread
    groove/counters) — a single Compound tool makes the boolean far more
    fragile on sewn meshes, and a failed groove must not cost the pilot
    bore's analytic cylinder. Returns ``(solid, applied, failed_names,
    degraded)`` — degraded maps feature name → which sub-cutter failed;
    those features keep their analytic pilot but get an honest note."""
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


def export_analytic_step(
    mesh: trimesh.Trimesh,
    features: list[Any],
    out_path: Path,
    *,
    max_faces: int | None = None,
    log: Any = None,
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
    try:
        # OCC boolean cost scales with face count; quadric decimation also
        # collapses the sub-tolerance coincident vertices left by keep-restore
        # unions. 12k facets keep sub-0.1mm fidelity on typical lattice skins
        # while cutting sew/boolean time by ~4x.
        from cadx.cad.mesh_ops import decimate

        if len(mesh.faces) > 12000:
            mesh = decimate(mesh, 12000)
    except Exception:
        pass
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

    solid, applied, failed, degraded = _cut_analytic_features(solid, mesh, normed)
    if failed and not applied:
        # escalation: refine keep-restore unions leave float32-coincident
        # vertices at patch seams — watertight per trimesh but the sewn
        # B-Rep is boolean-dead near them. ``process(validate=True)`` splits
        # vertices at non-manifold junctions so OCC sewing re-joins by
        # position and the solid cuts cleanly. Re-cut ALL features on the
        # escalated solid so every analytic surface ends up in one body.
        esc = _validate_split_solid(mesh)
        if esc is not None:
            solid2, applied2, failed2, degraded2 = _cut_analytic_features(esc, mesh, normed)
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
        export_step(solid, str(out_path))
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
