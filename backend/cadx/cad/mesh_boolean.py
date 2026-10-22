"""Mesh-native booleans + B-Rep cleanup for faceted mesh solids.

Trusted runner-side helper — NOT part of cadlib's allowlisted import surface:
``runner_entry`` imports it (runner frames bypass the sandbox import guard)
and injects :func:`boolean` into ``cadlib.ops`` as the rescue backend for
``mesh_cut``/``mesh_fuse``/``mesh_intersect``/``trim_below``. It also carries
the deterministic anchor re-cut applied after generated code runs.

Why a mesh backend at all: OCC booleans on a 10-20k-facet sewn solid are
slow and fail in three distinct ways (``None`` results, sub-mm sliver debris,
full shatter). manifold3d does the same op in milliseconds and cannot
fragment the result. The cost: boolean outcomes are faceted — cutters lose
their analytic cylinder/cone surfaces, so the OCC path stays preferred and
this module is the rescue lane.

The fast path relies on ``_cadx_mesh`` attributes: ``load_mesh_solid()``
attaches the source trimesh to the imported solid and every boolean result
carries its manifold mesh forward — a chain of ``mesh_cut`` calls therefore
never tessellates the body (OCC tessellation of a sewn solid costs >2 min
at the 15k-face cap). Analytic tool shapes tessellate in milliseconds.

Deliberately no ``cadx`` imports: this module must also load inside the
sandboxed runner where only ``<cad>/`` is on ``sys.path``.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any, Literal

MESH_ATTR = "_cadx_mesh"

# tessellating an un-backed B-Rep is the slow fallback — a sewn solid near
# the cap costs minutes inside the sandbox, so past this the backend refuses
# and the caller falls back to the OCC op instead
MAX_FACES_FOR_TESS = 20_000

_BooleanKind = Literal["cut", "fuse", "intersect"]


def _warmup() -> None:
    """Pull every lazily-imported dependency into sys.modules AND smoke-test
    the full pipeline once — required inside the sandbox: the import guard
    blocks any deferred import once model.py frames sit on the call stack,
    so trimesh's internal submodules (boolean/repair/exchange) and the OCP
    wrappers must be resolved here while only runner frames are live."""
    import trimesh.boolean
    import trimesh.ray
    import trimesh.repair  # noqa: F401
    from build123d import Box
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing  # noqa: F401
    from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter  # noqa: F401
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt  # noqa: F401
    from OCP.ShapeFix import ShapeFix_Shape  # noqa: F401
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain  # noqa: F401
    from OCP.StlAPI import StlAPI_Reader  # noqa: F401

    boolean(Box(2, 2, 2), Box(1, 1, 1), "cut")  # tessellate→manifold→sew→unify


def _as_volume(mesh: Any) -> Any:
    """Weld + repair a tessellation into a manifold volume, or raise.

    OCC tessellations of sewn solids have hairline cracks at sew-tolerance
    gaps (vertices coincident within ~1e-4 but not shared) — merge_vertices
    fixes almost all of them; the rest get normals/fill_holes before giving
    up so the caller can fall back to OCC."""
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if mesh.is_volume:
        return mesh
    with contextlib.suppress(Exception):
        mesh.fix_normals()
    with contextlib.suppress(Exception):
        import trimesh

        trimesh.repair.fill_holes(mesh)
    if not mesh.is_volume:
        raise ValueError("tessellation is not a closed volume")
    return mesh


def shape_to_mesh(shape: Any, *, linear_tol: float = 0.02, angular_tol: float = 0.1) -> Any:
    """Tessellate a B-Rep shape into a manifold trimesh volume."""
    import numpy as np
    import trimesh

    if len(shape.faces()) > MAX_FACES_FOR_TESS:
        raise ValueError(
            f"shape has {len(shape.faces())} faces — OCC tessellation too slow; "
            "keep the mesh-backed chain (mesh_cut/mesh_fuse) instead"
        )
    verts, tris = shape.tessellate(linear_tol, angular_tol)
    tm = trimesh.Trimesh(
        vertices=np.asarray([(v.X, v.Y, v.Z) for v in verts], dtype=np.float64),
        faces=np.asarray(tris, dtype=np.int64),
        process=False,
    )
    if not len(tm.faces):
        raise ValueError("tessellation produced no triangles")
    return _as_volume(tm)


def _operand_mesh(shape: Any) -> Any:
    """Reuse the manifold mesh a previous boolean attached, else tessellate."""
    attached = getattr(shape, MESH_ATTR, None)
    if attached is not None and len(getattr(attached, "faces", [])):
        return attached
    return shape_to_mesh(shape)


def attach_mesh(shape: Any, mesh: Any) -> Any:
    """Tag a shape with its authoritative trimesh backing for fast booleans."""
    with contextlib.suppress(Exception):
        setattr(shape, MESH_ATTR, mesh)
    return shape


def sew_mesh(mesh: Any) -> Any:
    """Watertight trimesh → build123d Solid via STL + OCCT sewing.

    Same pipeline as ``mesh_ops.mesh_to_solid``'s tail — kept duplicated
    (self-contained) because this module must import inside the sandbox where
    ``cadx.cad.mesh_ops`` is not reachable. Raises ValueError on failure.
    """
    import numpy as np
    from build123d import Shape, Shell, Solid
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS_Shape

    # Drop sliver facets: manifold boolean output keeps triangles whose
    # vertices quantize to nearly the same float32 point — a sub-1e-5 mm
    # edge makes OCC's STL reader abort the whole file ("command not done").
    # Removing them only opens sub-tolerance gaps the sewing tolerance covers.
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    if len(tri):
        e0 = tri[:, 0] - tri[:, 1]
        e1 = tri[:, 1] - tri[:, 2]
        e2 = tri[:, 2] - tri[:, 0]
        min_edge = np.minimum(
            np.linalg.norm(e0, axis=1),
            np.minimum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)),
        )
        keep = min_edge > 1e-5
        if not keep.all():
            mesh = mesh.copy()
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()

    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "mesh.stl"
        mesh.export(str(stl), file_type="stl")
        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        if not reader.Read(shape, str(stl)):
            raise ValueError("StlAPI_Reader failed")
        sewing = BRepBuilderAPI_Sewing(1e-4)
        sewing.Add(shape)
        sewing.Perform()
        fixed = ShapeFix_Shape(sewing.SewedShape()).Shape()
        wrapped = Shape.cast(fixed)
        shells = wrapped.shells()
        if not shells:
            raise ValueError("no closed shell found in mesh")
        shell = max(shells, key=lambda sh: sh.area)
        if not isinstance(shell, Shell):
            shell = Shell(shell.wrapped)
        shell = shell.fix()
    return Solid(Solid._make_solid(shell))


def boolean(a: Any, b: Any, kind: _BooleanKind) -> Any:
    """manifold3d boolean between two build123d shapes → sewn Solid.

    Raises on any failure — callers (cadlib mesh_* ops) keep the OCC result
    or their degraded fallback when this raises. The returned solid carries
    its manifold mesh on ``_cadx_mesh`` so the next boolean in a chain skips
    tessellation entirely.
    """
    import trimesh

    am = _operand_mesh(a)
    bm = _operand_mesh(b)
    if kind == "cut":
        res = am.difference(bm, engine="manifold")
    elif kind == "fuse":
        res = trimesh.boolean.union([am, bm], engine="manifold")
    else:
        res = am.intersection(bm, engine="manifold")
    if res is None or not len(getattr(res, "faces", [])):
        raise ValueError(f"manifold {kind} produced an empty mesh")
    out = sew_mesh(res)
    with contextlib.suppress(Exception):
        out = unify_same_domain(out)
    return attach_mesh(out, res)


def unify_same_domain(shape: Any) -> Any:
    """Merge coplanar faces / colinear edges of a (possibly faceted) solid.

    A flat plate sewn from triangles keeps 2 faces per side after this —
    subsequent OCC booleans get dramatically cheaper and the STEP export
    carries real planar faces instead of facet noise. Raises on failure;
    callers validate volume/solids before adopting the result.
    """
    from build123d import Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    unifier = ShapeUpgrade_UnifySameDomain(shape.wrapped, True, True, False)
    unifier.SetSafeInputMode(True)
    unifier.Build()
    res = Shape.cast(unifier.Shape())
    if not len(res.solids()):
        raise ValueError("unify produced no solids")
    return res


def unify_checked(shape: Any, *, vol_tol: float = 0.02) -> Any:
    """unify_same_domain with a correctness gate — returns input on any
    suspicion (volume drift > vol_tol, lost solids, exceptions)."""
    try:
        res = unify_same_domain(shape)
        if abs(res.volume - shape.volume) > max(1.0, abs(shape.volume) * vol_tol):
            return shape
        if len(res.solids()) != len(shape.solids()):
            return shape
        # unify only merges coplanar faces — geometry is identical, so the
        # attached manifold mesh stays exact and saves a re-tessellation
        src = getattr(shape, MESH_ATTR, None)
        if src is not None:
            with contextlib.suppress(Exception):
                setattr(res, MESH_ATTR, src)
        return res
    except Exception:
        return shape


def _axis_vec(axis: str) -> tuple[int, tuple[float, float, float]]:
    """axis label ('+z', '-x', ...) → (axis index, unit direction)."""
    idx = {"x": 0, "y": 1, "z": 2}[axis[-1]]
    sign = -1.0 if axis[0] == "-" else 1.0
    vec = [0.0, 0.0, 0.0]
    vec[idx] = sign
    return idx, (vec[0], vec[1], vec[2])


def _plane_coords(hole: dict[str, Any], idx: int) -> tuple[float, float] | None:
    """(u, v) position of the bore on the plane perpendicular to its axis.

    +z/-z anchors carry x/y directly; side anchors carry the full in-plane
    position in `pos` ([x, y, z] with the axial entry possibly None)."""
    if idx == 2:
        return float(hole["x"]), float(hole["y"])
    pos = hole.get("pos")
    if not pos or len(pos) != 3:
        return None
    others = [i for i in range(3) if i != idx]
    u, v = pos[others[0]], pos[others[1]]
    if u is None or v is None:
        return None
    return float(u), float(v)


def _surface_hits(shape: Any, origin: Any, direction: tuple[float, float, float]) -> list[float]:
    """Line-vs-B-Rep intersections; returns the ray parameter t of every hit
    (t = distance from `origin` along the unit `direction`)."""
    from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
    from OCP.gp import gp_Dir, gp_Lin

    inter = BRepIntCurveSurface_Inter()
    inter.Init(shape.wrapped, gp_Lin(origin, gp_Dir(*direction)), 1e-6)
    ts: list[float] = []
    while inter.More():
        p = inter.Pnt()
        ts.append(
            float(
                (p.X() - origin.X()) * direction[0]
                + (p.Y() - origin.Y()) * direction[1]
                + (p.Z() - origin.Z()) * direction[2]
            )
        )
        inter.Next()
    return ts


def tunnel_open(
    shape: Any,
    u: float,
    v: float,
    axis: str,
    axial_span: tuple[float, float] | None = None,
) -> bool:
    """True when a ray through (u, v) along `axis` meets NO material — i.e. a
    through-tunnel exists at the anchor position. Any surface hit means the
    nominal bore is missing, blocked or undersized.

    ``axial_span`` optionally restricts the check to the axial interval a
    bounded bore must pierce — an anchor carrying its declared axial centre
    (``pos[idx]``) and ``depth_mm`` only needs [pos-depth/2, pos+depth/2]
    free; material further along the same ray (a strut or a second wall
    beyond the declared host plate) is not a blockage and must not trigger
    a re-cut that pierces it.

    Two probes: solids carrying their source trimesh use `ray.intersects_-
    location` (the OCC ray on 30k-facet bodies is slow and reports phantom
    hits); analytic bodies use `is_inside` sampling — the OCC ray reports
    grazing hits at the far face plane on drilled analytical bores."""
    idx, direction = _axis_vec(axis)
    bb = shape.bounding_box()
    mins = [bb.min.X, bb.min.Y, bb.min.Z]
    maxs = [bb.max.X, bb.max.Y, bb.max.Z]
    span = maxs[idx] - mins[idx]
    margin = 50.0
    others = [i for i in range(3) if i != idx]
    # ray starts outside on the axis-sign face and travels INTO the part:
    # +z probes come down from above the bbox, -z probes come up from below
    start = (mins[idx] - margin) if axis[0] == "-" else (maxs[idx] + margin)
    d = tuple(-c for c in direction)
    span_lo = min(axial_span) - 1e-4 if axial_span is not None else None
    span_hi = max(axial_span) + 1e-4 if axial_span is not None else None

    mesh = getattr(shape, MESH_ATTR, None)
    if mesh is not None:
        origin = [0.0, 0.0, 0.0]
        origin[others[0]], origin[others[1]] = u, v
        origin[idx] = start
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[list(d)],
            multiple_hits=True,
        )
        for p in locs:
            t = (
                (float(p[0]) - origin[0]) * d[0]
                + (float(p[1]) - origin[1]) * d[1]
                + (float(p[2]) - origin[2]) * d[2]
            )
            if margin - 1e-4 < t < margin + span + 1e-4:
                if (
                    span_lo is not None
                    and span_hi is not None
                    and not (span_lo <= float(p[idx]) <= span_hi)
                ):
                    continue  # material outside the declared bore interval
                return False
        return True

    # analytic: sample interior points — thin plates must still count, so
    # the pitch adapts to the span (~2 mm resolution, 7 samples minimum).
    # With an axial_span only the declared bore interval is probed; an
    # interval that misses the part entirely reports open.
    lo, hi = mins[idx], maxs[idx]
    if axial_span is not None:
        lo = max(lo, min(axial_span))
        hi = min(hi, max(axial_span))
    probe_span = hi - lo
    if probe_span <= 0:
        return True
    n = max(7, int(probe_span / 2))
    for i in range(1, n + 1):
        p = [0.0, 0.0, 0.0]
        p[others[0]], p[others[1]] = u, v
        p[idx] = lo + probe_span * i / (n + 1)
        if shape.is_inside(tuple(p)):
            return False
    return True


def _pair_hits(ts: list[float], tol: float = 0.2) -> list[tuple[float, float]]:
    """Sorted surface hits → material intervals (entry, exit) pairs.

    Faceted bodies report grazing duplicates at seam/tangent facets — hits
    closer than `tol` collapse into one crossing. A trailing unpaired hit
    (ray exiting at the bbox boundary) is dropped."""
    ts = sorted(ts)
    deduped: list[float] = []
    for t in ts:
        if deduped and t - deduped[-1] < tol:
            continue
        deduped.append(t)
    return [(deduped[i], deduped[i + 1]) for i in range(0, len(deduped) - 1, 2)]


def _material_intervals(shape: Any, u: float, v: float, axis: str) -> list[tuple[float, float]]:
    """Continuous material intervals along the axis ray through (u, v), in
    part-frame coordinates (entry, exit) — trimesh path for solids carrying
    their source mesh, `is_inside` sampling otherwise."""
    idx, direction = _axis_vec(axis)
    bb = shape.bounding_box()
    mins = [bb.min.X, bb.min.Y, bb.min.Z]
    maxs = [bb.max.X, bb.max.Y, bb.max.Z]
    span = maxs[idx] - mins[idx]
    if span <= 0:
        return []
    margin = 50.0
    others = [i for i in range(3) if i != idx]
    start = (mins[idx] - margin) if axis[0] == "-" else (maxs[idx] + margin)
    d = tuple(-c for c in direction)

    mesh = getattr(shape, MESH_ATTR, None)
    if mesh is not None:
        origin = [0.0, 0.0, 0.0]
        origin[others[0]], origin[others[1]] = u, v
        origin[idx] = start
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[list(d)],
            multiple_hits=True,
        )
        ts: list[float] = []
        for p in locs:
            t = (
                (float(p[0]) - origin[0]) * d[0]
                + (float(p[1]) - origin[1]) * d[1]
                + (float(p[2]) - origin[2]) * d[2]
            )
            if margin - 1e-4 < t < margin + span + 1e-4:
                ts.append(start + d[idx] * t)  # axial coord of the hit
        return _pair_hits(ts)

    n = max(7, int(span / 2))
    edges: list[tuple[float, float]] = []
    lo: float | None = None
    for i in range(1, n + 1):
        p = [0.0, 0.0, 0.0]
        p[others[0]], p[others[1]] = u, v
        a = mins[idx] + span * i / (n + 1)
        p[idx] = a
        try:
            cur = bool(shape.is_inside(tuple(p)))
        except Exception:
            cur = False
        if cur and lo is None:
            lo = a
        elif not cur and lo is not None:
            edges.append((lo, a))
            lo = None
    if lo is not None:
        edges.append((lo, mins[idx] + span))
    return edges


def _cut_span(
    hole: dict[str, Any],
    idx: int,
    intervals: list[tuple[float, float]],
    lo_decl: float | None,
) -> tuple[float, float] | None:
    """(lo, hi) of the material interval the cutter must pierce for a
    through-bore — the interval containing the anchor's declared axial
    centre, else the interval nearest it (a separate strut on the same ray
    is deliberately left uncut)."""
    if not intervals:
        return None
    if lo_decl is not None:
        for lo, hi in intervals:
            if lo - 0.25 <= lo_decl <= hi + 0.25:
                return lo, hi
        return min(intervals, key=lambda iv: abs((iv[0] + iv[1]) / 2 - lo_decl))
    # no declared centre: the plate the bore ENTERS through — the axis-sign
    # face (+z enters from the top, -z from the bottom)
    if len(intervals) == 1:
        return intervals[0]
    neg = str(hole.get("axis") or "+z")[0] == "-"
    return min(intervals, key=lambda iv: iv[0]) if neg else max(intervals, key=lambda iv: iv[1])


def _bore_cutter(
    hole: dict[str, Any],
    idx: int,
    u: float,
    v: float,
    bb: Any,
    margin: float,
    material_span: tuple[float, float] | None = None,
) -> Any:
    """Positioned cylinder cutter along the anchor axis through the bbox.

    Blind anchors (`blind` flag or `depth_mm` from a declared bore depth)
    cut only the stated depth from the entry face. Host-span-bounded
    through-anchors keep the plate's axial centre but pierce the ACTUAL
    material interval there (`material_span`) — a mesh pad thicker than
    the declared plate depth must still yield a through-bore."""
    from build123d import Align, Axis, Cylinder, Pos

    mins = [bb.min.X, bb.min.Y, bb.min.Z]
    maxs = [bb.max.X, bb.max.Y, bb.max.Z]
    depth = hole.get("depth_mm")
    pos = hole.get("pos")
    axial = (
        pos[idx]
        if isinstance(pos, (list, tuple)) and len(pos) == 3 and pos[idx] is not None
        else None
    )
    if depth:
        if (
            not hole.get("blind")
            and material_span is not None
            and material_span[1] - material_span[0] > float(depth) + 1e-3
        ):
            span = (material_span[1] - material_span[0]) + 2 * margin
            lo = material_span[0] - margin
        elif axial is not None:
            # the anchor carries the bore's axial centre (spec anchors:
            # host-plate mid) — centre the cutter on it so a blind/bounded
            # bore lands on its plate instead of the envelope's axis-sign
            # face, which can sit 100mm away on a multi-body part
            span = float(depth) + margin
            lo = float(axial) - span / 2
        else:
            # blind bore entering from the axis-sign face — the cutter
            # (always aligned +axis) overhangs ABOVE the entry face only;
            # the blind bottom must land exactly at the stated depth
            neg = str(hole.get("axis", "+"))[0] == "-"
            entry = mins[idx] if neg else maxs[idx]
            span = float(depth) + margin / 2
            lo = entry - margin / 2 if neg else entry - float(depth)
    else:
        span = (maxs[idx] - mins[idx]) + 2 * margin
        lo = mins[idx] - margin
    cyl = Cylinder(
        radius=float(hole["dia"]) / 2,
        height=span,
        # base-anchored: pos marks the LOW end of the cutter along its axis
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    if idx == 0:
        cyl = cyl.rotate(Axis.Y, 90)
    elif idx == 1:
        cyl = cyl.rotate(Axis.X, -90)
    pos = [0.0, 0.0, 0.0]
    pos[idx] = lo
    others = [i for i in range(3) if i != idx]
    pos[others[0]], pos[others[1]] = u, v
    return Pos(*pos) * cyl


def _recut_span(
    h: dict[str, Any], shape: Any, idx: int, u: float, v: float
) -> tuple[float, float] | None:
    """Material interval a bounded through-cutter must pierce (None = keep
    the declared depth bound)."""
    if not h.get("depth_mm") or h.get("blind"):
        return None
    try:
        pos = h.get("pos")
        axial = pos[idx] if isinstance(pos, (list, tuple)) and len(pos) == 3 else None
        return _cut_span(
            h, idx, _material_intervals(shape, u, v, str(h.get("axis") or "+z")), axial
        )
    except Exception:
        return None


def _declared_axial_span(h: dict[str, Any], idx: int) -> tuple[float, float] | None:
    """(lo, hi) axial interval a bounded anchor bore must occupy — its
    declared axial centre ``pos[idx]`` ± ``depth_mm``/2. ``None`` for plain
    through anchors (no declared centre/depth): the whole-bbox tunnel check
    stays in place for them."""
    depth = h.get("depth_mm")
    pos = h.get("pos")
    if depth is None or not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return None
    try:
        axial = float(pos[idx])
        half = float(depth) / 2.0
    except (TypeError, ValueError):
        return None
    if half <= 0:
        return None
    return (axial - half, axial + half)


def recut_missing_holes(
    shape: Any,
    holes: list[dict[str, Any]],
    *,
    margin: float = 5.0,
) -> tuple[Any, list[str]]:
    """Deterministically re-cut anchored bores the generated code missed.

    For every anchor whose ray finds material, a cylinder cutter is applied —
    OCC first (keeps analytic bore walls when the body is analytic), the
    manifold mesh backend as rescue for faceted bodies. Returns
    (shape, applied descriptions); the input shape is returned unchanged when
    nothing needed cutting or every attempt failed.
    """
    from build123d import Compound

    blocked: list[tuple[dict[str, Any], int, float, float]] = []
    for h in holes:
        try:
            if str(h.get("kind") or "hole") == "dimple":
                continue
            # only authoritative measurements may force a cut — mesh-derived
            # and non-authoritative ("image-approx") anchors are hints the
            # critique handles, not features worth drilling blindly. Anchors
            # sourced from the spec ("spec", "mesh+spec", ...) carry declared
            # intent: a spec bore missing from geometry MUST be cut.
            src = str(h.get("source") or "")
            if src and "spec" not in src and src not in ("image", "image+mesh"):
                continue
            if float(h.get("dia") or 0.0) <= 0:
                continue
            axis = str(h.get("axis") or "+z")
            idx, _dir = _axis_vec(axis)
            uv = _plane_coords(h, idx)
            if uv is None:
                continue
            # a bounded anchor only owes an open tunnel across its declared
            # bore interval — a body further along the same ray is not a
            # blockage and must not trigger a re-cut that pierces it
            if tunnel_open(
                shape, uv[0], uv[1], axis, _declared_axial_span(h, idx)
            ):
                continue
            blocked.append((h, idx, uv[0], uv[1]))
        except Exception:
            continue
    if not blocked:
        return shape, []

    bb = shape.bounding_box()
    cutters = [
        _bore_cutter(h, idx, u, v, bb, margin, _recut_span(h, shape, idx, u, v))
        for h, idx, u, v in blocked
    ]
    labels = [f"Ø{float(h['dia']):g}@{h.get('axis', '+z')}" for h, _idx, _u, _v in blocked]

    # OCC first — on a mostly-analytic result the bore walls stay analytic
    try:
        tool = cutters[0] if len(cutters) == 1 else Compound(children=list(cutters))
        res = shape - tool
        if res is not None and res.solids() and res.volume >= shape.volume * 0.3:
            return res, labels
    except Exception:
        pass

    # manifold rescue: all cutters on the mesh level, sew back once
    try:
        tm = _operand_mesh(shape)
        for c in cutters:
            res_m = tm.difference(_operand_mesh(c), engine="manifold")
            if res_m is not None and len(getattr(res_m, "faces", [])):
                tm = res_m
        out = sew_mesh(tm)
        with contextlib.suppress(Exception):
            out = unify_same_domain(out)
        if out.solids() and out.volume >= shape.volume * 0.3:
            return attach_mesh(out, tm), labels
    except Exception:
        pass

    return shape, []
