"""cadlib ops — defensive wrappers around raw build123d operations.

These don't create or cut a feature of their own; they make a fragile OCC
call graceful. ``safe_fillet`` retries with a shrinking radius instead of
crashing the whole run on an empty/stale edge selection.
"""

import math
from typing import Sequence  # noqa: UP035 — keep cadlib's sandbox import allow-list

from build123d import Align, Box, BuildPart, Edge, GeomType, Pos, Shape, chamfer, fillet

# runner_entry injects the manifold3d backend (`cadx.cad.mesh_boolean`) for
# hybrid runs — kept as an injected callable so cadlib itself never imports
# trimesh/manifold3d/OCP (the sandbox guard forbids it).
_BOOLEAN_BACKEND = None

# A boolean that moved less than this fraction of min(body, tool) volume is a
# silent no-op, not a feature. 1e-4 keeps real small cuts (chamfers, shallow
# pockets — typically ≥1% of the tool) passing while a grazing touch that
# shaves ~0.01 mm^3 still raises.
_NOOP_DELTA_FRACTION = 1e-4


def set_boolean_backend(fn) -> None:
    """Install a ``fn(body, tool, kind) -> Shape`` mesh-native boolean used as
    the rescue lane when an OCC boolean on a faceted solid fails, returns
    ``None`` or sheds wreckage. ``kind`` is ``"cut"``, ``"fuse"`` or
    ``"intersect"``."""
    global _BOOLEAN_BACKEND
    _BOOLEAN_BACKEND = fn


def _backend_boolean(body, tool, kind: str):
    """Backend boolean or None — unavailable/injected-off/failed all fold to
    the OCC fallback path in the caller."""
    if _BOOLEAN_BACKEND is None:
        return None
    try:
        r = _BOOLEAN_BACKEND(body, tool, kind)
    except Exception:
        return None
    if r is None or not r.solids():
        return None
    return r


def _bboxes_disjoint(body: Shape, tool: Shape) -> bool:
    """True when the bounding boxes share no volume — a cheap reject that runs
    before any boolean so a tool placed outside the part (the classic
    "offsets from an end face mistaken for absolute coordinates" bug) fails
    immediately instead of passing every downstream volume check."""
    bb = body.bounding_box()
    tb = tool.bounding_box()
    return (
        tb.max.X <= bb.min.X
        or tb.min.X >= bb.max.X
        or tb.max.Y <= bb.min.Y
        or tb.min.Y >= bb.max.Y
        or tb.max.Z <= bb.min.Z
        or tb.min.Z >= bb.max.Z
    )


def _noop_delta_eps(body: Shape, tool: Shape) -> float:
    """Volume change below which a "successful" boolean is a silent no-op.

    ``min`` of the two volumes because a legitimate tool can be mostly
    outside the body (a long bore cutter through a thin wall) or mostly
    inside it — the smaller volume bounds the most a real feature can move.
    """
    return max(1e-6, min(abs(tool.volume), body.volume) * _NOOP_DELTA_FRACTION)


def _require_delta(body: Shape, tool: Shape, result: Shape, kind: str) -> None:
    """Guard a boolean's success path: a result that passes the floor check
    but moved less material than :func:`_noop_delta_eps` is a silent no-op
    (grazing cut, engulfed fuse) — raise so the codegen repair loop gets an
    actionable error instead of shipping a part missing the feature."""
    if kind == "cut":
        delta = body.volume - result.volume
        why = "removed"
        hint = "the tool only grazes the body — check the feature's position and depth"
    else:
        delta = result.volume - body.volume
        why = "added"
        hint = "the tool is engulfed by the body or only grazes it — check its position/size"
    if delta <= _noop_delta_eps(body, tool):
        raise ValueError(
            f"mesh_{kind} {why} no material — {hint}; the part is centred at "
            "origin, so 'N mm from an end' is not an absolute coordinate"
        )


def bbox_dims(shape) -> list[float]:
    """[dx, dy, dz] extents of a shape's bounding box in mm.

    Generated code kept guessing OCP-style BoundBox attributes
    (``xlen``, ``xmax``, ``bb.size.x`` — all invalid in build123d);
    this helper removes the need to touch the BoundBox API at all.
    """
    s = shape.bounding_box().size
    return [float(s.X), float(s.Y), float(s.Z)]


def largest_solid(shape):
    """Highest-volume solid of ``shape`` — itself when already single."""
    solids = shape.solids()
    if not solids:
        raise ValueError("shape contains no solids")
    return max(solids, key=lambda s: s.volume)


def collapse_solids(shape):
    """Collapse a fragmented boolean result to its dominant solid.

    Boolean ops on faceted mesh solids (``load_mesh_solid()``) can shed
    sub-mm slivers: a 28-solid compound whose largest fragment holds >90%
    of the volume. Keeping the dominant solid recovers the intended part;
    the slivers are boolean debris, not features.
    """
    solids = shape.solids()
    if len(solids) <= 1:
        return shape
    return largest_solid(shape)


_FILLET_MAX_EDGES = 300


def safe_fillet(edges, radius: float, min_radius: float = 0.2) -> float:
    """Fillet ``edges``, halving the radius until OCC accepts it.

    build123d raises ``ValueError`` both when the edge selection is empty
    (stale list after ``locate``/``translate``, overly strict filter) and when
    the radius does not fit. Both cases are retried with a smaller radius —
    an empty selection never fits, so it short-circuits to ``0.0`` without
    burning OCC calls. Faceted/frozen TopoDS shapes raise OCC-layer
    exceptions (``Standard_Failure``/``RuntimeError``), so the retry catches
    broadly instead of just ``ValueError``.

    Returns the applied radius, or ``0.0`` when nothing could be filleted —
    the caller can then fall back (e.g. keep sharp edges) instead of dying.

        fillet_r = safe_fillet(p.edges().filter_by(Axis.Z), radius=3)
        if not fillet_r:
            pass  # no fillet applied — part stays valid
    """
    try:
        n_edges = len(edges)
    except TypeError:
        n_edges = 0
    if n_edges == 0:
        # fast path — the shrink loop would still burn one OCC fillet call
        # per halving step on a selection that can never accept a radius
        return 0.0
    if BuildPart._get_context() is None:
        raise RuntimeError(
            "safe_fillet() only edits an active BuildPart and returns the applied radius, "
            "not a Shape; use safe_fillet_shape(shape, edges, radius) in algebra mode"
        )
    if n_edges > _FILLET_MAX_EDGES:
        # a selection this large is a faceted mesh body — OCC grinds on
        # thousands of triangle edges for minutes (live hang: >480s pipeline
        # timeout). Skip immediately; filleting belongs on the analytic
        # sub-part BEFORE mesh_fuse, or on a chamfer-in-mesh workaround.
        print(
            f"safe_fillet: skipped — selection holds {n_edges} edges "
            f"(limit {_FILLET_MAX_EDGES}); faceted mesh bodies cannot be "
            "filleted — fillet the analytic sub-part before mesh_fuse instead"
        )
        return 0.0
    r = float(radius)
    while r >= min_radius:
        try:
            fillet(edges, radius=r)
            return r
        except Exception:
            # faceted/frozen TopoDS shapes raise Standard_Failure / RuntimeError
            # from the OCC layer, not ValueError — a radius retry may still
            # succeed, so every kernel-side failure folds into the shrink loop
            r = round(r / 2, 3)
    return 0.0


def safe_fillet_shape(
    shape: Shape, edges: Sequence[Edge], radius: float, min_radius: float = 0.2
) -> tuple[Shape, float]:
    """Functionally fillet an algebra-mode shape and return ``(shape, radius)``.

    ``safe_fillet`` is a BuildPart operation: outside a builder, build123d's
    ``fillet`` returns a new Shape. This companion preserves that return value
    so code cannot mistake a successful radius for a mutated input shape.
    ``edges`` must be selected from ``shape`` before calling this function.
    """
    try:
        n_edges = len(edges)
    except TypeError:
        n_edges = 0
    if n_edges == 0 or n_edges > _FILLET_MAX_EDGES:
        return shape, 0.0
    if BuildPart._get_context() is not None:
        raise RuntimeError(
            "safe_fillet_shape() is for algebra mode outside BuildPart; "
            "use safe_fillet(edges, radius) inside the builder"
        )
    r = float(radius)
    while r >= min_radius:
        try:
            result = fillet(edges, radius=r)
            if isinstance(result, Shape) and result.solids():
                return result, r
        except Exception:
            pass
        r = round(r / 2, 3)
    return shape, 0.0


def safe_chamfer(edges, length: float, min_length: float = 0.2) -> float:
    """Chamfer ``edges``, halving the leg length until OCC accepts it.

    Same shrink-to-fit contract as :func:`safe_fillet`: an empty selection
    short-circuits to ``0.0``, oversized selections on faceted bodies are
    refused, and every kernel-side failure folds into the retry loop.
    Returns the applied length — ``0.0`` when nothing could be chamfered,
    so the caller can fall back instead of dying.

        chamfer_l = safe_chamfer(concave_edges(axis="y"), length=2)
    """
    try:
        n_edges = len(edges)
    except TypeError:
        n_edges = 0
    if n_edges == 0:
        return 0.0
    if BuildPart._get_context() is None:
        raise RuntimeError(
            "safe_chamfer() only edits an active BuildPart and returns the applied "
            "length, not a Shape; use safe_chamfer_shape(shape, edges, length) "
            "in algebra mode"
        )
    if n_edges > _FILLET_MAX_EDGES:
        print(
            f"safe_chamfer: skipped — selection holds {n_edges} edges "
            f"(limit {_FILLET_MAX_EDGES}); faceted mesh bodies cannot be "
            "chamfered — chamfer the analytic sub-part before mesh_fuse instead"
        )
        return 0.0
    r = float(length)
    while r >= min_length:
        try:
            chamfer(edges, length=r)
            return r
        except Exception:
            r = round(r / 2, 3)
    return 0.0


def safe_chamfer_shape(
    shape: Shape, edges: Sequence[Edge], length: float, min_length: float = 0.2
) -> tuple[Shape, float]:
    """Functionally chamfer an algebra-mode shape and return ``(shape, length)``.

    Companion to :func:`safe_chamfer` outside a ``BuildPart`` — see
    :func:`safe_fillet_shape` for the contract. ``edges`` must be selected
    from ``shape`` before calling this function.
    """
    try:
        n_edges = len(edges)
    except TypeError:
        n_edges = 0
    if n_edges == 0 or n_edges > _FILLET_MAX_EDGES:
        return shape, 0.0
    if BuildPart._get_context() is not None:
        raise RuntimeError(
            "safe_chamfer_shape() is for algebra mode outside BuildPart; "
            "use safe_chamfer(edges, length) inside the builder"
        )
    r = float(length)
    while r >= min_length:
        try:
            result = chamfer(edges, length=r)
            if isinstance(result, Shape) and result.solids():
                return result, r
        except Exception:
            pass
        r = round(r / 2, 3)
    return shape, 0.0


def _unwrap_shape(shape_or_builder=None):
    """Resolve ``shape_or_builder`` to a concrete :class:`Shape`.

    Accepts a Shape/Part (returned as-is), a ``BuildPart`` builder (its
    ``.part``), or ``None`` — which reads the active BuildPart context's
    current part, so helpers work bare inside ``with BuildPart() as p:``.
    Returns whatever it was given when nothing resolves further.
    """
    part = getattr(shape_or_builder, "part", None)
    if part is not None:
        return part
    if shape_or_builder is None:
        ctx = BuildPart._get_context()
        if ctx is not None:
            return ctx.part
    return shape_or_builder


def _edge_kind(shape: Shape, edge: Edge, eps: float = 0.25) -> str:
    """'convex' | 'concave' | 'tangent' — classify an edge without any OCP
    convexity API (the sandbox allow-list forbids it).

    Probes a ring of ``eps``-offset points in the plane perpendicular to the
    edge's mid-tangent against ``shape.is_inside``: a re-entrant edge (inner
    fillet root, groove corner, L-bracket inner join) is buried in material
    on most sides (>=5 of 8 probes); a proud outer edge has material on few
    (<=3); anything in between is a tangent/smooth join.
    """
    try:
        p = edge.position_at(0.5)
        t = edge.tangent_at(0.5)
        tn = math.sqrt(t.X * t.X + t.Y * t.Y + t.Z * t.Z)
        if tn <= 1e-9:
            return "tangent"
        tx, ty, tz = t.X / tn, t.Y / tn, t.Z / tn
        # stable in-plane basis (u, w) perp. t: z-up as reference unless
        # the tangent is near-vertical, where t cross z degenerates
        rx, ry, rz = (0.0, 0.0, 1.0) if abs(tz) <= 0.9 else (0.0, 1.0, 0.0)
        ux, uy, uz = (ty * rz - tz * ry, tz * rx - tx * rz, tx * ry - ty * rx)
        un = math.sqrt(ux * ux + uy * uy + uz * uz)
        if un <= 1e-9:
            return "tangent"
        ux, uy, uz = ux / un, uy / un, uz / un
        wx, wy, wz = (ty * uz - tz * uy, tz * ux - tx * uz, tx * uy - ty * ux)
        inside = 0
        for k in range(8):
            ang = 2.0 * math.pi * k / 8
            c, s = math.cos(ang), math.sin(ang)
            q = (
                p.X + eps * (c * ux + s * wx),
                p.Y + eps * (c * uy + s * wy),
                p.Z + eps * (c * uz + s * wz),
            )
            try:
                if shape.is_inside(q):
                    inside += 1
            except Exception:
                continue
        if inside <= 3:
            return "convex"
        if inside >= 5:
            return "concave"
        return "tangent"
    except Exception:
        return "tangent"


def _circle_normal(edge: Edge):
    """Unit normal of a circular rim edge — ``Edge.normal()`` where the
    kernel exposes it (planar edges), else t(0) cross t(0.25): two tangents
    a quarter-turn apart always span the rim plane."""
    normal = getattr(edge, "normal", None)
    if callable(normal):
        try:
            n = normal()
            if n is not None:
                return n.normalized()
        except Exception:
            pass
    n = edge.tangent_at(0.0).cross(edge.tangent_at(0.25))
    return n.normalized()


def edges_at(
    shape_or_builder=None,
    *,
    axis=None,
    x=None,
    y=None,
    z=None,
    x_range=None,
    y_range=None,
    z_range=None,
    geom=None,
    tol: float = 0.6,
) -> list[Edge]:
    """Filter a part's edges by geometry and position — the sandbox-safe way
    to build a chamfer/fillet selection without hand-rolled OCP code.

    ``axis`` (``"x"``/``"y"``/``"z"``, signs ignored) keeps edges whose
    mid-tangent is axis-dominant (``|component| > 0.98``). ``x``/``y``/``z``
    match the edge's bbox centre within ``±tol``; ``*_range`` keep centres
    inside ``[lo, hi]``. ``geom`` narrows by geometry kind: ``"line"`` /
    ``"circle"``. ``shape_or_builder=None`` inside an active ``BuildPart``
    reads the builder's current part.

        inner = edges_at(p.part, axis="y", x=-36, z=6, tol=1)
    """
    s = _unwrap_shape(shape_or_builder)
    if s is None:
        return []
    ax = str(axis).lstrip("+-").lower() if axis is not None else ""
    ai = {"x": 0, "y": 1, "z": 2}.get(ax)
    want_geom = None
    if geom is not None:
        want_geom = {
            "line": GeomType.LINE,
            "circle": GeomType.CIRCLE,
        }.get(str(geom).lower())

    def _in_range(c, rng) -> bool:
        lo, hi = rng
        return (lo is None or c >= float(lo)) and (hi is None or c <= float(hi))

    out: list[Edge] = []
    for e in s.edges():
        try:
            if want_geom is not None and e.geom_type != want_geom:
                continue
            if ai is not None:
                t = e.tangent_at(0.5)
                comps = (abs(t.X), abs(t.Y), abs(t.Z))
                if comps[ai] <= 0.98:
                    continue
            c = e.center()
            if x is not None and abs(c.X - float(x)) > tol:
                continue
            if y is not None and abs(c.Y - float(y)) > tol:
                continue
            if z is not None and abs(c.Z - float(z)) > tol:
                continue
            if x_range is not None and not _in_range(c.X, x_range):
                continue
            if y_range is not None and not _in_range(c.Y, y_range):
                continue
            if z_range is not None and not _in_range(c.Z, z_range):
                continue
            out.append(e)
        except Exception:
            continue
    return out


def concave_edges(
    shape_or_builder=None,
    *,
    axis=None,
    x_range=None,
    y_range=None,
    z_range=None,
) -> list[Edge]:
    """Re-entrant edges (inner fillet roots, groove/pocket corners, the
    L-bracket inner join) selected like :func:`edges_at` — the targets a
    fillet or chamfer is usually aimed at. Each candidate is classified by
    :func:`_edge_kind`'s inside-probe ring."""
    s = _unwrap_shape(shape_or_builder)
    if s is None:
        return []
    return [
        e
        for e in edges_at(s, axis=axis, x_range=x_range, y_range=y_range, z_range=z_range)
        if _edge_kind(s, e) == "concave"
    ]


def convex_edges(
    shape_or_builder=None,
    *,
    axis=None,
    x_range=None,
    y_range=None,
    z_range=None,
) -> list[Edge]:
    """Proud outer edges (part silhouette, boss rims) — the counterpart of
    :func:`concave_edges` for exterior rounding/chamfering."""
    s = _unwrap_shape(shape_or_builder)
    if s is None:
        return []
    return [
        e
        for e in edges_at(s, axis=axis, x_range=x_range, y_range=y_range, z_range=z_range)
        if _edge_kind(s, e) == "convex"
    ]


def hole_rim_edges(
    shape_or_builder=None,
    *,
    diameter=None,
    axis=None,
    z=None,
    tol: float = 0.5,
) -> list[Edge]:
    """Circular edges — the rims of bores, counterbores, countersink mouths
    and boss shoulders. ``diameter`` matches the rim's circle within ``±tol``
    (a Ø5 bore produces two ~Ø5 rims); ``z`` matches the rim's height;
    ``axis`` requires the rim plane's normal to run along it — a +z bore's
    rims answer ``axis="z"``.

        rims = hole_rim_edges(diameter=5, axis="z", z=plate_top, tol=0.6)
    """
    s = _unwrap_shape(shape_or_builder)
    if s is None:
        return []
    ai = None
    if axis is not None:
        ai = {"x": 0, "y": 1, "z": 2}.get(str(axis).lstrip("+-").lower())
    out: list[Edge] = []
    for e in edges_at(s, geom="circle"):
        try:
            if diameter is not None:
                try:
                    r = float(e.radius)
                except Exception:
                    continue
                if abs(2.0 * r - float(diameter)) > tol:
                    continue
            if z is not None and abs(e.center().Z - float(z)) > tol:
                continue
            if ai is not None:
                n = _circle_normal(e)
                if abs((n.X, n.Y, n.Z)[ai]) <= 0.98:
                    continue
            out.append(e)
        except Exception:
            continue
    return out


def mesh_intersect(body, tool):
    """``body & tool`` that raises instead of silently returning ``None``.

    OCC boolean intersections on faceted mesh solids (``load_mesh_solid()``)
    return ``None`` when the kernel produces no result — no exception, just a
    ``None`` that later shows up as "no result assigned". Raising here keeps
    the failure at the boolean line where it belongs.
    """
    try:
        r = body & tool
    except Exception:
        r = None
    if r is None:
        r = _backend_boolean(body, tool, "intersect")
    if r is None:
        raise ValueError(
            "boolean intersection produced no solid — on faceted mesh bodies "
            "prefer `body -= tool` (cut) or `trim_below(body, z)` over `body & tool`"
        )
    return collapse_solids(r)


def mesh_cut(body: Shape, tool: Shape) -> Shape:
    """``body - tool`` guarded against the mesh-solid failure modes.

    OCC booleans on faceted mesh solids can (a) return ``None``, (b) shed
    sub-mm sliver solids, and (c) SHATTER the whole body — a cut observed
    to turn a 19 cm^3 solid into 0.1 cm^3 of debris. Case (a) raises here;
    (b) is dropped via ``collapse_solids``; for (c) a result far below
    ``body - tool`` volume is wreckage, not a part, and the backend rescue
    lane gets the op before it counts as failed.

    A cut that removes nothing is an error, not a no-op: raises
    ``ValueError`` when the tool's bounding box does not overlap the body's
    (misplaced feature — observed: bore cutters at x=75/85 on a part
    spanning x ∈ [-42.6, +39.4] passed every check and the holes were never
    cut), when the result's removed volume is negligible, or when OCC
    shattered and no backend could rescue the op.

        body = mesh_cut(body, Pos(x, y, -1) * Cylinder(radius=2.5, height=10))
    """
    if _bboxes_disjoint(body, tool):
        raise ValueError(
            "mesh_cut removed no material — the tool does not overlap the body "
            "(check feature coordinates vs the part's bounding box; the part is "
            "centred at origin, so 'N mm from the -X end' is NOT an absolute X)"
        )
    try:
        r = body - tool
    except Exception:
        r = None
    floor = max(0.0, body.volume - abs(tool.volume)) * 0.3
    if r is not None and r.volume >= floor:
        out = collapse_solids(r)
        if body.volume - out.volume > _noop_delta_eps(body, tool):
            return out
        # OCC "succeeded" but removed nothing — the faceted-solid silent
        # no-op; the backend lane can still deliver the feature
    # OCC failed (None/exception), produced wreckage, or no-oped — the
    # mesh-native backend cannot shatter and never returns None, so give it
    # the op before declaring the cut undeliverable
    rb = _backend_boolean(body, tool, "cut")
    if rb is not None and rb.volume >= floor:
        out = collapse_solids(rb)
        _require_delta(body, tool, out, "cut")
        return out
    if r is None:
        raise ValueError(
            "boolean cut produced no solid — the tool may be disjoint from "
            "the body or the faceted mesh solid rejected the boolean"
        )
    raise ValueError(
        "mesh_cut removed no material — the tool overlaps the body but no "
        "boolean could deliver the cut (OCC produced a no-op or wreckage, "
        "no backend rescued it); refusing to silently keep the un-cut body "
        "(check the tool's size and position)"
    )


def mesh_fuse(body: Shape, tool: Shape) -> Shape:
    """``body + tool`` with the same guards as :func:`mesh_cut`.

    A fuse cannot shrink the body — a result far below the input volume is
    boolean wreckage and goes to the backend rescue lane. A fuse that adds
    nothing is an error, not a no-op: raises ``ValueError`` when the tool
    is disjoint from the body, fully engulfed by it, or when the boolean
    shattered and no backend could rescue the op.
    """
    if _bboxes_disjoint(body, tool):
        raise ValueError(
            "mesh_fuse added no material — the tool does not overlap the body "
            "(check feature coordinates vs the part's bounding box; the part is "
            "centred at origin, so 'N mm from an end' is NOT an absolute X)"
        )
    try:
        r = body + tool
    except Exception:
        r = None
    if r is not None and r.volume >= body.volume * 0.3:
        out = collapse_solids(r)
        if out.volume - body.volume > _noop_delta_eps(body, tool):
            return out
        # as with cut: an OCC result that added nothing is a silent no-op —
        # let the backend lane try before raising
    rb = _backend_boolean(body, tool, "fuse")
    if rb is not None and rb.volume >= body.volume * 0.3:
        out = collapse_solids(rb)
        _require_delta(body, tool, out, "fuse")
        return out
    if r is None:
        raise ValueError("boolean fuse produced no solid — the tool must overlap the body")
    raise ValueError(
        "mesh_fuse added no material — the tool overlaps the body but no "
        "boolean could deliver the fuse (engulfed tool, or OCC produced a "
        "no-op or wreckage with no backend rescue); refusing to silently "
        "keep the un-fused body (check the tool's size and position)"
    )


def trim_below(body, z: float = 0.0, margin: float = 400.0):
    """Cut everything below the plane ``z`` — the mesh-safe way to flatten the
    underside. ``body & Box(..., Align.MIN)`` expresses the same intent but
    silently returns ``None`` on faceted mesh solids; cutting a box below the
    plane works on both B-Rep and mesh bodies.

    Falls back to the un-trimmed body when the cut fails or fragments —
    the dominant piece of a shattered cut carries damaged seams that break
    every later boolean, while a stray sub-plane facet is only cosmetic.

        body = trim_below(body, z=0)   # guarantees min Z >= z when it succeeds
    """
    cutter = Pos(0, 0, z - margin) * Box(
        margin, margin, margin, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    try:
        r = body - cutter
    except Exception:
        r = None
    if r is not None and len(r.solids()) <= len(body.solids()):
        return r
    rb = _backend_boolean(body, cutter, "cut")
    if rb is not None:
        return rb
    return body


def clip_to_box(
    body: Shape,
    x: float | None = None,
    y: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    tol: float = 0.01,
) -> Shape:
    """Trim ``body`` to the declared envelope — the safe way to force a mesh
    solid inside its ``[L, W, H]`` spec after hand-tuned boolean ops left
    overhangs.

    ``x``/``y`` are HALF-extents (the part is centred at origin): material
    beyond ±x / ±y is removed; ``z_min``/``z_max`` are absolute Z bounds.
    Each of the six sides is checked against ``body.bounding_box()`` first
    and only cut where the body actually exceeds the bound by more than
    ``tol`` mm — a side already inside gets NO cutter, which is the point:
    ``mesh_cut`` raises on a non-overlapping cutter, so blind
    "cut every side" loops drown in `removed no material` errors. A side
    whose checked cut still removes nothing is skipped, not fatal.

        body = clip_to_box(body, x=L/2, y=W/2, z_min=0)   # trim to envelope
    """
    big = 500.0  # cutter half-extent — covers any overhang at part scale
    bb = body.bounding_box()
    cutters: list[Shape] = []
    if x is not None:
        if bb.max.X - x > tol:
            cutters.append(Pos(x + big / 2, 0, 0) * Box(big, big, big))
        if -bb.min.X - x > tol:
            cutters.append(Pos(-x - big / 2, 0, 0) * Box(big, big, big))
    if y is not None:
        if bb.max.Y - y > tol:
            cutters.append(Pos(0, y + big / 2, 0) * Box(big, big, big))
        if -bb.min.Y - y > tol:
            cutters.append(Pos(0, -y - big / 2, 0) * Box(big, big, big))
    if z_max is not None and bb.max.Z - z_max > tol:
        cutters.append(Pos(0, 0, z_max + big / 2) * Box(big, big, big))
    for cutter in cutters:
        try:
            body = mesh_cut(body, cutter)
        except ValueError:
            # the bbox said "overhang" but the kernel removed nothing
            # (grazing sliver under the delta eps) — skip the side instead
            # of crashing the whole trim
            continue
    if z_min is not None and z_min - bb.min.Z > tol:
        body = trim_below(body, z=z_min)  # sliver-safe already
    return collapse_solids(body)
