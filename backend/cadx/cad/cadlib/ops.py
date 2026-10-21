"""cadlib ops — defensive wrappers around raw build123d operations.

These don't create or cut a feature of their own; they make a fragile OCC
call graceful. ``safe_fillet`` retries with a shrinking radius instead of
crashing the whole run on an empty/stale edge selection.
"""

from typing import Sequence  # noqa: UP035 — keep cadlib's sandbox import allow-list

from build123d import Align, Box, BuildPart, Edge, Pos, Shape, fillet

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
