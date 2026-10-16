"""cadlib ops — defensive wrappers around raw build123d operations.

These don't create or cut a feature of their own; they make a fragile OCC
call graceful. ``safe_fillet`` retries with a shrinking radius instead of
crashing the whole run on an empty/stale edge selection.
"""

from build123d import Align, Box, Pos, fillet


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


def safe_fillet(edges, radius: float, min_radius: float = 0.2) -> float:
    """Fillet ``edges``, halving the radius until OCC accepts it.

    build123d raises ``ValueError`` both when the edge selection is empty
    (stale list after ``locate``/``translate``, overly strict filter) and when
    the radius does not fit. Both cases are retried with a smaller radius —
    an empty selection never fits, so the loop ends at ``min_radius``.

    Returns the applied radius, or ``0.0`` when nothing could be filleted —
    the caller can then fall back (e.g. keep sharp edges) instead of dying.

        fillet_r = safe_fillet(p.edges().filter_by(Axis.Z), radius=3)
        if not fillet_r:
            pass  # no fillet applied — part stays valid
    """
    r = float(radius)
    while r >= min_radius:
        try:
            fillet(edges, radius=r)
            return r
        except ValueError:
            r = round(r / 2, 3)
    return 0.0


def mesh_intersect(body, tool):
    """``body & tool`` that raises instead of silently returning ``None``.

    OCC boolean intersections on faceted mesh solids (``load_mesh_solid()``)
    return ``None`` when the kernel produces no result — no exception, just a
    ``None`` that later shows up as "no result assigned". Raising here keeps
    the failure at the boolean line where it belongs.
    """
    r = body & tool
    if r is None:
        raise ValueError(
            "boolean intersection produced no solid — on faceted mesh bodies "
            "prefer `body -= tool` (cut) or `trim_below(body, z)` over `body & tool`"
        )
    return collapse_solids(r)


def mesh_cut(body, tool):
    """``body - tool`` guarded against the mesh-solid failure modes.

    OCC booleans on faceted mesh solids can (a) return ``None``, (b) shed
    sub-mm sliver solids, and (c) SHATTER the whole body — a cut observed
    to turn a 19 cm^3 solid into 0.1 cm^3 of debris. Case (a) raises here;
    (b) is dropped via ``collapse_solids``; for (c) the un-cut body is
    returned so the missing feature can be retried instead of shipping a
    fragment: a cut cannot remove more than the tool's volume, so a result
    far below ``body - tool`` volume is wreckage, not a part.

        body = mesh_cut(body, Pos(x, y, -1) * Cylinder(radius=2.5, height=10))
    """
    r = body - tool
    if r is None:
        raise ValueError(
            "boolean cut produced no solid — the tool may be disjoint from "
            "the body or the faceted mesh solid rejected the boolean"
        )
    floor = max(0.0, body.volume - abs(tool.volume)) * 0.3
    if r.volume < floor:
        return body
    return collapse_solids(r)


def mesh_fuse(body, tool):
    """``body + tool`` with the same guards as :func:`mesh_cut`.

    A fuse cannot shrink the body — a result far below the input volume is
    boolean wreckage, so the un-fused body is returned instead.
    """
    r = body + tool
    if r is None:
        raise ValueError("boolean fuse produced no solid — the tool must overlap the body")
    if r.volume < body.volume * 0.3:
        return body
    return collapse_solids(r)


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
    r = body - cutter
    if r is None or len(r.solids()) > len(body.solids()):
        return body
    return r
