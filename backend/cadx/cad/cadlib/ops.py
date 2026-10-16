"""cadlib ops — defensive wrappers around raw build123d operations.

These don't create or cut a feature of their own; they make a fragile OCC
call graceful. ``safe_fillet`` retries with a shrinking radius instead of
crashing the whole run on an empty/stale edge selection.
"""

from build123d import Align, Box, Pos, fillet


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
    return r


def trim_below(body, z: float = 0.0, margin: float = 400.0):
    """Cut everything below the plane ``z`` — the mesh-safe way to flatten the
    underside. ``body & Box(..., Align.MIN)`` expresses the same intent but
    silently returns ``None`` on faceted mesh solids; cutting a box below the
    plane works on both B-Rep and mesh bodies.

        body = trim_below(body, z=0)   # guarantees min Z >= z
    """
    cutter = Pos(0, 0, z - margin) * Box(
        margin, margin, margin, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    return body - cutter
