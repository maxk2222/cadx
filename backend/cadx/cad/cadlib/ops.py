"""cadlib ops — defensive wrappers around raw build123d operations.

These don't create or cut a feature of their own; they make a fragile OCC
call graceful. ``safe_fillet`` retries with a shrinking radius instead of
crashing the whole run on an empty/stale edge selection.
"""

from build123d import fillet


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
