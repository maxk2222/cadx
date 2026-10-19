"""Declared-connection span coverage.

A feature whose position names a from-to path ("strut from (x=-55, z=35) on
the leg to (x=+15, z=95) on the arm underside") must carry material along
that path — a bar that misses its host floats, and the visual critique has
repeatedly called a detached strut "correctly connected" (bracket hybrid
run: strut started at x=-5 instead of -55 and the critic still passed it).

Sampling the solid's interior along the declared centreline is LLM-free
evidence: a coverage below the threshold means the declared connection is
not built.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from cadx.cad.poshint import parse_coord_points

# "from … to …" / "von … bis/nach/zu …" — the connector that turns two
# declared points into a path (a feature with several instances lists the
# same connectors between instances is NOT a span — see _VOID_TYPES and the
# minimum point count below)
_SPAN_LINK_RE = re.compile(r"\b(?:from|von)\b[^.]*?\b(?:to|bis|nach|zu)\b", re.I)

# void/subtractive feature types never span material — their positions mark
# cut centres, not connection paths
_VOID_TYPES = {
    "hole",
    "through_hole",
    "bore",
    "tap",
    "thread",
    "threaded_hole",
    "threaded_bore",
    "counterbore",
    "countersink",
    "slot",
    "pocket",
    "recess",
    "cutout",
    "opening",
    "dimple",
}

# centreline sample count and the fraction that must sit inside material;
# end samples sit inside the host bodies on a correct part, so a detached
# span fails long before the threshold
_N_SAMPLES = 25
_PASS_COVERAGE = 0.85


def _pt3(d: dict[str, float]) -> list[float] | None:
    """Coord point → [x, y, z] — needs at least two declared axes; a missing
    axis defaults to 0 (the centred contract axis)."""
    xs = [d.get(k) for k in ("x", "y", "z")]
    if sum(v is not None for v in xs) < 2:
        return None
    return [float(v) if v is not None else 0.0 for v in xs]


def declared_spans(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{name, points}] for features declaring a from-to connection path."""
    out: list[dict[str, Any]] = []
    for f in features:
        if str(f.get("type") or "").lower() in _VOID_TYPES:
            continue
        text = " ".join(str(f.get(k) or "") for k in ("position", "description")).strip()
        if not _SPAN_LINK_RE.search(text):
            continue
        pts = [_pt3(p) for p in parse_coord_points(text)]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            continue
        out.append({"name": str(f.get("name") or f.get("type") or "span"), "points": pts})
    return out


def span_coverage(shape: Any, p0: list[float], p1: list[float]) -> float:
    """Fraction of centreline samples inside the solid (0..1)."""
    inside = 0
    for i in range(_N_SAMPLES):
        t = i / (_N_SAMPLES - 1)
        pt = [p0[k] + t * (p1[k] - p0[k]) for k in range(3)]
        try:
            if shape.is_inside(pt):
                inside += 1
        except Exception:
            continue
    return inside / _N_SAMPLES


def span_checks(shape: Any, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{name, coverage, points, ok}] per declared connection span."""
    out: list[dict[str, Any]] = []
    for sp in declared_spans(features):
        pts = sp["points"]
        covs = [span_coverage(shape, a, b) for a, b in itertools.pairwise(pts)]
        cov = min(covs) if covs else 1.0
        out.append(
            {
                "name": sp["name"],
                "coverage": round(cov, 3),
                "points": pts,
                "ok": cov >= _PASS_COVERAGE,
            }
        )
    return out
