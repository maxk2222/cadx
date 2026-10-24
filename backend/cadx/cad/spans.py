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


def span_coverage(shape: Any, p0: list[float], p1: list[float], *, mesh: Any = None) -> float:
    """Fraction of centreline samples inside the solid (0..1)."""
    import numpy as np

    ts = [i / (_N_SAMPLES - 1) for i in range(_N_SAMPLES)]
    pts = [[p0[k] + t * (p1[k] - p0[k]) for k in range(3)] for t in ts]
    mesh = mesh if mesh is not None else getattr(shape, "_cadx_mesh", None)
    if mesh is not None:
        # trimesh ray-parity containment — OCC is_inside on a sewn faceted
        # solid costs ~200ms/point and grinds the analyze step for minutes
        try:
            inside = np.asarray(mesh.contains(np.asarray(pts)), dtype=bool)
            return float(inside.sum()) / _N_SAMPLES
        except Exception:
            pass
    inside = 0
    for pt in pts:
        try:
            if shape.is_inside(pt):
                inside += 1
        except Exception:
            continue
    return inside / _N_SAMPLES


def span_checks(
    shape: Any,
    features: list[dict[str, Any]],
    *,
    mesh: Any = None,
    cancelled: Any = None,
) -> list[dict[str, Any]]:
    """[{name, coverage, points, ok}] per declared connection span."""
    out: list[dict[str, Any]] = []
    for sp in declared_spans(features):
        if cancelled is not None and cancelled():
            break
        pts = sp["points"]
        covs = [span_coverage(shape, a, b, mesh=mesh) for a, b in itertools.pairwise(pts)]
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


# ---------------------------------------------------------------------------
# declared plate footprints — the visual critic cannot tell a 30x20 pad from
# a 40x30 blob when the outer surface looks flat (hybrid run: mesh lug pads
# stayed image-organic at ~2x the declared footprint while renders looked
# "fine"). Sampling a band of space around the declared z range is LLM-free
# evidence of the overshoot.

# grid pitch for the band samples and the margin around the declared box —
# the window must include the host's surroundings but not the rest of the
# part; a leg rising at the pad's inner edge lives inside the window yet
# stays within tolerance (its cross-section is smaller than the pad's)
_BAND_PITCH_MM = 1.5
_BAND_WINDOW_MM = 12.0
# allowed footprint growth per axis before flagging (relative + absolute):
# below this the host "follows" the declaration with organic tolerance
_OVERSHOOT_REL = 0.35
_OVERSHOOT_ABS_MM = 4.0


def _band_cells_mesh(
    mesh: Any, xs: list[float], ys: list[float], zmid: float
) -> set[tuple[int, int]]:
    """Grid cells inside the source mesh at the band mid height — trimesh's
    ray-parity `contains` is orders of magnitude faster than OCC is_inside
    on sewn faceted solids (~200 ms/point)."""
    import numpy as np

    pts = np.array([[x, y, zmid] for x in xs for y in ys], dtype=float)
    try:
        inside = np.asarray(mesh.contains(pts), dtype=bool)
    except Exception:
        return set()
    n_y = len(ys)
    return {(int(k) // n_y, int(k) % n_y) for k in np.flatnonzero(inside)}


def _band_cells_occ(
    shape: Any, xs: list[float], ys: list[float], zmid: float
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            try:
                if shape.is_inside((x, y, zmid)):
                    out.add((i, j))
            except Exception:
                continue
    return out


def _band_extent(
    shape: Any,
    xs: list[float],
    ys: list[float],
    zmid: float,
    *,
    mesh: Any = None,
) -> tuple[float, float] | None:
    """(x_span, y_span) of material at the horizontal band plane z=zmid
    inside the sampling window."""
    mesh = mesh if mesh is not None else getattr(shape, "_cadx_mesh", None)
    cells = (
        _band_cells_mesh(mesh, xs, ys, zmid)
        if mesh is not None
        else _band_cells_occ(shape, xs, ys, zmid)
    )
    if not cells:
        return None
    ix = [c[0] for c in cells]
    iy = [c[1] for c in cells]
    return (
        (max(ix) - min(ix) + 1) * _BAND_PITCH_MM,
        (max(iy) - min(iy) + 1) * _BAND_PITCH_MM,
    )


# "x=-45" / "x=+45" / "x=±45" single-axis centre declarations — a point,
# not a range (range syntax "x=-60..-55" is handled by parse_ranges first)
_POINT_RE = re.compile(r"\b([xyz])\s*=\s*[±+]?\s*(-?\d+(?:\.\d+)?)(?!\s*\.\.)", re.I)


def _parse_points(text: str) -> dict[str, float]:
    return {m.group(1).lower(): float(m.group(2)) for m in _POINT_RE.finditer(text or "")}


def _stated_dim(spec_dict: dict[str, Any] | None, key: str) -> float | None:
    """Stated overall dim of a serialized PartSpec ({"value","unit","source"})
    — only source="stated" is a real user-given envelope."""
    d = (spec_dict or {}).get(key)
    if isinstance(d, dict) and d.get("source") == "stated":
        try:
            return float(d["value"])
        except (TypeError, ValueError, KeyError):
            return None
    return None


def plate_checks(
    shape: Any,
    features: list[dict[str, Any]],
    spec_dict: dict[str, Any] | None = None,
    *,
    mesh: Any = None,
    cancelled: Any = None,
) -> list[dict[str, Any]]:
    """[{name, declared, measured, ok}] — plates/pads whose declared x/y/z
    ranges bound a flat host. Material at the plate's band mid-height must
    not sprawl past the declared footprint.

    Hybrid exemption: on a mesh-derived solid (source trimesh attached as
    `_cadx_mesh`) the organic body inside a pad's window legitimately spans
    up to the *stated overall* dim — compare_dims already enforces that
    envelope. Flagging it as overshoot whipsaws the agent into cutting the
    part below its own declared size (observed: 40mm organic pad vs a
    pad-dims width of 20 while the spec's width_y stayed stated=40)."""
    from cadx.agent.spechost import parse_ranges

    bb = shape.bounding_box()
    # only mesh-derived solids get the envelope exemption — a pure-CAD part
    # has no organic base, overshoot stays a real error there
    mesh_derived = getattr(shape, "_cadx_mesh", None) is not None
    stated = (
        (_stated_dim(spec_dict, "length_x"), _stated_dim(spec_dict, "width_y"))
        if mesh_derived
        else (None, None)
    )
    out: list[dict[str, Any]] = []
    for f in features:
        if cancelled is not None and cancelled():
            break
        if str(f.get("type") or "").lower() in _VOID_TYPES:
            continue
        text = " ".join(str(f.get(k) or "") for k in ("position", "description", "name"))
        rng = parse_ranges(text)
        # flat pads are usually declared as centre point + dims
        # ("centered at x=-45, y=0, z=0..6" + {length,width,thickness}) —
        # synthesise the missing footprint ranges from centre ± dim/2
        pts = _parse_points(text)
        dims = f.get("dims") or {}
        if "x" not in rng and "x" in pts and dims.get("length"):
            half = float(dims["length"]) / 2
            rng["x"] = (pts["x"] - half, pts["x"] + half)
        if "y" not in rng and "y" in pts and dims.get("width"):
            half = float(dims["width"]) / 2
            rng["y"] = (pts["y"] - half, pts["y"] + half)
        if "x" not in rng or "y" not in rng or "z" not in rng:
            continue  # no declared bounded plate — nothing to check
        dx, dy, dz = rng["x"], rng["y"], rng["z"]
        # skip degenerate declarations (a point, or a band spanning the body)
        if dx[1] - dx[0] < 2 or dy[1] - dy[0] < 2 or dz[1] - dz[0] < 1:
            continue
        zmid = (dz[0] + dz[1]) / 2
        x0, x1 = dx[0] - _BAND_WINDOW_MM, dx[1] + _BAND_WINDOW_MM
        y0, y1 = dy[0] - _BAND_WINDOW_MM, dy[1] + _BAND_WINDOW_MM
        x0, x1 = max(x0, float(bb.min.X)), min(x1, float(bb.max.X))
        y0, y1 = max(y0, float(bb.min.Y)), min(y1, float(bb.max.Y))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = [x0 + i * _BAND_PITCH_MM for i in range(int((x1 - x0) / _BAND_PITCH_MM) + 1)]
        ys = [y0 + i * _BAND_PITCH_MM for i in range(int((y1 - y0) / _BAND_PITCH_MM) + 1)]
        ext = _band_extent(shape, xs, ys, zmid, mesh=mesh)
        if ext is None:
            continue
        over_x = ext[0] - (dx[1] - dx[0])
        over_y = ext[1] - (dy[1] - dy[0])
        # exempt only material that stays inside the stated overall envelope —
        # a part wider than its own declared envelope is still a real error
        if stated[0] is not None and ext[0] <= stated[0] + 0.5:
            over_x = 0.0
        if stated[1] is not None and ext[1] <= stated[1] + 0.5:
            over_y = 0.0
        ok = over_x <= max(_OVERSHOOT_REL * (dx[1] - dx[0]), _OVERSHOOT_ABS_MM) and over_y <= max(
            _OVERSHOOT_REL * (dy[1] - dy[0]), _OVERSHOOT_ABS_MM
        )
        out.append(
            {
                "name": str(f.get("name") or f.get("type") or "plate"),
                "declared": {"x": dx, "y": dy, "z": dz},
                "measured_span_mm": {"x": round(ext[0], 1), "y": round(ext[1], 1)},
                "z_mid": round(zmid, 2),
                "ok": ok,
            }
        )
    return out
