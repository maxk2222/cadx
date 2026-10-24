"""Optional quiddity-based feature recognition, merged into MeasuredFeatures.

quiddity (Apache-2.0, the recogniser behind build123d-mcp) sees the
relationships the analytic face scan in ``features.py`` cannot express:
hole/slot PATTERNS (bolt circles, linear arrays, rect grids), bored bosses,
and typed chamfer/fillet/countersink records.

This module is a pure adapter: ``build_recognition_evidence`` returns plain
dicts keyed by ``MeasuredFeatures`` field names, plus summary counts used by
``features._merge_recognition`` for union/de-dupe. quiddity is an OPTIONAL
dependency — the executor image may lag pyproject — so every entry point
degrades to ``{}`` instead of raising.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from typing import Any

from build123d import Shape

from cadx.cad.features import _coaxial, _dominant_axis
from cadx.logging import get_logger

log = get_logger(__name__)

try:  # optional dependency — the executor image may lag pyproject
    import quiddity
except Exception:
    quiddity = None  # type: ignore[assignment]

# faceted/hybrid solids carry thousands of tiny faces: recognition is slow and
# wrong there — the mesh fallback in features.py owns that case
_MAX_RECOGNITION_FACES = 5000
# wall-clock budget for one recognition run. quiddity is advisory-only yet its
# per-face analysis explodes on helical thread B-Reps (measured >6 min on a
# 929-face screw) — it runs in a SPAWNED child (fork deadlocks the OCC child
# on inherited locks) so a timeout can be HARD-killed — a worker thread
# cannot, and the leftover grind used to starve every other pipeline call in
# the container. 0 disables the subprocess and keeps the in-process call.
_RECOGNITION_BUDGET_S = float(os.environ.get("CADX_RECOGNITION_BUDGET_S", "45") or 45)
# cap emitted records so a pathological part cannot bloat res.features JSON
_MAX_RECORDS = 64
# a countersink cone quiddity also reports as a hole's `cbore` step is the same
# feature — diameters agree to ~1 mm and the axes coincide
_CSK_CBORE_DIA_TOL = 1.0
# two boss records on the same axis line within this XY distance are the same
# physical boss seen twice (quiddity emits ±axis records for one standoff)
_BOSS_LOC_TOL = 1.0
_BOSS_DIA_TOL = 0.3

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _pt(v: Any) -> list[float]:
    """3D point/vector-ish record value → rounded [x, y, z], else []."""
    try:
        out = [round(float(c), 3) for c in v]
    except (TypeError, ValueError):
        return []
    return out if len(out) == 3 else []


def _num(v: Any) -> float:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return 0.0


def _member_diameter(members: list[Any]) -> float:
    """Mean member bore diameter (0 when members carry none, e.g. slots)."""
    dias = [_num(getattr(m, "diameter", 0.0)) for m in members]
    dias = [d for d in dias if d > 0]
    return round(sum(dias) / len(dias), 3) if dias else 0.0


def _hole_pattern_dict(pat: Any) -> dict[str, Any]:
    """Normalise a quiddity hole-pattern record (BoltCircle, LinearArray,
    RectGrid, RectangularHoleSet, …) onto the MeasuredPattern shape."""
    members = list(getattr(pat, "holes", ()) or ())
    kind = _CAMEL_RE.sub("_", type(pat).__name__).lower()
    kind = {
        "bolt_circle": "bolt_circle",
        "linear_array": "linear_array",
        "rect_grid": "rect_grid",
        "rectangular_hole_set": "rect_set",
    }.get(kind, kind)
    axis = ""
    if members:
        m_ax = _pt(getattr(members[0], "axis", None))
        axis = _dominant_axis(m_ax) if m_ax else ""
    direction = getattr(pat, "direction", None)
    if not axis and direction is not None:
        dir_pt = _pt(direction)
        axis = _dominant_axis(dir_pt) if dir_pt else ""
    rows = int(getattr(pat, "rows", 0) or 0)
    cols = int(getattr(pat, "cols", 0) or 0)
    pitch = _num(getattr(pat, "pitch", 0.0)) or _num(getattr(pat, "row_pitch", 0.0))
    return {
        "kind": kind,
        "count": len(members) or rows * cols,
        "axis": axis,
        "center": _pt(getattr(pat, "center", None)),
        "diameter": _num(getattr(pat, "diameter", 0.0)),  # PCD on bolt circles
        "width": _num(getattr(pat, "width", 0.0)),
        "height": _num(getattr(pat, "height", 0.0)),
        "pitch": pitch,
        "rows": rows,
        "cols": cols,
        "angle": _num(getattr(pat, "angle", 0.0)),
        "member_diameter": _member_diameter(members),
        "members": [p for p in (_pt(getattr(m, "location", None)) for m in members) if p],
    }


def _slot_pattern_dict(pat: Any) -> dict[str, Any]:
    """Normalise a quiddity slot-pattern record (SlotArray, SlotGrid,
    OrientedSlotArray, OrientedSlotGrid)."""
    members = list(getattr(pat, "slots", ()) or ())
    kind = _CAMEL_RE.sub("_", type(pat).__name__).lower()
    direction = getattr(pat, "direction", None)
    dir_pt = _pt(direction)
    axis = _dominant_axis(dir_pt) if dir_pt else ""
    rows = int(getattr(pat, "rows", 0) or 0)
    cols = int(getattr(pat, "cols", 0) or 0)
    pitch = _num(getattr(pat, "pitch", 0.0)) or _num(getattr(pat, "row_pitch", 0.0))
    # Slot records have no 3D centre (w_center/lo/hi); OrientedSlot does.
    centers = [c for c in (_pt(getattr(m, "center", None)) for m in members) if c]
    return {
        "kind": kind,
        "count": len(members) or rows * cols,
        "axis": axis,
        "center": _pt(getattr(pat, "center", None)),
        "pitch": pitch,
        "rows": rows,
        "cols": cols,
        "angle": _num(getattr(pat, "angle", 0.0)),
        "member_diameter": _member_diameter(members)
        or _num(getattr(members[0], "width", 0.0) if members else 0.0),
        "members": centers,
    }


def _same_boss(a: Any, b: Any) -> bool:
    """Two boss records on the same axis line at the same XY with matching
    diameter are one physical boss recognised twice (±axis duplicates)."""
    if abs(_num(a.diameter) - _num(b.diameter)) > _BOSS_DIA_TOL:
        return False
    la, lb = _pt(a.location), _pt(b.location)
    if not la or not lb:
        return False
    ax, bx = _pt(a.axis), _pt(b.axis)
    if not ax or not bx or _dominant_axis(ax)[-1] != _dominant_axis(bx)[-1]:
        return False
    # perpendicular distance between the two axis lines is ~0 when XY matches
    d = [lb[i] - la[i] for i in range(3)]
    par = sum(d[i] * ax[i] for i in range(3))
    perp = float(sum((d[i] - par * ax[i]) ** 2 for i in range(3))) ** 0.5
    return bool(perp <= _BOSS_LOC_TOL)


def _bored_bosses(bosses: list[Any], holes: list[Any]) -> list[dict[str, Any]]:
    """Pair protruding bosses with coaxial bores — a bored boss is the
    relationship neither the boss list nor the hole list expresses alone."""
    unique: list[Any] = []
    for b in bosses:
        dupe = next((u for u in unique if _same_boss(u, b)), None)
        if dupe is None:
            unique.append(b)
        elif _num(b.height) > _num(dupe.height):
            unique[unique.index(dupe)] = b
    h_axes = {
        id(h): (_pt(h.location), _pt(h.axis))
        for h in holes
        if _pt(getattr(h, "location", None)) and _pt(getattr(h, "axis", None))
    }
    out: list[dict[str, Any]] = []
    for b in unique[:_MAX_RECORDS]:
        b_axis, b_loc = _pt(b.axis), _pt(b.location)
        if not b_axis or not b_loc:
            continue
        # the main bore is the largest coaxial hole inside the boss OD
        bore = max(
            (
                h
                for h in holes
                if id(h) in h_axes
                and 0 < _num(h.diameter) < _num(b.diameter) - _BOSS_DIA_TOL
                and _coaxial(b_loc, b_axis, h_axes[id(h)][0], h_axes[id(h)][1])
            ),
            key=lambda h: _num(h.diameter),
            default=None,
        )
        if bore is None:
            continue
        out.append(
            {
                "axis": _dominant_axis(b_axis),
                "location": b_loc,
                "boss_diameter": _num(b.diameter),
                "bore_diameter": _num(bore.diameter),
                "boss_height": _num(b.height),
                "bore_depth": _num(bore.depth),
            }
        )
    return out


def _countersink_count(holes: list[Any], countersinks: list[Any]) -> int:
    """Dedicated countersink records plus hole.csink flares not already
    covered by one (a hole's csink IS the same cone as a CounterSink record)."""
    n = len(countersinks)
    for h in holes:
        sink = getattr(h, "csink", None)
        if not sink:
            continue
        h_ax, h_loc = _pt(h.axis), _pt(h.location)
        if not h_ax or not h_loc:
            continue
        if any(
            _coaxial(h_loc, h_ax, _pt(c.location), _pt(c.axis))
            for c in countersinks
            if _pt(c.location) and _pt(c.axis)
        ):
            continue
        n += 1
    return n


def _counterbore_count(holes: list[Any], countersinks: list[Any]) -> int:
    """Holes carrying a stepped cbore — minus cbores that are actually the
    same flare as a countersink record (coaxial, matching outer diameter)."""
    n = 0
    for h in holes:
        cbore = getattr(h, "cbore", None)
        if not cbore:
            continue
        cb_dia = _num(
            cbore.get("diameter") if isinstance(cbore, dict) else getattr(cbore, "diameter", 0.0)
        )
        h_ax, h_loc = _pt(h.axis), _pt(h.location)
        if not h_ax or not h_loc:
            continue
        is_csk = any(
            _coaxial(h_loc, h_ax, _pt(c.location), _pt(c.axis))
            and abs(cb_dia - _num(c.major_diameter)) <= _CSK_CBORE_DIA_TOL
            for c in countersinks
            if _pt(c.location) and _pt(c.axis)
        )
        if not is_csk:
            n += 1
    return n


def _try(fn: Any, default: Any) -> Any:
    """Best-effort extraction — one malformed record family must not cost the
    whole inventory (recognition is advisory evidence, never load-bearing)."""
    try:
        return fn()
    except Exception:
        return default


_WARMED = False


def _warm_quiddity() -> None:
    """Pay quiddity's lazy first-call init once, early — the first recognition
    call in a fresh executor process otherwise pays it inside a measured
    context (spawned children re-import anyway, but the in-process fallback
    path must not surprise the pipeline with a multi-second init)."""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    try:
        from build123d import Box

        quiddity.build_recognition_result(Box(1, 1, 1))
    except Exception:
        pass


def _recognition_child(brep_path: str, out_path: str) -> None:
    """Spawned worker: rebuilds the solid from its BREP file, runs quiddity,
    writes the evidence dict as JSON.

    Spawn (fresh interpreter) is mandatory — a forked child inherits OCC /
    allocator locks held by dead threads and deadlocks before quiddity even
    starts (observed: ~0 CPU for the full budget on a trivial part). A spawned
    child re-imports cleanly and can be hard-killed on timeout; a crash inside
    it only costs the advisory evidence, never the pipeline."""
    import json
    from pathlib import Path

    from build123d import import_brep

    try:
        shape = import_brep(brep_path)
        rec = _evidence_dict(quiddity.build_recognition_result(shape))
    except Exception:
        rec = {}
    with contextlib.suppress(Exception):
        Path(out_path).write_text(json.dumps(rec))


def _run_budgeted(shape: Shape, budget_s: float, cancelled: Any = None) -> dict[str, Any] | None:
    """Run recognition in a spawned child with a hard wall-clock budget.

    Returns the evidence dict, ``{}`` on child failure/timeout, or ``None``
    when the spawn path is unavailable (caller may then run in-process)."""
    import json
    import multiprocessing as mp
    import tempfile
    from pathlib import Path

    from build123d import export_brep

    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="cadx-recognition-") as td:
            brep = Path(td) / "shape.brep"
            out = Path(td) / "rec.json"
            export_brep(shape, str(brep))
            proc = ctx.Process(target=_recognition_child, args=(str(brep), str(out)), daemon=True)
            try:
                proc.start()
            except Exception:
                return None  # spawn unavailable — caller falls back in-process
            deadline = time.monotonic() + budget_s
            while proc.is_alive() and time.monotonic() < deadline:
                proc.join(min(1.0, max(0.05, deadline - time.monotonic())))
                if cancelled is not None and cancelled():
                    break
            if proc.is_alive():
                proc.terminate()
                proc.join(5.0)
                log.warning(
                    "recognition.budget_exceeded",
                    budget_s=budget_s,
                    faces=len(shape.faces()) if hasattr(shape, "faces") else -1,
                )
                return {}
            try:
                rec = json.loads(out.read_text())
            except Exception:
                rec = {}
            return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}


def build_recognition_evidence(
    shape: Shape, *, budget_s: float | None = None, cancelled: Any = None
) -> dict[str, Any]:
    """Run quiddity on a B-Rep solid and return measured-feature evidence.

    Returns ``{}`` when quiddity is not installed, the shape is too large for
    bounded recognition (> _MAX_RECOGNITION_FACES faces — faceted/hybrid
    solids stay owned by the mesh fallback), recognition itself fails, or the
    run exceeds ``budget_s`` (default ``_RECOGNITION_BUDGET_S``; the advisory
    pass must never grind the pipeline past its deadline).
    Otherwise returns the additive MeasuredFeatures fields as plain dicts plus
    union-summary keys consumed by ``features._merge_recognition``.
    """
    if quiddity is None:
        return {}
    try:
        if len(shape.faces()) > _MAX_RECOGNITION_FACES:
            return {}
    except Exception:
        return {}
    _warm_quiddity()
    budget = _RECOGNITION_BUDGET_S if budget_s is None else float(budget_s)
    if budget > 0:
        rec = _run_budgeted(shape, budget, cancelled=cancelled)
        if rec is not None:
            return rec
        # spawn unavailable — in-process fallback keeps the evidence on hosts
        # where the child cannot be started (budget cannot be enforced there)
    try:
        result = quiddity.build_recognition_result(shape)
    except Exception:
        return {}
    return _evidence_dict(result)


def _evidence_dict(result: Any) -> dict[str, Any]:
    """Plain-dict projection of a quiddity recognition result."""
    holes = _try(lambda: list(getattr(result, "holes", ()) or ()), [])
    bosses = _try(lambda: list(getattr(result, "bosses", ()) or ()), [])
    countersinks = _try(lambda: list(getattr(result, "countersinks", ()) or ()), [])
    chamfers = _try(lambda: list(getattr(result, "chamfers", ()) or ()), [])
    fillets = _try(lambda: list(getattr(result, "fillets", ()) or ()), [])
    blends = _try(lambda: list(getattr(result, "blends", ()) or ()), [])
    hole_pats = _try(lambda: list(getattr(result, "hole_patterns", ()) or ()), [])
    slot_pats = _try(lambda: list(getattr(result, "slot_patterns", ()) or ()), [])
    oslot_pats = _try(lambda: list(getattr(result, "oriented_slot_patterns", ()) or ()), [])

    fillet_radii = sorted(
        {_num(getattr(f, "radius", 0.0)) for f in fillets}
        | {_num(getattr(b, "radius", 0.0)) for b in blends} - {0.0}
    )

    return {
        "hole_patterns": [
            p for p in (_try(lambda p=p: _hole_pattern_dict(p), None) for p in hole_pats) if p
        ][:_MAX_RECORDS],
        "slot_patterns": [
            p
            for p in (_try(lambda p=p: _slot_pattern_dict(p), None) for p in slot_pats + oslot_pats)
            if p
        ][:_MAX_RECORDS],
        "bored_bosses": _try(lambda: _bored_bosses(bosses, holes), []),
        "chamfers": [
            {
                "axis": str(getattr(c, "axis", "") or ""),
                "leg1": _num(getattr(c, "leg1", 0.0)),
                "leg2": _num(getattr(c, "leg2", 0.0)),
                "angle": _num(getattr(c, "angle", 0.0)),
                "at": _pt(getattr(c, "at", None)),
            }
            for c in chamfers[:_MAX_RECORDS]
        ],
        "fillets": [
            {
                "axis": str(getattr(f, "axis", "") or ""),
                "radius": _num(getattr(f, "radius", 0.0)),
                "at": _pt(getattr(f, "at", None)),
                "side": str(getattr(f, "side", "") or ""),
            }
            for f in fillets[:_MAX_RECORDS]
        ],
        # union summaries — merged as max()/rescue in features.py, never added
        "countersink_count": _try(lambda: _countersink_count(holes, countersinks), 0),
        "counterbore_count": _try(lambda: _counterbore_count(holes, countersinks), 0),
        "chamfer_face_count": len(chamfers),
        "fillet_radii": fillet_radii,
    }
