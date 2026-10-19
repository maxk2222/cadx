"""Trusted geometry comparison: version diffs, fit checks, alignment checks.

Foundation for "what changed between v3 and v4" diffs and future assembly
interference checks. Ported from build123d-mcp's compare/clearance/align
tools, but returning structured dicts instead of JSON strings and keeping
every OCC operation defensive.

Deliberately no ``cadx`` imports: pure build123d/OCP so the module can also
load inside the sandboxed runner (imports are lazy for the same reason —
``OCP`` is not on the generated-code allowlist, but runner frames bypass the
import guard).

Defensive contract: no public function raises. Failures return a dict with
an ``"error"`` key; OCC boolean failures degrade to bbox/volume-only
results flagged via ``"boolean": "failed"`` (or ``"skipped"`` when the face
count exceeds :data:`MAX_FACES_FOR_BOOLEAN` — OCC booleans get quadratic
past ~20k faces, so big faceted bodies only get the cheap comparison).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from typing import Any, Literal

# Past this, OCC booleans are too slow/risky to run unattended — same cap as
# mesh_boolean.MAX_FACES_FOR_TESS.
MAX_FACES_FOR_BOOLEAN = 20_000

# mm³ — boolean noise floor: overlap below this counts as touching, not
# interfering. OCC intersection slivers land around 1e-9; a real clash on
# mm-scale parts is orders of magnitude above.
_VOL_EPS = 1e-6

# mm — bbox-face / COM coincidence tolerance for "aligned"/"coaxial".
_FACE_TOL_MM = 1e-3

# Relative tolerance for declaring two principal moments equal, i.e. the
# shape is inertia-symmetric about the third axis (2% — loose enough for
# faceted solids, tight enough that a 10x10x30 box still reads as symmetric).
_SYM_REL_TOL = 0.02

# cos(2°) — parallelism gate for two symmetry axes.
_AXIS_PARALLEL_COS = 0.99939

_BooleanStatus = Literal["ok", "failed", "skipped"]
_AXES = ("X", "Y", "Z")


def _as_compound(shape: Any) -> Any:
    """Normalise a shape to a Compound.

    Whether a shape wraps a bare Solid or a Compound changes boolean and
    distance answers (empty intersections on bare solids have returned
    ShapeLists with no ``.volume`` in older builds). Same normalisation as
    build123d-mcp's ``measure._as_compound``.
    """
    from build123d import Compound

    if type(shape.wrapped).__name__ == "TopoDS_Compound":
        return shape
    return Compound([shape])


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _bbox_dict(bb: Any) -> dict[str, list[float]]:
    return {
        "min": [round(bb.min.X, 4), round(bb.min.Y, 4), round(bb.min.Z, 4)],
        "max": [round(bb.max.X, 4), round(bb.max.Y, 4), round(bb.max.Z, 4)],
        "size": [round(bb.size.X, 4), round(bb.size.Y, 4), round(bb.size.Z, 4)],
    }


def _safe_volume(result: Any) -> float | None:
    """Volume of a boolean result; None if the op produced nothing usable."""
    if result is None:
        return None
    try:
        return float(result.volume)
    except Exception:
        return None


def _cut(a: Any, b: Any) -> tuple[Any | None, float | None]:
    """``a - b`` returning (shape, volume); (None, None) on OCC failure."""
    try:
        res = a - b
        vol = _safe_volume(res)
        if vol is None:
            return None, None
        return res, vol
    except Exception:
        return None, None


def _intersect(a: Any, b: Any) -> tuple[Any | None, float | None]:
    """``a & b`` returning (shape, volume); (None, None) on OCC failure."""
    try:
        res = a & b
        vol = _safe_volume(res)
        if vol is None:
            return None, None
        return res, vol
    except Exception:
        return None, None


def _mass_props(shape: Any) -> dict[str, Any] | None:
    """COM + principal inertia moments/axes via GProp; None on failure.

    The axis list is aligned with the moment list: ``axes[i]`` is the
    principal axis belonging to ``moments[i]``.
    """
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.wrapped, props)
        com = props.CentreOfMass()
        pp = props.PrincipalProperties()
        moments = pp.Moments()
        dirs = (
            pp.FirstAxisOfInertia(),
            pp.SecondAxisOfInertia(),
            pp.ThirdAxisOfInertia(),
        )
        return {
            "com": [float(com.X()), float(com.Y()), float(com.Z())],
            "moments": [float(m) for m in moments],
            "axes": [[float(d.X()), float(d.Y()), float(d.Z())] for d in dirs],
        }
    except Exception:
        return None


def _symmetry_axis(props: dict[str, Any] | None) -> list[float] | None:
    """Axis of (inertia-)symmetry for a shape, or None.

    A solid axisymmetric about an axis has two equal principal moments; the
    unique moment's axis is the symmetry axis. All-equal moments (sphere,
    cube) means every axis works — too ambiguous to report, so None.
    """
    if props is None:
        return None
    m = props["moments"]
    if m is None or len(m) != 3 or min(m) <= 0:
        return None

    def close(i: int, j: int) -> bool:
        return abs(m[i] - m[j]) <= _SYM_REL_TOL * max(abs(m[i]), abs(m[j]))

    if close(0, 1) and close(1, 2):
        return None  # isotropic — any axis, nothing to report
    # the pair that matches -> the leftover index is the symmetry axis
    for pair, axis_idx in (((0, 1), 2), ((1, 2), 0), ((0, 2), 1)):
        if close(*pair):
            return props["axes"][axis_idx]
    return None


def _axis_index(axis: str) -> int | None:
    return _AXES.index(axis) if axis in _AXES else None


def shape_compare(a: Any, b: Any) -> dict[str, Any]:
    """Compare two shapes: volumes, bboxes, solid counts, material diff.

    ``added_volume`` is material in ``b`` not in ``a`` (``b - a``),
    ``removed_volume`` is material in ``a`` not in ``b`` (``a - b``).
    ``added_bbox``/``removed_bbox`` locate the changed material. On boolean
    failure or a >20k-face shape, those keys are None and ``"boolean"``
    carries ``"failed"``/``"skipped"`` — the bbox/volume fields still stand.
    """
    try:
        ca, cb = _as_compound(a), _as_compound(b)
        bb_a, bb_b = ca.bounding_box(), cb.bounding_box()
        faces_a, faces_b = len(ca.faces()), len(cb.faces())
        vol_a, vol_b = float(ca.volume), float(cb.volume)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    box_a, box_b = _bbox_dict(bb_a), _bbox_dict(bb_b)
    result: dict[str, Any] = {
        "volume_a": _round(vol_a),
        "volume_b": _round(vol_b),
        "volume_delta": _round(vol_b - vol_a),
        "bbox_a": box_a,
        "bbox_b": box_b,
        "bbox_delta": {
            "min": [_round(box_b["min"][i] - box_a["min"][i]) for i in range(3)],
            "max": [_round(box_b["max"][i] - box_a["max"][i]) for i in range(3)],
            "size": [_round(box_b["size"][i] - box_a["size"][i]) for i in range(3)],
        },
        "solids_a": len(ca.solids()),
        "solids_b": len(cb.solids()),
        "faces_a": faces_a,
        "faces_b": faces_b,
        "added_volume": None,
        "removed_volume": None,
        "added_bbox": None,
        "removed_bbox": None,
        "added_solids": 0,
        "removed_solids": 0,
    }

    if max(faces_a, faces_b) > MAX_FACES_FOR_BOOLEAN:
        result["boolean"] = "skipped"
        result["note"] = (
            f"face count {max(faces_a, faces_b)} exceeds {MAX_FACES_FOR_BOOLEAN} — "
            "bbox/volume comparison only"
        )
        return result

    added, added_vol = _cut(cb, ca)
    removed, removed_vol = _cut(ca, cb)
    if added_vol is None or removed_vol is None:
        result["boolean"] = "failed"
        result["note"] = "OCC boolean failed — bbox/volume comparison only"
        # still report whichever half succeeded
        if added_vol is not None:
            result["added_volume"] = _round(added_vol)
        if removed_vol is not None:
            result["removed_volume"] = _round(removed_vol)
        return result

    result["boolean"] = "ok"
    result["added_volume"] = _round(added_vol)
    result["removed_volume"] = _round(removed_vol)
    for key, shape in (("added", added), ("removed", removed)):
        if shape is not None and len(shape.solids()):
            try:
                result[f"{key}_bbox"] = _bbox_dict(shape.bounding_box())
                result[f"{key}_solids"] = len(shape.solids())
            except Exception:
                pass
    return result


def fit_check(a: Any, b: Any, clearance_mm: float = 0.0) -> dict[str, Any]:
    """Does ``a`` fit against/inside ``b`` without interference?

    Assembly semantics: ``a`` and ``b`` are two parts (or a part and the
    cavity/housing it mates with). ``fits`` is True when they don't overlap
    AND the minimum surface gap is at least ``clearance_mm``.

    Clearance approach: the *exact* minimum surface distance
    (``Shape.distance_to`` / ``BRepExtrema_DistShapeShape``) is compared
    against ``clearance_mm`` — deliberately NOT an ``offset``/inflated copy
    of ``b``. ``BRepOffsetAPI_MakeOffsetShape`` is fragile on curved and
    faceted geometry (fails outright or self-intersects), while the distance
    query is cheap and exact. ``min_distance >= clearance_mm`` is equivalent
    to ``a ∩ offset(b, clearance) == ∅`` without ever building the offset.

    ``interference_volume`` is ``a ∩ b`` — the clash material that would
    have to be removed for ``a`` to fit. When the bounding boxes are
    disjoint the intersection is provably empty and the boolean is skipped
    (``"boolean": "skipped"``); when bounding boxes overlap the boolean
    always runs, because ``distance_to`` alone cannot distinguish "inside a
    cavity" from "embedded in material".
    """
    out: dict[str, Any] = {
        "fits": None,
        "clearance_mm": clearance_mm,
        "method": "min_distance",
        "min_distance": None,
        "interference_volume": None,
        "interference_bbox": None,
        # containment diagnostics (build123d-mcp clearance parity) — let a
        # caller answer "is a inside b's envelope" as well as "does a clash
        # with b" without a second call
        "a_volume_outside_b": None,
        "b_volume_outside_a": None,
        "boolean": "skipped",
    }
    try:
        ca, cb = _as_compound(a), _as_compound(b)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    try:
        dist = float(ca.distance_to(cb))
        out["min_distance"] = _round(dist, 6)
    except Exception:
        dist = None

    faces = 0
    with contextlib.suppress(Exception):
        faces = max(len(ca.faces()), len(cb.faces()))
    if faces > MAX_FACES_FOR_BOOLEAN:
        out["note"] = (
            f"face count {faces} exceeds {MAX_FACES_FOR_BOOLEAN} — "
            "interference not computed; fits decided on distance only"
        )
        if dist is None:
            out["error"] = "distance query failed and boolean skipped — cannot decide fit"
            return out
        # dist > 0 proves disjoint; dist ~0 is undecidable without a boolean
        out["fits"] = bool(dist > _FACE_TOL_MM and dist >= clearance_mm)
        return out

    # Bounding-box-disjoint shapes can't intersect — skip the boolean
    # entirely. Note dist>0 alone is NOT enough: distance_to() returns the
    # surface gap for a contained shape too, and can't tell "inside a
    # cavity" from "embedded in material" — only the boolean can.
    try:
        bb_a, bb_b = ca.bounding_box(), cb.bounding_box()
        disjoint = any(
            getattr(bb_a.max, ax) < getattr(bb_b.min, ax) - _FACE_TOL_MM
            or getattr(bb_b.max, ax) < getattr(bb_a.min, ax) - _FACE_TOL_MM
            for ax in "XYZ"
        )
    except Exception:
        disjoint = False
    if disjoint:
        out["interference_volume"] = 0.0
        try:
            out["a_volume_outside_b"] = _round(float(ca.volume))
            out["b_volume_outside_a"] = _round(float(cb.volume))
        except Exception:
            pass
        if dist is not None:
            out["fits"] = bool(dist >= clearance_mm)
        else:
            # disjoint bboxes already prove the zero-clearance case
            out["fits"] = clearance_mm <= 0
        return out

    inter, inter_vol = _intersect(ca, cb)
    _a_out, a_out_vol = _cut(ca, cb)
    _b_out, b_out_vol = _cut(cb, ca)
    out["a_volume_outside_b"] = _round(a_out_vol)
    out["b_volume_outside_a"] = _round(b_out_vol)
    if inter_vol is None:
        out["boolean"] = "failed"
        # The intersect may fail where the cut still worked — vol(a∩b) ==
        # vol(a) - vol(a-b), so recover the interference volume from the
        # cut when possible (only the interference bbox is lost).
        try:
            vol_a = float(ca.volume)
        except Exception:
            vol_a = None
        if a_out_vol is not None and vol_a is not None:
            est = max(0.0, vol_a - a_out_vol)
            out["interference_volume"] = _round(est)
            out["note"] = "intersection failed — interference inferred as vol(a) - vol(a-b)"
            clear_ok = dist is None or dist >= clearance_mm
            out["fits"] = bool(est <= _VOL_EPS and clear_ok)
            return out
        out["interference_volume"] = None
        out["fits"] = None
        out["note"] = "booleans failed — fits undecidable"
        if dist is None:
            out["error"] = "OCC boolean and distance query both failed"
        return out

    out["boolean"] = "ok"
    out["interference_volume"] = _round(inter_vol)
    if inter_vol > _VOL_EPS and inter is not None:
        with contextlib.suppress(Exception):
            out["interference_bbox"] = _bbox_dict(inter.bounding_box())
    clear_ok = dist is None or dist >= clearance_mm
    out["fits"] = bool(inter_vol <= _VOL_EPS and clear_ok)
    return out


def align_check(
    a: Any,
    b: Any,
    axes: str | Sequence[str] = "XYZ",
    tol_mm: float = 0.1,
) -> dict[str, Any]:
    """Alignment of ``a`` against the axes/bbox of ``b``.

    For each requested axis: signed offset of ``a``'s center of mass
    relative to ``b``'s, plus the min/max bbox-face deltas and whether both
    faces coincide within ``tol_mm``.

    ``coaxial`` is True when both shapes are inertia-symmetric about
    (near-)parallel axes and their centers of mass coincide in the plane
    perpendicular to that axis within ``tol_mm`` — i.e. two cylinders on a
    common centerline read coaxial even with different lengths/positions
    along the axis. Non-axisymmetric shapes report per-axis offsets with
    ``coaxial=False`` and a ``coaxial_reason``.
    """
    try:
        norm_axes = [ax.upper() for ax in axes]
        bad = [ax for ax in norm_axes if _axis_index(ax) is None]
        if bad or not norm_axes:
            return {"error": f"Invalid axes {bad or axes!r} — use X, Y, Z."}
        ca, cb = _as_compound(a), _as_compound(b)
        bb_a, bb_b = ca.bounding_box(), cb.bounding_box()
        pa, pb = _mass_props(ca), _mass_props(cb)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    com_a = pa["com"] if pa else _center_fallback(ca)
    com_b = pb["com"] if pb else _center_fallback(cb)
    if com_a is None or com_b is None:
        return {"error": "could not compute centers of mass"}

    mins_a = (bb_a.min.X, bb_a.min.Y, bb_a.min.Z)
    mins_b = (bb_b.min.X, bb_b.min.Y, bb_b.min.Z)
    maxs_a = (bb_a.max.X, bb_a.max.Y, bb_a.max.Z)
    maxs_b = (bb_b.max.X, bb_b.max.Y, bb_b.max.Z)

    per_axis: dict[str, dict[str, Any]] = {}
    for ax in norm_axes:
        i = _axis_index(ax)
        assert i is not None
        min_d = mins_a[i] - mins_b[i]
        max_d = maxs_a[i] - maxs_b[i]
        per_axis[ax] = {
            "com_offset": _round(com_a[i] - com_b[i], 6),
            "min_face_delta": _round(min_d, 6),
            "max_face_delta": _round(max_d, 6),
            "faces_aligned": abs(min_d) <= tol_mm and abs(max_d) <= tol_mm,
        }

    com_delta = [com_a[i] - com_b[i] for i in range(3)]
    com_dist = math.sqrt(sum(d * d for d in com_delta))

    axis_a, axis_b = _symmetry_axis(pa), _symmetry_axis(pb)
    coaxial = False
    coaxial_reason = ""
    if axis_a is None or axis_b is None:
        coaxial_reason = "one or both shapes are not axisymmetric"
    else:
        dot = abs(sum(axis_a[i] * axis_b[i] for i in range(3)))
        if dot < _AXIS_PARALLEL_COS:
            coaxial_reason = "symmetry axes are not parallel"
        else:
            # perpendicular COM offset: component of com_delta ⊥ the axis
            along = sum(com_delta[i] * axis_a[i] for i in range(3))
            perp = math.sqrt(max(0.0, com_dist * com_dist - along * along))
            coaxial = perp <= tol_mm
            if not coaxial:
                coaxial_reason = f"centers offset {perp:.4g} mm perpendicular to the axis"

    return {
        "axes": per_axis,
        "com_offset": [_round(d, 6) for d in com_delta],
        "com_distance": _round(com_dist, 6),
        "symmetry_axis_a": axis_a,
        "symmetry_axis_b": axis_b,
        "coaxial": coaxial,
        "coaxial_reason": coaxial_reason,
        "tol_mm": tol_mm,
    }


def _center_fallback(shape: Any) -> list[float] | None:
    try:
        c = shape.center()
        return [float(c.X), float(c.Y), float(c.Z)]
    except Exception:
        return None


def _region_hint(bbox: dict[str, list[float]] | None, solids: int) -> str:
    """Compact 'where' for a diff region, e.g. 'at z∈[40.0, 60.0]'."""
    if bbox is None:
        return ""
    spans = ", ".join(
        f"{ax.lower()}∈[{bbox['min'][i]:g}, {bbox['max'][i]:g}]" for i, ax in enumerate(_AXES)
    )
    regions = f" ({solids} regions)" if solids > 1 else ""
    return f"at {spans}{regions}"


def diff_summary(a: Any, b: Any, *, cmp: dict[str, Any] | None = None) -> str:
    """One-line human summary of what changed from ``a`` to ``b``.

    e.g. ``"1000.0 → 1500.0 mm³ (Δ +500.0); +500.0 mm³ added at z∈[8, 10]"``.
    Never raises — a failure yields ``"compare failed: <err>"``.

    ``cmp`` optionally carries a precomputed :func:`shape_compare` result —
    callers that already ran the compare shouldn't pay for the OCC work
    twice.
    """
    try:
        cmp = cmp if cmp is not None else shape_compare(a, b)
        if "error" in cmp:
            return f"compare failed: {cmp['error']}"
        parts = [f"{cmp['volume_a']:g} → {cmp['volume_b']:g} mm³ (Δ {cmp['volume_delta']:+g})"]
        if cmp["boolean"] == "ok":
            added, removed = cmp["added_volume"], cmp["removed_volume"]
            if added and added > _VOL_EPS:
                hint = _region_hint(cmp["added_bbox"], cmp["added_solids"])
                parts.append(f"+{added:g} mm³ added {hint}".rstrip())
            if removed and removed > _VOL_EPS:
                hint = _region_hint(cmp["removed_bbox"], cmp["removed_solids"])
                parts.append(f"-{removed:g} mm³ removed {hint}".rstrip())
            if (not added or added <= _VOL_EPS) and (not removed or removed <= _VOL_EPS):
                parts.append("no material change")
            if cmp["solids_a"] != cmp["solids_b"]:
                parts.append(f"solids {cmp['solids_a']} → {cmp['solids_b']}")
        else:
            parts.append(f"booleans {cmp['boolean']} — bbox/volume only")
        return "; ".join(parts)
    except Exception as exc:  # last-resort guard — summary must never raise
        return f"compare failed: {type(exc).__name__}: {exc}"
