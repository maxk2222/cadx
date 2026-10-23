"""verify_spec — canonical PASS/FAIL/UNVERIFIED conformance report.

Ported concept from build123d-mcp's ``verify_spec``: a single report that
answers "code executes" vs "geometry conforms" per requirement tier,
consolidating the verification evidence the executor pipeline already
assembles (``ExecResult``: analysis / features / dfm / step roundtrip) into
one verdict the worker ships as the canonical ``verify_spec`` entry on
``Version.issues``.

Tiers:
- structural: execution succeeded, single valid solid, no located BRepCheck
  defects, STEP roundtrip clean.
- envelope:   measured bbox vs *stated* spec dims within tolerance
  (``dim_tolerance_pct``/``dim_tolerance_mm`` — the same band ``compare_dims``
  uses worker-side).
- features:   spec features vs measured geometry. The ``compare_features``
  findings are passed in via ``feature_findings`` (never recomputed here);
  hole-pattern evidence is reported when recognition actually ran.
  ``_feature_placement`` adds per-feature checks — void axis/kind, rib/boss
  inside the stated envelope, parseable position hints (``x=12`` offsets,
  ``75mm from -x end`` end-relative offsets resolved against the measured
  bbox, ``centred in y`` mid-plane hints, ``+y``/``front`` face hints) — the
  l_bracket trap class a legal-ish overall envelope hides.
- robustness: DfM warnings + declared min-wall — hints tier. UNVERIFIED when
  the evidence is absent; never FAIL just because a detector was skipped
  (a faceted-solid mesh fallback already owns its own tier).

Pure function over plain dicts — no geometry imports — so the executor
pipeline (``engine.py``) and tests call the same code path.
"""

from __future__ import annotations

import re
from typing import Any

from cadx.cad.poshint import parse_coord_points, parse_position_hints, resolve_end_hint

Status = str  # "PASS" | "FAIL" | "UNVERIFIED"
TIER_STRUCTURAL = "structural"
TIER_ENVELOPE = "envelope"
TIER_FEATURES = "features"
TIER_ROBUSTNESS = "robustness"

# spec vocabulary for repeated-hole arrangements — mirrors
# features._PATTERN_RE (kept local so this module stays dependency-light)
_PATTERN_RE = re.compile(
    r"bolt.?circle|pitch.?circle|\bpcd\b|pattern|array|grid|loch(?:bild|kreis|reihe)",
    re.I,
)

_MAX_EVIDENCE = 8

# spec feature types whose `axis` is the through-direction of a void —
# mirrors features._HOLE_TYPES (kept local, dependency-light)
_HOLE_KIND_TYPES = {
    "hole",
    "insert",
    "thread",
    "bearing_seat",
    "through_hole",
    "blind_hole",
    "countersink_hole",
    "counterbore_hole",
    "through_hole_with_countersink",
    "clearance_hole",
    "tapped_hole",
}
# additive protrusion features that must stay inside the stated envelope —
# ribs standing proud of the spec width is the l_bracket trap
_PROTRUSION_TYPES = {"rib", "gusset", "boss", "tube", "pin", "post", "stud", "standoff"}

# axis declarations in free text: "x-axis", "along z" — deliberately narrow so
# "x = 12" style offsets never read as an axis declaration
_AXIS_TEXT_RE = re.compile(
    r"\b([xyz])[\s-]*(?:axis|achse)\b"
    r"|\balong\s+(?:the\s+)?([xyz])(?:[\s-]*axis)?\b"
    r"|\bin\s+(?:der\s+)?([xyz])[\s-]*richtung\b"
    r"|\bdurch\s+([xyz])\b",
    re.I,
)
# "y faces"/"+x side" — the envelope face a protrusion feature sits on
_HINT_AXIS_RE = re.compile(r"(?<![\w])[+-]?([xyz])[\s-]*(?:faces?|sides?|seite)\b", re.I)
_POS_TOL_MM = 2.0  # "x=12" offsets verified within ±2 mm
_POS_GROUP_TOL_MM = 1.5  # a declared point must own one centre within ±1.5 mm
_POS_SIDE_EPS = 0.5  # a centre this close to the mid-plane is ambiguous


def _position_coords(text: str) -> dict[str, float]:
    """Coordinate hints with ranges normalized by the shared position parser."""
    coords, _faces, _ends, _nears = parse_position_hints(text)
    return coords


def _check(name: str, tier: str, status: Status, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "tier": tier, "status": status, "evidence": evidence}


def _dim_value(dim: Any) -> float | None:
    """Numeric value of a spec Dim (dict dump or object)."""
    v = dim.get("value") if isinstance(dim, dict) else getattr(dim, "value", None)
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def _dim_source(dim: Any) -> str:
    return str(dim.get("source") if isinstance(dim, dict) else getattr(dim, "source", "") or "")


def _wants_pattern(f: Any) -> bool:
    """Spec feature that declares a repeated-hole arrangement."""
    get = f.get if isinstance(f, dict) else lambda k, d="": getattr(f, k, d)
    text = f"{get('type', '')} {get('name', '')} {get('description', '')}"
    return str(get("type", "")).lower() == "pattern" or bool(_PATTERN_RE.search(text))


def _axis_letter(axis: Any) -> str:
    """'x'|'y'|'z' for an axis value ('+z', '-X', 'z' all -> letters)."""
    a = str(axis or "").strip().lower().lstrip("+-")
    return a if a in ("x", "y", "z") else ""


def _spec_axis(f: dict[str, Any]) -> str:
    """Declared through-axis letter of a spec feature — structured ``axis``
    field first, then a narrow axis-phrase regex on position/description.
    Positional phrases ("one hole ON +Y axis" — a hole placed on the +Y
    face, not drilled along Y) are masked before the regex runs so they
    cannot fabricate an axis declaration."""
    ax = _axis_letter(f.get("axis"))
    if ax:
        return ax
    text = f"{f.get('position') or ''} {f.get('description') or ''}"
    clean = re.sub(
        r"\bon\s+(?:the\s+)?[+-]?[xyz][\s-]*(?:axis|achse)\b",
        " ",
        text,
        flags=re.I,
    )
    m = _AXIS_TEXT_RE.search(clean)
    if m:
        return next(g for g in m.groups() if g).lower()
    return ""


# slot free-text direction phrases: for a SLOT an axis phrase in
# position/description names the long axis it runs along ("slot along z",
# "Langloch in Z-Richtung") — the elongation direction, NOT the pierce
# direction a hole's axis phrase names. Phrases such as "drill along X"
# explicitly name the through direction and must not be mistaken for the
# slot's long axis. The structured `axis` field keeps its
# through-normal meaning (featurebind maps it to oblong_slot's `normal=`).
_AXIS_SLOT_ELONG_RE = re.compile(
    r"\b([xyz])[\s-]*(?:axis|achse)\b"
    r"|\balong\s+(?:the\s+)?([xyz])(?:[\s-]*axis)?\b"
    r"|\bin\s+(?:der\s+)?([xyz])[\s-]*richtung\b",
    re.I,
)
_AXIS_SLOT_THROUGH_RE = re.compile(
    r"\bdurch\s+([xyz])\b"
    r"|\b(?:drill|drilled|drilling|bore|bored|pierce|pierced)\s+along\s+([xyz])\b"
    r"|\bthrough\s+([xyz])(?:[\s-]*axis)?\b",
    re.I,
)


def _slot_declared_dirs(f: dict[str, Any]) -> tuple[str, str]:
    """(through, elongation) axis letters a slot spec feature declares.

    ``f.axis`` is the wall normal the slot pierces — the through direction.
    A free-text axis phrase in position/description names the ELONGATION
    for slots (the phrase ``_spec_axis`` would wrongly read as the bore
    axis). A bare "durch x" phrase stays a through declaration.
    """
    text = f"{f.get('position') or ''} {f.get('description') or ''}".lower()
    through = _axis_letter(f.get("axis"))
    through_match = _AXIS_SLOT_THROUGH_RE.search(text)
    if not through and through_match:
        through = next(group for group in through_match.groups() if group)
    elong_text = _AXIS_SLOT_THROUGH_RE.sub(" ", text)
    m = _AXIS_SLOT_ELONG_RE.search(elong_text)
    elong = next((g for g in m.groups() if g), "") if m else ""
    return through, elong


def _slot_elong_letter(e: dict[str, Any]) -> str:
    """Axis letter a measured slot-like void is elongated along —
    elongated hole_groups carry ``elong_axis``; for grooves/slot patterns
    the entry's own ``axis`` IS the long direction."""
    if e.get("elongated") or "elong_axis" in e:
        return _axis_letter(e.get("elong_axis"))
    return _axis_letter(e.get("axis"))


def _measured_ids(entries: list[dict[str, Any]]) -> list[str]:
    return [str(e["id"]) for e in entries if e.get("id")]


def _axes_of(entries: list[dict[str, Any]]) -> list[str]:
    return sorted(a for a in {_axis_letter(e.get("axis")) for e in entries} if a)


def _kind_label(e: dict[str, Any]) -> str:
    if e.get("_void_kind"):
        return str(e["_void_kind"])
    if "elongated" in e:
        return "slot" if e.get("elongated") else "hole"
    if e.get("kind"):
        return f"pattern:{e['kind']}"
    return "feature"


def _stated_envelope(spec: dict[str, Any] | None) -> dict[str, float]:
    """Stated spec dims only — inferred/assumed dims are declared intent, not
    a hard envelope a protruding feature can violate."""
    out: dict[str, float] = {}
    for ax, key in (("x", "length_x"), ("y", "width_y"), ("z", "height_z")):
        dim = (spec or {}).get(key)
        if dim is None or _dim_source(dim) != "stated":
            continue
        v = _dim_value(dim)
        if v is not None:
            out[ax] = v
    return out


def _feature_hint_axes(f: dict[str, Any]) -> set[str]:
    """Axis letters the feature can protrude along — its ``axis`` field plus
    face hints ('outer y sides', '+x face') in position/description."""
    out: set[str] = set()
    ax = _axis_letter(f.get("axis"))
    if ax:
        out.add(ax)
    text = f"{f.get('position') or ''} {f.get('description') or ''}"
    out |= {m.group(1) for m in _HINT_AXIS_RE.finditer(text.lower())}
    _c, faces, _ends, _nears = parse_position_hints(text)
    out |= {fa[1] for fa in faces}
    return out


def _void_kind_axis_check(
    i: int,
    name: str,
    ftype: str,
    f: dict[str, Any],
    cands: list[dict[str, Any]],
    other: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    """Axis + kind check for one void spec feature (hole/slot/cutout/pocket).

    FAIL when same-kind voids exist but none runs along the stated axis, and
    when only the WRONG kind was measured (a round hole never satisfies a
    slot spec). UNVERIFIED when no void evidence exists at all."""
    want_ax = _spec_axis(f)
    check_name = f"feat_{i}_axis" if want_ax else f"feat_{i}_kind"
    base: dict[str, Any] = {"feature": name, "feature_type": ftype}
    if want_ax:
        base["expected_axis"] = want_ax
    all_voids = cands + other
    if not all_voids:
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "UNVERIFIED",
                {
                    **base,
                    "reason": "no measured void features — the feature is absent or unmeasurable",
                },
            )
        )
        return
    if not cands:
        kinds = sorted({_kind_label(e) for e in all_voids})
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "FAIL",
                {
                    **base,
                    "reason": (
                        f"kind mismatch — spec '{ftype}' cannot be satisfied by "
                        f"the measured {', '.join(kinds)} void(s)"
                    ),
                    "measured_kinds": kinds,
                    "measured_axes": _axes_of(all_voids),
                    "measured_ids": _measured_ids(all_voids),
                },
            )
        )
        return
    if not want_ax:
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "PASS",
                {
                    **base,
                    "measured_axes": _axes_of(cands),
                    "measured_ids": _measured_ids(cands),
                },
            )
        )
        return
    slotish = "slot" in ftype or "langloch" in ftype
    matching = [
        g
        for g in cands
        if _axis_letter(g.get("axis")) == want_ax
        # a slot's declared direction is ambiguous — `axis` names the wall
        # normal (through) while free text usually names the long axis;
        # either measured direction may satisfy it (through_axis == axis)
        or (slotish and _axis_letter(g.get("elong_axis")) == want_ax)
    ]
    ev = {
        **base,
        "measured_axes": _axes_of(cands),
        "measured_ids": _measured_ids(matching or cands),
    }
    if slotish:
        elong_axes = sorted(a for a in {_axis_letter(g.get("elong_axis")) for g in cands} if a)
        if elong_axes:
            ev["measured_elong_axes"] = elong_axes
    if other:
        ev["other_kind_axes"] = _axes_of(other)
    if matching:
        checks.append(_check(check_name, TIER_FEATURES, "PASS", ev))
    else:
        ev["reason"] = f"no measured {ftype} runs along the {want_ax} axis"
        checks.append(_check(check_name, TIER_FEATURES, "FAIL", ev))


def _slot_orientation_check(
    i: int,
    name: str,
    ftype: str,
    f: dict[str, Any],
    cands: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    """Slot elongation-direction check — the leg-slot phantom class.

    A measured slot group reports ``elong_axis`` (the in-plane direction
    the void is stretched along) separately from ``axis``/``through_axis``
    (the pierce direction) — the single ``axis`` label used to let a
    critique read the through direction as the elongation and flag a
    correct part. The spec side declares the elongation with a free-text
    direction phrase ("slot along z", "Langloch in Z-Richtung") while
    ``f.axis`` stays the through normal; an explicit elongation phrase is
    checked strictly against ``elong_axis``. A bare ``f.axis``/``durch x``
    declaration is ambiguous — when it lands on the through direction the
    feat_{i}_axis check owns it and orientation stays UNVERIFIED rather
    than second-guessing which direction the spec meant.
    """
    through_w, elong_w = _slot_declared_dirs(f)
    strict = bool(elong_w)  # free-text direction = explicit elongation decl
    want = elong_w or through_w
    if not want or not cands:
        return
    check_name = f"feat_{i}_orientation"
    ev: dict[str, Any] = {
        "feature": name,
        "feature_type": ftype,
        "expected_axis": want,
    }
    elong_axes = sorted(a for a in {_slot_elong_letter(e) for e in cands} if a)
    if elong_axes:
        ev["measured_elong_axes"] = elong_axes
    elong_hits = [e for e in cands if _slot_elong_letter(e) == want]
    if elong_hits:
        ev["measured_ids"] = _measured_ids(elong_hits)
        checks.append(_check(check_name, TIER_FEATURES, "PASS", ev))
        return
    if not strict:
        through_hits = [e for e in cands if _axis_letter(e.get("axis")) == want]
        if through_hits:
            ev["measured_ids"] = _measured_ids(through_hits)
            ev["reason"] = (
                "declared axis matches the slot's through direction — no "
                "separate elongation declaration to verify"
            )
            checks.append(_check(check_name, TIER_FEATURES, "UNVERIFIED", ev))
            return
    ev["measured_ids"] = _measured_ids(cands)
    if not elong_axes:
        ev["reason"] = "no measured slot reports an elongation direction"
        checks.append(_check(check_name, TIER_FEATURES, "UNVERIFIED", ev))
        return
    ev["reason"] = f"slot elongated along +{elong_axes[0]}, spec declares {want}"
    checks.append(_check(check_name, TIER_FEATURES, "FAIL", ev))


def _bbox_bounds(bbox_min: Any, bbox_max: Any) -> dict[str, tuple[float, float]]:
    """Measured part bbox as ``{axis: (lo, hi)}`` — the reference end/edge
    that "75mm from -x end"-style hints resolve against."""
    if not (
        isinstance(bbox_min, list | tuple)
        and isinstance(bbox_max, list | tuple)
        and len(bbox_min) == 3
        and len(bbox_max) == 3
    ):
        return {}
    try:
        return {a: (float(bbox_min[j]), float(bbox_max[j])) for j, a in enumerate("xyz")}
    except (TypeError, ValueError):
        return {}


def _centre_coord_check(
    centers: list[dict[str, float]], wants: dict[str, list[float]]
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Check only coordinate axes actually measured for each void centre.

    A through-axis coordinate is not a centre coordinate for a tunnel and
    must not be filled with the global bbox midpoint.
    """
    violations: list[dict[str, Any]] = []
    verified = 0
    unverified: list[dict[str, Any]] = []
    for ax, ax_wants in sorted(wants.items()):
        values = [float(c[ax]) for c in centers if ax in c]
        if not values:
            unverified.extend({"axis": ax, "expected_mm": want} for want in ax_wants)
            continue
        for want in ax_wants:
            nearest = min(values, key=lambda value: abs(value - want))
            delta = abs(nearest - want)
            if delta <= _POS_TOL_MM:
                verified += 1
            else:
                violations.append(
                    {
                        "axis": ax,
                        "expected_mm": want,
                        "measured_mm": round(nearest, 3),
                        "delta_mm": round(delta, 3),
                    }
                )
    return violations, verified, unverified


def _point_group_check(
    centers: list[dict[str, float]], groups: list[dict[str, float]]
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Multi-point check: each declared coordinate group must own ONE
    measured centre within ±_POS_GROUP_TOL_MM on every observable axis.
    Missing measured axes are UNVERIFIED, never compared against a
    placeholder coordinate."""
    violations: list[dict[str, Any]] = []
    verified = 0
    unverified: list[dict[str, Any]] = []
    for g in groups:
        want = {ax: v for ax, v in g.items() if ax in ("x", "y", "z")}
        if not want:
            continue
        candidates = [c for c in centers if any(ax in c for ax in want)]
        if not candidates:
            unverified.append(
                {
                    "point": {ax: round(v, 3) for ax, v in want.items()},
                    "reason": "no point axes measured",
                }
            )
            continue
        best = min(
            candidates,
            key=lambda c: (
                -sum(ax in c for ax in want),
                max(abs(float(c[ax]) - v) for ax, v in want.items() if ax in c),
            ),
        )
        measured_axes = [ax for ax in want if ax in best]
        missing_axes = [ax for ax in want if ax not in best]
        dev = {ax: abs(float(best[ax]) - want[ax]) for ax in measured_axes}
        bad = {ax: delta for ax, delta in dev.items() if delta > _POS_GROUP_TOL_MM}
        verified += len(measured_axes)
        if bad:
            violations.append(
                {
                    "point": {ax: round(v, 3) for ax, v in g.items()},
                    "measured_by_axis": {ax: round(float(best[ax]), 3) for ax in measured_axes},
                    "delta_mm": round(max(bad.values()), 3),
                }
            )
        if missing_axes:
            unverified.append(
                {
                    "point": {ax: round(v, 3) for ax, v in want.items()},
                    "axes": missing_axes,
                    "reason": "position lies along the void's through axis or lacks measurement",
                }
            )
    return violations, verified, unverified


def _spec_void_dia(f: dict[str, Any]) -> float | None:
    """Declared void diameter of a spec feature (diameter|dia|bore_diameter)."""
    dims = f.get("dims")
    if not isinstance(dims, dict):
        return None
    for key in ("diameter", "dia", "bore_diameter"):
        v = dims.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool) and v > 0:
            return float(v)
    return None


def _void_dia(e: dict[str, Any]) -> float | None:
    """Bore diameter of a measured void entry — hole_groups carry
    ``diameter``; patterns carry ``member_diameter`` (``diameter`` there is
    the pitch circle)."""
    for key in ("dia", "member_diameter", "diameter", "bore_diameter"):
        v = e.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool) and v > 0:
            return float(v)
    return None


def _spec_slot_dims(f: dict[str, Any]) -> tuple[float | None, float | None]:
    """Declared (width, length) of a slot spec feature — dims keys
    ``slot_width``/``width`` and ``slot_length``/``length``."""
    dims = f.get("dims")
    if not isinstance(dims, dict):
        return None, None

    def _num(*keys: str) -> float | None:
        for k in keys:
            v = dims.get(k)
            if isinstance(v, int | float) and not isinstance(v, bool) and v > 0:
                return float(v)
        return None

    return _num("slot_width", "width"), _num("slot_length", "length")


def _slot_dims_hit(f: dict[str, Any], e: dict[str, Any]) -> bool:
    """True when a measured elongated void's span dims match the slot
    feature's declared width/length within the same 30% band as the dia
    match. The measured ``diameter`` of a slot is an equivalent-area value
    (≈10.5 for a 6.5x15 slot) that never equals the declared width — the
    dia check alone flags a correctly cut slot as unrequested. Only runs
    when the measured span fields are populated (older payloads lack
    them)."""
    if not e.get("elongated"):
        return False
    mw, ml = e.get("width_mm"), e.get("length_mm")
    if not (
        isinstance(mw, int | float)
        and not isinstance(mw, bool)
        and mw > 0
        and isinstance(ml, int | float)
        and not isinstance(ml, bool)
        and ml > 0
    ):
        return False
    sw, sl = _spec_slot_dims(f)
    pairs = [(d, m) for d, m in ((sw, mw), (sl, ml)) if d is not None]
    return bool(pairs) and all(abs(m - d) <= 0.3 * d for d, m in pairs)


def _dia_matched(f: dict[str, Any], cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Narrow void candidates to the diameter-matched subset (within 30% of
    the spec feature's declared dia) so a differently-sized void cannot
    satisfy this feature's position/axis checks by pooling. Falls back to
    all candidates when nothing matches — checking something beats nothing."""
    decl = _spec_void_dia(f)
    if decl is None:
        return cands
    matched = [e for e in cands if (d := _void_dia(e)) is not None and abs(d - decl) <= 0.3 * decl]
    return matched or cands


def _void_center_maps(e: dict[str, Any]) -> list[dict[str, float]]:  # noqa: C901
    """Axis-named coordinates for every measured void centre.

    New ``centers_xyz`` rows explicitly carry ``None`` on an unmeasured
    through axis. Legacy XY centers remain supported: for +z they represent
    x/y; for +x only y is trustworthy; for +y only x is trustworthy.
    """
    xyz = e.get("centers_xyz")
    maps: list[dict[str, float]] = []
    if isinstance(xyz, list):
        for row in xyz:
            if not isinstance(row, list | tuple) or len(row) != 3:
                continue
            point = {
                ax: float(value)
                for ax, value in zip("xyz", row, strict=True)
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            if point:
                maps.append(point)
    if maps:
        return maps

    axis = _axis_letter(e.get("through_axis") or e.get("axis"))
    legacy = e.get("centers")
    if isinstance(legacy, list):
        for row in legacy:
            if not isinstance(row, list | tuple) or len(row) < 2:
                continue
            if axis == "x":
                maps.append({"y": float(row[1])})
            elif axis == "y":
                maps.append({"x": float(row[0])})
            else:
                maps.append({"x": float(row[0]), "y": float(row[1])})

    # Recognition patterns carry absolute 3D centers; unlike a tunnel's
    # projected centers, their axes are not fabricated from a bbox midpoint.
    for key in ("members", "center"):
        raw = e.get(key)
        if key == "center":
            raw = [raw] if isinstance(raw, list | tuple) else []
        if not isinstance(raw, list | tuple):
            continue
        for row in raw:
            if not isinstance(row, list | tuple):
                continue
            if len(row) >= 3:
                maps.append(
                    {
                        ax: float(value)
                        for ax, value in zip("xyz", row[:3], strict=False)
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    }
                )
            elif len(row) >= 2:
                maps.append({"x": float(row[0]), "y": float(row[1])})
    return maps


def _void_kind_compatible(feature: dict[str, Any], measured: dict[str, Any]) -> bool:
    """Do not let a slot's unmeasured through-axis coordinate make a round
    hole look like the spec feature's positional match (or vice versa)."""
    ftype = str(feature.get("type") or "").lower()
    wants_slot = "slot" in ftype or "langloch" in ftype
    measured_kind = _kind_label(measured)
    is_slot = measured_kind == "slot" or (
        measured_kind.startswith("pattern:") and "slot" in measured_kind
    )
    if wants_slot and not is_slot:
        return False
    return not (is_slot and not wants_slot and ftype not in {"cutout", "pattern"})


def _structured_feature_centers(feature: dict[str, Any]) -> list[dict[str, float]]:
    """Absolute centers from ``Feature.centers``; partial XYZ is allowed."""
    raw = feature.get("centers")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        point = {
            ax: float(row[ax])
            for ax in "xyz"
            if isinstance(row.get(ax), int | float) and not isinstance(row.get(ax), bool)
        }
        if point:
            out.append(point)
    return out


def _void_pos_wants(
    f: dict[str, Any], bounds: dict[str, tuple[float, float]]
) -> dict[str, list[float]]:
    """Resolved position wants from structured centers or free-text hints."""
    text = str(f.get("position") or "").strip() or str(f.get("description") or "")
    coords = _position_coords(text)
    _raw_coords, _faces, ends, nears = parse_position_hints(text)
    count = int(f.get("count") or 1)
    groups = parse_coord_points(text) if count > 1 else []
    structured = _structured_feature_centers(f)
    wants: dict[str, list[float]] = {ax: [] for ax in "xyz"}
    for ax in "xyz":
        # an end offset carries the user's stated semantics; a contradicting
        # model-computed coord ("75mm from -x end (x=20)") is dropped so it
        # cannot spawn a second, wrong want
        bnd = bounds.get(ax)
        end_vals = [
            resolve_end_hint(ax, str(e["end"]), float(e["offset_mm"]), *bnd)
            for e in ends
            if str(e["axis"]) == ax and bnd is not None
        ]
        # Structured centers are the authoritative per-instance targets.
        # Otherwise preserve all repeated text coordinates.
        coord_vals = (
            sorted({g[ax] for g in structured if ax in g})
            if structured
            else (
                sorted({g[ax] for g in groups if ax in g})
                if len(groups) >= 2
                else ([coords[ax]] if ax in coords else [])
            )
        )
        for coord_v in coord_vals:
            if end_vals and all(abs(coord_v - v) > 1.0 for v in end_vals):
                continue
            wants[ax].append(coord_v)
        for v in end_vals:
            if v not in wants[ax]:
                wants[ax].append(v)
    dia = _spec_void_dia(f)
    inset = max(1.5 * dia, 6.0) if dia else 6.0
    for n in nears:
        ax = str(n.get("axis") or "")
        if ax in wants and not wants[ax]:
            bnd = bounds.get(ax)
            if bnd is not None:
                lo, hi = bnd
                wants[ax].append(
                    hi - inset if str(n.get("end") or "").startswith("+") else lo + inset
                )
    return {ax: vals for ax, vals in wants.items() if vals}


def _near_band_check(
    nears: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
    centers: list[dict[str, float]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """'near the +x end' proximity hints resolve to an end-band region: the
    outermost 30% of the measured bbox span on that axis. The centre nearest
    the declared end must sit inside the band; without a measurement on that
    axis the hint stays unverifiable."""
    violations: list[dict[str, Any]] = []
    verified = 0
    unverified: list[dict[str, Any]] = []
    for n in nears:
        nax = str(n.get("axis") or "")
        bnd = bounds.get(nax)
        vals = [float(c[nax]) for c in centers if nax in c]
        if bnd is None or bnd[1] - bnd[0] <= 0 or not vals:
            unverified.append({"near": str(n.get("end")), "axis": nax})
            continue
        lo, hi = bnd
        plus = str(n.get("end") or "").startswith("+")
        edge = lo + (0.7 if plus else 0.3) * (hi - lo)
        best = max(vals) if plus else min(vals)
        if (plus and best >= edge) or (not plus and best <= edge):
            verified += 1
        else:
            violations.append(
                {
                    "near": str(n.get("end")),
                    "axis": nax,
                    "measured_mm": round(best, 3),
                    "band_edge_mm": round(edge, 3),
                }
            )
    return violations, verified, unverified


def _face_side_check(
    faces: list[tuple[str, str]], centers: list[dict[str, float]]
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """'+x face'/'front' side hints: the outermost measured centre must sit
    on the declared side of the mid-plane. A mid-plane centre is ambiguous
    (not wrong); missing coordinate axes stay UNVERIFIED."""
    violations: list[dict[str, Any]] = []
    verified = 0
    unverified: list[dict[str, Any]] = []
    for face in faces:
        vals = [float(c[face[1]]) for c in centers if face[1] in c]
        if not vals:
            unverified.append({"face": face, "axis": face[1]})
            continue
        best = max(vals) if face[0] == "+" else min(vals)
        if (face[0] == "+" and best > _POS_SIDE_EPS) or (face[0] == "-" and best < -_POS_SIDE_EPS):
            verified += 1
        elif abs(best) <= _POS_SIDE_EPS:
            continue  # mid-plane centre — ambiguous, not wrong
        else:
            violations.append({"face": face, "measured_mm": round(best, 3)})
    return violations, verified, unverified


def _void_position_check(  # noqa: C901
    i: int,
    name: str,
    f: dict[str, Any],
    cands: list[dict[str, Any]],
    bbox_min: Any,
    bbox_max: Any,
    checks: list[dict[str, Any]],
) -> None:
    """Verify the measured centre honours machine-readable position hints
    ('x=12', '75mm from -x end', 'centred in y', '+y', 'front') —
    conservative: only unambiguously parseable hints are checked, everything
    else is UNVERIFIED. End-relative offsets resolve to absolute centred
    coords via the measured bbox ('-x end +75mm' -> x=xmin+75); without bbox
    bounds they stay unverifiable, never a guess. 'near the +x end'
    proximity hints resolve to an end-band region check (the outermost 30%
    of the measured bbox span). Axis-aware centers only verify coordinates
    that were measured; an unobserved through-axis coordinate is UNVERIFIED
    rather than compared against a fabricated bbox midpoint."""
    pos = str(f.get("position") or "").strip()
    structured = _structured_feature_centers(f)
    if not pos and not structured:
        return
    check_name = f"feat_{i}_position"
    hint_text = pos or str(f.get("description") or "")
    coords = _position_coords(hint_text)
    _raw_coords, faces, ends, nears = parse_position_hints(hint_text)
    count = int(f.get("count") or 1)
    text_groups = parse_coord_points(hint_text) if count > 1 else []
    want_ax = _spec_axis(f)
    if want_ax:
        # position evidence only counts centres of voids on the declared
        # through axis — a differently-oriented bore cannot satisfy it
        ax_cands = [g for g in cands if _axis_letter(g.get("axis")) == want_ax]
        cands = ax_cands or cands
        if "blind" not in str(f.get("type") or "").lower():
            # the through axis is not an in-plane centre coordinate —
            # "through z=0..6" parses as z=3.0 but a tunnel's z centre is
            # never measured (centers_xyz marks it None)
            coords.pop(want_ax, None)
            text_groups = [
                {k: v for k, v in g.items() if k != want_ax} for g in text_groups
            ]
    groups = structured or text_groups
    base: dict[str, Any] = {"feature": name, "position": pos[:160] or None}
    if not structured and not coords and not faces and not ends and not nears:
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "UNVERIFIED",
                {**base, "reason": "position is not machine-parseable"},
            )
        )
        return
    expected: dict[str, Any] = {"coords": coords, "faces": faces}
    if structured:
        expected["points"] = structured
    elif len(text_groups) >= 2:
        expected["points"] = groups
    if ends:
        expected["ends"] = ends
    if nears:
        expected["nears"] = nears
    centers = [c for g in cands for c in _void_center_maps(g)]
    if not centers:
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "UNVERIFIED",
                {**base, "expected": expected, "reason": "no measured centres for the feature"},
            )
        )
        return
    bounds = _bbox_bounds(bbox_min, bbox_max)
    violations: list[dict[str, Any]] = []
    verified = 0
    unverified: list[dict[str, Any]] = []
    # multi-point declarations ("x=-25, y=0 and x=+45, y=0") check each
    # group against ONE centre — the flat coords dict would only check
    # the last point and let a missing duplicate pass
    n_coord_hints = sum(len(group) for group in groups) if groups else len(coords)
    wants: dict[str, list[float]] = {}
    use_point_groups = bool(structured) or len(text_groups) >= 2
    if use_point_groups:
        gv, gn, gu = _point_group_check(centers, groups)
        violations += gv
        verified += gn
        unverified += gu
    else:
        for ax, want in coords.items():
            wants.setdefault(ax, []).append(want)
    for e in ends:
        bnd = bounds.get(e["axis"])
        if bnd is not None:
            wants.setdefault(e["axis"], []).append(
                resolve_end_hint(e["axis"], e["end"], e["offset_mm"], *bnd)
            )
    if structured:
        structured_axes = {ax for point in structured for ax in point}
        for ax in structured_axes:
            wants.pop(ax, None)
        ends = [e for e in ends if e["axis"] not in structured_axes]
        nears = [n for n in nears if n["axis"] not in structured_axes]
    w_viol, w_ver, w_unverified = _centre_coord_check(centers, wants)
    violations += w_viol
    verified += w_ver
    unverified += w_unverified
    f_viol, f_ver, f_unverified = _face_side_check(faces, centers)
    violations += f_viol
    verified += f_ver
    unverified += f_unverified
    near_violations, near_verified, near_unverified = _near_band_check(nears, bounds, centers)
    violations += near_violations
    verified += near_verified
    unverified += near_unverified
    ev = {
        **base,
        "expected": expected,
        "measured_centers_by_axis": [
            {ax: round(value, 3) for ax, value in c.items()} for c in centers[:6]
        ],
        "measured_ids": _measured_ids(cands),
    }
    if unverified:
        ev["unverified"] = unverified[:_MAX_EVIDENCE]
    if violations:
        ev["violations"] = violations[:_MAX_EVIDENCE]
        checks.append(_check(check_name, TIER_FEATURES, "FAIL", ev))
    elif unverified:
        ev["reason"] = "one or more requested position axes were not measured"
        checks.append(_check(check_name, TIER_FEATURES, "UNVERIFIED", ev))
    elif verified and n_coord_hints + len(faces) + len(ends) + len(nears) == verified:
        checks.append(_check(check_name, TIER_FEATURES, "PASS", ev))
    else:
        ev["reason"] = (
            "some position hints are not verifiable from measured centres"
            if verified
            else "position hints could not be verified from measured centres"
        )
        checks.append(_check(check_name, TIER_FEATURES, "UNVERIFIED", ev))


def _protrusion_check(
    i: int,
    name: str,
    ftype: str,
    f: dict[str, Any],
    stated_env: dict[str, float],
    bosses: list[dict[str, Any]],
    bbox: Any,
    tol_pct: float,
    tol_mm: float,
    checks: list[dict[str, Any]],
) -> None:
    """Rib/boss envelope check: an additive feature's outer extent must lie
    inside the *stated* spec dims — ribs proud of the 40 mm width are the
    l_bracket trap. UNVERIFIED when no measured extent can locate it."""
    check_name = f"feat_{i}_envelope"
    base: dict[str, Any] = {"feature": name, "feature_type": ftype}
    if not stated_env:
        checks.append(
            _check(
                check_name,
                TIER_FEATURES,
                "UNVERIFIED",
                {**base, "reason": "spec carries no stated envelope dims to check against"},
            )
        )
        return
    hint_axes = _feature_hint_axes(f)
    env_axes = [a for a in "xyz" if a in stated_env]
    check_axes = [a for a in env_axes if a in hint_axes] or env_axes
    bbox3 = bbox if isinstance(bbox, list | tuple) and len(bbox) == 3 else None
    measured_extent: dict[str, float] = {}
    protrusions: list[dict[str, Any]] = []
    if bbox3 is not None:
        for a in check_axes:
            got = float(bbox3["xyz".index(a)])
            measured_extent[a] = round(got, 3)
            tol = max(stated_env[a] * tol_pct / 100.0, tol_mm)
            if got - stated_env[a] > tol:
                protrusions.append(
                    {
                        "axis": a,
                        "envelope_mm": stated_env[a],
                        "measured_mm": round(got, 3),
                        "over_mm": round(got - stated_env[a], 3),
                    }
                )
    # boss-type features carry per-entry extents — a boss larger than the
    # stated dim it must fit inside is direct protrusion evidence
    boss_ids: list[str] = []
    if ftype in _PROTRUSION_TYPES - {"rib", "gusset"}:
        for b in bosses:
            bid = str(b.get("id") or "")
            if bid:
                boss_ids.append(bid)
            b_ax = _axis_letter(b.get("axis"))
            dia = b.get("diameter")
            hgt = b.get("height")
            for pax, want in stated_env.items():
                tol = max(want * tol_pct / 100.0, tol_mm)
                extent, val = ("boss_height", hgt) if pax == b_ax else ("boss_diameter", dia)
                if isinstance(val, int | float) and not isinstance(val, bool) and val - want > tol:
                    protrusions.append(
                        {
                            "axis": pax,
                            "envelope_mm": want,
                            "measured_mm": round(float(val), 3),
                            "extent": extent,
                            "boss_id": bid or None,
                        }
                    )
    ev: dict[str, Any] = {
        **base,
        "envelope_mm": {a: stated_env[a] for a in check_axes},
        "measured_extent_mm": measured_extent,
    }
    if boss_ids:
        ev["measured_ids"] = boss_ids
    located = bbox3 is not None or bool(boss_ids)
    if protrusions:
        ev["protrusions"] = protrusions[:_MAX_EVIDENCE]
        checks.append(_check(check_name, TIER_FEATURES, "FAIL", ev))
    elif located:
        checks.append(_check(check_name, TIER_FEATURES, "PASS", ev))
    else:
        ev["reason"] = "no measured extent locates the feature"
        checks.append(_check(check_name, TIER_FEATURES, "UNVERIFIED", ev))


def _feature_placement(
    spec: dict[str, Any],
    measured: dict[str, Any],
    bbox: Any,
    bbox_min: Any,
    bbox_max: Any,
    tol_pct: float,
    tol_mm: float,
    checks: list[dict[str, Any]],
) -> None:
    """Per-feature placement checks — the l_bracket trap class: a part can
    satisfy the overall envelope while a slot is bored through the wrong
    axis or a rib stands proud of the stated width. Each spec feature gets
    its own check(s); evidence always carries expected AND measured values
    (measured entries cited by their stable ``id``) for the improve step."""
    spec_feats = (spec or {}).get("features") or []
    groups = [g for g in (measured or {}).get("hole_groups") or [] if isinstance(g, dict)]
    round_groups = [g for g in groups if not g.get("elongated")]
    slot_groups = [g for g in groups if g.get("elongated")]
    hole_pats = [p for p in (measured or {}).get("hole_patterns") or [] if isinstance(p, dict)]
    slot_pats = [p for p in (measured or {}).get("slot_patterns") or [] if isinstance(p, dict)]
    bosses = [b for b in (measured or {}).get("bosses") or [] if isinstance(b, dict)]

    # structural (AAG) void features — tagged so kind mismatch reporting
    # can cite "pocket"/"groove"/"notch" instead of a generic label
    def _tagged(key: str, kind: str) -> list[dict[str, Any]]:
        return [
            {**e, "_void_kind": kind}
            for e in (measured or {}).get(key) or []
            if isinstance(e, dict)
        ]

    pockets = _tagged("pockets", "pocket")
    grooves = _tagged("grooves", "groove")
    notches = _tagged("notches", "notch")
    stated_env = _stated_envelope(spec)

    for i, f in enumerate(spec_feats):
        if not isinstance(f, dict):
            continue
        ftype = str(f.get("type") or "").lower()
        name = str(f.get("name") or f.get("type") or f"feature_{i}")
        if ftype in {"fillet", "round"}:
            _fillet_check(i, name, f, measured, checks)
            continue
        if ftype in _PROTRUSION_TYPES:
            _protrusion_check(i, name, ftype, f, stated_env, bosses, bbox, tol_pct, tol_mm, checks)
            continue
        if ftype in _HOLE_KIND_TYPES or "hole" in ftype:
            cands, other = (
                round_groups + hole_pats,
                slot_groups + slot_pats + pockets + grooves + notches,
            )
        elif "slot" in ftype or "langloch" in ftype:
            cands, other = slot_groups + slot_pats + grooves, round_groups + hole_pats
        elif "notch" in ftype:
            cands, other = notches + grooves, groups + hole_pats + slot_pats
        elif ftype == "pocket":
            # a "pocket" whose text describes a through-opening (window/
            # cutout/Durchgang) is a cutout semantically — it must match
            # measured through-voids, not only AAG floor pockets
            blob = f"{name} {f.get('description') or ''}".lower()
            if any(k in blob for k in ("through", "window", "cutout", "fenst", "durchg")):
                cands, other = groups + hole_pats + slot_pats + pockets + grooves + notches, []
            else:
                cands, other = (
                    pockets,
                    groups + hole_pats + slot_pats + grooves + notches,
                )
        elif ftype == "cutout":
            cands, other = groups + hole_pats + slot_pats + pockets + grooves + notches, []
        else:
            continue
        # narrow to the diameter-matched subset so a differently-sized void
        # cannot satisfy this feature's position/axis checks by pooling;
        # `other`/kind-mismatch reporting stays on the full void set
        pos_cands = _dia_matched(f, cands)
        _void_kind_axis_check(i, name, ftype, f, pos_cands, other, checks)
        _void_position_check(i, name, f, pos_cands, bbox_min, bbox_max, checks)
        if "slot" in ftype or "langloch" in ftype:
            _slot_orientation_check(i, name, ftype, f, pos_cands, checks)

    _unrequested_voids_check(
        spec_feats,
        round_groups + slot_groups + hole_pats,
        _bbox_bounds(bbox_min, bbox_max),
        checks,
    )


def _fillet_check(
    i: int,
    name: str,
    feature: dict[str, Any],
    measured: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Require explicit evidence for requested rounds; missing evidence on a
    reliable B-Rep is a failure, while approximate/faceted evidence stays
    UNVERIFIED rather than being treated as absence."""
    dims = feature.get("dims") if isinstance(feature.get("dims"), dict) else {}
    want = next(
        (
            float(dims[k])
            for k in ("radius", "fillet_radius", "r")
            if isinstance(dims.get(k), int | float)
            and not isinstance(dims.get(k), bool)
            and float(dims[k]) > 0
        ),
        None,
    )
    radii: list[float] = []
    for value in measured.get("rounds") or []:
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            radii.append(float(value))
    fillets = [row for row in measured.get("fillets") or [] if isinstance(row, dict)]
    radii.extend(
        float(row["radius"])
        for row in fillets
        if isinstance(row.get("radius"), int | float)
        and not isinstance(row.get("radius"), bool)
        and row["radius"] > 0
    )
    evidence: dict[str, Any] = {
        "feature": name,
        "expected_radius_mm": want,
        "measured_radii_mm": sorted(round(r, 3) for r in radii),
    }
    if not radii:
        approximate = bool(measured.get("approximate"))
        evidence["reason"] = (
            "fillet surface not detected on reliable B-Rep"
            if not approximate
            else "fillet recognition unavailable on approximate/faceted geometry"
        )
        checks.append(
            _check(
                f"feat_{i}_fillet",
                TIER_FEATURES,
                "UNVERIFIED" if approximate else "FAIL",
                evidence,
            )
        )
        return
    if want is None:
        checks.append(_check(f"feat_{i}_fillet", TIER_FEATURES, "PASS", evidence))
        return
    tolerance = max(0.5, 0.2 * want) if measured.get("approximate") else max(0.25, 0.1 * want)
    matches = [r for r in radii if abs(r - want) <= tolerance]
    if matches:
        evidence["matched_radius_mm"] = min(matches, key=lambda r: abs(r - want))
        checks.append(_check(f"feat_{i}_fillet", TIER_FEATURES, "PASS", evidence))
    else:
        evidence["reason"] = f"no measured fillet radius within {tolerance:g} mm of spec"
        checks.append(_check(f"feat_{i}_fillet", TIER_FEATURES, "FAIL", evidence))


def _unrequested_voids_check(  # noqa: C901
    spec_feats: list[Any],
    voids: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
    checks: list[dict[str, Any]],
) -> None:
    """Extra-void check: a measured void no spec feature asked for (a
    duplicate bore codegen forgot to suppress) is a placement failure even
    when every declared feature verifies. A void counts as matched when its
    dia is within 30% of some hole-like spec feature's declared dia, when a
    centre sits within 4 mm of a resolved spec position, or when a spec
    void feature declares neither dia nor position (in doubt = matched).
    Dia-matched groups may still carry excess centres beyond the declared
    count — the duplicate-in-one-group case. Conservative throughout: only
    voids with dia >= 3 mm are counted (chamfer/countersink pseudo-voids are
    never unrequested) and anything ambiguous counts as matched. An
    elongated void whose measured ``width_mm``/``length_mm`` spans match a
    slot feature's declared dims counts as matched too — its
    equivalent-area ``diameter`` never equals the declared width."""
    void_types = _HOLE_KIND_TYPES | {"pocket", "cutout"}
    feats = [
        f
        for f in spec_feats
        if isinstance(f, dict)
        and (
            (ft := str(f.get("type") or "").lower()) in void_types
            or "hole" in ft
            or "slot" in ft
            or "langloch" in ft
            or "notch" in ft
            or _wants_pattern(f)
        )
    ]
    if not feats:
        return
    info: list[dict[str, Any]] = []
    for f in feats:
        try:
            count = int(f.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        info.append(
            {
                "dia": _spec_void_dia(f),
                "wants": _void_pos_wants(f, bounds),
                "count": count,
                "feature": f,
            }
        )
    # an unconstrained spec void (no dia, no resolvable position) could be
    # any measured void — it can never prove one unrequested
    wildcard = any(i["dia"] is None and not i["wants"] for i in info)
    unmatched: list[dict[str, Any]] = []
    n_voids = 0
    for e in voids:
        dia = _void_dia(e)
        if dia is None or dia < 3.0:
            continue  # pseudo-void — never unrequested
        n_voids += 1
        if wildcard:
            continue
        dia_feats = [
            i for i in info if i["dia"] is not None and abs(dia - i["dia"]) <= 0.3 * i["dia"]
        ]
        centers = _void_center_maps(e)
        pos_hit = False
        position_candidates = dia_feats or [
            i for i in info if _void_kind_compatible(i["feature"], e)
        ]
        for item in position_candidates:
            if not item["wants"]:
                continue
            for center in centers:
                measured_axes = [ax for ax in item["wants"] if ax in center]
                if not measured_axes or all(
                    any(abs(center[ax] - value) <= 4.0 for value in item["wants"][ax])
                    for ax in measured_axes
                ):
                    # Ambiguous/unmeasured coordinates are conservative:
                    # they must not turn an otherwise requested void into an
                    # "unrequested" failure.
                    pos_hit = True
                    break
            if pos_hit:
                break
        # a slot's equivalent-area dia never matches its declared width —
        # a slot-typed feature declaring matching width/length dims counts
        # the elongated void as requested before the dia check runs
        slot_hit = any(_slot_dims_hit(i["feature"], e) for i in info)
        entry: dict[str, Any] = {
            "id": str(e.get("id") or ""),
            "dia": round(dia, 3),
        }
        if centers:
            entry["centers_by_axis"] = [
                {ax: round(value, 3) for ax, value in c.items()} for c in centers[:6]
            ]
        if not dia_feats and not pos_hit and not slot_hit:
            entry["reason"] = "no spec feature declares this void"
            unmatched.append(entry)
        elif dia_feats and centers:
            counts = [i["count"] for i in dia_feats]
            if counts and all(c > 0 for c in counts) and len(centers) > sum(counts):
                entry["reason"] = f"{len(centers)} voids exceed the declared count of {sum(counts)}"
                unmatched.append(entry)
    ev: dict[str, Any] = {"measured_voids": n_voids, "spec_void_features": len(feats)}
    if unmatched:
        ev["unmatched_ids"] = [u["id"] for u in unmatched]
        ev["unmatched"] = unmatched[:_MAX_EVIDENCE]
        checks.append(_check("unrequested_voids", TIER_FEATURES, "FAIL", ev))
    else:
        ev["reason"] = "every measured void corresponds to a spec feature"
        checks.append(_check("unrequested_voids", TIER_FEATURES, "PASS", ev))


def _threads(res: dict[str, Any], spec: dict[str, Any] | None, checks: list[dict[str, Any]]) -> None:
    """Explicit thread verification: declared standard parts / thread features
    must leave geometry evidence matching the executed detail mode — visible
    preview markers in preview mode, real helix objects in export mode."""
    declared_parts = [
        row for row in ((spec or {}).get("standard_parts") or []) if isinstance(row, dict)
    ]
    features = [row for row in ((spec or {}).get("features") or []) if isinstance(row, dict)]
    if not declared_parts and not any(
        "thread" in str(row.get("type") or "").lower() or row.get("thread_spec")
        for row in features
    ):
        return
    expected = 0
    try:
        from types import SimpleNamespace

        from cadx.cad.standard_parts import expected_thread_count
        from cadx.llm.schemas import Feature, StandardPartSpec

        spec_obj = SimpleNamespace(
            standard_parts=[StandardPartSpec.model_validate(row) for row in declared_parts],
            features=[Feature.model_validate(row) for row in features],
        )
        expected = expected_thread_count(spec_obj)
    except Exception:
        expected = 0
    if not expected:
        return
    detail_mode = str(res.get("detail_mode") or "preview")
    real = int(res.get("thread_count") or 0)
    markers = int(res.get("preview_thread_count") or 0)
    ev: dict[str, Any] = {
        "expected": expected,
        "detail_mode": detail_mode,
        "thread_count": real,
        "preview_thread_count": markers,
        "declared": [
            {
                "class": part.get("class_name"),
                "type": part.get("fastener_type"),
                "size": part.get("size"),
            }
            for part in declared_parts
        ],
    }
    if detail_mode == "export":
        if real >= expected:
            checks.append(_check("thread_features", TIER_FEATURES, "PASS", ev))
        else:
            ev["reason"] = (
                f"{real} real thread feature(s) measured; {expected} required "
                "for export — a smooth/preview shaft must not ship as final"
            )
            checks.append(_check("thread_features", TIER_FEATURES, "FAIL", ev))
        return
    if real + markers >= expected:
        checks.append(_check("thread_features", TIER_FEATURES, "PASS", ev))
    else:
        ev["reason"] = "no thread geometry and no visible preview-thread metadata"
        checks.append(_check("thread_features", TIER_FEATURES, "FAIL", ev))


def _structural(
    res: dict[str, Any], checks: list[dict[str, Any]], spec: dict[str, Any] | None = None
) -> None:
    ok = bool(res.get("ok"))
    evidence: dict[str, Any] = {"ok": ok}
    if res.get("error"):
        evidence["error"] = str(res["error"])[:300]
    checks.append(_check("executes", TIER_STRUCTURAL, "PASS" if ok else "FAIL", evidence))

    analysis = res.get("analysis") or {}
    feats = res.get("features") or {}
    brep_source = str(analysis.get("source") or "brep") == "brep"
    if not analysis:
        checks.append(
            _check(
                "single_valid_solid",
                TIER_STRUCTURAL,
                "UNVERIFIED",
                {"reason": "no analysis evidence — execution produced no shape"},
            )
        )
        checks.append(
            _check(
                "brep_defects", TIER_STRUCTURAL, "UNVERIFIED", {"reason": "no analysis evidence"}
            )
        )
    else:
        solid_count = int(analysis.get("solid_count") or 0)
        is_valid = bool(analysis.get("is_valid"))
        declared_parts = (spec or {}).get("standard_parts") or []
        declared_names = {
            str(part.get("class_name") or "")
            for part in declared_parts
            if isinstance(part, dict)
        }
        bom_names = {
            str(part.get("part_class") or "")
            for part in (res.get("bom") or [])
            if isinstance(part, dict)
        }
        declared_threaded_part = (
            bool(declared_names & bom_names) and int(res.get("thread_count") or 0) > 0
        )
        expected_assembly = len(declared_parts) > 1 or any(
            isinstance(part, dict) and int(part.get("quantity") or 1) > 1
            for part in declared_parts
        )
        valid_component_set = solid_count == 1 or declared_threaded_part or expected_assembly
        checks.append(
            _check(
                "single_valid_solid",
                TIER_STRUCTURAL,
                "PASS" if is_valid and valid_component_set else "FAIL",
                {
                    "is_valid": is_valid,
                    "solid_count": solid_count,
                    "declared_standard_component_set": valid_component_set,
                    "watertight": analysis.get(
                        "watertight_after_repair", analysis.get("watertight")
                    ),
                },
            )
        )
        defects = [d for d in (feats.get("defects") or []) if isinstance(d, dict)]
        located = [d for d in defects if "error" not in d]
        if not brep_source:
            # faceted/mesh source — BRepCheck never ran, nothing to locate
            checks.append(
                _check(
                    "brep_defects",
                    TIER_STRUCTURAL,
                    "UNVERIFIED",
                    {"reason": "no B-Rep (tessellated/mesh source) — BRepCheck did not run"},
                )
            )
        elif located:
            checks.append(
                _check(
                    "brep_defects",
                    TIER_STRUCTURAL,
                    "FAIL",
                    {"defects": located[:_MAX_EVIDENCE]},
                )
            )
        elif is_valid:
            checks.append(
                _check(
                    "brep_defects",
                    TIER_STRUCTURAL,
                    "PASS",
                    {"reason": "no BRepCheck defects located"},
                )
            )
        else:
            # the solid-validity FAIL above already carries the verdict; the
            # defect locator simply produced no coordinates this run
            checks.append(
                _check(
                    "brep_defects",
                    TIER_STRUCTURAL,
                    "UNVERIFIED",
                    {"reason": "invalid B-Rep but defect location produced no evidence"},
                )
            )

    gate = feats.get("step_roundtrip") or {}
    if not brep_source:
        checks.append(
            _check(
                "step_roundtrip",
                TIER_STRUCTURAL,
                "UNVERIFIED",
                {"reason": "no B-Rep (tessellated/mesh source) — STEP gate did not run"},
            )
        )
    elif gate.get("ok") is True:
        checks.append(
            _check(
                "step_roundtrip",
                TIER_STRUCTURAL,
                "PASS",
                {k: gate[k] for k in ("reimported_valid", "volume_delta_pct") if k in gate},
            )
        )
    elif gate.get("warning") or gate.get("ok") is False:
        checks.append(
            _check(
                "step_roundtrip",
                TIER_STRUCTURAL,
                "FAIL",
                {"warning": str(gate.get("warning") or "STEP roundtrip failed")[:400]},
            )
        )
    else:
        checks.append(
            _check(
                "step_roundtrip",
                TIER_STRUCTURAL,
                "UNVERIFIED",
                {
                    "reason": str(gate.get("skipped") or gate.get("error") or "gate did not run")[
                        :300
                    ]
                },
            )
        )


def _envelope(
    spec: dict[str, Any] | None,
    res: dict[str, Any],
    tol_pct: float,
    tol_mm: float,
    checks: list[dict[str, Any]],
    *,
    bound_only: bool = False,
) -> None:
    bbox = (res.get("analysis") or {}).get("bbox_size")
    if not spec:
        checks.append(
            _check("envelope", TIER_ENVELOPE, "UNVERIFIED", {"reason": "no spec supplied"})
        )
        return
    emitted = False
    for ax, key in (("x", "length_x"), ("y", "width_y"), ("z", "height_z")):
        dim = spec.get(key)
        if dim is None:
            continue
        want = _dim_value(dim)
        if want is None:
            continue
        emitted = True
        name = f"envelope_{ax}"
        source = _dim_source(dim)
        if source != "stated":
            # inferred/assumed dims are declared intent but not a hard
            # requirement — same convention as worker-side compare_dims
            checks.append(
                _check(
                    name,
                    TIER_ENVELOPE,
                    "UNVERIFIED",
                    {
                        "reason": f"dim source is '{source or 'unset'}', not a stated requirement",
                        "expected_mm": want,
                    },
                )
            )
            continue
        if not isinstance(bbox, list | tuple) or len(bbox) != 3:
            checks.append(
                _check(
                    name,
                    TIER_ENVELOPE,
                    "UNVERIFIED",
                    {"reason": "no measured bbox", "expected_mm": want},
                )
            )
            continue
        actual = float(bbox["xyz".index(ax)])
        tol = max(want * tol_pct / 100.0, tol_mm)
        if bound_only:
            # topopt/refine result: the stated dim is an upper bound — the
            # optimized part must fit inside the envelope, shrinking below
            # it is the point of the exercise, not a defect
            over = actual - want
            checks.append(
                _check(
                    name,
                    TIER_ENVELOPE,
                    "PASS" if over <= tol else "FAIL",
                    {
                        "expected_mm": want,
                        "actual_mm": round(actual, 3),
                        "over_mm": round(over, 3),
                        "tol_mm": round(tol, 3),
                        "bound": "upper",
                    },
                )
            )
            continue
        delta = abs(actual - want)
        checks.append(
            _check(
                name,
                TIER_ENVELOPE,
                "PASS" if delta <= tol else "FAIL",
                {
                    "expected_mm": want,
                    "actual_mm": round(actual, 3),
                    "delta_mm": round(delta, 3),
                    "tol_mm": round(tol, 3),
                },
            )
        )
    if not emitted:
        checks.append(
            _check(
                "envelope",
                TIER_ENVELOPE,
                "UNVERIFIED",
                {"reason": "spec carries no dimension requirements"},
            )
        )


def _features(
    spec: dict[str, Any] | None,
    measured: dict[str, Any],
    feature_findings: list[dict[str, Any]] | None,
    checks: list[dict[str, Any]],
) -> None:
    spec_feats = (spec or {}).get("features") or []
    if not spec_feats:
        return
    if not measured:
        checks.append(
            _check(
                "spec_features",
                TIER_FEATURES,
                "UNVERIFIED",
                {"reason": "feature measurement produced no evidence"},
            )
        )
        return
    if feature_findings is None:
        checks.append(
            _check(
                "spec_features",
                TIER_FEATURES,
                "UNVERIFIED",
                {
                    "reason": "compare_features findings not supplied",
                    "spec_features": len(spec_feats),
                },
            )
        )
    else:
        majors = [f for f in feature_findings if str(f.get("severity")) == "major"]
        checks.append(
            _check(
                "spec_features",
                TIER_FEATURES,
                "FAIL" if majors else "PASS",
                {
                    "spec_features": len(spec_feats),
                    "major_findings": len(majors),
                    "findings": feature_findings[:_MAX_EVIDENCE],
                },
            )
        )
    # declared connection spans: "strut from A to B" must carry material
    # along the whole declared path — a detached bar passes every hole/bbox
    # check while missing its host (bracket run: strut built from x=-5
    # instead of the leg at x=-55, critic still called it connected)
    for i, sp in enumerate((measured.get("spans") or [])[:8]):
        if not isinstance(sp, dict):
            continue
        cov = sp.get("coverage")
        ok = bool(sp.get("ok")) and cov is not None and float(cov) >= 0.85
        checks.append(
            _check(
                f"span_{i}",
                TIER_FEATURES,
                "PASS" if ok else "FAIL",
                {
                    "name": sp.get("name"),
                    "coverage": cov,
                    "points": sp.get("points"),
                    "required": "material along the declared connection path",
                },
            )
        )
    if not any(_wants_pattern(f) for f in spec_feats):
        return
    if not measured.get("recognition_ran"):
        checks.append(
            _check(
                "hole_patterns",
                TIER_FEATURES,
                "UNVERIFIED",
                {"reason": "feature recognition did not run (recognition_ran false)"},
            )
        )
        return
    patterns = measured.get("hole_patterns") or []
    if patterns:
        checks.append(
            _check(
                "hole_patterns",
                TIER_FEATURES,
                "PASS",
                {
                    "patterns": [
                        {
                            k: p.get(k)
                            for k in ("id", "kind", "count", "diameter", "pitch")
                            if p.get(k) not in (None, "", 0, 0.0)
                        }
                        for p in patterns[:6]
                        if isinstance(p, dict)
                    ]
                },
            )
        )
    else:
        checks.append(
            _check(
                "hole_patterns",
                TIER_FEATURES,
                "FAIL",
                {"reason": "spec declares a hole pattern but none was recognised"},
            )
        )


def _robustness(
    spec: dict[str, Any] | None, res: dict[str, Any], checks: list[dict[str, Any]]
) -> None:
    dfm = res.get("dfm")
    warnings = (dfm or {}).get("warnings") or []
    if dfm is None:
        checks.append(
            _check(
                "dfm",
                TIER_ROBUSTNESS,
                "UNVERIFIED",
                {"reason": "DfM did not run (light pipeline or skipped)"},
            )
        )
    else:
        high = [w for w in warnings if w.get("severity") == "high"]
        checks.append(
            _check(
                "dfm",
                TIER_ROBUSTNESS,
                "FAIL" if high else "PASS",
                {
                    "warnings": len(warnings),
                    "high": len(high),
                    "high_warnings": [str(w.get("message")) for w in high[:4]],
                },
            )
        )
    want_wall = (spec or {}).get("min_wall_mm")
    if want_wall is None:
        return
    if dfm is None:
        checks.append(
            _check(
                "min_wall",
                TIER_ROBUSTNESS,
                "UNVERIFIED",
                {
                    "reason": "no wall-thickness probe evidence (DfM skipped)",
                    "expected_min_mm": want_wall,
                },
            )
        )
        return
    # indirect evidence — the DfM thin_wall heuristic owns this check
    walls = [w for w in warnings if w.get("code") == "thin_wall"]
    checks.append(
        _check(
            "min_wall",
            TIER_ROBUSTNESS,
            "FAIL" if walls else "PASS",
            {
                "expected_min_mm": want_wall,
                "thin_wall_warnings": [str(w.get("message")) for w in walls[:4]],
            },
        )
    )


def verify_spec(
    spec: dict[str, Any] | None,
    res: dict[str, Any],
    *,
    feature_findings: list[dict[str, Any]] | None = None,
    measured_features: dict[str, Any] | None = None,
    dim_tolerance_pct: float = 2.0,
    dim_tolerance_mm: float = 0.5,
    envelope_bound_only: bool = False,
) -> dict[str, Any]:
    """Build the canonical conformance report for one executed candidate.

    ``spec`` is the minimal spec sub-dict shipped with the execute payload
    (dims as Dim dumps under ``length_x``/``width_y``/``height_z``, a
    ``features`` list of Feature dumps, optional ``min_wall_mm``). ``res`` is
    the ExecResult dict (``ok``/``analysis``/``features``/``dfm``...).
    ``feature_findings`` are the already-computed ``compare_features`` results;
    ``measured_features`` overrides ``res["features"]`` as the feature-tier
    evidence source (mesh pipeline passes the adapted mesh measurements).
    ``envelope_bound_only`` switches the envelope tier to bound semantics for
    topopt/refine results — stated dims are upper bounds the optimized part
    must fit inside, not targets to match.

    Returns ``{"status": "PASS"|"FAIL"|"UNVERIFIED", "checks": [...],
    "summary": str}`` — plain JSON-serializable dicts only.
    """
    checks: list[dict[str, Any]] = []
    measured = measured_features if measured_features is not None else (res.get("features") or {})

    _structural(res, checks, spec)
    _threads(res, spec, checks)
    _envelope(
        spec, res, dim_tolerance_pct, dim_tolerance_mm, checks, bound_only=envelope_bound_only
    )
    _features(spec, measured, feature_findings, checks)
    if (spec or {}).get("features"):
        analysis = res.get("analysis") or {}
        _feature_placement(
            spec,
            measured,
            analysis.get("bbox_size"),
            analysis.get("bbox_min"),
            analysis.get("bbox_max"),
            dim_tolerance_pct,
            dim_tolerance_mm,
            checks,
        )
    _robustness(spec, res, checks)

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_unv = sum(1 for c in checks if c["status"] == "UNVERIFIED")
    status: Status = "FAIL" if n_fail else ("PASS" if n_pass else "UNVERIFIED")
    failing = [c["name"] for c in checks if c["status"] == "FAIL"]
    parts = [f"{n_pass} pass", f"{n_fail} fail", f"{n_unv} unverified"]
    summary = f"verify_spec {status} — " + ", ".join(parts)
    if failing:
        summary += f"; failing: {', '.join(failing[:4])}"
    if n_unv:
        summary += " (UNVERIFIED = evidence absent, not a failure)"
    return {"status": status, "checks": checks, "summary": summary}
