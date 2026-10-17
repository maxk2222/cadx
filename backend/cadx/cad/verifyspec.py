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

from cadx.cad.poshint import parse_position_hints, resolve_end_hint

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
    r"\b([xyz])[\s-]*axis\b|\balong\s+(?:the\s+)?([xyz])(?:[\s-]*axis)?\b", re.I
)
# "y faces"/"+x side" — the envelope face a protrusion feature sits on
_HINT_AXIS_RE = re.compile(r"(?<![\w])[+-]?([xyz])[\s-]*(?:faces?|sides?|seite)\b", re.I)
_POS_TOL_MM = 2.0  # "x=12" offsets verified within ±2 mm
_POS_SIDE_EPS = 0.5  # a centre this close to the mid-plane is ambiguous


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
    field first, then a narrow axis-phrase regex on position/description."""
    ax = _axis_letter(f.get("axis"))
    if ax:
        return ax
    text = f"{f.get('position') or ''} {f.get('description') or ''}"
    m = _AXIS_TEXT_RE.search(text)
    if m:
        return next(g for g in m.groups() if g).lower()
    return ""


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
    _c, faces, _ends = parse_position_hints(text)
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
    matching = [g for g in cands if _axis_letter(g.get("axis")) == want_ax]
    ev = {
        **base,
        "measured_axes": _axes_of(cands),
        "measured_ids": _measured_ids(matching or cands),
    }
    if other:
        ev["other_kind_axes"] = _axes_of(other)
    if matching:
        checks.append(_check(check_name, TIER_FEATURES, "PASS", ev))
    else:
        ev["reason"] = f"no measured {ftype} runs along the {want_ax} axis"
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
    centers: list[Any], wants: dict[str, list[float]]
) -> tuple[list[dict[str, Any]], int]:
    """Per-axis absolute-coord check against measured [x, y] centres: each
    expected value passes when the nearest centre is within _POS_TOL_MM.
    z stays unverifiable — centres carry no z."""
    violations: list[dict[str, Any]] = []
    verified = 0
    for ax, ax_wants in sorted(wants.items()):
        if ax == "z":
            continue
        idx = 0 if ax == "x" else 1
        for want in ax_wants:
            nearest = min(centers, key=lambda c: abs(float(c[idx]) - want))
            delta = abs(float(nearest[idx]) - want)
            if delta <= _POS_TOL_MM:
                verified += 1
            else:
                violations.append(
                    {
                        "axis": ax,
                        "expected_mm": want,
                        "measured_mm": round(float(nearest[idx]), 3),
                        "delta_mm": round(delta, 3),
                    }
                )
    return violations, verified


def _void_position_check(
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
    bounds they stay unverifiable, never a guess. Measured void centres
    carry plan-view [x, y] only, so z constraints stay unverifiable."""
    pos = str(f.get("position") or "").strip()
    if not pos:
        return
    check_name = f"feat_{i}_position"
    coords, faces, ends = parse_position_hints(pos)
    base: dict[str, Any] = {"feature": name, "position": pos[:160]}
    if not coords and not faces and not ends:
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
    if ends:
        expected["ends"] = ends
    centers = [
        c
        for g in cands
        for c in (g.get("centers") or [])
        if isinstance(c, list | tuple) and len(c) >= 2
    ]
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
    wants: dict[str, list[float]] = {}
    for ax, want in coords.items():
        wants.setdefault(ax, []).append(want)
    for e in ends:
        bnd = bounds.get(e["axis"])
        if bnd is not None:
            wants.setdefault(e["axis"], []).append(
                resolve_end_hint(e["axis"], e["end"], e["offset_mm"], *bnd)
            )
    violations, verified = _centre_coord_check(centers, wants)
    for face in faces:
        if face[1] == "z":
            continue  # not verifiable from [x, y] centres
        idx = 0 if face[1] == "x" else 1
        vals = [float(c[idx]) for c in centers]
        best = max(vals) if face[0] == "+" else min(vals)
        if (face[0] == "+" and best > _POS_SIDE_EPS) or (face[0] == "-" and best < -_POS_SIDE_EPS):
            verified += 1
        elif abs(best) <= _POS_SIDE_EPS:
            continue  # mid-plane centre — ambiguous, not wrong
        else:
            violations.append({"face": face, "measured_mm": round(best, 3)})
    ev = {
        **base,
        "expected": expected,
        "measured_centers": [[round(float(v), 3) for v in c] for c in centers[:6]],
        "measured_ids": _measured_ids(cands),
    }
    if violations:
        ev["violations"] = violations[:_MAX_EVIDENCE]
        checks.append(_check(check_name, TIER_FEATURES, "FAIL", ev))
    elif verified and len(coords) + len(faces) + len(ends) == verified:
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
        if ftype in _PROTRUSION_TYPES:
            _protrusion_check(i, name, ftype, f, stated_env, bosses, bbox, tol_pct, tol_mm, checks)
            continue
        if ftype in _HOLE_KIND_TYPES or "hole" in ftype:
            cands, other = (
                round_groups + hole_pats,
                slot_groups + slot_pats + pockets + grooves + notches,
            )
        elif "slot" in ftype:
            cands, other = slot_groups + slot_pats + grooves, round_groups + hole_pats
        elif "notch" in ftype:
            cands, other = notches + grooves, groups + hole_pats + slot_pats
        elif ftype == "pocket":
            cands, other = (
                pockets,
                groups + hole_pats + slot_pats + grooves + notches,
            )
        elif ftype == "cutout":
            cands, other = groups + hole_pats + slot_pats + pockets + grooves + notches, []
        else:
            continue
        _void_kind_axis_check(i, name, ftype, f, cands, other, checks)
        _void_position_check(i, name, f, cands, bbox_min, bbox_max, checks)


def _structural(res: dict[str, Any], checks: list[dict[str, Any]]) -> None:
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
        checks.append(
            _check(
                "single_valid_solid",
                TIER_STRUCTURAL,
                "PASS" if is_valid and solid_count == 1 else "FAIL",
                {
                    "is_valid": is_valid,
                    "solid_count": solid_count,
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
) -> dict[str, Any]:
    """Build the canonical conformance report for one executed candidate.

    ``spec`` is the minimal spec sub-dict shipped with the execute payload
    (dims as Dim dumps under ``length_x``/``width_y``/``height_z``, a
    ``features`` list of Feature dumps, optional ``min_wall_mm``). ``res`` is
    the ExecResult dict (``ok``/``analysis``/``features``/``dfm``...).
    ``feature_findings`` are the already-computed ``compare_features`` results;
    ``measured_features`` overrides ``res["features"]`` as the feature-tier
    evidence source (mesh pipeline passes the adapted mesh measurements).

    Returns ``{"status": "PASS"|"FAIL"|"UNVERIFIED", "checks": [...],
    "summary": str}`` — plain JSON-serializable dicts only.
    """
    checks: list[dict[str, Any]] = []
    measured = measured_features if measured_features is not None else (res.get("features") or {})

    _structural(res, checks)
    _envelope(spec, res, dim_tolerance_pct, dim_tolerance_mm, checks)
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
