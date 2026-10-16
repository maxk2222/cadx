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
- robustness: DfM warnings + declared min-wall — hints tier. UNVERIFIED when
  the evidence is absent; never FAIL just because a detector was skipped
  (a faceted-solid mesh fallback already owns its own tier).

Pure function over plain dicts — no geometry imports — so the executor
pipeline (``engine.py``) and tests call the same code path.
"""

from __future__ import annotations

import re
from typing import Any

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
                            for k in ("kind", "count", "diameter", "pitch")
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
