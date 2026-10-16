"""Design-robustness audit: perturb each numeric PARAMS value ±ε and re-run.

Ported from build123d-mcp's ``design_audit`` (the Arko-T §6 "design to edit,
not a shape to render" idea): every other gate inspects the executed geometry;
this one inspects the *parameterisation* that produced it. A param whose small
change collapses the solid or drifts the volume reveals a brittle design.

The audit is orchestration only — each perturbation re-executes the SAME code
through the dependency-injected ``runner`` callable (production wires the
sandbox pipeline with ``params_override``; tests inject a fake). Per-run errors
and timeouts are captured into that param's entry and never abort the audit.
"""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cadx.cad.allowlist import check_code
from cadx.db.models import ProjectSettings
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

# a params_override -> exec-result-dict callable; see make_pipeline_runner
Runner = Callable[[dict[str, object]], Mapping[str, Any]]

# geometry drift beyond which a successful perturbation is flagged brittle —
# comfortably above the ±ε change a well-behaved param legitimately produces
DEFAULT_DRIFT_PCT = 10.0

_VERDICT_RANK = {"robust": 0, "brittle": 1, "fails": 2}


def _is_numeric(v: object) -> bool:
    # bools are ints in Python but are categorical switches, not dimensions
    return isinstance(v, int | float) and not isinstance(v, bool)


def _perturbed(value: float, epsilon: float, sign: int) -> float:
    """value * (1 ± ε) — int params get an integer delta of at least ±1 so a
    ±5% nudge on e.g. count=3 still perturbs (and stays an int for code that
    feeds it to range()/repeat)."""
    if isinstance(value, int):
        return value + sign * max(1, round(abs(value) * epsilon))
    return value * (1 + sign * epsilon)


def make_pipeline_runner(
    code: str,
    base_dir: Path,
    *,
    references: list[Any] | None = None,
    settings: ProjectSettings | None = None,
    runtime: RuntimeSettings | None = None,
    mesh_solid: Path | None = None,
    mesh_bbox: list[float] | None = None,
    anchors: dict[str, Any] | None = None,
) -> Runner:
    """Production runner: light pipeline run (sandbox + analysis, no renders/
    DfM) per perturbation, each into its own subdir of ``base_dir``."""
    from cadx.cad.engine import run_pipeline

    counter = itertools.count()

    def _run(overrides: dict[str, object]) -> Mapping[str, Any]:
        res = run_pipeline(
            code=code,
            out_dir=base_dir / f"run{next(counter):02d}",
            params_override=dict(overrides) or None,
            references=references,
            settings=settings,
            runtime=runtime,
            name="audit",
            mesh_solid=mesh_solid,
            mesh_bbox=mesh_bbox,
            anchors=anchors,
            light=True,
        )
        return res.to_dict()

    return _run


def _summarize_run(d: Mapping[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Distil an exec-result dict to the fields the audit reports on, with
    volume/dims deltas vs the baseline run."""
    analysis = d.get("analysis") or {}
    volume = analysis.get("volume")
    dims = analysis.get("bbox_size") or []
    warnings = list(
        dict.fromkeys(list(d.get("warnings") or []) + list(analysis.get("warnings") or []))
    )
    out: dict[str, Any] = {
        "ok": bool(d.get("ok")),
        "error": d.get("error"),
        "is_valid": analysis.get("is_valid"),
        "volume": volume,
        "bbox_size": dims,
        "warnings": warnings,
        "duration_s": (d.get("timings") or {}).get("execute_s"),
    }
    if baseline is not None:
        bv = baseline.get("volume")
        out["volume_delta_pct"] = (
            round((volume - bv) / abs(bv) * 100, 2)
            if isinstance(volume, int | float) and isinstance(bv, int | float) and bv
            else None
        )
        bd = baseline.get("bbox_size") or []
        out["dims_delta_pct"] = [
            round((v - b) / abs(b) * 100, 2) if isinstance(v, int | float) and b else None
            for v, b in zip(dims, bd, strict=False)
        ]
    return out


def _safe_run(runner: Runner, overrides: dict[str, object]) -> dict[str, Any]:
    """One perturbation run — a raising/timing-out runner must never abort
    the audit; the failure becomes this run's entry."""
    try:
        d = runner(overrides)
        return dict(d) if isinstance(d, Mapping) else {"ok": False, "error": "bad runner result"}
    except Exception as e:
        return {"ok": False, "error": f"runner raised {type(e).__name__}: {e}"[:300]}


def _verdict(
    plus: dict[str, Any], minus: dict[str, Any], baseline: dict[str, Any], drift_pct: float
) -> tuple[str, str]:
    base_warnings = set(baseline.get("warnings") or [])
    for side, r in (("plus", plus), ("minus", minus)):
        if not r["ok"]:
            return "fails", f"{side} run failed: {(r.get('error') or 'unknown')[:160]}"
        if r.get("is_valid") is False:
            return "fails", f"{side} run produced invalid geometry"
        if r.get("volume") is None:
            return "fails", f"{side} run produced no measurable geometry"
    triggers: list[str] = []
    for side, r in (("plus", plus), ("minus", minus)):
        vd = r.get("volume_delta_pct")
        if vd is not None and abs(vd) > drift_pct:
            triggers.append(f"{side}: volume drifted {vd:+.1f}%")
        for axis, dd in zip("xyz", r.get("dims_delta_pct") or [], strict=False):
            if dd is not None and abs(dd) > drift_pct:
                triggers.append(f"{side}: bbox {axis} drifted {dd:+.1f}%")
        new_warnings = [w for w in r["warnings"] if w not in base_warnings]
        if new_warnings:
            triggers.append(f"{side}: {new_warnings[0][:100]}")
    if triggers:
        return "brittle", "; ".join(triggers)[:300]
    return "robust", "both perturbations stayed within drift tolerance"


def audit_params(
    code: str,
    params: Mapping[str, object],
    *,
    epsilon: float = 0.05,
    max_params: int = 8,
    drift_pct: float = DEFAULT_DRIFT_PCT,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Perturb each numeric PARAMS value ±epsilon and re-run the validity gate.

    ``params`` are the baseline values to audit around (typically the declared
    PARAMS merged with any job overrides); names not declared in the code's
    PARAMS literal are ignored — the sandbox would reject them anyway.

    Returns ``{"status", "baseline", "params", "audited", "skipped", ...}`` with
    per-param ``{"value", "plus", "minus", "verdict", "reason"}`` entries.
    """
    if not 0 < epsilon < 1:
        return {"status": "fails", "error": "epsilon must be in (0, 1)", "audited": 0}
    if max_params < 1:
        return {"status": "fails", "error": "max_params must be >= 1", "audited": 0}

    check = check_code(code)
    if not check.ok:
        return {
            "status": "fails",
            "error": "static check failed: " + "; ".join(check.errors[:3])[:300],
            "audited": 0,
            "params": {},
            "skipped": [],
        }
    declared = check.params

    # baseline values: declared defaults overridden by the supplied params
    effective: dict[str, object] = {k: params.get(k, v) for k, v in declared.items()}
    skipped: list[dict[str, str]] = [
        {"name": k, "reason": "not declared in PARAMS"} for k in params if k not in declared
    ]

    candidates: list[str] = []
    for name, value in effective.items():
        if not _is_numeric(value):
            skipped.append({"name": name, "reason": "non-numeric / categorical"})
        elif isinstance(value, float) and value == 0.0:
            skipped.append({"name": name, "reason": "zero value — ±ε is a no-op"})
        else:
            candidates.append(name)
    audited_names, capped = candidates[:max_params], candidates[max_params:]
    skipped += [{"name": n, "reason": "max_params cap"} for n in capped]

    if not audited_names:
        return {
            "status": "robust",
            "audited": 0,
            "params": {},
            "skipped": skipped,
            "epsilon": epsilon,
            "note": "no numeric parameters to audit",
        }

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if runner is None:
        tmp = tempfile.TemporaryDirectory(prefix="cadx_audit_")
        runner = make_pipeline_runner(code, Path(tmp.name))
    try:
        baseline_overrides = {k: effective[k] for k in declared}
        baseline = _summarize_run(_safe_run(runner, baseline_overrides), None)
        if not baseline["ok"] or baseline.get("is_valid") is False:
            return {
                "status": "fails",
                "baseline": baseline,
                "audited": 0,
                "params": {},
                "skipped": skipped,
                "epsilon": epsilon,
                "note": "baseline run failed — parameter robustness is not decidable",
            }

        report_params: dict[str, Any] = {}
        worst = "robust"
        for name in audited_names:
            value = effective[name]
            assert isinstance(value, int | float)  # narrowed by _is_numeric above
            entry: dict[str, Any] = {"value": value}
            runs: dict[str, dict[str, Any]] = {}
            for side, sign in (("plus", 1), ("minus", -1)):
                pv = _perturbed(value, epsilon, sign)
                ov = {**baseline_overrides, name: pv}
                run = _summarize_run(_safe_run(runner, ov), baseline)
                run["override"] = pv
                runs[side] = run
                entry[side] = run
            verdict, reason = _verdict(runs["plus"], runs["minus"], baseline, drift_pct)
            entry["verdict"] = verdict
            entry["reason"] = reason
            report_params[name] = entry
            if _VERDICT_RANK[verdict] > _VERDICT_RANK[worst]:
                worst = verdict
            log.info(
                "design_audit.param",
                param=name,
                verdict=verdict,
                reason=reason[:120],
            )

        return {
            "status": worst,
            "baseline": {
                "ok": baseline["ok"],
                "is_valid": baseline.get("is_valid"),
                "volume": baseline.get("volume"),
                "bbox_size": baseline.get("bbox_size"),
                "warnings": baseline.get("warnings") or [],
            },
            "params": report_params,
            "audited": len(audited_names),
            "skipped": skipped,
            "epsilon": epsilon,
            "drift_pct": drift_pct,
        }
    finally:
        if tmp is not None:
            tmp.cleanup()
