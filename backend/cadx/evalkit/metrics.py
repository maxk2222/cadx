"""Metrics harvest — StageOutcome -> the metrics block persisted in
stages/<id>.json and aggregated by report/baseline.

Driver-agnostic by construction: both drivers normalize into StageOutcome,
so every metric is defined once here.
"""

from __future__ import annotations

from typing import Any

from cadx.evalkit.gates import verify_spec_report
from cadx.evalkit.models import StageOutcome
from cadx.evalkit.stats import iterations, step_stats


def harvest(outcome: StageOutcome) -> dict[str, Any]:
    analysis = outcome.analysis or {}
    usage = outcome.usage or {}
    report = verify_spec_report({"mesh": outcome.mesh or {}}, outcome.issues)
    fem = outcome.fem or {}
    top = (outcome.mesh or {}).get("topopt") if isinstance(outcome.mesh, dict) else None
    bbox = analysis.get("bbox_size")
    watertight = analysis.get("watertight_after_repair")
    if watertight is None:
        watertight = analysis.get("watertight")
    return {
        "wall_s": round(outcome.wall_s, 1),
        "status": outcome.status,
        "score": outcome.score,
        "iterations": max(outcome.iterations, iterations(outcome.trace_records)),
        "resolved_mode": outcome.resolved_mode,
        "mode_reason": " | ".join(
            p
            for p in (
                str((outcome.plan or {}).get("approach_reason") or ""),
                outcome.mode_note,
            )
            if p
        ),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "calls": int(usage.get("calls") or 0),
        },
        "step_stats": step_stats(outcome.trace_records),
        "execs": int(outcome.extras.get("execs") or 0),
        "exec_s": round(float(outcome.extras.get("exec_s") or 0.0), 1),
        "bbox": [round(float(x), 2) for x in bbox] if bbox else None,
        "features": outcome.features,
        "watertight": bool(watertight) if watertight is not None else None,
        "verify_spec": {
            "status": str(report["status"]) if report else None,
            "fails": [
                c.get("name")
                for c in (report or {}).get("checks") or []
                if isinstance(c, dict) and c.get("status") == "FAIL"
            ],
            "unverified": [
                c.get("name")
                for c in (report or {}).get("checks") or []
                if isinstance(c, dict) and c.get("status") == "UNVERIFIED"
            ],
        },
        "fem": {
            "worst_safety_factor": fem.get("worst_safety_factor"),
            "safety_factor": fem.get("safety_factor"),
            "solve_ok": fem.get("solve_ok") if "solve_ok" in fem else fem.get("ok"),
            "cases": [
                {k: c.get(k) for k in ("name", "safety_factor", "solve_ok") if k in c}
                for c in (fem.get("cases") or [])[:8]
                if isinstance(c, dict)
            ],
        },
        "topopt": (
            {
                k: top.get(k)
                for k in (
                    "volfrac_achieved",
                    "volfrac",
                    "sf_verified",
                    "sf_export",
                    "sf_homog",
                    "sf_floor",
                    "n_bodies",
                    "bottleneck_mm2",
                    "load_path_ok",
                )
                if k in top
            }
            if top
            else None
        ),
        "issues": [
            f"{i.get('what', '')} {i.get('fix_hint', '')}".strip()
            for i in outcome.issues or []
            if isinstance(i, dict)
        ][:20],
        "critique_score": (outcome.critique or {}).get("score"),
        "assets": outcome.asset_ids,
        "artifacts": outcome.artifacts,
    }
