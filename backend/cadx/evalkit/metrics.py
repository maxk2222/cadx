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
    topopt_block: dict[str, Any] | None = None
    if top:
        topopt_block = {
            k: top.get(k)
            for k in (
                "volfrac_achieved",
                "volfrac",
                "sf_verified",
                "sf_export",
                "sf_homog",
                "sf_verified_basis",
                "sf_floor",
                "n_bodies",
                "bottleneck_mm2",
                "load_path_ok",
                "enclosed_void_mm3",
                "n_sealed_cavities",
                "largest_void_mm3",
                "void_frac_of_solid",
                "sealed_cavities_mm3",
                "sealed_voids_filled_mm3",
                "shell_min_wall_mm",
                "void_pinned_cells_total",
                "void_pinned_mm3",
                "void_pin_iters",
                "void_pin_sym_mirrored",
                "void_pinning_applied",
                "warm_start",
                "guide_solver",
                "rescue_dropped_frac",
            )
            if k in top
        }
    else:
        # keep-base: the agent rejected every candidate — report the best
        # rejected run's numbers so gates measure a real sf/volfrac rather
        # than a bare "no topopt happened" null
        kept = (
            (outcome.mesh or {}).get("topopt_kept_base") if isinstance(outcome.mesh, dict) else None
        ) or {}
        rejected = kept.get("rejected_runs") or []
        best = max(
            (r for r in rejected if isinstance(r, dict)),
            key=lambda r: float(r.get("worst_sf") or 0.0),
            default=None,
        )
        if best is not None:
            # a solved sf_verified of 0.0 IS a verdict — never or-chain it away
            ver = best.get("verification") or {}
            sf_verified = ver.get("sf_verified")
            if sf_verified is None:
                sf_verified = best.get("worst_sf")
            topopt_block = {
                "kept_base": True,
                "volfrac_achieved": best.get("achieved"),
                "sf_verified": sf_verified,
                "sf_export": best.get("sf_export"),
                "n_bodies": best.get("n_bodies_export"),
                "load_path_ok": best.get("load_path_ok"),
                # sealed-void telemetry of the best rejected run — the
                # compact report mirrors the gate-side kept-base fallback
                "enclosed_void_mm3": ver.get("enclosed_void_mm3"),
                "n_sealed_cavities": ver.get("n_sealed_cavities"),
                "largest_void_mm3": ver.get("largest_void_mm3"),
                "sealed_cavities_mm3": ver.get("sealed_cavities_mm3"),
                "void_frac_of_solid": ver.get("void_frac_of_solid"),
                "export_grid": ver.get("export_grid"),
            }
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
        "features_available": (
            outcome.features_available
            if outcome.features_available is not None
            else (True if outcome.features is not None else None)
        ),
        "features_error": outcome.features_error,
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
        "topopt": topopt_block,
        "issues": [
            f"{i.get('what', '')} {i.get('fix_hint', '')}".strip()
            for i in outcome.issues or []
            if isinstance(i, dict)
        ][:20],
        "critique_score": (outcome.critique or {}).get("score"),
        "assets": outcome.asset_ids,
        "artifacts": outcome.artifacts,
    }
