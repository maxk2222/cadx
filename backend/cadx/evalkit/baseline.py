"""Baselines — committed snapshots of eval results for regression diffs.

`baseline save <label>` condenses the latest run per (case, variant) under
`evals/results/` into `evals/baselines/<label>.json`. `baseline diff a b`
(or `report --baseline`) flags gate flips and metric regressions.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cadx.evalkit.models import CaseResult
from cadx.evalkit.paths import BASELINES_DIR, RESULTS_DIR
from cadx.evalkit.report import load_results

# metrics compared numerically in diffs (higher-is-better -> lower-is-better)
_HIGHER_BETTER = {"score", "pass_rate"}
_LOWER_BETTER = {"wall_s", "prompt_tokens", "completion_tokens", "exec_s"}
_REGRESS_FIELDS = {
    "score": ("higher", 3.0),
    "wall_s": ("lower", 0.20),  # >20% slower = regression
    "prompt_tokens": ("lower", 0.25),
    "completion_tokens": ("lower", 0.25),
}


def _case_cell(res: CaseResult) -> dict[str, Any]:
    """The per-(case,variant) snapshot stored in a baseline."""
    ptok = ctok = 0
    scores: list[float] = []
    stages: dict[str, Any] = {}
    for s in res.stages:
        u = (s.metrics or {}).get("usage") or {}
        ptok += int(u.get("prompt_tokens") or 0)
        ctok += int(u.get("completion_tokens") or 0)
        if (s.metrics or {}).get("score") is not None:
            scores.append(float(s.metrics["score"]))
        stages[s.stage_id] = {
            "status": s.status,
            "passed": s.passed,
            "inconclusive_gates": sum(g.ok is None for g in s.gates),
            "wall_s": s.wall_s,
            "score": (s.metrics or {}).get("score"),
            "gates": {g.name: g.ok for g in s.gates},
        }
    decided = [s for s in res.stages if s.passed is not None]
    return {
        "passed": res.passed,
        "inconclusive_runs": int(res.passed is None),
        "run_id": res.run_id,
        "wall_s": res.wall_s,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "score": statistics.fmean(scores) if scores else None,
        "pass_rate": (
            sum(1 for s in decided if s.passed is True) / len(decided) if decided else None
        ),
        "stages": stages,
    }


def save(label: str, *, root: Path | None = None, out_dir: Path | None = None) -> Path:
    """Latest run per (case, variant) -> `baselines/<label>.json`."""
    results = load_results(root or RESULTS_DIR)
    if not results:
        raise ValueError(f"no results under {root or RESULTS_DIR}")
    # latest per cell
    cells: dict[tuple[str, str], CaseResult] = {}
    for r in sorted(results, key=lambda x: x.finished_at or ""):
        cells[(r.case, r.variant)] = r
    payload = {
        "schema_version": 1,
        "label": label,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cells": {
            f"{case}|{variant}": _case_cell(r) for (case, variant), r in sorted(cells.items())
        },
    }
    d = out_dir or BASELINES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{label}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def load_baseline(label: str, *, out_dir: Path | None = None) -> dict[str, Any]:
    path = (out_dir or BASELINES_DIR) / f"{label}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Diff:
    cell: str
    kind: str  # "gate_flip" | "gate_inconclusive" | "pass_flip" | "metric"
    detail: str


def diff(a: dict[str, Any], b: dict[str, Any]) -> list[Diff]:
    """Compare two baseline payloads; a regression list (empty = ok)."""
    out: list[Diff] = []
    cells_a = a.get("cells") or {}
    cells_b = b.get("cells") or {}
    for key in sorted(set(cells_a) | set(cells_b)):
        ca, cb = cells_a.get(key), cells_b.get(key)
        if ca is None:
            out.append(Diff(key, "missing", f"not in {a.get('label')}"))
            continue
        if cb is None:
            out.append(Diff(key, "missing", f"not in {b.get('label')}"))
            continue
        if ca.get("passed") != cb.get("passed"):
            out.append(Diff(key, "pass_flip", f"passed {ca.get('passed')} -> {cb.get('passed')}"))
        for field, (direction, thresh) in _REGRESS_FIELDS.items():
            va, vb = ca.get(field), cb.get(field)
            if va is None or vb is None:
                continue
            va, vb = float(va), float(vb)
            bad = va > 0 and vb > va * (1 + thresh) if direction == "lower" else vb < va - thresh
            if bad:
                out.append(Diff(key, "metric", f"{field} {va:g} -> {vb:g} (regression)"))
        stages_a = ca.get("stages") or {}
        stages_b = cb.get("stages") or {}
        for sid in sorted(set(stages_a) | set(stages_b)):
            ga = (stages_a.get(sid) or {}).get("gates") or {}
            gb = (stages_b.get(sid) or {}).get("gates") or {}
            for gname in sorted(set(ga) | set(gb)):
                if ga.get(gname) is True and gb.get(gname) is False:
                    out.append(Diff(key, "gate_flip", f"{sid}/{gname}: pass -> fail"))
                elif ga.get(gname) is True and gb.get(gname) is None:
                    out.append(
                        Diff(
                            key,
                            "gate_inconclusive",
                            f"{sid}/{gname}: pass -> evidence unavailable",
                        )
                    )
    return out


def render_diff(diffs: list[Diff], label_a: str, label_b: str) -> str:
    if not diffs:
        return f"baseline diff {label_a} -> {label_b}: no regressions"
    lines = [f"baseline diff {label_a} -> {label_b}: {len(diffs)} regression(s)", ""]
    for d in diffs:
        lines.append(f"- [{d.kind}] {d.cell}: {d.detail}")
    return "\n".join(lines)
