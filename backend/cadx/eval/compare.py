"""Aggregation for eval run summaries — the compare report's data layer.

`cadx eval` writes one `result.json` per run under
`<eval_out>/<bench>/[<variant>/]<run_id>/`; this module loads those trees,
groups by (variant, benchmark) and renders the markdown table that
`scripts/eval_compare.py` prints. Everything is tolerant of missing dirs,
crashed runs (`result: null`) and fields absent from older summaries.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

RESULT_FILE = "result.json"


@dataclass
class CompareRow:
    """One variant x benchmark cell of the comparison table."""

    variant: str
    benchmark: str
    runs: int
    passed: int
    exec_ok: int  # runs that reached ≥1 sandbox execution
    mean_score: float | None
    mean_iterations: float | None
    prompt_tokens: int
    completion_tokens: int
    thinking_calls: int
    thinking_completion_tokens: int
    mean_wall_s: float | None
    mean_exec_s: float | None


def load_run_summaries(root: Path) -> list[dict[str, Any]]:
    """Read every result.json under `root`; malformed files are skipped."""
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob(RESULT_FILE)):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("eval_compare.bad_summary", path=str(p), error=str(e))
            continue
        if isinstance(d, dict):
            d["_path"] = str(p)
            out.append(d)
    return out


def _step_totals(result: dict[str, Any]) -> dict[str, int]:
    """Sum per-step token stats — the reasoning split only exists on
    summaries whose trace steps carried the `thinking` flag."""
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_calls": 0,
        "thinking_prompt_tokens": 0,
        "thinking_completion_tokens": 0,
    }
    for step in (result.get("step_stats") or {}).values():
        if not isinstance(step, dict):
            continue
        for k in totals:
            totals[k] += int(step.get(k) or 0)
    return totals


def aggregate(summaries: list[dict[str, Any]]) -> list[CompareRow]:
    """Group run summaries into variant x benchmark rows."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in summaries:
        key = (str(s.get("variant") or "default"), str(s.get("benchmark") or "?"))
        groups.setdefault(key, []).append(s)

    rows: list[CompareRow] = []
    for (variant, bench), ss in sorted(groups.items()):
        passed = exec_ok = 0
        ptok = ctok = thk_calls = thk_ctok = 0
        scores: list[float] = []
        iters: list[float] = []
        walls: list[float] = []
        exec_s: list[float] = []
        for s in ss:
            r = s.get("result") or {}  # crashed runs persist result=None
            if not isinstance(r, dict):
                r = {}
            if r.get("passed"):
                passed += 1
            if int(r.get("execs") or 0) > 0:
                exec_ok += 1
            if r.get("score") is not None:
                scores.append(float(r["score"]))
            if r.get("iterations"):
                iters.append(float(r["iterations"]))
            wall = r.get("wall_s", s.get("wall_s"))
            if wall is not None:
                walls.append(float(wall))
            if r.get("exec_s") is not None:
                exec_s.append(float(r["exec_s"]))
            steps = _step_totals(r)
            # top-level token fields win; fall back to the step_stats sum for
            # summaries written before the fields existed
            ptok += int(r.get("prompt_tokens") or 0) or steps["prompt_tokens"]
            ctok += int(r.get("completion_tokens") or 0) or steps["completion_tokens"]
            thk_calls += steps["thinking_calls"]
            thk_ctok += steps["thinking_completion_tokens"]
        rows.append(
            CompareRow(
                variant=variant,
                benchmark=bench,
                runs=len(ss),
                passed=passed,
                exec_ok=exec_ok,
                mean_score=statistics.fmean(scores) if scores else None,
                mean_iterations=statistics.fmean(iters) if iters else None,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                thinking_calls=thk_calls,
                thinking_completion_tokens=thk_ctok,
                mean_wall_s=statistics.fmean(walls) if walls else None,
                mean_exec_s=statistics.fmean(exec_s) if exec_s else None,
            )
        )
    return rows


def _f(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    if digits == 0:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def render_markdown(rows: list[CompareRow]) -> str:
    """Render the variant x benchmark comparison as a markdown table."""
    headers = [
        "variant",
        "benchmark",
        "runs",
        "pass",
        "exec-ok",
        "score",
        "iters",
        "ptok",
        "ctok",
        "reasoning ctok",
        "wall s",
        "exec s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.variant,
                    r.benchmark,
                    str(r.runs),
                    f"{r.passed}/{r.runs}",
                    f"{r.exec_ok}/{r.runs}",
                    _f(r.mean_score),
                    _f(r.mean_iterations, 1),
                    str(r.prompt_tokens),
                    str(r.completion_tokens),
                    str(r.thinking_completion_tokens),
                    _f(r.mean_wall_s),
                    _f(r.mean_exec_s),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def rows_to_dicts(rows: list[CompareRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
