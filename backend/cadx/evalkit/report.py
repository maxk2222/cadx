"""Report — aggregate `result.json` trees into variant x case rows.

Ported from the old compare.py; reads the v2 CaseResult schema
(`evals/results/` for durable runs, `evals/scratch/` only when explicitly
included).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cadx.evalkit.models import CaseResult
from cadx.evalkit.stats import step_totals
from cadx.logging import get_logger

log = get_logger(__name__)

RESULT_FILE = "result.json"


def load_results(*roots: Path) -> list[CaseResult]:
    """Read every result.json under the given roots; malformed files are
    skipped with a warning."""
    out: list[CaseResult] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob(RESULT_FILE)):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("eval.report.bad_result", path=str(p), error=str(e))
                continue
            if isinstance(d, dict):
                try:
                    res = CaseResult.from_dict(d)
                except Exception as e:
                    log.warning("eval.report.bad_schema", path=str(p), error=str(e))
                    continue
                res.meta["_path"] = str(p)
                out.append(res)
    return out


@dataclass
class ReportRow:
    variant: str
    case: str
    driver: str
    runs: int
    passed: int
    inconclusive: int
    mean_score: float | None
    mean_iterations: float | None
    prompt_tokens: int
    completion_tokens: int
    thinking_completion_tokens: int
    mean_wall_s: float | None
    mean_exec_s: float | None
    stage_pass: dict[str, float] = field(default_factory=dict)  # stage -> pass rate
    stage_inconclusive: dict[str, int] = field(default_factory=dict)


def aggregate(results: list[CaseResult]) -> list[ReportRow]:
    groups: dict[tuple[str, str, str], list[CaseResult]] = {}
    for r in results:
        key = (r.variant or "default", r.case, r.driver or "?")
        groups.setdefault(key, []).append(r)

    rows: list[ReportRow] = []
    for (variant, case, driver), rs in sorted(groups.items()):
        passed = 0
        inconclusive = 0
        ptok = ctok = thk = 0
        scores: list[float] = []
        iters: list[float] = []
        walls: list[float] = []
        exec_s: list[float] = []
        stage_hits: dict[str, list[bool]] = {}
        stage_inconclusive: dict[str, int] = {}
        for res in rs:
            if res.passed is True:
                passed += 1
            elif res.passed is None:
                inconclusive += 1
            walls.append(res.wall_s)
            for s in res.stages:
                if s.passed is None:
                    stage_inconclusive[s.stage_id] = stage_inconclusive.get(s.stage_id, 0) + 1
                else:
                    stage_hits.setdefault(s.stage_id, []).append(s.passed)
                m = s.metrics or {}
                if m.get("score") is not None:
                    scores.append(float(m["score"]))
                if m.get("iterations"):
                    iters.append(float(m["iterations"]))
                if m.get("exec_s") is not None:
                    exec_s.append(float(m["exec_s"]))
                u = m.get("usage") or {}
                steps = step_totals(m)
                ptok += int(u.get("prompt_tokens") or 0) or steps["prompt_tokens"]
                ctok += int(u.get("completion_tokens") or 0) or steps["completion_tokens"]
                thk += steps["thinking_completion_tokens"]
        rows.append(
            ReportRow(
                variant=variant,
                case=case,
                driver=driver,
                runs=len(rs),
                passed=passed,
                inconclusive=inconclusive,
                mean_score=statistics.fmean(scores) if scores else None,
                mean_iterations=statistics.fmean(iters) if iters else None,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                thinking_completion_tokens=thk,
                mean_wall_s=statistics.fmean(walls) if walls else None,
                mean_exec_s=statistics.fmean(exec_s) if exec_s else None,
                stage_pass={k: sum(v) / len(v) for k, v in sorted(stage_hits.items())},
                stage_inconclusive=dict(sorted(stage_inconclusive.items())),
            )
        )
    return rows


def _f(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def render_markdown(rows: list[ReportRow], *, title: str = "eval report") -> str:
    headers = [
        "variant",
        "case",
        "driver",
        "runs",
        "pass",
        "score",
        "iters",
        "ptok",
        "ctok",
        "reasoning ctok",
        "wall s",
        "exec s",
    ]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.variant,
                    r.case,
                    r.driver,
                    str(r.runs),
                    f"{r.passed}/{r.runs} (inconclusive {r.inconclusive})",
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
    # per-stage pass rates
    if any(r.stage_pass or r.stage_inconclusive for r in rows):
        lines += [
            "",
            "| variant | case | stage | pass rate (decided runs) | inconclusive runs |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            for sid in sorted(set(r.stage_pass) | set(r.stage_inconclusive)):
                rate = r.stage_pass.get(sid)
                rate_text = f"{rate:.0%}" if rate is not None else "-"
                count = r.stage_inconclusive.get(sid, 0)
                lines.append(
                    f"| {r.variant} | {r.case} | {sid} | {rate_text} | {count} |"
                )
    return "\n".join(lines)


def rows_to_dicts(rows: list[ReportRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
