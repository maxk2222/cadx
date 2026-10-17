#!/usr/bin/env python3
"""Harvest verified CADX eval runs into SFT pairs (axolotl messages JSONL).

Reads per-run `result.json` summaries under the eval output root
(`$CADX_EVAL_OUT` / `--root`, default <repo>/data/eval_out), follows each
`version_dir` into `data/projects/<pid>/versions/<vid>/` and harvests:

* ``codegen`` pairs — real stored ``system``+``user`` messages of a
  code-producing step (``codegen`` / ``improve*``) paired with the code it
  produced, but ONLY when that exact code was executed and the sandbox
  attempt reported ``ok=true`` AND the run passed with
  ``score >= --min-score`` (``dim_ok``/``watertight``/``code_ok``/
  ``features_ok`` must not be false where present).
* ``repair`` pairs (ILF-style) — for every exec attempt that FAILED
  (``iterations/<n>/a<k>/_run/model.py`` + ``result.json`` with
  ``ok=false``) followed by a LATER attempt that PASSED, pair
  ``(broken code + error + fix instructions) -> fixed code``. The prompt is
  the real stored ``repair_*`` user message when its embedded
  ``PREVIOUS CODE`` block matches the broken snapshot
  (``meta.prompt_source="trace"``); otherwise a synthesized minimal
  prompt (``meta.prompt_source="synthesized"``) carrying the part spec from
  the attempt's ``_run/input.json`` anchors.

Attempt code lives at ``iterations/<iter>/a<attempt>/_run/model.py`` with
the sandbox verdict next to it in ``result.json``; the submitted code of
each LLM call is inside the trace record's JSON ``response``
(``code`` / ``full_code`` keys) and is verified against ``model.py``
snapshots — unexecuted code is never harvested.

Secrets: nothing outside message text is copied; image blocks are
flattened to ``<image>`` placeholders and every emitted text field passes
a ``_contains_secret`` guard — matching rows are dropped and counted.

Usage:
    python scripts/export_sft.py --root data/eval_out --out sft_pairs.jsonl \
        [--min-score 85] [--kinds codegen,repair] [--include-project-versions]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CODE_STEPS = ("codegen", "improve")  # improve_2, improve_3, ... match via startswith
REPAIR_STEP_PREFIX = "repair"

# Belt & braces: nothing we copy should ever contain credentials, but scan
# every emitted text field anyway and drop the row on any match.
_SECRET_RE = re.compile(
    r"(?i)("
    r"sk-ant-[a-z0-9_-]{8,}|sk-or-[a-z0-9_-]{8,}|sk-[a-z0-9]{16,}"
    r"|ghp_[a-z0-9]{20,}|gho_[a-z0-9]{20,}|xox[baprs]-[a-z0-9-]{10,}"
    r"|akia[0-9a-z]{16}"
    r"|-----begin [a-z ]*private key"
    r"|api[_-]?key[\"'\s]*[:=]|apikey[\"'\s]*[:=]"
    r"|secret[_-]?key[\"'\s]*[:=]|client[_-]?secret[\"'\s]*[:=]"
    r"|access[_-]?token[\"'\s]*[:=]|auth[_-]?token[\"'\s]*[:=]"
    r"|refresh[_-]?token[\"'\s]*[:=]|id[_-]?token[\"'\s]*[:=]"
    r"|bearer\s+[a-z0-9._~+/=-]{12,}"
    r"|password[\"'\s]*[:=]|passwd[\"'\s]*[:=]"
    r")"
)

_LINE_NO_RE = re.compile(r"^\s*\d+\s*[│|]\s?")

_FALLBACK_REPAIR_SYSTEM = (
    "You are an expert mechanical CAD engineer writing build123d code. "
    "The user shows you code that failed in the sandbox executor, the "
    "error it produced, and the part specification. Return the complete "
    "corrected program — code only."
)

_FALLBACK_REPAIR_INSTRUCTION = (
    "The code below failed in the sandbox executor. Apply a minimal fix "
    "and return the complete corrected program."
)


@dataclass
class Attempt:
    """One sandbox execution: iterations/<iter>/a<n>/_run/model.py + result.json."""

    iteration: int
    index: int
    code: str
    ok: bool | None
    error: str | None
    run_dir: Path


@dataclass
class VersionCtx:
    """Everything harvested from one version dir."""

    version_dir: Path
    project_id: str
    version_id: str
    benchmark: str
    variant: str
    run_id: str
    score: float | None
    passed: bool
    dim_ok: bool | None
    watertight: bool | None
    code_ok: bool | None
    features_ok: bool | None
    records: list[dict] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)


def _contains_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text))


def _flatten_content(content: object) -> str:
    """Message content -> plain text; image blocks become an ``<image>`` line."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text") or ""))
            elif btype == "image_url":
                parts.append("<image>")
            elif btype and "text" in block:
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p != "")
    return ""


def _norm_code(code: str) -> str:
    return code.strip()


def _extract_code(response_text: str) -> str | None:
    """Code produced by a codegen/improve/repair response (code | full_code)."""
    if not response_text.strip():
        return None
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("code") or payload.get("full_code")
    return code.strip() if isinstance(code, str) and code.strip() else None


def _iter_trace_records(version_dir: Path) -> list[dict]:
    jl = version_dir / "trace.jsonl"
    if jl.is_file():
        records = []
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
    tj = version_dir / "trace.json"
    if tj.is_file():
        try:
            data = json.loads(tj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else [data]
    return []


def _iter_attempts(version_dir: Path) -> list[Attempt]:
    attempts: list[Attempt] = []
    for model_py in version_dir.glob("iterations/*/a*/_run/model.py"):
        m = re.search(r"iterations/(\d+)/a(\d+)/_run/model\.py$", str(model_py))
        if not m:
            continue
        res_path = model_py.parent / "result.json"
        ok: bool | None = None
        error: str | None = None
        if res_path.is_file():
            try:
                res = json.loads(res_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                res = {}
            ok = res.get("ok")
            err = res.get("error")
            error = str(err) if err else None
        try:
            code = model_py.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not code:
            continue
        attempts.append(
            Attempt(
                iteration=int(m.group(1)),
                index=int(m.group(2)),
                code=code,
                ok=ok,
                error=error,
                run_dir=model_py.parent,
            )
        )
    return sorted(attempts, key=lambda a: (a.iteration, a.index))


def _resolve_version_dir(raw: str | None, projects_root: Path) -> Path | None:
    """Map stored version_dir (often container path /data/projects/...) to disk."""
    if not raw:
        return None
    p = Path(raw)
    if p.is_dir():
        return p
    m = re.search(r"projects[/\\]([^/\\]+)[/\\]versions[/\\]([^/\\]+)", raw)
    if m:
        cand = projects_root / m.group(1) / "versions" / m.group(2)
        if cand.is_dir():
            return cand
    return None


def _spec_from_attempt(attempt: Attempt) -> str | None:
    """Part spec stored in _run/input.json anchors (fallback prompt material)."""
    inp = attempt.run_dir / "input.json"
    if not inp.is_file():
        return None
    try:
        data = json.loads(inp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    spec = (data.get("anchors") or {}).get("spec")
    if not spec:
        return None
    return json.dumps(spec, indent=1, ensure_ascii=False)


def _previous_code_block(user_text: str) -> str | None:
    """Extract the PREVIOUS CODE python fence from a repair user message."""
    m = re.search(r"PREVIOUS CODE[^\n]*\n```python\n(.*?)```", user_text, re.DOTALL)
    if not m:
        return None
    lines = [_LINE_NO_RE.sub("", line) for line in m.group(1).splitlines()]
    return "\n".join(lines).strip()


def _record_prompt_messages(rec: dict) -> list[dict]:
    """Stored messages -> emitted [{role, content}] (system+user only, flattened)."""
    out: list[dict] = []
    for msg in rec.get("messages") or []:
        role = msg.get("role")
        if role not in ("system", "user"):
            continue  # drop tool/mid-loop assistant turns — single-turn SFT rows
        text = _flatten_content(msg.get("content"))
        if text:
            out.append({"role": role, "content": text})
    return out


def _user_text(rec: dict) -> str:
    return "\n".join(
        _flatten_content(m.get("content"))
        for m in rec.get("messages") or []
        if m.get("role") == "user"
    )


def _codegen_pairs(ctx: VersionCtx, min_score: float) -> Iterator[tuple[list[dict], dict]]:
    """Yield (messages_prefix, meta) for each verified code-producing step."""
    if not ctx.passed:
        return
    if ctx.score is None or ctx.score < min_score:
        return
    for flag in (ctx.dim_ok, ctx.watertight, ctx.code_ok, ctx.features_ok):
        if flag is False:
            return
    ok_codes = {_norm_code(a.code) for a in ctx.attempts if a.ok}
    seen_steps: set[tuple[str, int, str]] = set()
    for rec in ctx.records:
        step = str(rec.get("step") or "")
        if not step.startswith(CODE_STEPS):
            continue
        code = _extract_code(str(rec.get("response") or ""))
        if not code:
            continue
        dedup_key = (step, int(rec.get("iteration") or 0), code)
        if dedup_key in seen_steps:
            continue
        seen_steps.add(dedup_key)
        if _norm_code(code) not in ok_codes:
            continue  # produced code failed exec or never ran — not verified
        prefix = _record_prompt_messages(rec)
        if not any(m["role"] == "user" for m in prefix):
            continue
        messages = [*prefix, {"role": "assistant", "content": code}]
        meta = {
            "kind": "codegen",
            "benchmark": ctx.benchmark,
            "variant": ctx.variant,
            "score": ctx.score,
            "run_id": ctx.run_id,
            "iteration": rec.get("iteration"),
            "step": step,
            "project_id": ctx.project_id,
            "version_id": ctx.version_id,
            "model": rec.get("model"),
            "prompt_source": "trace",
        }
        yield messages, meta


def _find_repair_record(ctx: VersionCtx, broken: Attempt, fixed: Attempt) -> dict | None:
    """Repair record whose PREVIOUS CODE block matches the broken snapshot."""
    target = _norm_code(broken.code)
    for rec in ctx.records:
        step = str(rec.get("step") or "")
        if not step.startswith(REPAIR_STEP_PREFIX):
            continue
        it = int(rec.get("iteration") or 0)
        if not (broken.iteration <= it <= fixed.iteration):
            continue
        prev = _previous_code_block(_user_text(rec))
        if prev and _norm_code(prev) == target:
            return rec
    return None


def _synthesized_repair_prompt(ctx: VersionCtx, broken: Attempt) -> list[dict]:
    parts = [_FALLBACK_REPAIR_INSTRUCTION, ""]
    spec = _spec_from_attempt(broken)
    if spec:
        parts += ["PART SPEC (JSON):", spec, ""]
    parts += [
        "BROKEN CODE:",
        f"```python\n{broken.code}\n```",
        "",
        "EXECUTION ERROR:",
        broken.error or "(unknown error)",
    ]
    return [
        {"role": "system", "content": _FALLBACK_REPAIR_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _repair_pairs(ctx: VersionCtx) -> Iterator[tuple[list[dict], dict]]:
    """Yield (messages, meta) for failed-attempt -> later-passing-attempt pairs."""
    atts = ctx.attempts
    for i, broken in enumerate(atts):
        if broken.ok is not False or not broken.error:
            continue
        fixed = next((a for a in atts[i + 1 :] if a.ok), None)
        if fixed is None:
            continue  # no later pass — nothing verified to learn from
        if _norm_code(fixed.code) == _norm_code(broken.code):
            continue  # e.g. autofix params_override retry — no code change
        rec = _find_repair_record(ctx, broken, fixed)
        if rec is not None:
            prefix = _record_prompt_messages(rec)
            prompt_source = "trace"
            step = str(rec.get("step") or "")
            model = rec.get("model")
        else:
            prefix = _synthesized_repair_prompt(ctx, broken)
            prompt_source = "synthesized"
            step = "repair"
            model = None
        if not prefix:
            continue
        messages = [*prefix, {"role": "assistant", "content": fixed.code}]
        meta = {
            "kind": "repair",
            "benchmark": ctx.benchmark,
            "variant": ctx.variant,
            "score": ctx.score,
            "run_id": ctx.run_id,
            "iteration": broken.iteration,
            "fixed_iteration": fixed.iteration,
            "step": step,
            "project_id": ctx.project_id,
            "version_id": ctx.version_id,
            "model": model,
            "prompt_source": prompt_source,
        }
        yield messages, meta


def _row_is_clean(messages: list[dict], meta: dict) -> bool:
    for m in messages:
        if _contains_secret(str(m.get("content") or "")):
            return False
    return not any(_contains_secret(str(v)) for v in meta.values() if isinstance(v, str))


def _default_root() -> Path:
    env = os.environ.get("CADX_EVAL_OUT")
    if env:
        return Path(env)
    for cand in ("data/eval_out", "backend/data/eval_out", "eval_out"):
        p = REPO_ROOT / cand
        if p.is_dir():
            return p
    return REPO_ROOT / "data" / "eval_out"


def _load_run_ctx(result_path: Path, projects_root: Path) -> VersionCtx | None:
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    res = data.get("result") or {}
    raw_vd = res.get("version_dir") or data.get("version_dir")
    vdir = _resolve_version_dir(raw_vd, projects_root)
    if vdir is None:
        return None
    score = res.get("score")
    return VersionCtx(
        version_dir=vdir,
        project_id=str(data.get("project_id") or vdir.parent.parent.name),
        version_id=vdir.name,
        benchmark=str(res.get("name") or data.get("benchmark") or result_path.parent.name),
        variant=str(res.get("variant") or data.get("variant") or "default"),
        run_id=str(data.get("run_id") or result_path.parent.name),
        score=float(score) if isinstance(score, int | float) else None,
        passed=bool(res.get("passed")),
        dim_ok=res.get("dim_ok"),
        watertight=res.get("watertight"),
        code_ok=res.get("code_ok"),
        features_ok=res.get("features_ok"),
    )


def _critique_verdict(records: list[dict]) -> tuple[bool | None, float | None]:
    """Best pass/score verdict from critique records (for project versions)."""
    passed: bool | None = None
    score: float | None = None
    for rec in records:
        step = str(rec.get("step") or "")
        if not step.startswith("critique"):
            continue
        parsed = rec.get("parsed") or {}
        if "final_passed" in parsed:
            passed = bool(parsed["final_passed"])
            fs = parsed.get("final_score")
            if isinstance(fs, int | float):
                score = float(fs)
            continue
        try:
            resp = json.loads(str(rec.get("response") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(resp, dict) and "passed" in resp:
            passed = bool(resp["passed"])
            s = resp.get("score")
            if isinstance(s, int | float):
                score = float(s)
    return passed, score


def _project_version_ctx(
    version_dir: Path, projects_root: Path, min_score: float
) -> VersionCtx | None:
    records = _iter_trace_records(version_dir)
    if not records:
        return None
    passed, score = _critique_verdict(records)
    if passed is not True:
        return None
    if score is not None and score < min_score:
        return None
    attempts = _iter_attempts(version_dir)
    ok_codes = {_norm_code(a.code) for a in attempts if a.ok}
    has_verified_code = any(
        (c := _extract_code(str(r.get("response") or ""))) and _norm_code(c) in ok_codes
        for r in records
        if str(r.get("step") or "").startswith(CODE_STEPS)
    )
    if not has_verified_code:
        return None
    project_id = version_dir.parent.parent.name
    return VersionCtx(
        version_dir=version_dir,
        project_id=project_id,
        version_id=version_dir.name,
        benchmark="project",
        variant="live",
        run_id=version_dir.name,
        score=score,
        passed=True,
        dim_ok=None,
        watertight=None,
        code_ok=None,
        features_ok=None,
    )


def _collect_contexts(
    root: Path, projects_root: Path, args: argparse.Namespace
) -> tuple[list[VersionCtx], list[Path]]:
    result_paths = sorted(root.rglob("result.json")) if root.is_dir() else []
    ctxs: list[VersionCtx] = []
    seen_vdirs: set[Path] = set()
    for rp in result_paths:
        ctx = _load_run_ctx(rp, projects_root)
        if ctx is None:
            print(f"warn: skipping {rp} (unresolvable version_dir)", file=sys.stderr)
            continue
        seen_vdirs.add(ctx.version_dir)
        ctx.records = _iter_trace_records(ctx.version_dir)
        ctx.attempts = _iter_attempts(ctx.version_dir)
        ctxs.append(ctx)

    if args.include_project_versions and projects_root.is_dir():
        for vdir in sorted(projects_root.glob("*/versions/*")):
            if vdir in seen_vdirs or not vdir.is_dir():
                continue
            ctx = _project_version_ctx(vdir, projects_root, args.min_score)
            if ctx is None:
                continue
            ctx.records = _iter_trace_records(vdir)
            ctx.attempts = _iter_attempts(vdir)
            ctxs.append(ctx)
    return ctxs, result_paths


def _emit_rows(ctxs: list[VersionCtx], kinds: set[str], min_score: float) -> tuple[list[dict], int]:
    rows: list[dict] = []
    seen_pairs: set[str] = set()
    dropped_secret = 0
    for ctx in ctxs:
        emitters: list[Iterable[tuple[list[dict], dict]]] = []
        if "codegen" in kinds:
            emitters.append(_codegen_pairs(ctx, min_score))
        if "repair" in kinds:
            emitters.append(_repair_pairs(ctx))
        for emitter in emitters:
            for messages, meta in emitter:
                if not _row_is_clean(messages, meta):
                    dropped_secret += 1
                    continue
                user = next((m["content"] for m in messages if m["role"] == "user"), "")
                code = messages[-1]["content"]
                key = hashlib.sha256(f"{user}\x00{code}".encode()).hexdigest()
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rows.append({"messages": messages, "meta": meta})
    return rows, dropped_secret


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="eval output root (default: $CADX_EVAL_OUT or <repo>/data/eval_out)",
    )
    ap.add_argument(
        "--projects-root",
        type=Path,
        default=None,
        help="projects dir holding version dirs (default: <root>/../projects)",
    )
    ap.add_argument("--out", type=Path, required=True, help="output JSONL path")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="min run score for codegen pairs (default: 0 — any passed run)",
    )
    ap.add_argument(
        "--kinds",
        default="codegen,repair",
        help="comma-separated pair kinds to emit (default: codegen,repair)",
    )
    ap.add_argument(
        "--include-project-versions",
        action="store_true",
        help="also harvest data/projects version dirs not covered by eval runs "
        "(pass inferred from trace critique verdicts; meta.benchmark='project')",
    )
    args = ap.parse_args()

    root = (args.root or _default_root()).resolve()
    projects_root = (
        args.projects_root.resolve() if args.projects_root else (root.parent / "projects").resolve()
    )
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if not kinds <= {"codegen", "repair"}:
        ap.error(f"--kinds must be subset of codegen,repair (got {sorted(kinds)})")

    ctxs, result_paths = _collect_contexts(root, projects_root, args)
    rows, dropped_secret = _emit_rows(ctxs, kinds, args.min_score)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        k = (str(row["meta"]["kind"]), str(row["meta"]["benchmark"]))
        counts[k] = counts.get(k, 0) + 1
    print(f"runs scanned: {len(result_paths)}  versions harvested: {len(ctxs)}")
    for (kind, bench), n in sorted(counts.items()):
        print(f"  {kind:<8} {bench:<28} {n}")
    print(f"total pairs: {len(rows)}  dropped(secret-guard): {dropped_secret}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
