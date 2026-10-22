"""Run orchestration — driver-agnostic case execution.

For each enabled stage: resolve parent, dispatch via driver, harvest
metrics, evaluate gates, persist `stages/<id>.json` immediately (crash-safe).
At the end write `result.json` + `summary.md` + `meta.json` into the run
dir (evals/results/<case>/<ts>_<label> for http, evals/scratch for local).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cadx.evalkit.gates import run_gates
from cadx.evalkit.metrics import harvest
from cadx.evalkit.models import CaseResult, StageOutcome, StageResult
from cadx.evalkit.paths import DEFAULT_VARIANT, REPO_ROOT, RESULTS_DIR, SCRATCH_DIR
from cadx.evalkit.stats import run_dir
from cadx.evalkit.suite import EvalCase, EvalStage
from cadx.logging import get_logger

log = get_logger(__name__)


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return sha, dirty
    except Exception:
        return "", False


def _settings_sha(settings: dict[str, Any]) -> str:
    raw = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _resolve_images(paths: list[str]) -> list[Path]:
    """Repo-relative image paths -> absolute existing files (same convention
    as the old harness: repo root first, then backend/)."""
    out: list[Path] = []
    for rel in paths:
        p = (REPO_ROOT / rel).resolve()
        if not p.exists():
            alt = REPO_ROOT / "backend" / rel.removeprefix("backend/")
            if alt.exists():
                p = alt
        if p.exists():
            out.append(p)
        else:
            log.warning("eval.image_missing", path=rel)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, indent=1), encoding="utf-8")


def _summary_md(res: CaseResult) -> str:
    lines = [
        f"# eval {res.case} ({res.driver}, variant={res.variant})",
        "",
        f"- passed: **{res.passed}**  wall: {res.wall_s:.0f}s  run: `{res.run_id}`",
        f"- project: `{res.project}` ({res.project_id or '-'})",
        "",
        "| stage | type | status | passed | score | wall s | gates failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in res.stages:
        fails = [g.name for g in s.gates if not g.ok]
        lines.append(
            f"| {s.stage_id} | {s.stage_type} | {s.status} | {s.passed} | "
            f"{(s.metrics or {}).get('score')} | {s.wall_s:.0f} | {', '.join(fails) or '-'} |"
        )
    if res.error:
        lines += ["", f"**error:** {res.error}"]
    return "\n".join(lines)


async def run_case(
    case: EvalCase,
    *,
    driver: str,
    label: str = "",
    variant: str | None = None,
    variant_overrides: dict[str, Any] | None = None,
    settings_overrides: dict[str, Any] | None = None,
    stages_filter: set[str] | None = None,
    from_version: str | None = None,
    from_dir: Path | None = None,
    api_url: str = "http://localhost:8080",
    repeat_index: int = 0,
    keep_artifacts: bool = True,
    fail_fast: bool = True,
    verbose: bool = False,
) -> CaseResult:
    """Execute one case under one variant; returns the persisted CaseResult."""
    t0 = time.monotonic()
    root = SCRATCH_DIR if driver == "local" else RESULTS_DIR
    out = run_dir(root, case.name, label or (variant or DEFAULT_VARIANT), repeat_index)
    out.mkdir(parents=True, exist_ok=True)

    res = CaseResult(
        case=case.name,
        driver=driver,
        variant=variant or DEFAULT_VARIANT,
        run_id=out.name,
        passed=None,
        wall_s=0.0,
        project=case.project,
        variant_overrides=variant_overrides or {},
        settings_overrides=settings_overrides or {},
    )

    sha, dirty = _git_sha()
    res.meta = {
        "git_sha": sha,
        "git_dirty": dirty,
        "suite_case": case.name,
        "difficulty": case.difficulty,
        "driver": driver,
    }

    stages = case.enabled_stages(stages_filter)
    if not stages:
        res.error = "no enabled stages selected"
        _persist(res, out)
        return res

    # ---- driver setup -----------------------------------------------------
    if driver == "local":
        from cadx.evalkit.local import (
            LocalCtx,
            LocalDriver,
            bootstrap,
            import_parent,
            seed_project,
            snapshot_runtime,
            stage_assets,
        )

        lctx = LocalCtx(out)
        prev_url = await bootstrap(out)
        runtime = await snapshot_runtime(prev_url)
        overrides = {**(variant_overrides or {}), **case.runtime}
        if overrides:
            runtime = runtime.model_copy(update=overrides)
            res.meta["runtime_overrides"] = overrides
        await seed_project(lctx, case.project, case.settings, runtime)
        res.project_id = str(lctx.project_id)
        # case images -> scratch assets
        img_paths = _resolve_images(case.images)
        if img_paths:
            lctx.case_assets = await stage_assets(lctx, img_paths)
        if from_dir:
            await import_parent(lctx, Path(from_dir))
        drv: Any = LocalDriver(verbose=verbose)
        drv.patch_executor()
        dctx: Any = lctx
    else:
        from cadx.evalkit.http import DriverCtx, HttpDriver

        hctx = DriverCtx()
        hctx.out_dir = out
        drv = HttpDriver(api_url, verbose=verbose)
        if not await drv.health():
            res.error = f"api unreachable at {api_url}"
            _persist(res, out)
            return res
        # variant swap: settings backup -> apply -> restore in finally
        backup: dict[str, Any] | None = None
        overrides = {**(variant_overrides or {}), **case.runtime}
        if overrides or settings_overrides:
            try:
                backup = await drv.settings_get()
                _write_json(out / "settings.backup.json", backup)
                await drv.settings_put({**overrides, **(settings_overrides or {})})
                res.meta["runtime_overrides"] = {**overrides, **(settings_overrides or {})}
            except Exception as e:
                res.error = f"settings swap failed: {e}"
                _persist(res, out)
                return res
        try:
            proj = await drv.find_or_create_project(
                case.project, {"_eval": True, "clarification_mode": "assume", **case.settings}
            )
            hctx.project_id = str(proj["id"])
            res.project_id = hctx.project_id
            img_paths = _resolve_images(case.images)
            if img_paths:
                assets = await drv.upload_assets(hctx.project_id, img_paths)
                hctx.case_assets = [str(a["id"]) for a in assets]
            if from_version:
                hctx.parent_version_id = from_version
            dctx = hctx

            await _run_stages(case, stages, drv, dctx, res, out, fail_fast, verbose)
        finally:
            if backup is not None:
                try:
                    from cadx.api.schemas import SettingsUpdate

                    allowed = set(SettingsUpdate.model_fields)
                    await drv.settings_put({k: v for k, v in backup.items() if k in allowed})
                except Exception as e:
                    log.warning("eval.settings_restore_failed", error=str(e)[:120])
            await drv.close()
        res.wall_s = round(time.monotonic() - t0, 1)
        res.passed = all(s.passed for s in res.stages) if res.stages else None
        _persist(res, out)
        return res

    # local driver path
    try:
        await _run_stages(case, stages, drv, dctx, res, out, fail_fast, verbose)
    finally:
        drv.unpatch_executor()
    res.wall_s = round(time.monotonic() - t0, 1)
    res.passed = all(s.passed for s in res.stages) if res.stages else None
    _persist(res, out)
    return res


async def _run_stages(
    case: EvalCase,
    stages: list[EvalStage],
    drv: Any,
    dctx: Any,
    res: CaseResult,
    out: Path,
    fail_fast: bool,
    verbose: bool,
) -> None:
    last_id: str | None = None
    for stage in stages:
        if verbose:
            log.info("eval.stage", case=case.name, stage=stage.id, type=stage.type)
        # per-stage images -> assets (http uploads; local scratch rows)
        if stage.images:
            paths = _resolve_images(stage.images)
            if res.driver == "local" and paths:
                from cadx.evalkit.local import stage_assets

                ids = await stage_assets(dctx, paths)
                dctx.case_assets = [*dctx.case_assets, *ids]
            elif res.driver == "http" and paths:
                assets = await drv.upload_assets(dctx.project_id, paths)
                dctx.case_assets = [*dctx.case_assets, *[str(a["id"]) for a in assets]]

        t0 = time.monotonic()
        try:
            outcome: StageOutcome = await drv.dispatch(stage, dctx)
        except Exception as e:
            outcome = StageOutcome(
                status="failed", wall_s=time.monotonic() - t0, error=f"{type(e).__name__}: {e}"
            )
        wall = outcome.wall_s or (time.monotonic() - t0)
        metrics = harvest(outcome)
        gates = run_gates(stage.gates, outcome, wall)
        passed = (
            bool(gates)
            and all(g.ok for g in gates)
            and outcome.status
            not in (
                "failed",
                "cancelled",
                "timeout",
            )
        )
        if not gates:
            passed = outcome.status == "succeeded"
        sres = StageResult(
            stage_id=stage.id,
            stage_type=stage.type,
            status=outcome.status,
            passed=passed,
            wall_s=round(wall, 1),
            version_id=outcome.version_id,
            job_id=outcome.job_id,
            metrics=metrics,
            gates=gates,
            artifacts=outcome.artifact_files or list(outcome.artifacts.values())[:10],
            error=outcome.error,
        )
        res.stages.append(sres)
        _write_json(out / "stages" / f"{stage.id}.json", dataclasses.asdict(sres))
        _persist(res, out)  # crash-safe: refresh result.json after every stage
        last_id = stage.id
        if fail_fast and passed is False:
            res.error = f"stage {stage.id} failed — stopping (fail-fast)"
            break
    _ = last_id


def _persist(res: CaseResult, out: Path) -> None:
    res.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json(out / "result.json", res.to_dict())
    _write_json(out / "meta.json", res.meta)
    (out / "summary.md").write_text(_summary_md(res), encoding="utf-8")
