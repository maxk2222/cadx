"""Run orchestration — driver-agnostic case execution.

For each enabled stage: resolve parent, dispatch via driver, harvest
metrics, evaluate gates, persist `stages/<id>.json` immediately (crash-safe).
At the end write `result.json` + `summary.md` + `meta.json` into the run
dir (evals/results/<case>/<ts>_<label> for http, evals/scratch for local).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cadx.evalkit.features import ensure_features, needs_features
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


def _atomic_write(path: Path, text: str) -> None:
    """Crash-safe file write: temp file in the same dir + fsync + os.replace.

    A crash mid-write leaves either the previous version or the new one —
    never a truncated file. Temp names are unique per call so concurrent
    writers cannot collide."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, default=str, indent=1))


def _summary_md(res: CaseResult) -> str:
    lines = [
        f"# eval {res.case} ({res.driver}, variant={res.variant})",
        "",
        f"- status: **{res.status}**  verdict: **{_verdict_label(res.passed)}**  "
        f"wall: {res.wall_s:.0f}s  run: `{res.run_id}`",
        f"- project: `{res.project}` ({res.project_id or '-'})",
        "",
        "| stage | type | status | verdict | score | wall s | gates failed | inconclusive |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in res.stages:
        fails = [g.name for g in s.gates if g.ok is False]
        inconclusive = [g.name for g in s.gates if g.ok is None]
        lines.append(
            f"| {s.stage_id} | {s.stage_type} | {s.status} | {_verdict_label(s.passed)} | "
            f"{(s.metrics or {}).get('score')} | {s.wall_s:.0f} | "
            f"{', '.join(fails) or '-'} | {', '.join(inconclusive) or '-'} |"
        )
    if res.error:
        lines += ["", f"**error:** {res.error}"]
    return "\n".join(lines)


def _verdict_label(value: bool | None) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "INCONCLUSIVE"


def _aggregate_verdict(values: list[bool | None]) -> bool | None:
    """A definite failure dominates; otherwise preserve any missing verdict."""
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


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
        res.status = "failed"
        _persist(res, out, final=True)
        return res

    # persist the run as `running` BEFORE potentially long driver setup —
    # a crash afterwards must still show an in-progress run, not a missing
    # or finished-looking result.
    _persist(res, out)

    # ---- driver setup -----------------------------------------------------
    if driver == "local":
        drv, dctx = None, None
        try:
            drv, dctx = await _setup_local(
                case, res, out, variant_overrides, from_dir, verbose
            )
        except Exception as e:
            if drv is not None:
                with contextlib.suppress(Exception):
                    drv.unpatch_executor()
            res.status = "failed"
            res.error = f"{type(e).__name__}: {e}"
            res.wall_s = round(time.monotonic() - t0, 1)
            _persist(res, out, final=True)
            return res
    else:
        from cadx.evalkit.http import DriverCtx, HttpDriver

        hctx = DriverCtx()
        hctx.out_dir = out
        drv = HttpDriver(api_url, verbose=verbose)
        # variant swap: settings backup -> apply -> restore in finally
        overrides = {**(variant_overrides or {}), **case.runtime}
        backup: dict[str, Any] | None = None
        try:
            if not await drv.health():
                res.error = f"api unreachable at {api_url}"
                res.status = "failed"
                _persist(res, out, final=True)
                return res
            backup = await _swap_settings(drv, res, out, overrides, settings_overrides)
            if res.error:
                res.status = "failed"
                _persist(res, out, final=True)
                return res
            proj = await drv.find_or_create_project(
                case.project, {"eval_mode": True, "clarification_mode": "assume", **case.settings}
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
        except Exception as e:
            # unexpected runner/driver error — the run is failed, NOT merely
            # a gate failure; persist whatever partial state exists.
            res.status = "failed"
            res.error = f"{type(e).__name__}: {e}"
        finally:
            if backup is not None:
                try:
                    from cadx.api.schemas import SettingsUpdate

                    allowed = set(SettingsUpdate.model_fields)
                    await drv.settings_put({k: v for k, v in backup.items() if k in allowed})
                except Exception as e:
                    log.warning("eval.settings_restore_failed", error=str(e)[:120])
            await drv.close()
        if res.status == "running":
            res.status = "finished"
        res.wall_s = round(time.monotonic() - t0, 1)
        res.passed = _aggregate_verdict([s.passed for s in res.stages]) if res.stages else None
        _persist(res, out, final=True)
        return res

    # local driver path
    try:
        await _run_stages(case, stages, drv, dctx, res, out, fail_fast, verbose)
    except Exception as e:
        res.status = "failed"
        res.error = f"{type(e).__name__}: {e}"
    finally:
        drv.unpatch_executor()
    if res.status == "running":
        res.status = "finished"
    res.wall_s = round(time.monotonic() - t0, 1)
    res.passed = _aggregate_verdict([s.passed for s in res.stages]) if res.stages else None
    _persist(res, out, final=True)
    return res


async def _swap_settings(
    drv: Any,
    res: CaseResult,
    out: Path,
    overrides: dict[str, Any],
    settings_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Backup live settings, apply eval overrides; returns the backup (None
    when nothing needed swapping). On failure sets res.error."""
    if not overrides and not settings_overrides:
        return None
    backup: dict[str, Any] | None = None
    try:
        backup = await drv.settings_get()
        _write_json(out / "settings.backup.json", backup)
        await drv.settings_put({**overrides, **(settings_overrides or {})})
        res.meta["runtime_overrides"] = {**overrides, **(settings_overrides or {})}
    except Exception as e:
        res.error = f"settings swap failed: {e}"
    # return the backup even on failure — a put that errored after the server
    # committed still needs restoring
    return backup


async def _setup_local(
    case: EvalCase,
    res: CaseResult,
    out: Path,
    variant_overrides: dict[str, Any] | None,
    from_dir: Path | None,
    verbose: bool,
) -> tuple[Any, Any]:
    """Bootstrap the scratch (local) driver: project, runtime overrides,
    assets, patched executor. Returns (driver, driver ctx)."""
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
    return drv, lctx


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
    for pos, stage in enumerate(stages):
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
        # measured features are not persisted on Version — recover them from
        # the exported artifacts when a feature-family gate is present
        if needs_features(stage.gates):
            await ensure_features(outcome)
        metrics = harvest(outcome)
        gates = run_gates(stage.gates, outcome, wall)
        if outcome.status in ("failed", "cancelled", "timeout"):
            passed: bool | None = False
        elif not gates:
            passed = outcome.status == "succeeded"
        else:
            passed = _aggregate_verdict([g.ok for g in gates])
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
            # remaining stages never ran — record them explicitly instead of
            # silently omitting them from the report
            for skipped in stages[pos + 1 :]:
                res.stages.append(
                    StageResult(
                        stage_id=skipped.id,
                        stage_type=skipped.type,
                        status="skipped",
                        passed=None,
                        wall_s=0.0,
                    )
                )
            break
    _ = last_id


def _persist(res: CaseResult, out: Path, *, final: bool = False) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    res.updated_at = now
    if final:
        res.finished_at = now
    _write_json(out / "result.json", res.to_dict())
    _write_json(out / "meta.json", res.meta)
    _atomic_write(out / "summary.md", _summary_md(res))
