"""Executor ports: in-process (CLI/tests) and queue-based (worker -> cad-executor)."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import arq
from arq.connections import ArqRedis, create_pool
from arq.jobs import Job, JobStatus

from cadx.cad.tasks import (
    run_execute_payload,
    run_fem_payload,
    run_mesh_payload,
    run_optimize_payload,
    run_parametrize_payload,
)
from cadx.config import get_settings
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

CAD_QUEUE = "cad"
CAD_LIGHT_QUEUE = "cad-light"
# codegen-loop latency lane — these funcs are short rlimit-bounded tasks and
# must not queue behind a long heavy solve (optimize/topopt/fem) on the
# serial 'cad' lane. Keep in sync with CadLightWorkerSettings.functions.
_LIGHT_FUNCS = frozenset({"execute_version", "audit_params", "edit_feature", "export_all"})


def exec_error(res: dict[str, Any], default: str = "executor failed") -> str:
    """User-facing error for an executor result dict.

    A queue-side timeout carries ``timed_out`` (the task itself is still
    running until its cancel flag lands) — surface that as a cancelled task,
    not a raw failure string, wherever the error reaches mode_note /
    Version.error / job logs.
    """
    if res.get("timed_out"):
        return "executor timed out (task cancelled)"
    return str(res.get("error") or default)


class InProcessExecutor:
    """Runs the CAD pipeline in the current process (CLI, tests, dev without queues)."""

    async def cancel_flag(self, job_id: str) -> bool:
        return False

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_execute_payload, payload)

    async def audit_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_audit_payload

        return await asyncio.to_thread(run_audit_payload, payload)

    async def process_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_mesh_payload, payload)

    async def parametrize_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_parametrize_payload, payload)

    async def fem_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_fem_payload, payload)

    async def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_optimize_payload, payload)

    async def topopt_resample(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_topopt_resample_payload

        return await asyncio.to_thread(run_topopt_resample_payload, payload)

    async def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_refine_payload

        return await asyncio.to_thread(run_refine_payload, payload)

    async def organic(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_organic_payload

        return await asyncio.to_thread(run_organic_payload, payload)

    async def edit_feature(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Verified parametric feature edit (M9). ``feature_id=None`` lists
        editable through-holes; ``dry_run`` runs the gate without writing."""
        from cadx.cad.tasks import run_edit_feature_payload

        return await asyncio.to_thread(run_edit_feature_payload, payload)

    async def export_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        import uuid as _uuid

        from cadx import storage
        from cadx.cad.engine import export_from_brep, export_from_mesh

        def _run() -> dict[str, Any]:
            pid = _uuid.UUID(payload["project_id"])
            vid = _uuid.UUID(payload["version_id"])
            art_dir = str(payload.get("art_dir") or "")
            it_dir = (
                storage.resolve(art_dir)
                if art_dir
                else storage.iteration_dir(pid, vid, int(payload["iteration"]))
            )
            brep = it_dir / "_run" / "model.brep"
            flat = it_dir / "_run" / "flat_profile.brep"
            stl = it_dir / "_run" / "model.stl"
            out_dir = storage.export_dir(pid, vid)
            out_dir.mkdir(parents=True, exist_ok=True)
            if brep.exists():
                exports = export_from_brep(
                    brep,
                    out_dir,
                    name=payload.get("name", "model"),
                    flat_brep=flat if flat.exists() else None,
                )
            elif stl.exists():
                exports = export_from_mesh(stl, out_dir, name=payload.get("name", "model"))
            else:
                return {"ok": False, "error": "no model.brep or model.stl found"}
            rel = storage.relpath(out_dir)
            return {
                "ok": True,
                "exports": {k: (f"{rel}/{v}" if v else None) for k, v in exports.items()},
            }

        return await asyncio.to_thread(_run)


# post-optimizer pipeline allowance for optimize tasks — extraction,
# multi-realization export verification, finishing and serialize happen
# AFTER the optimizer's own topopt_timeout_s bound (see
# engine.run_topopt_pipeline). Sized for the heaviest observed tail
# (3 robust realizations x several load cases on a dense grid).
_OPT_TAIL_S = 900.0

# consecutive poll failures tolerated before giving up — a redis blip must
# not strand a multi-hour cad task: surfacing the transport error tells the
# agent "call failed" while the arq job keeps running, and the agent's retry
# then queues BEHIND its own still-running task (observed: one blip →
# double-submit → 45min queue delay → stage timeout)
_POLL_FAIL_PATIENCE_S = 120.0
_QUEUE_HARD_FACTOR = 1.5


def _remaining_task_budget(
    timeout_s: float, queue_cap_s: float, enqueued_at: float, started_at: float | None
) -> float:
    now = time.monotonic()
    return (
        timeout_s - (now - started_at)
        if started_at is not None
        else queue_cap_s - (now - enqueued_at)
    )


def _with_queue_meta(
    result: dict[str, Any],
    enqueued_at: float,
    started_at: float | None,
    *,
    queue_timed_out: bool | None = None,
    queued_s: float | None = None,
) -> dict[str, Any]:
    if queued_s is None:
        if started_at is not None:
            queued_s = started_at - enqueued_at
        elif result.get("queue_timed_out"):
            queued_s = time.monotonic() - enqueued_at
    result.setdefault("queued_s", round(queued_s, 2) if queued_s is not None else None)
    result.setdefault("queue_start_observed", started_at is not None)
    if queue_timed_out is not None:
        result.setdefault("queue_timed_out", queue_timed_out)
    return result


async def _fetch_job_result(job: Job, *, retry: bool = True) -> dict[str, Any]:
    try:
        result = await job.result()
    except Exception as e:
        if retry:
            try:
                await asyncio.sleep(2.0)
                result = await job.result()
            except Exception:
                return {"ok": False, "error": f"executor task failed: {e}"}
        else:
            return {"ok": False, "error": f"executor task failed: {e}"}
    return result if isinstance(result, dict) else {"ok": False, "error": "bad result"}


async def _cancel_task_and_wait(
    pool: ArqRedis,
    job: Job,
    job_id: str,
    func: str,
    timeout_s: float,
    enqueued_at: float,
    started_at: float | None,
) -> dict[str, Any]:
    # the cad task is still running — set the TASK-level cancel flag it
    # polls between stages (and mid-SIMP) so the timeout doesn't leave it
    # burning a cad slot until it finishes on its own. The job-level key
    # cancel:{payload['job_id']} belongs to user cancellation only —
    # setting it here bricked every subsequent task of the job (observed:
    # one pipeline_hang turned all later iterations into 'cancelled').
    try:
        await pool.setex(f"cancel:{job_id}", 3600, b"1")
    except Exception:
        log.warning("executor.cancel_flag_failed", func=func, job_id=str(job_id))
    queue_timed_out = started_at is None
    queued_s = time.monotonic() - enqueued_at if queue_timed_out else started_at - enqueued_at
    # grace poll: the task honours the flag at its next checkpoint and
    # typically returns a partial result seconds later — worth waiting a
    # bounded window instead of reporting a bare timeout (an optimize run
    # exports its partial density on the way out)
    grace = min(120.0, timeout_s * 0.15)
    g_waited = 0.0
    while g_waited < grace:
        try:
            status = await job.status()
        except Exception:
            status = None
        if status == JobStatus.in_progress and started_at is None:
            started_at = time.monotonic()
            queued_s = started_at - enqueued_at
        if status == JobStatus.complete:
            try:
                result = await job.result()
            except Exception as e:
                result = {"ok": False, "error": f"executor task failed: {e}"}
                return _with_queue_meta(
                    result,
                    enqueued_at,
                    started_at,
                    queue_timed_out=queue_timed_out,
                    queued_s=queued_s,
                )
            if isinstance(result, dict):
                result["timed_out"] = True
                result.setdefault("status", "partial")
            else:
                result = {"ok": False, "timed_out": True, "error": "bad result"}
            return _with_queue_meta(
                result,
                enqueued_at,
                started_at,
                queue_timed_out=queue_timed_out,
                queued_s=queued_s,
            )
        await asyncio.sleep(5.0)
        g_waited += 5.0
    timeout_phase = "queue" if queue_timed_out else "execution"
    return _with_queue_meta(
        {
            "ok": False,
            "timed_out": True,
            "task_maybe_running": True,
            "queue_timed_out": queue_timed_out,
            "queued_s": round(queued_s, 2),
            "error": f"executor task '{func}' timed out in {timeout_phase} after "
            f"{timeout_s}s budget (queued {queued_s:.1f}s)",
        },
        enqueued_at,
        started_at,
        queue_timed_out=queue_timed_out,
        queued_s=queued_s,
    )


class QueueExecutor:
    """Enqueues work on the 'cad'/'cad-light' arq queues and polls the job
    until complete. ``emit_cb`` (the job's ``emit``) gets a one-shot
    "queued for executor" + "executor running" event per task so the UI can
    distinguish a poll-wait from an actual build — never fatal on failure."""

    def __init__(
        self,
        pool: ArqRedis | None = None,
        *,
        runtime_settings: RuntimeSettings | None = None,
        emit_cb: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self._pool = pool
        self.runtime_settings = runtime_settings or RuntimeSettings()
        self._emit = emit_cb

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            self._pool = await create_pool(
                arq.connections.RedisSettings.from_dsn(get_settings().redis_url)
            )
        return self._pool

    async def _run_task(
        self, func: str, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        job_id = f"{func}-{uuid.uuid4()}"
        # task-scoped identity for the cancel flag — the timeout below must
        # kill THIS task only, never the whole cadx job (a job-level flag
        # silently cancelled every later execute for an hour)
        payload["_task_id"] = job_id
        queue = CAD_LIGHT_QUEUE if func in _LIGHT_FUNCS else CAD_QUEUE
        enqueued_at = time.monotonic()
        try:
            await pool.enqueue_job(func, payload, _queue_name=queue, _job_id=job_id)
        except Exception as e:
            # submit never reached the queue — no task exists, a retry is
            # safe and must be told so explicitly (see poll-failure case)
            return {
                "ok": False,
                "error": f"executor submit failed — task not running, safe to retry: {e!r}",
            }
        job = Job(job_id, pool)
        started_at: float | None = None
        if self._emit is not None:
            with contextlib.suppress(Exception):
                pos = await pool.zcard(f"arq:queue:{queue}")
                label = f"queued for executor (pos {pos})"
                await self._emit("exec", 0, label, sub_label=label)
        queue_cap_s = timeout_s * _QUEUE_HARD_FACTOR

        # capped exponential backoff: fast tasks answer on the first short
        # polls, long ones settle to a 5s cadence instead of 1 Hz for hours
        delay = 0.5
        poll_fail_waited = 0.0
        while True:
            remaining = _remaining_task_budget(timeout_s, queue_cap_s, enqueued_at, started_at)
            if remaining <= 0:
                break
            try:
                status = await job.status()
            except Exception as e:
                # transport/redis error — the task itself is unaffected, so a
                # blip must not surface as "call failed" (the agent's blind
                # retry would queue behind this very task). Tolerate a
                # bounded streak; past it, report MAYBE-running so the agent
                # never resubmits a duplicate.
                poll_fail_waited += delay
                if poll_fail_waited >= _POLL_FAIL_PATIENCE_S:
                    return _with_queue_meta(
                        {
                            "ok": False,
                            "task_maybe_running": True,
                            "error": f"executor status poll failed ({e!r}) — task may "
                            "still be running, do NOT resubmit",
                        },
                        enqueued_at,
                        started_at,
                    )
                await asyncio.sleep(
                    min(
                        delay,
                        max(
                            0.0,
                            _remaining_task_budget(timeout_s, queue_cap_s, enqueued_at, started_at),
                        ),
                    )
                )
                continue
            poll_fail_waited = 0.0
            if status == JobStatus.complete:
                return _with_queue_meta(await _fetch_job_result(job), enqueued_at, started_at)
            if status == JobStatus.in_progress and started_at is None:
                started_at = time.monotonic()
                if self._emit is not None:
                    with contextlib.suppress(Exception):
                        await self._emit(
                            "exec", 0, "executor running", sub_label="executor running"
                        )
            await asyncio.sleep(
                min(
                    delay,
                    max(
                        0.0,
                        _remaining_task_budget(timeout_s, queue_cap_s, enqueued_at, started_at),
                    ),
                )
            )
            delay = min(delay * 1.5, 5.0)
        return await _cancel_task_and_wait(
            pool, job, job_id, func, timeout_s, enqueued_at, started_at
        )

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.exec_timeout_s * 2 + 120
        return await self._run_task("execute_version", payload, timeout)

    async def audit_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        # baseline + 2 light pipeline runs per audited param
        runs = 2 * int(payload.get("max_params") or 8) + 1
        timeout = self.runtime_settings.exec_timeout_s * runs + 120
        return await self._run_task("audit_params", payload, timeout)

    async def process_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.exec_timeout_s + self.runtime_settings.mesh_timeout_s + 120
        return await self._run_task("process_mesh", payload, timeout)

    async def parametrize_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        # reconstruction + one full exec pipeline — same envelope as execute
        timeout = self.runtime_settings.exec_timeout_s * 2 + 120
        return await self._run_task("parametrize_mesh", payload, timeout)

    async def fem_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("fem_analyze", payload, timeout)

    async def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        # topopt_timeout_s bounds the OPTIMIZER wall-clock inside the task
        # (all SIMP/BESO/MMA passes incl. the safety retry). The pipeline's
        # post-optimizer tail — extraction, nominal/eroded/dilated export
        # verification, finishing, serialize — needs its own allowance
        # (observed ~400-800s); otherwise a run that lands exactly at its
        # optimizer budget returns the verified artifact AFTER the caller
        # already reported a bare timeout with no run_index.
        timeout = self.runtime_settings.topopt_timeout_s + _OPT_TAIL_S + 120
        result = await self._run_task("optimize", payload, timeout)
        if result.get("queued_s") is not None and isinstance(result.get("topopt"), dict):
            result["topopt"]["queued_s"] = result["queued_s"]
        return result

    async def topopt_resample(self, payload: dict[str, Any]) -> dict[str, Any]:
        # re-extraction + FEM verify only — no SIMP loop, much cheaper
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("topopt_resample", payload, timeout)

    async def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        # booleans + optional FEM verify + optional analytic STEP export —
        # feature finishing is exec-class work of its own (plug/drill/restore
        # booleans), so a maxed finishing stage must fit alongside the maxed
        # FEM and STEP stages
        timeout = (
            self.runtime_settings.exec_timeout_s
            + self.runtime_settings.fem_timeout_s
            + self.runtime_settings.analytic_step_timeout_s
            + 120
        )
        return await self._run_task("refine", payload, timeout)

    async def organic(self, payload: dict[str, Any]) -> dict[str, Any]:
        # voxelize + field gen + extraction — no FEM/SIMP, same envelope
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("organic", payload, timeout)

    async def edit_feature(self, payload: dict[str, Any]) -> dict[str, Any]:
        # recognition + one boolean (+defeature for shrinks) + renders/exports —
        # comfortably inside the normal exec envelope
        timeout = self.runtime_settings.exec_timeout_s + 120
        return await self._run_task("edit_feature", payload, timeout)

    async def export_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._run_task("export_all", payload, 300)

    async def cancel_flag(self, job_id: str) -> bool:
        """The orchestrator checks cancellation via ctx.check_cancel; nothing extra here."""
        return False
