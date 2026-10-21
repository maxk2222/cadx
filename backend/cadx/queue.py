"""arq workers: queue "agent" (LLM/orchestration) and queue "cad" (execution)."""

from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

from cadx.agent.chat_worker import reap_orphaned_chats, run_chat_turn
from cadx.agent.worker_jobs import (
    run_agent_job,
    run_code_job,
    run_edit_feature_job,
    run_fem_job,
    run_mesh_edit_job,
    run_meshgen_job,
    run_optimize_job,
    run_organic_job,
    run_params_job,
    run_reexport_job,
    run_refine_job,
    run_topopt_resample_job,
)
from cadx.cad import resources
from cadx.cad.tasks import (
    audit_params,
    edit_feature,
    execute_version,
    export_all,
    fem_analyze,
    optimize,
    organic,
    parametrize_mesh,
    process_mesh,
    refine,
    topopt_resample,
)
from cadx.config import get_settings


def _redis() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


async def _startup_common(ctx: dict[str, Any]) -> None:
    from arq.connections import create_pool

    ctx["arq_redis"] = await create_pool(_redis())
    try:  # ensure the runtime row exists even if the worker boots before the api
        from cadx.db.session import session_scope
        from cadx.runtime_config import ensure_runtime_seeded

        async with session_scope() as db:
            if await ensure_runtime_seeded(db):
                await db.commit()
    except Exception:
        pass


async def agent_startup(ctx: dict[str, Any]) -> None:
    await _startup_common(ctx)
    # reaper: a worker restart orphans in-flight arq tasks (max_tries=1), but
    # their job/version rows keep status='running' forever — fail them here.
    # Only the AGENT-queue worker may reap: every Job row is enqueued on
    # "agent" (cad-queue arq tasks carry no Job row), so an executor restart
    # must not mark live agent jobs as failed.
    try:
        from datetime import UTC, datetime

        from sqlmodel import col, select

        from cadx.db.models import Job, JobStatus, Version, VersionStatus
        from cadx.db.session import session_scope

        async with session_scope() as db:
            stale = (
                (await db.execute(select(Job).where(col(Job.status) == JobStatus.running)))
                .scalars()
                .all()
            )
            for j in stale:
                j.status = JobStatus.failed
                j.finished_at = datetime.now(UTC)
                j.log = (j.log or "") + "failed: worker restarted — orphaned task\n"
                v = await db.get(Version, j.version_id)
                if v is not None and v.status in (
                    VersionStatus.queued,
                    VersionStatus.running,
                ):
                    v.status = VersionStatus.failed
                    v.error = "worker restarted — orphaned job"
            if stale:
                await db.commit()
    except Exception:
        pass
    # chat reaper: orphaned turn_running flags + unanswered pending messages
    await reap_orphaned_chats(ctx)


async def cad_startup(ctx: dict[str, Any]) -> None:
    from cadx.cad import resources

    # cgroup-aware thread defaults for everything the executor process spawns
    # or runs in-process (BLAS, OCCT pools, llvmpipe raster)
    resources.configure_thread_env(os.environ)
    resources.log_detected()
    await _startup_common(ctx)
    # publish the executor's view of the world so the settings API shows the
    # numbers the 0=auto knobs actually resolve to (the api container sees a
    # different cgroup)
    try:
        from cadx.db.models import AppSetting
        from cadx.db.session import session_scope

        async with session_scope() as db:
            row = await db.get(AppSetting, "resources_detected")
            if row is None:
                db.add(AppSetting(key="resources_detected", value=resources.detected()))
            else:
                row.value = resources.detected()
            await db.commit()
    except Exception:
        pass


async def worker_shutdown(ctx: dict[str, Any]) -> None:
    redis = ctx.get("arq_redis")
    if redis is not None:
        await redis.aclose()


class AgentWorkerSettings:
    """Run with: uv run arq cadx.queue.AgentWorkerSettings"""

    functions = [  # noqa: RUF012
        run_agent_job,
        run_chat_turn,
        run_edit_feature_job,
        run_params_job,
        run_reexport_job,
        run_optimize_job,
        run_topopt_resample_job,
        run_mesh_edit_job,
        run_fem_job,
        run_refine_job,
        run_organic_job,
        run_code_job,
        run_meshgen_job,
    ]
    on_startup = agent_startup
    on_shutdown = worker_shutdown
    redis_settings = _redis()
    queue_name = "agent"
    max_jobs = 4
    job_timeout = get_settings().agent_job_timeout_s
    max_tries = 1


class CadWorkerSettings:
    """Run with: uv run arq cadx.queue.CadWorkerSettings"""

    functions = [  # noqa: RUF012
        execute_version,
        audit_params,
        edit_feature,
        export_all,
        process_mesh,
        parametrize_mesh,
        fem_analyze,
        optimize,
        topopt_resample,
        refine,
        organic,
    ]
    on_startup = cad_startup
    on_shutdown = worker_shutdown
    redis_settings = _redis()
    queue_name = "cad"
    # cad tasks are CPU-heavy — scale with the detected (cgroup-aware) cpu
    # budget; ~1 job per 4 cores, at least the historical 2. Override via
    # CADX_CAD_MAX_JOBS.
    max_jobs = int(os.environ.get("CADX_CAD_MAX_JOBS") or max(2, resources.detect_cpus() // 4))
    # generous outer cap — per-job limits come from RuntimeSettings payload
    # (topopt_timeout_s up to 2700s+); QueueExecutor enforces the real timeout
    job_timeout = 7200
    # like the agent queue: never auto-retry. An in-flight job whose worker was
    # killed (cgroup OOM, restart) stays "in-progress" until job_timeout (2h),
    # then arq would requeue it as a zombie that re-runs the same heavy task on
    # a fresh worker — often killing it again and confusing callers that already
    # timed out. Callers (QueueExecutor) own the real timeout and resubmit
    # deliberately.
    max_tries = 1
