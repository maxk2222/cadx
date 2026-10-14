"""arq workers: queue "agent" (LLM/orchestration) and queue "cad" (execution)."""

from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings

from cadx.agent.worker_jobs import (
    run_agent_job,
    run_fem_job,
    run_mesh_edit_job,
    run_optimize_job,
    run_params_job,
    run_reexport_job,
    run_topopt_resample_job,
)
from cadx.cad.tasks import (
    execute_version,
    export_all,
    fem_analyze,
    optimize,
    process_mesh,
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


async def cad_startup(ctx: dict[str, Any]) -> None:
    await _startup_common(ctx)


async def worker_shutdown(ctx: dict[str, Any]) -> None:
    redis = ctx.get("arq_redis")
    if redis is not None:
        await redis.aclose()


class AgentWorkerSettings:
    """Run with: uv run arq cadx.queue.AgentWorkerSettings"""

    functions = [  # noqa: RUF012
        run_agent_job,
        run_params_job,
        run_reexport_job,
        run_optimize_job,
        run_topopt_resample_job,
        run_mesh_edit_job,
        run_fem_job,
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
        export_all,
        process_mesh,
        fem_analyze,
        optimize,
        topopt_resample,
    ]
    on_startup = cad_startup
    on_shutdown = worker_shutdown
    redis_settings = _redis()
    queue_name = "cad"
    max_jobs = 2
    # generous outer cap — per-job limits come from RuntimeSettings payload
    # (topopt_timeout_s up to 2700s+); QueueExecutor enforces the real timeout
    job_timeout = 7200
    # idempotent recompute tasks — a container restart mid-task may retry once
    max_tries = 2
