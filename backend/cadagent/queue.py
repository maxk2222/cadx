"""arq workers: queue "agent" (LLM/orchestration) and queue "cad" (execution)."""

from __future__ import annotations

from typing import Any

from arq.connections import RedisSettings

from cadagent.agent.worker_jobs import run_agent_job, run_params_job, run_reexport_job
from cadagent.cad.tasks import execute_version, export_all, process_mesh
from cadagent.config import get_settings


def _redis() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


async def agent_startup(ctx: dict[str, Any]) -> None:
    from arq.connections import create_pool

    ctx["arq_redis"] = await create_pool(_redis())


async def worker_shutdown(ctx: dict[str, Any]) -> None:
    redis = ctx.get("arq_redis")
    if redis is not None:
        await redis.aclose()


class AgentWorkerSettings:
    """Run with: uv run arq cadagent.queue.AgentWorkerSettings"""

    functions = [run_agent_job, run_params_job, run_reexport_job]  # noqa: RUF012
    on_startup = agent_startup
    on_shutdown = worker_shutdown
    redis_settings = _redis()
    queue_name = "agent"
    max_jobs = 4
    job_timeout = 3600
    max_tries = 1


class CadWorkerSettings:
    """Run with: uv run arq cadagent.queue.CadWorkerSettings"""

    functions = [execute_version, export_all, process_mesh]  # noqa: RUF012
    on_startup = agent_startup
    on_shutdown = worker_shutdown
    redis_settings = _redis()
    queue_name = "cad"
    max_jobs = 2
    job_timeout = 600
    max_tries = 1
