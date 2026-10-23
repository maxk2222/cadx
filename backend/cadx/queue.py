"""arq workers: queue "agent" (LLM/orchestration) and queue "cad" (execution)."""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime
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
from cadx.logging import get_logger

log = get_logger(__name__)


def _redis() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


HB_KEY_PREFIX = "cadx:hb:"
HB_INTERVAL_S = 15.0
HB_TTL_S = 45

# queued rows this fresh are ignored by the sweeper — a spawn commits the Job
# row before enqueuing, so a sweep must not race the in-flight enqueue
SWEEP_INTERVAL_S = 60.0
SWEEP_GRACE_S = 90.0


async def _heartbeat_loop(ctx: dict[str, Any], role: str) -> None:
    """Publish this worker's liveness to redis (`cadx:hb:<role>`, TTL-gated).

    Runs inside the arq worker loop — if the loop wedges, the key expires and
    /api/health reports the worker as stale; the host watchdog restarts it.
    The loop must never crash the worker, so every error is swallowed."""
    redis = ctx["arq_redis"]
    while True:
        with contextlib.suppress(Exception):
            await redis.setex(f"{HB_KEY_PREFIX}{role}", HB_TTL_S, datetime.now(UTC).isoformat())
        await asyncio.sleep(HB_INTERVAL_S)


async def _arq_live(redis: Any, key_id: str) -> bool:
    """queued or running: `arq:job:` (survives the whole run) or a TTL'd
    job_key with a live `arq:in-progress:`. result_key is NOT liveness —
    see the result-key branch in _sweep_job."""
    if redis is None:
        return True  # can't prove it's missing — don't double-fire
    try:
        return bool(await redis.exists(f"arq:job:{key_id}", f"arq:in-progress:{key_id}"))
    except Exception:
        return True


async def _arq_result_exists(redis: Any, key_id: str) -> bool:
    if redis is None:
        return False
    with contextlib.suppress(Exception):
        return bool(await redis.exists(f"arq:result:{key_id}"))
    return False


async def _fail_lost(db: Any, j: Any, why: str) -> None:
    from cadx.db.models import JobStatus, Version, VersionStatus

    j.status = JobStatus.failed
    j.finished_at = datetime.now(UTC)
    j.log = (j.log or "") + f"failed: {why}\n"
    v = await db.get(Version, j.version_id)
    if v is not None and v.status in (VersionStatus.queued, VersionStatus.running):
        v.status = VersionStatus.failed
        v.error = why


async def _sweep_job(redis: Any, db: Any, j: Any) -> None:
    """One stale `queued` row: skip live entries, honest-fail finished ones,
    replay the payload envelope for truly lost ones."""
    from cadx.api.deps import enqueue

    payload = j.payload or {}
    func, args = payload.get("func"), payload.get("args")
    if not (func and isinstance(args, dict)):
        await _fail_lost(db, j, "job lost from queue — predates payload capture")
        return
    if await _arq_live(redis, str(j.id)):
        return  # entry still queued or running — nothing lost
    if await _arq_result_exists(redis, str(j.id)):
        # a result without a live entry means the task RAN and finished —
        # its DB status write was lost (db outage mid-task). Re-enqueueing
        # would run the whole job a second time; honest surface is failed.
        await _fail_lost(db, j, "queue entry finished but status transition lost")
        log.info("queue.sweeper.lost_transition", job=str(j.id), func=func)
        return
    args = {"job_id": str(j.id), "version_id": str(j.version_id), **args}
    ok = await enqueue(func, args, dedup_id=str(j.id))
    if ok is None:
        return  # real enqueue failure — next sweep retries
    j.log = (j.log or "") + "re-enqueued by queue sweeper (queue loss)\n"
    log.info("queue.sweeper.reenqueued", job=str(j.id), func=func)


async def _sweep_once(ctx: dict[str, Any]) -> None:
    """Re-enqueue `queued` Job rows whose arq entry is missing (redis loss).

    `payload={"func","args"}` replays verbatim; NULL payloads (pre-migration
    rows) fail honestly — JobType→func is ambiguous (edit maps to two runners).
    arq deduplicates on _job_id (job_key OR recent result_key), so re-adding a
    surviving entry is a no-op. Chats with orphaned turn_running / unanswered
    pending messages get `run_chat_turn` re-enqueued — dedup rides on the
    pending message id, so a new user message can never be swallowed by an
    old arq result key (results linger ~1h)."""
    from sqlmodel import col, select

    from cadx.db.models import Job, JobStatus
    from cadx.db.session import session_scope

    redis = ctx.get("arq_redis")
    cutoff = datetime.now(UTC)

    def _age_s(ts: datetime) -> float:
        # postgres returns tz-aware; sqlite (tests/dev) stores naive — treat
        # naive as utc so the grace arithmetic never throws
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (cutoff - ts).total_seconds()

    async with session_scope() as db:
        stale = (
            (await db.execute(select(Job).where(col(Job.status) == JobStatus.queued)))
            .scalars()
            .all()
        )
        for j in stale:
            if _age_s(j.created_at) < SWEEP_GRACE_S:
                continue
            await _sweep_job(redis, db, j)
        # chat orphans: flag left running / pending user message never answered
        await _sweep_chats(db, _age_s, redis)
        await db.commit()


async def _chat_turn_alive(redis: Any, chat_id: Any) -> bool:
    """True when a run_chat_turn task for this chat is queued or in-flight.

    Turns are enqueued under deterministic `chat:<chat>:<msg>` keys, so a
    live turn leaves `arq:job:chat:<id>:*` (job_key survives until task end)
    or `arq:in-progress:chat:<id>:*`. Unknown == alive: clearing a flag that
    a real turn still holds would unlock a second concurrent turn."""
    if redis is None:
        return True
    try:
        for prefix in ("arq:job:chat:", "arq:in-progress:chat:"):
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=f"{prefix}{chat_id}:*", count=200)
                if keys:
                    return True
                if not cursor:
                    break
    except Exception:
        return True
    return False


async def _sweep_chats(db: Any, age_s: Any, redis: Any) -> None:
    """Re-enqueue turns for pending user messages the queue lost, and clear
    turn_running flags that no live task owns anymore.

    turn_running can NOT be cleared on age alone: chat.updated_at refreshes
    only between LLM rounds, so a single slow call (>grace) would look dead
    while a turn is mid-flight — clearing the flag then unlocks a second
    concurrent turn (the atomic claim is the only real turn mutex). Only a
    chat with no live arq key at all is a proven orphan."""
    from sqlmodel import col, select

    from cadx.api.deps import chat_turn_dedup_key, enqueue
    from cadx.db.models import Chat, ChatMessage, ChatRole

    chats = (await db.execute(select(Chat).where(col(Chat.turn_running).is_(True)))).scalars().all()
    for c in chats:
        if age_s(c.updated_at) < SWEEP_GRACE_S:
            continue
        if await _chat_turn_alive(redis, c.id):
            continue
        c.turn_running = False
        log.info("chat.sweeper.clear_orphan", chat=str(c.id))
    pending_msgs = (
        (
            await db.execute(
                select(ChatMessage)
                .where(col(ChatMessage.pending).is_(True))
                .where(col(ChatMessage.role) == ChatRole.user)
                .order_by(col(ChatMessage.created_at))
            )
        )
        .scalars()
        .all()
    )
    for m in pending_msgs:
        if age_s(m.created_at) < SWEEP_GRACE_S:
            continue
        if await _chat_turn_alive(redis, m.chat_id):
            continue  # a live turn drains pending messages itself every round
        key = chat_turn_dedup_key(m.chat_id, m.id)
        if redis is not None:
            try:
                # arq:result: is deliberately NOT part of the liveness check —
                # a claim-lost task writes one too, and arq dedup honors it
                # for ~1h, which would suppress this re-enqueue forever.
                if await redis.exists(f"arq:job:{key}", f"arq:in-progress:{key}"):
                    continue
                await redis.delete(f"arq:result:{key}")
            except Exception:
                continue  # redis flaky — retry next sweep
        ok = await enqueue(
            "run_chat_turn",
            {"chat_id": str(m.chat_id)},
            dedup_id=key,
        )
        if ok is not None:
            log.info("chat.sweeper.reenqueued", chat=str(m.chat_id), msg=str(m.id))


async def _sweep_loop(ctx: dict[str, Any]) -> None:
    """Periodic orphan repair — startup reaping only covers a *worker* restart,
    not a redis restart mid-operation."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_S)
        try:
            await _sweep_once(ctx)
        except Exception:
            log.exception("queue.sweep_failed")


async def _startup_common(ctx: dict[str, Any], role: str) -> None:
    from arq.connections import create_pool

    ctx["arq_redis"] = await create_pool(_redis())
    ctx["bg_tasks"] = [
        asyncio.create_task(_heartbeat_loop(ctx, role), name=f"hb:{role}"),
    ]
    try:  # ensure the runtime row exists even if the worker boots before the api
        from cadx.db.session import session_scope
        from cadx.runtime_config import ensure_runtime_seeded

        async with session_scope() as db:
            if await ensure_runtime_seeded(db):
                await db.commit()
    except Exception:
        pass


async def agent_startup(ctx: dict[str, Any]) -> None:
    await _startup_common(ctx, "agent")
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
    # queue sweeper: re-enqueues queued jobs / orphaned chat turns the arq side
    # lost (redis restart mid-run — startup reaping can't see that)
    ctx["bg_tasks"].append(asyncio.create_task(_sweep_loop(ctx), name="queue-sweeper"))


async def cad_startup(ctx: dict[str, Any]) -> None:
    from cadx.cad import resources

    # cgroup-aware thread defaults for everything the executor process spawns
    # or runs in-process (BLAS, OCCT pools, llvmpipe raster)
    resources.configure_thread_env(os.environ)
    resources.log_detected()
    await _startup_common(ctx, "cad")
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
    tasks = ctx.get("bg_tasks") or []
    for t in tasks:
        t.cancel()
    if tasks:  # let the sweeper/heartbeat finish their in-flight iteration
        await asyncio.gather(*tasks, return_exceptions=True)
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
