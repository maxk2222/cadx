"""Job inspection, clarification answers, cancellation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx.api.deps import get_pool, get_session
from cadx.api.schemas import AnswerRequest, EnqueueResponse, JobListItem, JobOut
from cadx.db.models import (
    AgentTrace,
    Job,
    JobStatus,
    Part,
    Project,
    Version,
    VersionStatus,
)
from cadx.spawn import SpawnError, enqueue_resume, spawn_restart

router = APIRouter(tags=["jobs"])


def _out(j: Job) -> JobOut:
    return JobOut(
        id=j.id,
        version_id=j.version_id,
        type=j.type,
        status=j.status,
        step=j.step,
        iteration=j.iteration,
        progress=j.progress,
        log=j.log,
        live=j.live,
        questions=j.questions,
        cancel_requested=j.cancel_requested,
        finish_requested=j.finish_requested,
        started_at=j.started_at,
        finished_at=j.finished_at,
        created_at=j.created_at,
    )


async def _queue_positions(jobs: list[Job]) -> dict[uuid.UUID, tuple[int, int]]:
    """arq sorted-set rank for queued jobs — `arq:queue:agent` holds the
    deterministic job ids (deps.enqueue keys arq jobs by payload.job_id)."""
    ids = [j.id for j in jobs if j.status == JobStatus.queued]
    if not ids:
        return {}
    try:
        pool = await get_pool()
        async with pool.pipeline(transaction=False) as pipe:
            pipe.zcard("arq:queue:agent")
            ranks = [pipe.zrank("arq:queue:agent", str(i)) for i in ids]
            depth, *ranks = await pipe.execute()
        return {i: (r, depth) for i, r in zip(ids, ranks, strict=True) if r is not None}
    except Exception:
        return {}


@router.get("/jobs/events")
async def jobs_events(db: AsyncSession = Depends(get_session)):
    """Global job-lifecycle SSE — snapshot of active jobs, then one `jobs`
    event per create/status transition. MUST register before `/jobs/{job_id}`
    or 'events' is parsed as a job id."""
    from cadx.api.routers.sse import jobs_lifecycle_stream

    return StreamingResponse(
        jobs_lifecycle_stream(db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    out = _out(j)
    if (pos := (await _queue_positions([j])).get(j.id)) is not None:
        out.queue_position, out.queue_depth = pos
    return out


@router.get("/jobs/{job_id}/trace")
async def job_trace(
    job_id: uuid.UUID,
    limit: int = 5000,
    full: bool = False,
    db: AsyncSession = Depends(get_session),
):
    """AgentTrace records of a job — step/iteration/model/token/latency plus
    the `parsed` extras (thinking flags etc.). `messages`/`response` bulk is
    omitted unless `full=1` (eval harvesters only need the metrics)."""
    if not await db.get(Job, job_id):
        raise HTTPException(404, "job not found")
    rows = (
        (
            await db.execute(
                select(AgentTrace)
                .where(AgentTrace.job_id == job_id)
                .order_by(col(AgentTrace.created_at))
                .limit(max(1, min(limit, 50_000)))
            )
        )
        .scalars()
        .all()
    )
    out = []
    for r in rows:
        d = {
            "id": str(r.id),
            "step": r.step,
            "iteration": r.iteration,
            "model": r.model,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms,
            "parsed": r.parsed,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        if full:
            d["messages"] = r.messages
            d["response"] = r.response
        out.append(d)
    return out


async def job_list_items(
    db: AsyncSession, *, active: bool = False, limit: int = 200
) -> list[JobListItem]:
    """Shared JobListItem query — the REST list and the `/jobs/events`
    SSE snapshot emit identical rows."""
    q = (
        select(Job, Version, Project, Part)
        .join(Version, Job.version_id == Version.id)
        .join(Project, Version.project_id == Project.id)
        .outerjoin(Part, Version.part_id == Part.id)
        .order_by(col(Job.created_at).desc())
        .limit(limit)
    )
    if active:
        q = q.where(
            col(Job.status).in_([JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input])
        )
    rows = await db.execute(q)
    all_rows = rows.all()
    positions = await _queue_positions([r[0] for r in all_rows])
    out: list[JobListItem] = []
    for j, v, p, part in all_rows:
        d = _out(j).model_dump()
        if (pos := positions.get(j.id)) is not None:
            d["queue_position"], d["queue_depth"] = pos
        d["log"] = (d.get("log") or "")[-500:]  # tail only — full logs are heavy
        out.append(
            JobListItem(
                **d,
                version_number=v.number,
                version_status=v.status,
                prompt=(v.prompt or "")[:140],
                project_id=p.id,
                project_name=p.name,
                part_id=v.part_id,
                part_name=part.name if part else "",
            )
        )
    return out


@router.get("/jobs", response_model=list[JobListItem])
async def list_jobs(
    active: bool = False,
    limit: int = 200,
    db: AsyncSession = Depends(get_session),
):
    """Global job list for the Jobs page — newest first, optionally only
    non-terminal (queued/running/waiting_for_input) jobs."""
    return await job_list_items(db, active=active, limit=limit)


@router.get("/versions/{version_id}/jobs", response_model=list[JobOut])
async def version_jobs(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Job).where(Job.version_id == version_id).order_by(col(Job.created_at).desc())
    )
    return [_out(j) for j in rows.scalars().all()]


@router.post("/jobs/{job_id}/answer", response_model=EnqueueResponse)
async def answer(job_id: uuid.UUID, body: AnswerRequest, db: AsyncSession = Depends(get_session)):
    """Submit clarification answers -> spawns a resume job.

    The same Version is reused (it sits in waiting_for_input); a new Job of type
    `resume` continues from the stored spec plus the answers appended to the prompt.
    """
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")

    qa_lines = []
    questions = j.questions or []
    for i, a in enumerate(body.answers):
        q = questions[i]["question"] if i < len(questions) else (a.id or f"q{i + 1}")
        qa_lines.append(f"Q: {q}\nA: {a.answer}")
    answers_text = "\n".join(qa_lines)

    try:
        v, new_job = await enqueue_resume(db, j, answers_text)
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=new_job.id)


@router.post("/jobs/{job_id}/cancel")
async def cancel(job_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(409, f"job already {j.status}")
    j.cancel_requested = True
    if j.status == JobStatus.queued:
        j.status = JobStatus.cancelled
        v = await db.get(Version, j.version_id)
        if v:
            v.status = VersionStatus.cancelled
    await db.commit()
    # wake in-flight CAD executor tasks (topopt/fem check this key in-loop)
    try:
        pool = await get_pool()
        await pool.setex(f"cancel:{job_id}", 3600, b"1")
    except Exception:
        pass
    return {"ok": True}


@router.post("/jobs/{job_id}/finish")
async def finish(job_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Ask a running job to stop iterating early and finalize its best
    candidate instead of running the full budget."""
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j.status != JobStatus.running:
        raise HTTPException(409, f"job is {j.status}, not running")
    j.finish_requested = True
    await db.commit()
    return {"ok": True}


@router.post("/jobs/{job_id}/restart", response_model=EnqueueResponse)
async def restart(job_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Re-run a finished/failed/cancelled job with the same inputs.

    The dispatch lives in `spawn.spawn_restart` (shared with the MCP server)
    and keys off `job.type` first: resample/refine/organic/mesh_edit replay
    their stored requests into a fresh child of the same parent, fem and
    reexport spawn a new Job on the SAME version (returned without a new
    Version row), edit-feature jobs replay `run_edit_feature_job`, and
    generate/edit/params/optimize/code/meshgen follow their version-kind
    branches — unrecoverable inputs 409.
    """
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j.status not in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(409, f"job is {j.status} — cancel it before restarting")
    try:
        res = await spawn_restart(db, j)
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    if isinstance(res, Job):
        # same-version jobs (e.g. reexport) return the Job alone
        return EnqueueResponse(version_id=res.version_id, job_id=res.id)
    new_v, new_job = res
    return EnqueueResponse(version_id=new_v.id, job_id=new_job.id)
