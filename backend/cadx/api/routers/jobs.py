"""Job inspection, clarification answers, cancellation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx.api.deps import get_pool, get_session
from cadx.api.schemas import AnswerRequest, EnqueueResponse, JobListItem, JobOut
from cadx.db.models import (
    Job,
    JobStatus,
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


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    j = await db.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _out(j)


@router.get("/jobs", response_model=list[JobListItem])
async def list_jobs(
    active: bool = False,
    limit: int = 200,
    db: AsyncSession = Depends(get_session),
):
    """Global job list for the Jobs page — newest first, optionally only
    non-terminal (queued/running/waiting_for_input) jobs."""
    q = (
        select(Job, Version, Project)
        .join(Version, Job.version_id == Version.id)
        .join(Project, Version.project_id == Project.id)
        .order_by(col(Job.created_at).desc())
        .limit(limit)
    )
    if active:
        q = q.where(
            col(Job.status).in_([JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input])
        )
    rows = await db.execute(q)
    out: list[JobListItem] = []
    for j, v, p in rows.all():
        d = _out(j).model_dump()
        d["log"] = (d.get("log") or "")[-500:]  # tail only — full logs are heavy
        out.append(
            JobListItem(
                **d,
                version_number=v.number,
                version_status=v.status,
                prompt=(v.prompt or "")[:140],
                project_id=p.id,
                project_name=p.name,
            )
        )
    return out


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
