"""Job inspection, clarification answers, cancellation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadagent.api.deps import enqueue, get_session
from cadagent.api.schemas import AnswerRequest, EnqueueResponse, JobOut
from cadagent.db.models import Job, JobStatus, JobType, Version, VersionStatus

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
        questions=j.questions,
        cancel_requested=j.cancel_requested,
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
    if j.status != JobStatus.waiting_for_input:
        raise HTTPException(409, f"job is {j.status}, not waiting_for_input")
    v = await db.get(Version, j.version_id)
    if not v:
        raise HTTPException(404, "version not found")

    qa_lines = []
    questions = j.questions or []
    for i, a in enumerate(body.answers):
        q = questions[i]["question"] if i < len(questions) else (a.id or f"q{i + 1}")
        qa_lines.append(f"Q: {q}\nA: {a.answer}")
    answers_text = "\n".join(qa_lines)

    v.prompt = (v.prompt or "") + f"\n\nClarification answers:\n{answers_text}"
    v.status = VersionStatus.queued
    new_job = Job(
        version_id=v.id,
        type=JobType.resume,
        resume_state=j.resume_state,
    )
    db.add(new_job)
    await db.commit()
    await enqueue("run_agent_job", {"job_id": str(new_job.id), "version_id": str(v.id)})
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
    return {"ok": True}
