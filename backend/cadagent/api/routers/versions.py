"""Version tree + generation/edit/params/reexport endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadagent.api.deps import enqueue, get_session
from cadagent.api.schemas import (
    ArtifactOut,
    EditRequest,
    EnqueueResponse,
    GenerateRequest,
    ParamsRequest,
    VersionDetail,
    VersionSummary,
)
from cadagent.db.models import (
    AgentTrace,
    Artifact,
    Job,
    JobType,
    Project,
    Version,
    VersionKind,
    VersionStatus,
)

router = APIRouter(tags=["versions"])


def _artifact_out(a: Artifact) -> ArtifactOut:
    return ArtifactOut(
        id=a.id,
        kind=a.kind,
        view=a.view,
        path=a.path,
        size=a.size,
        url=f"/api/artifacts/{a.id}/file",
        meta=a.meta or {},
    )


async def _summary(v: Version, db: AsyncSession) -> VersionSummary:
    thumb = await db.execute(
        select(Artifact).where(Artifact.version_id == v.id, Artifact.kind == "thumbnail")
    )
    t = thumb.scalars().first()
    return VersionSummary(
        id=v.id,
        project_id=v.project_id,
        parent_id=v.parent_id,
        number=v.number,
        kind=v.kind,
        prompt=v.prompt,
        status=v.status,
        score=v.score,
        iterations=v.iterations,
        error=v.error,
        gen_mode=v.gen_mode,
        resolved_mode=v.resolved_mode,
        created_at=v.created_at,
        thumbnail_url=f"/api/artifacts/{t.id}/file" if t else None,
    )


async def _detail(v: Version, db: AsyncSession) -> VersionDetail:
    base = await _summary(v, db)
    rows = await db.execute(select(Artifact).where(Artifact.version_id == v.id))
    return VersionDetail(
        **base.model_dump(),
        spec=v.spec,
        plan=v.plan,
        code=v.code,
        params=v.params,
        params_schema=v.params_schema,
        analysis=v.analysis,
        critique=v.critique,
        dfm=v.dfm,
        mesh=v.mesh,
        usage=v.usage,
        asset_ids=v.asset_ids,
        artifacts=[_artifact_out(a) for a in rows.scalars().all()],
    )


async def _next_number(db: AsyncSession, project_id: uuid.UUID) -> int:
    c = await db.execute(
        select(func.coalesce(func.max(Version.number), 0)).where(Version.project_id == project_id)
    )
    return int(c.scalar_one()) + 1


async def _spawn(
    db: AsyncSession,
    project: Project,
    kind: VersionKind,
    job_type: JobType,
    func: str,
    prompt: str,
    parent: Version | None = None,
    asset_ids: list[uuid.UUID] | None = None,
    extra_version: dict | None = None,
    extra_job: dict | None = None,
) -> EnqueueResponse:
    v = Version(
        project_id=project.id,
        parent_id=parent.id if parent else None,
        number=await _next_number(db, project.id),
        kind=kind,
        prompt=prompt,
        asset_ids=[str(a) for a in (asset_ids or [])],
        **(extra_version or {}),
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=job_type, **(extra_job or {}))
    db.add(job)
    await db.commit()
    await enqueue(func, {"job_id": str(job.id), "version_id": str(v.id)})
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/projects/{project_id}/generate", response_model=EnqueueResponse)
async def generate(
    project_id: uuid.UUID, body: GenerateRequest, db: AsyncSession = Depends(get_session)
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    if not body.prompt.strip():
        raise HTTPException(422, "prompt required")
    gen_mode = body.mode or (project.settings or {}).get("gen_mode") or "auto"
    return await _spawn(
        db,
        project,
        VersionKind.generate,
        JobType.generate,
        "run_agent_job",
        body.prompt,
        asset_ids=body.asset_ids,
        extra_version={"gen_mode": gen_mode},
    )


@router.post("/versions/{version_id}/edit", response_model=EnqueueResponse)
async def edit(version_id: uuid.UUID, body: EditRequest, db: AsyncSession = Depends(get_session)):
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if parent.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot edit version in status {parent.status}")
    if not parent.code and parent.resolved_mode != "mesh":
        raise HTTPException(409, "version has no code to edit")
    project = await db.get(Project, parent.project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    gen_mode = (
        body.mode or parent.resolved_mode or (project.settings or {}).get("gen_mode") or "auto"
    )
    extra_job = {"resume_state": {"regenerate_mesh": True}} if body.regenerate_mesh else None
    return await _spawn(
        db,
        project,
        VersionKind.edit,
        JobType.edit,
        "run_agent_job",
        body.prompt,
        parent=parent,
        asset_ids=body.asset_ids,
        extra_version={"gen_mode": gen_mode},
        extra_job=extra_job,
    )


@router.post("/versions/{version_id}/params", response_model=EnqueueResponse)
async def params(
    version_id: uuid.UUID, body: ParamsRequest, db: AsyncSession = Depends(get_session)
):
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if not parent.code and parent.resolved_mode != "mesh":
        raise HTTPException(409, "version has no code")
    v = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=await _next_number(db, parent.project_id),
        kind=VersionKind.params,
        prompt="params change",
        params={**(parent.params or {}), **body.params},
        params_schema=parent.params_schema,
        code=parent.code,
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.params)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_params_job",
        {"job_id": str(job.id), "version_id": str(v.id), "params": body.params},
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/reexport", response_model=EnqueueResponse)
async def reexport(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    job = Job(version_id=v.id, type=JobType.reexport)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_reexport_job",
        {"job_id": str(job.id), "version_id": str(v.id), "iteration": v.iterations or 1},
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.get("/projects/{project_id}/versions", response_model=list[VersionSummary])
async def version_tree(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Version).where(Version.project_id == project_id).order_by(col(Version.number))
    )
    return [await _summary(v, db) for v in rows.scalars().all()]


@router.get("/versions/{version_id}", response_model=VersionDetail)
async def get_version(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    return await _detail(v, db)


@router.delete("/versions/{version_id}")
async def delete_version(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    children = await db.execute(select(Version).where(Version.parent_id == v.id))
    if children.scalars().first():
        raise HTTPException(409, "version has children")
    from cadagent import storage

    jids = select(Job.id).where(col(Job.version_id) == v.id)
    await db.execute(sa_delete(AgentTrace).where(col(AgentTrace.job_id).in_(jids)))
    await db.execute(sa_delete(Job).where(col(Job.version_id) == v.id))
    await db.execute(sa_delete(Artifact).where(col(Artifact.version_id) == v.id))
    storage.delete_version_dir(v.project_id, v.id)
    await db.delete(v)
    await db.commit()
    return {"ok": True}
