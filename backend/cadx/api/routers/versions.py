"""Version tree + generation/edit/params/reexport endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadx.api.deps import enqueue, get_session
from cadx.api.schemas import (
    ArtifactOut,
    EditRequest,
    EnqueueResponse,
    GenerateRequest,
    OptimizeRequest,
    ParamsRequest,
    RefineRequest,
    TopoptResampleRequest,
    VersionDetail,
    VersionSummary,
)
from cadx.db.models import (
    AgentTrace,
    Artifact,
    Job,
    JobStatus,
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


def _active_job_dict(j: Job | None) -> dict | None:
    """Compact live-job summary embedded in VersionSummary."""
    if j is None:
        return None
    return {
        "id": str(j.id),
        "type": str(j.type),
        "status": str(j.status),
        "step": j.step,
        "iteration": j.iteration,
        "progress": j.progress,
        "max_iterations": (j.live or {}).get("max_iterations"),
    }


async def _summary(v: Version, db: AsyncSession, active_job: Job | None = None) -> VersionSummary:
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
        active_job=_active_job_dict(active_job),
    )


async def _detail(v: Version, db: AsyncSession) -> VersionDetail:
    base = await _summary(v, db)
    rows = await db.execute(select(Artifact).where(Artifact.version_id == v.id))
    return VersionDetail(
        **base.model_dump(),
        options=v.options or {},
        spec=v.spec,
        plan=v.plan,
        code=v.code,
        params=v.params,
        params_schema=v.params_schema,
        analysis=v.analysis,
        critique=v.critique,
        dfm=v.dfm,
        bom=v.bom,
        issues=v.issues,
        mesh=v.mesh,
        fem=v.fem,
        usage=v.usage,
        asset_ids=v.asset_ids,
        library_part_ids=v.library_part_ids,
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
    library_part_ids: list[uuid.UUID] | None = None,
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
        library_part_ids=[str(p) for p in (library_part_ids or [])],
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
    opts = body.options.model_dump(exclude_none=True) if body.options else {}
    gen_mode = (
        "topopt"
        if opts.get("topopt")
        else (body.mode or (project.settings or {}).get("gen_mode") or "auto")
    )
    return await _spawn(
        db,
        project,
        VersionKind.generate,
        JobType.generate,
        "run_agent_job",
        body.prompt,
        asset_ids=body.asset_ids,
        library_part_ids=body.library_part_ids,
        extra_version={"gen_mode": gen_mode, "options": opts},
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
    opts = (
        body.options.model_dump(exclude_none=True)
        if body.options is not None
        else (parent.options or {})
    )
    gen_mode = (
        "topopt"
        if opts.get("topopt")
        else body.mode or parent.resolved_mode or (project.settings or {}).get("gen_mode") or "auto"
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
        library_part_ids=(
            body.library_part_ids
            if "library_part_ids" in body.model_fields_set
            else [uuid.UUID(p) for p in parent.library_part_ids]
        ),
        extra_version={"gen_mode": gen_mode, "options": opts},
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
        options=parent.options or {},
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


@router.post("/versions/{version_id}/optimize", response_model=EnqueueResponse)
async def optimize(
    version_id: uuid.UUID, body: OptimizeRequest, db: AsyncSession = Depends(get_session)
):
    """Spawn a topology-optimized child version (no LLM involved)."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if parent.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot optimize version in status {parent.status}")
    if parent.resolved_mode == "topopt" and not body.allow_reopt:
        raise HTTPException(
            409,
            "version is already topology-optimized — use allow_reopt to "
            "optimize it again (its mesh becomes the design domain), or "
            "POST /topopt-resample to re-extract the stored density field",
        )
    v = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=await _next_number(db, parent.project_id),
        kind=VersionKind.optimize,
        prompt=f"optimize {parent.prompt}"[:500],
        gen_mode="topopt",
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": True,
            "optimize_request": {
                "load_cases": body.load_cases,
                "optimize": body.optimize,
                "material": body.material,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.optimize)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_optimize_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "load_cases": body.load_cases,
            "optimize": body.optimize,
            "material": body.material,
        },
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/topopt-resample", response_model=EnqueueResponse)
async def topopt_resample(
    version_id: uuid.UUID, body: TopoptResampleRequest, db: AsyncSession = Depends(get_session)
):
    """Re-extract the stored SIMP density field at a new density threshold /
    min-member size — no new SIMP run. Creates an 'optimize' child version."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if parent.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot resample version in status {parent.status}")
    if not ((parent.mesh or {}).get("topopt") or {}).get("density_relpath"):
        raise HTTPException(409, "version has no stored density field — run optimize first")
    v = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=await _next_number(db, parent.project_id),
        kind=VersionKind.optimize,
        prompt=f"resample {parent.prompt}"[:500],
        gen_mode="topopt",
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": True,
            "optimize_request": {
                "optimize": {
                    "density_threshold": body.density_threshold,
                    "min_member_mm": body.min_member_mm or None,
                },
                "material": body.material,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.resample)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_topopt_resample_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "load_cases": body.load_cases,
            "keep_regions": body.keep_regions,
            "material": body.material,
            "optimize": {
                "density_threshold": body.density_threshold,
                "min_member_mm": body.min_member_mm or None,
            },
        },
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/refine", response_model=EnqueueResponse)
async def refine_version(
    version_id: uuid.UUID, body: RefineRequest, db: AsyncSession = Depends(get_session)
):
    """Post-TopOpt feature finishing: re-cut exact bores/threads/inserts at
    measured anchors and restore keep-region surfaces from the
    pre-optimization part. Creates a 'refine' child version (no LLM)."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if parent.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot refine version in status {parent.status}")
    if parent.kind == VersionKind.refine:
        raise HTTPException(409, "version is already refined — refine a sibling instead")
    v = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=await _next_number(db, parent.project_id),
        kind=VersionKind.refine,
        prompt=f"refine {parent.prompt}"[:500],
        gen_mode=parent.gen_mode,
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": (parent.options or {}).get("topopt", False),
            "refine_request": {
                "features": body.features,
                "load_cases": body.load_cases,
                "keep_regions": body.keep_regions,
                "material": body.material,
                "voxel_mm": body.voxel_mm,
                "restore_planes": body.restore_planes,
                "helical_threads": body.helical_threads,
                "auto_plane_restore": body.auto_plane_restore,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.refine)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_refine_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "features": body.features,
            "load_cases": body.load_cases,
            "keep_regions": body.keep_regions,
            "material": body.material,
            "voxel_mm": body.voxel_mm,
            "restore_planes": body.restore_planes,
            "helical_threads": body.helical_threads,
            "auto_plane_restore": body.auto_plane_restore,
        },
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
    versions = list(rows.scalars().all())
    vids = [v.id for v in versions]
    active: dict[uuid.UUID, Job] = {}
    if vids:
        jr = await db.execute(
            select(Job)
            .where(
                col(Job.version_id).in_(vids),
                col(Job.status).in_(
                    [JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input]
                ),
            )
            .order_by(col(Job.created_at).desc())
        )
        for j in jr.scalars().all():
            active.setdefault(j.version_id, j)  # newest non-terminal job wins
    return [await _summary(v, db, active.get(v.id)) for v in versions]


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
    from cadx import storage

    jids = select(Job.id).where(col(Job.version_id) == v.id)
    # signal in-flight tasks before deleting their rows — a worker polling
    # cancel between the two commits stops gracefully; afterwards the
    # missing job row cancels the task outright (no zombie runs)
    await db.execute(
        sa_update(Job).where(col(Job.version_id) == v.id).values(cancel_requested=True)
    )
    await db.commit()
    await db.execute(sa_delete(AgentTrace).where(col(AgentTrace.job_id).in_(jids)))
    await db.execute(sa_delete(Job).where(col(Job.version_id) == v.id))
    await db.execute(sa_delete(Artifact).where(col(Artifact.version_id) == v.id))
    storage.delete_version_dir(v.project_id, v.id)
    await db.delete(v)
    await db.commit()
    return {"ok": True}
