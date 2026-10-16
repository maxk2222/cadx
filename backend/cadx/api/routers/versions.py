"""Version tree + generation/edit/params/reexport endpoints."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx.api.deps import enqueue, get_session
from cadx.api.schemas import (
    ArtifactOut,
    EditFeatureRequest,
    EditRequest,
    EnqueueResponse,
    GenerateRequest,
    OptimizeRequest,
    OrganicRequest,
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
)
from cadx.spawn import (
    SpawnError,
    next_version_number,
    spawn_edit,
    spawn_edit_feature,
    spawn_generate,
    spawn_optimize,
    spawn_organic,
    spawn_params,
    spawn_refine,
    spawn_resample,
    spawn_version_job,
)

router = APIRouter(tags=["versions"])
log = structlog.get_logger()


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


# version/job spawning lives in cadx.spawn — shared with the chat agent's
# action tools so both entry points take the identical code path.
_next_number = next_version_number
_spawn = spawn_version_job


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
    try:
        v, job = await spawn_generate(
            db,
            project,
            body.prompt,
            gen_mode=body.mode or "",
            asset_ids=list(body.asset_ids or []),
            library_part_ids=list(body.library_part_ids or []),
            options=opts,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/edit", response_model=EnqueueResponse)
async def edit(version_id: uuid.UUID, body: EditRequest, db: AsyncSession = Depends(get_session)):
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    project = await db.get(Project, parent.project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    opts = (
        body.options.model_dump(exclude_none=True)
        if body.options is not None
        else (parent.options or {})
    )
    try:
        v, job = await spawn_edit(
            db,
            project,
            parent,
            body.prompt,
            gen_mode=body.mode or "",
            regenerate_mesh=body.regenerate_mesh,
            asset_ids=list(body.asset_ids or []),
            library_part_ids=(
                list(body.library_part_ids) if "library_part_ids" in body.model_fields_set else None
            ),
            options=opts,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/params", response_model=EnqueueResponse)
async def params(
    version_id: uuid.UUID, body: ParamsRequest, db: AsyncSession = Depends(get_session)
):
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    try:
        v, job = await spawn_params(db, parent, body.params)
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.get("/versions/{version_id}/editable-features")
async def list_editable_features(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Recognised features the parametric editor can resize (M9: plain
    through-holes). Runs quiddity recognition on the version's best B-Rep
    in-process — bounded by the module's face cap, ~seconds on big parts."""
    from cadx.spawn import feature_edit_brep_path

    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    src = feature_edit_brep_path(v)
    if src is None:
        return {"features": [], "reason": "version has no B-Rep solid to feature-edit"}

    def _scan() -> list[dict[str, Any]]:
        if src.suffix == ".step":
            from build123d import import_step

            shape = import_step(str(src))
        else:
            from build123d import import_brep

            shape = import_brep(str(src))
        from cadx.cad import editfeature

        return editfeature.find_editable_features(shape)

    try:
        features = await asyncio.to_thread(_scan)
    except Exception as e:
        log.warning("versions.editable_features_failed", version_id=str(version_id), error=str(e))
        return {"features": [], "reason": f"feature scan failed: {e}"}
    return {"features": features}


@router.post("/versions/{version_id}/edit-feature", response_model=EnqueueResponse)
async def edit_feature_version(
    version_id: uuid.UUID,
    body: EditFeatureRequest,
    db: AsyncSession = Depends(get_session),
):
    """Verified parametric edit of a recognised feature on the version's
    B-Rep (M9: plain through-holes, `parameter` is "diameter" only). Spawns a
    child version; the executor gate decides success — a refusal fails the
    child with the reason, never a mutated solid."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    try:
        v, job = await spawn_edit_feature(
            db,
            parent,
            feature_id=body.feature_id,
            parameter=body.parameter,
            value=body.value,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/optimize", response_model=EnqueueResponse)
async def optimize(
    version_id: uuid.UUID, body: OptimizeRequest, db: AsyncSession = Depends(get_session)
):
    """Spawn a topology-optimized child version (no LLM involved)."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    try:
        v, job = await spawn_optimize(
            db,
            parent,
            load_cases=body.load_cases,
            optimize=body.optimize,
            material=body.material,
            allow_reopt=body.allow_reopt,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
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
    try:
        v, job = await spawn_resample(
            db,
            parent,
            density_threshold=body.density_threshold,
            min_member_mm=body.min_member_mm,
            material=body.material,
            load_cases=body.load_cases,
            keep_regions=body.keep_regions,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
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
    try:
        v, job = await spawn_refine(
            db,
            parent,
            features=body.features,
            load_cases=body.load_cases,
            keep_regions=body.keep_regions,
            material=body.material,
            voxel_mm=body.voxel_mm,
            restore_planes=body.restore_planes,
            helical_threads=body.helical_threads,
            auto_plane_restore=body.auto_plane_restore,
        )
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/organic", response_model=EnqueueResponse)
async def organic_version(
    version_id: uuid.UUID, body: OrganicRequest, db: AsyncSession = Depends(get_session)
):
    """Procedural pseudo-topology restyle — converts the part into a
    deterministic organic structure (gyroid/trabecular/voronoi/branches)
    inside its own design domain. Creates an 'organic' child version
    (LLM-free). Aesthetic only — NOT structurally verified."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    organic = body.model_dump(mode="json", exclude={"anchors"})
    try:
        v, job = await spawn_organic(db, parent, organic)
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
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
