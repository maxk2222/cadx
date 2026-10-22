"""Version tree + generation/edit/params/reexport endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from pathlib import Path
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
    CompareResult,
    EditFeatureRequest,
    EditRequest,
    EnqueueResponse,
    GenerateRequest,
    MeshgenRequest,
    OptimizeRequest,
    OrganicRequest,
    ParamsDiffEntry,
    ParamsRequest,
    RefineRequest,
    TopoptResampleRequest,
    VersionDetail,
    VersionSummary,
)
from cadx.db.models import (
    AgentTrace,
    Artifact,
    ArtifactKind,
    Asset,
    AssetKind,
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
    spawn_meshgen,
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


@router.post("/projects/{project_id}/meshgen", response_model=EnqueueResponse)
async def meshgen(
    project_id: uuid.UUID, body: MeshgenRequest, db: AsyncSession = Depends(get_session)
):
    """Image->mesh generation over REST — same spawn path as the MCP tool."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    ids = list(dict.fromkeys(body.asset_ids))
    rows = await db.execute(
        select(Asset).where(col(Asset.id).in_(ids), Asset.project_id == project_id)
    )
    assets = list(rows.scalars().all())
    if len(assets) != len(ids):
        raise HTTPException(422, "one or more asset_ids not found in this project")
    bad = [a.id for a in assets if a.kind not in (AssetKind.image, AssetKind.generated_image)]
    if bad:
        raise HTTPException(422, f"assets are not images: {[str(b) for b in bad]}")
    try:
        v, job = await spawn_meshgen(
            db,
            project,
            asset_ids=ids,
            subject=body.subject,
            model=body.model,
            seed=body.seed,
            post=body.post or None,
            prompt=body.prompt,
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
    jid = await enqueue(
        "run_reexport_job",
        {"job_id": str(job.id), "version_id": str(v.id), "iteration": v.iterations or 1},
    )
    if jid is None:
        # no worker behind the row — mark it instead of leaving it queued
        job.status = JobStatus.failed
        await db.commit()
        raise HTTPException(503, "enqueue failed — worker queue unavailable")
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


# OCC isn't thread-safe — concurrent compares run serialized under this
# lock; the acquire timeout is what actually bounds pile-up (a running OCC
# thread can't be killed).
_OCC_LOCK = threading.Lock()
_COMPARE_TIMEOUT_S = 300.0


class _CompareBusy(RuntimeError):
    """The OCC compare slot did not free up within _COMPARE_TIMEOUT_S."""


async def _compare_geometry_path(v: Version, db: AsyncSession) -> Path | None:
    """Best loadable geometry artifact for shape diffing, or None.

    Prefers the version-level ``exports/model.step``; else the newest
    ``kind="step"`` Artifact row (optimize/resample/mesh versions keep STEP
    under ``iterations/<n>/exports/``); else the best triangle mesh (sewn
    into a solid inside the worker thread).
    """
    from cadx import storage

    step = storage.export_dir(v.project_id, v.id) / "model.step"
    if step.exists():
        return step
    rows = await db.execute(
        select(Artifact)
        .where(Artifact.version_id == v.id, Artifact.kind == ArtifactKind.step)
        .order_by(col(Artifact.created_at).desc())
    )
    for art in rows.scalars().all():
        with contextlib.suppress(ValueError):
            p = storage.resolve(art.path)
            if p.suffix.lower() in (".step", ".stp") and p.exists():
                return p
    rel = storage.best_mesh_relpath(v.project_id, v.id, v.iterations)
    return storage.resolve(rel) if rel else None


def _load_compare_shape(path: Path) -> Any:
    """STEP via build123d; anything else via trimesh -> sewn solid.

    The mesh path is bounded by MAX_FACES_FOR_BOOLEAN — an unbounded
    ``mesh_to_solid`` can sew a huge STL effectively forever.
    """
    if path.suffix.lower() in (".step", ".stp"):
        from build123d import import_step

        return import_step(str(path))
    from cadx.cad.compare import MAX_FACES_FOR_BOOLEAN
    from cadx.cad.mesh_ops import load_mesh, mesh_to_solid

    return mesh_to_solid(load_mesh(path), max_faces=MAX_FACES_FOR_BOOLEAN)


def _params_diff(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[ParamsDiffEntry]:
    """Sorted union of both versions' params keys, reduced to entries whose
    values differ (a/b null when the key is absent on that side)."""
    pa, pb = a or {}, b or {}
    return [
        ParamsDiffEntry(key=k, a=pa.get(k), b=pb.get(k))
        for k in sorted(set(pa) | set(pb))
        if pa.get(k) != pb.get(k)
    ]


@router.get("/versions/{version_id}/compare/{other_id}", response_model=CompareResult)
async def compare_versions(
    version_id: uuid.UUID, other_id: uuid.UUID, db: AsyncSession = Depends(get_session)
):
    """Version-to-version diff: geometry (volumes, bboxes, added/removed
    material) plus params. OCC work is synchronous and can take seconds, so
    it runs in a thread; any load/compare failure degrades to ok=False —
    the endpoint never 500s on OCC weirdness."""
    a = await db.get(Version, version_id)
    if not a:
        raise HTTPException(404, "version not found")
    b = await db.get(Version, other_id)
    if not b:
        raise HTTPException(404, "version not found")

    path_a, path_b = await _compare_geometry_path(a, db), await _compare_geometry_path(b, db)
    if path_a is None or path_b is None:
        missing = a.id if path_a is None else b.id
        raise HTTPException(409, f"version {missing} has no loadable geometry artifact")

    base: dict[str, Any] = {
        "a": str(version_id),
        "b": str(other_id),
        "params_diff": _params_diff(a.params, b.params),
    }

    def _run() -> dict[str, Any]:
        from cadx.cad.compare import diff_summary, shape_compare

        if not _OCC_LOCK.acquire(timeout=_COMPARE_TIMEOUT_S):
            raise _CompareBusy("compare busy — another comparison is still running")
        try:
            sa, sb = _load_compare_shape(path_a), _load_compare_shape(path_b)
            shape = shape_compare(sa, sb)
            return {"shape": shape, "diff_summary": diff_summary(sa, sb, cmp=shape)}
        finally:
            _OCC_LOCK.release()

    try:
        # bounded wait — a timed-out coroutine can't kill its worker thread,
        # which finishes (and releases the lock) in the background
        out = await asyncio.wait_for(asyncio.to_thread(_run), timeout=_COMPARE_TIMEOUT_S)
    except TimeoutError:
        log.warning("versions.compare_timeout", a=str(version_id), b=str(other_id))
        return CompareResult(
            **base,
            ok=False,
            diff_summary="",
            shape=None,
            error=f"compare timed out after {_COMPARE_TIMEOUT_S:g}s",
        )
    except Exception as e:
        log.warning("versions.compare_failed", a=str(version_id), b=str(other_id), error=str(e))
        return CompareResult(
            **base, ok=False, diff_summary="", shape=None, error=f"{type(e).__name__}: {e}"
        )

    shape = out["shape"]
    err = shape.get("error") if isinstance(shape, dict) else None
    return CompareResult(
        **base, ok=err is None, diff_summary=out["diff_summary"], shape=shape, error=err
    )


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
