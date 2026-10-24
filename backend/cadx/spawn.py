"""Shared version+job spawn helpers — single code path for the REST routers
and the chat agent's action tools.

Every function creates the Version+Job rows and enqueues the worker task, so
chat-spawned runs appear in the version tree, jobs dashboard and SSE channels
exactly like GUI-triggered ones. Guard failures raise `SpawnError` (carries an
HTTP status so routers can map it 1:1, chat tools map it to an error result).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select

from cadx import storage
from cadx.cad.allowlist import check_code
from cadx.cad.mesh_ops import OPS_REGISTRY, MeshPostParams
from cadx.db.models import (
    Job,
    JobStatus,
    JobType,
    Project,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import session_scope
from cadx.llm.schemas import ParamDef
from cadx.logging import get_logger

log = get_logger(__name__)


class SpawnError(Exception):
    """Spawn guard failure — `status` lets the router map it to HTTPException."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


async def _enqueue(func: str, payload: dict[str, Any]) -> None:
    from cadx.api.deps import enqueue

    if await enqueue(func, payload) is None:
        # deps.enqueue already logged the failure — without raising here the
        # freshly-committed Version/Job rows would sit `queued` forever with
        # no arq task behind them
        raise SpawnError("enqueue failed — worker queue unavailable", status=503)


async def _enqueue_or_fail(
    db: AsyncSession,
    enqueue: Callable[[str, dict[str, Any]], Awaitable[Any]],
    func: str,
    payload: dict[str, Any],
    *,
    version: Version | None = None,
    job: Job | None = None,
) -> None:
    """Enqueue the worker task, marking the freshly-spawned rows failed when
    the queue can't take the job — the boot reaper only reaps `running` jobs,
    so a `queued` row with no arq task would dangle forever.

    `job.payload` stores the verbatim {func, args} envelope BEFORE the enqueue
    attempt — the queue sweeper replays it when redis loses the arq entry."""
    if job is not None:
        from cadx.api.deps import payload_envelope

        job.payload = payload_envelope(func, payload)
        await db.commit()
    try:
        await enqueue(func, payload)
    except Exception:
        try:
            if version is not None:
                version.status = VersionStatus.failed
                version.error = "enqueue failed — worker queue unavailable"
            if job is not None:
                job.status = JobStatus.failed
            if version is not None or job is not None:
                await db.commit()
        except Exception:  # never mask the original enqueue error
            log.warning("enqueue_failure_mark_failed_error", func=func)
        raise


async def next_version_number(db: AsyncSession, project_id: uuid.UUID) -> int:
    c = await db.execute(
        select(func.coalesce(func.max(Version.number), 0)).where(Version.project_id == project_id)
    )
    return int(c.scalar_one()) + 1


async def spawn_version_job(
    db: AsyncSession,
    project: Project,
    *,
    kind: VersionKind,
    job_type: JobType,
    func: str,
    prompt: str,
    parent: Version | None = None,
    part_id: uuid.UUID | str | None = None,
    part_name: str | None = None,
    asset_ids: list[uuid.UUID | str] | None = None,
    library_part_ids: list[uuid.UUID | str] | None = None,
    extra_version: dict | None = None,
    extra_job: dict | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Generic Version+Job+enqueue — the old `_spawn` moved here verbatim.

    Part resolution for NEW ROOTS: `part_id`/`part_name` pick the owning
    part (validated/created via cadx.parts.resolve_part); omitted → a new
    part is auto-created from the prompt. Children always inherit
    `parent.part_id` — part args on a child spawn are rejected."""
    from cadx import parts as parts_svc  # lazy — parts.py imports SpawnError from here

    if parent is not None:
        if part_id is not None or part_name:
            raise SpawnError("part selection only applies to new root versions", status=409)
        resolved_part_id = parent.part_id
    else:
        resolved = await parts_svc.resolve_part(
            db,
            project.id,
            part_id=part_id,
            part_name=part_name,
            prompt_for_name=prompt,
        )
        resolved_part_id = resolved.id
    v = Version(
        project_id=project.id,
        part_id=resolved_part_id,
        parent_id=parent.id if parent else None,
        number=await next_version_number(db, project.id),
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
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        func,
        {"job_id": str(job.id), "version_id": str(v.id)},
        version=v,
        job=job,
    )
    return v, job


def _check_editable(v: Version, what: str) -> None:
    if v.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise SpawnError(f"cannot {what} version in status {v.status}")


async def spawn_generate(
    db: AsyncSession,
    project: Project,
    prompt: str,
    *,
    gen_mode: str = "auto",
    part_id: uuid.UUID | str | None = None,
    part_name: str | None = None,
    asset_ids: list[uuid.UUID | str] | None = None,
    library_part_ids: list[uuid.UUID | str] | None = None,
    options: dict | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    if not prompt.strip():
        raise SpawnError("prompt required", status=422)
    opts = options or {}
    mode = (
        "topopt"
        if opts.get("topopt")
        else gen_mode or (project.settings or {}).get("gen_mode") or "auto"
    )
    return await spawn_version_job(
        db,
        project,
        kind=VersionKind.generate,
        job_type=JobType.generate,
        func="run_agent_job",
        prompt=prompt,
        part_id=part_id,
        part_name=part_name,
        asset_ids=asset_ids,
        library_part_ids=library_part_ids,
        extra_version={"gen_mode": mode, "options": opts},
        enqueue_fn=enqueue_fn,
    )


async def spawn_edit(
    db: AsyncSession,
    project: Project,
    parent: Version,
    prompt: str,
    *,
    gen_mode: str | None = None,
    regenerate_mesh: bool = False,
    asset_ids: list[uuid.UUID | str] | None = None,
    library_part_ids: list[uuid.UUID | str] | None = None,
    options: dict | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    _check_editable(parent, "edit")
    if not parent.code and parent.resolved_mode != "mesh":
        raise SpawnError("version has no code to edit")
    opts = options if options is not None else (parent.options or {})
    mode = (
        "topopt"
        if opts.get("topopt")
        else gen_mode or parent.resolved_mode or (project.settings or {}).get("gen_mode") or "auto"
    )
    return await spawn_version_job(
        db,
        project,
        kind=VersionKind.edit,
        job_type=JobType.edit,
        func="run_agent_job",
        prompt=prompt,
        parent=parent,
        asset_ids=asset_ids,
        library_part_ids=(
            library_part_ids
            if library_part_ids is not None
            else [uuid.UUID(p) for p in parent.library_part_ids]
        ),
        extra_version={"gen_mode": mode, "options": opts},
        extra_job={"resume_state": {"regenerate_mesh": True}} if regenerate_mesh else None,
        enqueue_fn=enqueue_fn,
    )


async def spawn_params(
    db: AsyncSession,
    parent: Version,
    params: dict[str, Any],
    *,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    if not parent.code and parent.resolved_mode != "mesh":
        raise SpawnError("version has no code")
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.params,
        prompt="params change",
        params={**(parent.params or {}), **params},
        params_schema=parent.params_schema,
        code=parent.code,
        options=parent.options or {},
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.params)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_params_job",
        {"job_id": str(job.id), "version_id": str(v.id), "params": params},
        version=v,
        job=job,
    )
    return v, job


async def spawn_optimize(
    db: AsyncSession,
    parent: Version,
    *,
    load_cases: list | None,
    optimize: dict | None,
    material: str | None,
    allow_reopt: bool = False,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    _check_editable(parent, "optimize")
    if parent.resolved_mode == "topopt" and not allow_reopt:
        raise SpawnError(
            "version is already topology-optimized — pass allow_reopt to "
            "optimize it again, or topopt_resample to re-extract the stored "
            "density field"
        )
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.optimize,
        prompt=f"optimize {parent.prompt}"[:500],
        gen_mode="topopt",
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": True,
            "optimize_request": {
                "load_cases": load_cases,
                "optimize": optimize,
                "material": material,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.optimize)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_optimize_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "load_cases": load_cases,
            "optimize": optimize,
            "material": material,
        },
        version=v,
        job=job,
    )
    return v, job


async def spawn_resample(
    db: AsyncSession,
    parent: Version,
    *,
    density_threshold: float | None,
    min_member_mm: float | None,
    material: str | None,
    load_cases: list | None = None,
    keep_regions: list | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Re-extract the stored density field — works on optimize AND organic
    parents (organic versions carry a density_relpath too)."""
    _check_editable(parent, "resample")
    organic_parent = parent.kind == VersionKind.organic
    if not (
        ((parent.mesh or {}).get("topopt") or {}).get("density_relpath")
        or ((parent.mesh or {}).get("organic") or {}).get("density_relpath")
    ):
        raise SpawnError("version has no stored density field — run optimize first")
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.organic if organic_parent else VersionKind.optimize,
        prompt=f"resample {parent.prompt}"[:500],
        gen_mode="mesh" if organic_parent else "topopt",
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": True,
            "optimize_request": {
                "optimize": {
                    "density_threshold": density_threshold,
                    "min_member_mm": min_member_mm or None,
                },
                "material": material,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.resample)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_topopt_resample_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "load_cases": load_cases,
            "keep_regions": keep_regions,
            "material": material,
            "optimize": {
                "density_threshold": density_threshold,
                "min_member_mm": min_member_mm or None,
            },
        },
        version=v,
        job=job,
    )
    return v, job


async def spawn_refine(
    db: AsyncSession,
    parent: Version,
    *,
    features: list | None = None,
    load_cases: list | None = None,
    keep_regions: list | None = None,
    material: str | None = None,
    voxel_mm: float | None = None,
    restore_planes: bool | None = None,
    helical_threads: bool | None = None,
    auto_plane_restore: bool | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    _check_editable(parent, "refine")
    if parent.kind == VersionKind.refine:
        raise SpawnError("version is already refined — refine a sibling instead")
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.refine,
        prompt=f"refine {parent.prompt}"[:500],
        gen_mode=parent.gen_mode,
        asset_ids=parent.asset_ids or [],
        options={
            **(parent.options or {}),
            "topopt": (parent.options or {}).get("topopt", False),
            "refine_request": {
                "features": features,
                "load_cases": load_cases,
                "keep_regions": keep_regions,
                "material": material,
                "voxel_mm": voxel_mm,
                "restore_planes": restore_planes,
                "helical_threads": helical_threads,
                "auto_plane_restore": auto_plane_restore,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.refine)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_refine_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "features": features,
            "load_cases": load_cases,
            "keep_regions": keep_regions,
            "material": material,
            "voxel_mm": voxel_mm,
            "restore_planes": restore_planes,
            "helical_threads": helical_threads,
            "auto_plane_restore": auto_plane_restore,
        },
        version=v,
        job=job,
    )
    return v, job


async def spawn_organic(
    db: AsyncSession,
    parent: Version,
    organic: dict[str, Any],
    *,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    _check_editable(parent, "restyle")
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.organic,
        prompt=f"organic {parent.prompt}"[:500],
        gen_mode="mesh",
        asset_ids=parent.asset_ids or [],
        options={**(parent.options or {}), "organic_request": organic},
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.organic)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_organic_job",
        {"job_id": str(job.id), "version_id": str(v.id), "organic": organic},
        version=v,
        job=job,
    )
    return v, job


async def spawn_code(
    db: AsyncSession,
    project: Project,
    code: str,
    *,
    params: dict[str, Any] | None = None,
    parent: Version | None = None,
    part_id: uuid.UUID | str | None = None,
    part_name: str | None = None,
    prompt: str = "",
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Persist + execute externally-written build123d code (LLM-free, MCP).

    The code is contract-checked exactly like LLM output — it runs through
    the same sandboxed execute pipeline. Declared PARAMS become the
    version's params_schema so the GUI params panel works on code versions.
    """
    if not (code or "").strip():
        raise SpawnError("code required", status=422)
    check = check_code(code)
    if not check.ok:
        errs = (check.errors + check.sig_errors)[:3]
        raise SpawnError(
            "code check failed: " + "; ".join(errs or ["unknown"]),
            status=422,
        )
    params_schema = [
        ParamDef(name=k, value=v).model_dump()
        for k, v in check.params.items()
        if isinstance(v, str | int | float | bool)
    ]
    return await spawn_version_job(
        db,
        project,
        kind=VersionKind.code,
        job_type=JobType.code,
        func="run_code_job",
        prompt=prompt[:500],
        parent=parent,
        part_id=part_id,
        part_name=part_name,
        asset_ids=([uuid.UUID(a) for a in parent.asset_ids] if parent else None),
        library_part_ids=([uuid.UUID(p) for p in parent.library_part_ids] if parent else None),
        extra_version={
            "code": code,
            "params": params or {},
            "params_schema": params_schema,
        },
        enqueue_fn=enqueue_fn,
    )


async def spawn_meshgen(
    db: AsyncSession,
    project: Project,
    *,
    asset_ids: list[uuid.UUID | str],
    subject: str = "",
    model: str = "",
    seed: int = 0,
    post: dict | None = None,
    part_id: uuid.UUID | str | None = None,
    part_name: str | None = None,
    prompt: str = "",
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Image->mesh generation via the mesh sidecar (LLM-free, MCP).

    Generation inputs live in version.options so the job + restart can
    replay them verbatim; `post` is validated against MeshPostParams now so
    a bad shape fails before the version row exists.
    """
    if not asset_ids:
        raise SpawnError("asset_ids required — at least one image asset", status=422)
    try:
        MeshPostParams(**(post or {}))
    except Exception as e:
        raise SpawnError(f"invalid mesh post params: {e}", status=422) from e
    return await spawn_version_job(
        db,
        project,
        kind=VersionKind.meshgen,
        job_type=JobType.meshgen,
        func="run_meshgen_job",
        prompt=(prompt or subject or "mesh generation")[:500],
        part_id=part_id,
        part_name=part_name,
        asset_ids=asset_ids,
        extra_version={
            "gen_mode": "mesh",
            "options": {
                "subject": subject,
                "model": model,
                "seed": seed,
                "post": post or {},
            },
        },
        enqueue_fn=enqueue_fn,
    )


async def spawn_restart(  # noqa: C901
    db: AsyncSession,
    job: Job,
    *,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job] | Job:
    """Re-run a finished/failed/cancelled job with the same inputs — shared
    by the REST jobs router and the MCP server.

    Dispatch is on `job.type` first — several job types share a version kind
    with siblings of other types (resample jobs produce kind=optimize
    versions like optimize jobs do), and fem/reexport jobs have no child
    version at all — then falls back to the version-kind branches:

    resample      → new resample child replaying options.optimize_request.
    refine        → new refine child replaying options.refine_request.
    organic       → new organic child replaying options.organic_request.
    mesh_edit     → new mesh_edit child replaying the recorded params.ops.
    fem           → new fem Job on the SAME version (stored fem_request).
    reexport      → new reexport Job on the SAME version.
    edit+feature  → kind=edit version spawned by spawn_edit_feature replays
                    options.edit_feature_request (genuine LLM edits take the
                    kind branch below).
    generate/edit → new sibling version cloning the version's inputs; a
                    `resume` job stays `resume` so run_agent_job restores
                    its stored spec.
    params        → new params child carrying the same effective overrides.
    optimize      → new optimize child of the same parent (stored request
                    when the original run saved one, spec fallback otherwise).
    code          → new code version re-executing the stored code+params.
    meshgen       → new meshgen version replaying the stored gen options.
    anything else → SpawnError(409).
    """
    if job.status not in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
        raise SpawnError(f"job is {job.status} — cancel it before restarting")
    v = await db.get(Version, job.version_id)
    if not v:
        raise SpawnError("version not found", status=404)
    enqueue = enqueue_fn or _enqueue
    parent = await db.get(Version, v.parent_id) if v.parent_id else None
    opts = v.options or {}

    # --- job.type dispatch (before the kind branches — these types share
    # version kinds with siblings of other types) ---

    if job.type == JobType.resample:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        # resample children store the request as optimize_request (only the
        # optimize block + material — load cases/keep regions resolve from
        # the parent's params/spec at run time, as in the original run)
        req = opts.get("optimize_request") or {}
        opt = req.get("optimize") or {}
        return await spawn_resample(
            db,
            parent,
            density_threshold=opt.get("density_threshold"),
            min_member_mm=opt.get("min_member_mm"),
            material=req.get("material"),
            enqueue_fn=enqueue_fn,
        )

    if job.type == JobType.refine:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        req = opts.get("refine_request")
        if not isinstance(req, dict):
            raise SpawnError("version has no stored refine request — cannot restart")
        return await spawn_refine(
            db,
            parent,
            features=req.get("features"),
            load_cases=req.get("load_cases"),
            keep_regions=req.get("keep_regions"),
            material=req.get("material"),
            voxel_mm=req.get("voxel_mm"),
            restore_planes=req.get("restore_planes"),
            helical_threads=req.get("helical_threads"),
            auto_plane_restore=req.get("auto_plane_restore"),
            enqueue_fn=enqueue_fn,
        )

    if job.type == JobType.organic:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        organic = opts.get("organic_request")
        if not isinstance(organic, dict):
            raise SpawnError("version has no stored organic request — cannot restart")
        return await spawn_organic(db, parent, organic, enqueue_fn=enqueue_fn)

    if job.type == JobType.mesh_edit:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        # run_mesh_edit_job persists the applied op stack as version.params
        # (a MeshPostParams dump); a job that died before the run has no
        # recorded ops to replay
        recorded = v.params or {}
        ops = recorded.get("ops") or []
        if not ops:
            raise SpawnError("version has no recorded mesh ops — cannot restart")
        post = {k: val for k, val in recorded.items() if k != "ops"}
        return await spawn_mesh_edit(
            db, parent, ops=ops, post=post, note=v.prompt or "", enqueue_fn=enqueue_fn
        )

    if job.type == JobType.fem:
        # same-version job — the request lives on the job's resume_state
        # (spawn_fem); fem jobs spawned before that was stored have nothing
        # recoverable to replay
        req = (job.resume_state or {}).get("fem_request") or {}
        if not req.get("load_cases"):
            raise SpawnError("fem job has no stored load cases — cannot restart")
        return await spawn_fem(
            db,
            v,
            load_cases=req.get("load_cases"),
            keep_regions=req.get("keep_regions"),
            material=req.get("material"),
            enqueue_fn=enqueue_fn,
        )

    if job.type == JobType.reexport:
        # same-version job, like the reexport endpoint — no child version
        new_job = Job(version_id=v.id, type=JobType.reexport)
        db.add(new_job)
        await db.commit()
        await _enqueue_or_fail(
            db,
            enqueue,
            "run_reexport_job",
            {
                "job_id": str(new_job.id),
                "version_id": str(v.id),
                "iteration": v.iterations or 1,
            },
            job=new_job,
        )
        return new_job

    if (
        job.type == JobType.edit
        and v.kind == VersionKind.edit
        and isinstance(opts.get("edit_feature_request"), dict)
        and str(v.prompt or "").startswith("feature edit:")
    ):
        # a deterministic feature edit, not an LLM edit (the prompt check
        # keeps LLM-edit children — which inherit the parent's options — on
        # the kind branch below)
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        req = opts["edit_feature_request"]
        if req.get("feature_id") is None or req.get("value") is None:
            raise SpawnError("stored feature-edit request is incomplete — cannot restart")
        return await spawn_edit_feature(
            db,
            parent,
            feature_id=req["feature_id"],
            parameter=req.get("parameter") or "diameter",
            value=req["value"],
            enqueue_fn=enqueue_fn,
        )

    # --- version-kind dispatch (LLM-generated kinds + the rest) ---

    if v.kind in (VersionKind.generate, VersionKind.edit):
        new_v = Version(
            project_id=v.project_id,
            part_id=v.part_id,
            parent_id=v.parent_id,
            number=await next_version_number(db, v.project_id),
            kind=v.kind,
            prompt=v.prompt,
            asset_ids=v.asset_ids or [],
            library_part_ids=v.library_part_ids or [],
            gen_mode=v.gen_mode,
            options=v.options or {},
        )
        db.add(new_v)
        await db.flush()
        new_job = Job(
            version_id=new_v.id,
            # a resume job must stay `resume` — run_agent_job only restores
            # the stored spec for JobType.resume; re-typing it to
            # generate/edit silently dropped the resume semantics
            type=(
                JobType.resume
                if job.type == JobType.resume
                else JobType.edit
                if v.kind == VersionKind.edit
                else JobType.generate
            ),
            resume_state=job.resume_state,
        )
        db.add(new_job)
        await db.commit()
        await _enqueue_or_fail(
            db,
            enqueue,
            "run_agent_job",
            {"job_id": str(new_job.id), "version_id": str(new_v.id)},
            version=new_v,
            job=new_job,
        )
        return new_v, new_job

    if v.kind == VersionKind.params:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        new_v = Version(
            project_id=v.project_id,
            part_id=v.part_id,
            parent_id=v.parent_id,
            number=await next_version_number(db, v.project_id),
            kind=VersionKind.params,
            prompt="params change",
            params=v.params or {},
            params_schema=v.params_schema,
            code=v.code,
            options=v.options or {},
        )
        db.add(new_v)
        await db.flush()
        new_job = Job(version_id=new_v.id, type=JobType.params)
        db.add(new_job)
        await db.commit()
        await _enqueue_or_fail(
            db,
            enqueue,
            "run_params_job",
            {"job_id": str(new_job.id), "version_id": str(new_v.id), "params": v.params or {}},
            version=new_v,
            job=new_job,
        )
        return new_v, new_job

    if v.kind == VersionKind.optimize:
        if parent is None:
            raise SpawnError("parent version missing — cannot restart")
        req = opts.get("optimize_request") or {}
        new_v = Version(
            project_id=v.project_id,
            part_id=v.part_id,
            parent_id=parent.id,
            number=await next_version_number(db, v.project_id),
            kind=VersionKind.optimize,
            prompt=f"optimize {parent.prompt}"[:500],
            gen_mode="topopt",
            asset_ids=parent.asset_ids or [],
            options={**(parent.options or {}), "topopt": True, "optimize_request": req},
        )
        db.add(new_v)
        await db.flush()
        new_job = Job(version_id=new_v.id, type=JobType.optimize)
        db.add(new_job)
        await db.commit()
        await _enqueue_or_fail(
            db,
            enqueue,
            "run_optimize_job",
            {
                "job_id": str(new_job.id),
                "version_id": str(new_v.id),
                "load_cases": req.get("load_cases") or [],
                "optimize": req.get("optimize") or {},
                "material": req.get("material") or "",
            },
            version=new_v,
            job=new_job,
        )
        return new_v, new_job

    if v.kind == VersionKind.code:
        if not (v.code or "").strip():
            raise SpawnError("version has no code to re-run")
        project = await db.get(Project, v.project_id)
        if project is None:
            raise SpawnError("project not found", status=404)
        return await spawn_code(
            db,
            project,
            v.code or "",
            params=v.params or {},
            parent=parent,
            # root restart stays in its own part; a child inherits via parent
            # (part args on a child spawn are rejected by spawn_version_job)
            part_id=None if parent else v.part_id,
            prompt=v.prompt,
            enqueue_fn=enqueue_fn,
        )

    if v.kind == VersionKind.meshgen:
        project = await db.get(Project, v.project_id)
        if project is None:
            raise SpawnError("project not found", status=404)
        return await spawn_meshgen(
            db,
            project,
            asset_ids=v.asset_ids or [],
            subject=str(opts.get("subject") or ""),
            model=str(opts.get("model") or ""),
            seed=int(opts.get("seed") or 0),
            post=opts.get("post") or {},
            # root restart stays in its own part; a child inherits via parent
            part_id=None if parent else v.part_id,
            prompt=v.prompt,
            enqueue_fn=enqueue_fn,
        )

    raise SpawnError(f"cannot restart a '{v.kind}' version")


def validate_ops(ops: list[dict[str, Any]]) -> None:
    unknown = [o.get("op") for o in ops if o.get("op") not in OPS_REGISTRY]
    if unknown:
        raise SpawnError(
            f"unknown mesh op(s): {', '.join(sorted(set(unknown)))} — "
            f"available: {', '.join(sorted(OPS_REGISTRY))}",
            status=422,
        )


def mesh_edit_source(v: Version) -> None:
    """Guard: raises SpawnError when the version has no editable mesh."""
    if (v.resolved_mode or "") == "mesh":
        raw = storage.version_dir(v.project_id, v.id) / "mesh" / "mesh_raw.glb"
        if raw.exists():
            return
    if storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1) is None:
        raise SpawnError("version has no mesh artifact to edit")


async def spawn_mesh_edit(
    db: AsyncSession,
    parent: Version,
    *,
    ops: list[dict[str, Any]],
    post: dict | None = None,
    note: str = "",
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    _check_editable(parent, "edit")
    validate_ops(ops)
    mesh_edit_source(parent)
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.mesh_edit,
        prompt=(note or f"visual edit of v{parent.number}")[:500],
        gen_mode=parent.gen_mode,
        options=parent.options or {},
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.mesh_edit)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_mesh_edit_job",
        {"job_id": str(job.id), "version_id": str(v.id), "ops": ops, "post": post or {}},
        version=v,
        job=job,
    )
    return v, job


def feature_edit_brep_path(v: Version) -> Path | None:
    """Best B-Rep/STEP source on disk for feature editing, or None.

    Search order mirrors run_edit_feature_job's _parent_source_model:
    canonical mesh/solid.brep (written by prior edits/mesh versions),
    the newest iterations/*/_run/model.brep, then exports/model.step."""
    v_dir = storage.version_dir(v.project_id, v.id)
    if (v_dir / "mesh" / "solid.brep").exists():
        return v_dir / "mesh" / "solid.brep"
    it_root = v_dir / "iterations"
    candidates: list[Path] = []
    if it_root.is_dir():
        for it in it_root.iterdir():
            if not it.is_dir():
                continue
            if (it / "_run" / "model.brep").exists():
                candidates.append(it / "_run" / "model.brep")
            candidates.extend(
                sub / "_run" / "model.brep"
                for sub in it.iterdir()
                if sub.is_dir() and (sub / "_run" / "model.brep").exists()
            )
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if (v_dir / "exports" / "model.step").exists():
        return v_dir / "exports" / "model.step"
    return None


def feature_edit_source(v: Version) -> None:
    """Guard: raises SpawnError when the version has no analytic B-Rep to
    feature-edit (mesh-only versions can't be edited parametrically)."""
    if feature_edit_brep_path(v) is None:
        raise SpawnError("version has no B-Rep solid to feature-edit")


async def spawn_edit_feature(
    db: AsyncSession,
    parent: Version,
    *,
    feature_id: int | str,
    parameter: str,
    value: float,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Verified parametric edit of a recognised feature (M9). Creates a
    ``kind=edit`` child; the executor's gate decides whether it succeeds —
    the child version fails with the refusal otherwise."""
    _check_editable(parent, "edit")
    if str(parameter).strip().lower() != "diameter":
        raise SpawnError(
            f"unsupported parameter {parameter!r} — only 'diameter' is editable",
            status=422,
        )
    feature_edit_source(parent)
    v = Version(
        project_id=parent.project_id,
        part_id=parent.part_id,
        parent_id=parent.id,
        number=await next_version_number(db, parent.project_id),
        kind=VersionKind.edit,
        prompt=(f"feature edit: {parameter}={value} on feature {feature_id} (of v{parent.number})")[
            :500
        ],
        gen_mode=parent.gen_mode,
        asset_ids=parent.asset_ids or [],
        spec=parent.spec,
        code=parent.code,
        params=parent.params,
        params_schema=parent.params_schema,
        options={
            **(parent.options or {}),
            "edit_feature_request": {
                "feature_id": feature_id,
                "parameter": parameter,
                "value": value,
            },
        },
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.edit)
    db.add(job)
    await db.commit()
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_edit_feature_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "feature_id": feature_id,
            "parameter": parameter,
            "value": value,
        },
        version=v,
        job=job,
    )
    return v, job


async def spawn_fem(
    db: AsyncSession,
    version: Version,
    *,
    load_cases: list | None,
    keep_regions: list | None = None,
    material: str | None = None,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> Job:
    """Standalone FEM run on the SAME version (no child version).

    `load_cases` falls back to the version's spec-declared cases — the same
    cascade optimize uses (explicit request wins, else what the part was
    designed against). `load_cases_source` records which supplied them so
    restarts and eval reports can tell request cases from spec cases."""
    _check_editable(version, "analyse")
    if storage.best_mesh_relpath(version.project_id, version.id, version.iterations or 1) is None:
        raise SpawnError("version has no mesh artifact to analyse")
    lcs = load_cases or None
    source = "request" if lcs else "spec"
    if not lcs:
        spec_cases = (version.spec or {}).get("load_cases") or []
        lcs = [dict(c) for c in spec_cases if isinstance(c, dict)] or None
    if not lcs:
        raise SpawnError("at least one load case is required", status=422)
    job = Job(
        version_id=version.id,
        type=JobType.fem,
        # the Job hangs off the version itself (no child) — the request has to
        # live on the job row so spawn_restart can replay it
        resume_state={
            "fem_request": {
                "load_cases": lcs,
                "keep_regions": keep_regions,
                "material": material,
                "load_cases_source": source,
            }
        },
    )
    db.add(job)
    await db.commit()
    # only the new Job row is ours to fail — the version is the user's
    # existing (succeeded) version, untouched by this spawn
    await _enqueue_or_fail(
        db,
        enqueue_fn or _enqueue,
        "run_fem_job",
        {
            "job_id": str(job.id),
            "version_id": str(version.id),
            "load_cases": lcs,
            "keep_regions": keep_regions,
            "material": material,
        },
        job=job,
    )
    return job


async def enqueue_resume(
    db: AsyncSession,
    job: Job,
    answers_text: str,
    *,
    enqueue_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
) -> tuple[Version, Job]:
    """Submit clarification answers to a waiting_for_input job — spawns a
    `resume` job on the same version (extracted from jobs.py::answer)."""
    if job.status != JobStatus.waiting_for_input:
        raise SpawnError(f"job is {job.status}, not waiting_for_input")
    v = await db.get(Version, job.version_id)
    if not v:
        raise SpawnError("version not found", status=404)
    v.prompt = (v.prompt or "") + f"\n\nClarification answers:\n{answers_text}"
    v.status = VersionStatus.queued
    new_job = Job(version_id=v.id, type=JobType.resume, resume_state=job.resume_state)
    db.add(new_job)
    # the waiting job's lifecycle ends here — its continuation is the resume.
    # Leaving it waiting_for_input keeps the answer form live and lets a
    # double-submit (or a stale UI) spawn parallel resumes on the same spec.
    job.status = JobStatus.succeeded
    job.finished_at = datetime.now(UTC)
    job.log = (job.log or "") + f"answered — continued as job {new_job.id}\n"
    from cadx.api.deps import payload_envelope

    args = {"job_id": str(new_job.id), "version_id": str(v.id)}
    new_job.payload = payload_envelope("run_agent_job", args)
    await db.commit()
    try:
        await (enqueue_fn or _enqueue)("run_agent_job", args)
    except Exception:
        # fail the new job and put the version + original job back — the
        # original job is still waiting_for_input, so 'failed'/'succeeded'
        # would misrepresent it
        try:
            new_job.status = JobStatus.failed
            v.status = VersionStatus.waiting_for_input
            job.status = JobStatus.waiting_for_input
            job.finished_at = None
            await db.commit()
        except Exception:  # never mask the original enqueue error
            log.warning("enqueue_failure_mark_failed_error", func="run_agent_job")
        raise
    return v, new_job


# --------------------------------------------------------------------------- #
# waiting on spawned jobs (chat tools block here while the UI live-cards)      #
# --------------------------------------------------------------------------- #

TERMINAL_JOB_STATES = {
    JobStatus.succeeded,
    JobStatus.failed,
    JobStatus.cancelled,
    JobStatus.waiting_for_input,
}


def _compact_fem(fem: dict[str, Any]) -> dict[str, Any]:
    """Agent-sized FemReport digest — per-case verdict + where the loads
    and supports actually landed (resultant + centroids are the self-check
    for BC mistakes)."""
    cases = []
    for c in (fem.get("cases") or [])[:6]:
        rx = c.get("reactions") or {}
        cases.append(
            {
                "name": c.get("name"),
                "safety_factor": c.get("safety_factor"),
                "max_vm_mpa": c.get("max_vm_mpa"),
                "max_disp_mm": c.get("max_disp_mm"),
                "work_disp_mm": c.get("work_disp_mm"),
                "converged": c.get("converged"),
                "resultant_n": rx.get("resultant_n"),
                "fixed_centroid_mm": rx.get("fixed_centroid_mm"),
                "load_centroid_mm": _load_centroid(c.get("applied") or []),
                "warnings": (c.get("warnings") or [])[:4],
                "feature_peaks": (c.get("feature_peaks") or [])[:6],
            }
        )
    return {
        "worst_safety_factor": fem.get("worst_safety_factor"),
        "model": fem.get("model"),
        "cases": cases,
        "warnings": (fem.get("warnings") or [])[:5],
    }


def _load_centroid(applied: list[dict]) -> list[float] | None:
    """Force-weighted mean of the applied-load centroids — where the load
    actually attacked (vs. where it was declared)."""
    tot = 0.0
    acc = [0.0, 0.0, 0.0]
    for rec in applied:
        w = float(rec.get("force_mag_n") or 0.0)
        p = rec.get("centroid_mm")
        if w <= 0 or not isinstance(p, (list, tuple)) or len(p) < 3:
            continue
        for i in range(3):
            acc[i] += w * float(p[i])
        tot += w
    if tot <= 0:
        return None
    return [round(v / tot, 2) for v in acc]


def _compact_topopt(top: dict[str, Any]) -> dict[str, Any]:
    """Agent-sized TopOpt digest — headline safety factors + the flags
    that change the verdict (uniform baseline, load path, solver used)."""
    out = {
        "solver_used": top.get("solver_used") or (top.get("verification") or {}).get("solver_used"),
        "status": top.get("status"),
        "converged": top.get("converged"),
        "volfrac_achieved": top.get("volfrac_achieved"),
        "volfrac_export": top.get("volfrac_export"),
        "sf_verified": top.get("sf_verified"),
        "sf_export": top.get("sf_export"),
        "sf_mask": top.get("sf_mask"),
        "load_path_ok": top.get("load_path_ok"),
        "worse_than_uniform": top.get("worse_than_uniform"),
        "uniform_compliance_mj": top.get("uniform_compliance_mj"),
        "min_member_mm": top.get("min_member_mm"),
        "n_bodies_export": top.get("n_bodies_export"),
        "warnings": (top.get("warnings") or [])[:6],
    }
    return {k: v for k, v in out.items() if v is not None}


def _job_summary(job: Job, v: Version | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": job.status in (JobStatus.succeeded, JobStatus.waiting_for_input),
        "status": job.status.value,
        "job_id": str(job.id),
        "version_id": str(job.version_id),
    }
    if v is not None:
        out.update(
            version_number=v.number,
            version_status=v.status.value,
            kind=v.kind.value,
            score=v.score,
            iterations=v.iterations,
            resolved_mode=v.resolved_mode,
            error=v.error,
        )
        a = v.analysis or {}
        if a.get("bbox_size"):
            out["bbox_mm"] = a["bbox_size"]
        if a.get("volume") is not None:
            out["volume_mm3"] = a["volume"]
        if v.issues:
            out["issues"] = [i.get("what", "") for i in v.issues[:8]]
        # structural evidence the agent must see — a fem/optimize job's
        # verdict lives on the version, not the job row; without these
        # blocks fem_check/optimize_version returned a bare status
        if job.type == JobType.fem and (v.fem or {}).get("cases"):
            out["fem"] = _compact_fem(v.fem or {})
        if job.type in (JobType.optimize, JobType.resample, JobType.refine):
            top = (v.mesh or {}).get("topopt") or {}
            if top:
                out["topopt"] = _compact_topopt(top)
    if job.status == JobStatus.waiting_for_input:
        out["questions"] = job.questions or []
        out["note"] = (
            "job needs clarification — relay these questions to the user, then "
            "call answer_questions(job_id, answers) with their replies"
        )
    elif job.status == JobStatus.failed:
        out["note"] = "job failed — explain the error and suggest a fix or retry"
    return out


async def wait_job_terminal(
    job_id: str | uuid.UUID,
    *,
    cancel_check: Callable[[], Awaitable[bool]] | None = None,
    interrupt_check: Callable[[], Awaitable[bool]] | None = None,
    timeout_s: float = 3600.0,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    """Poll a spawned job until terminal (or cancelled/interrupted).

    The caller (a chat turn) stays blocked while the UI renders the job's own
    live card. `interrupt_check` (a new pending user message) returns early
    with status "running" — the job continues in the background and can be
    re-awaited via the `wait_job` tool.
    """
    jid = _u(job_id)
    t0 = time.monotonic()
    while True:
        async with session_scope() as db:
            job = await db.get(Job, jid)
            v = await db.get(Version, job.version_id) if job else None
            if job is None:
                return {"ok": False, "status": "missing", "error": "job not found"}
            if job.status in TERMINAL_JOB_STATES:
                return _job_summary(job, v)
        if cancel_check is not None and await cancel_check():
            async with session_scope() as db:
                j = await db.get(Job, jid)
                if j is not None and j.status in (JobStatus.queued, JobStatus.running):
                    j.cancel_requested = True
                    await db.commit()
            return {
                "ok": False,
                "status": "cancelled",
                "job_id": str(jid),
                "version_id": str(job.version_id),
                "note": "cancelled by user",
            }
        if interrupt_check is not None and await interrupt_check():
            return {
                "ok": True,
                "status": "running",
                "job_id": str(jid),
                "version_id": str(job.version_id),
                "note": (
                    "user sent a new message — the job keeps running in the "
                    "background; check it later with job_status or wait_job"
                ),
            }
        if time.monotonic() - t0 > timeout_s:
            return {
                "ok": False,
                "status": "timeout",
                "job_id": str(jid),
                "version_id": str(job.version_id),
                "note": f"still running after {timeout_s:.0f}s — use wait_job to re-check",
            }
        await asyncio.sleep(poll_s)
