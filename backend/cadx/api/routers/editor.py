"""Visual editor endpoints: mesh-ops previews, mesh-edit children, FEM region
preview + standalone FEM check, and GUI editor-state persistence.

Previews enqueue work on the 'cad' queue directly and poll for the result —
they run seconds, not minutes, so a synchronous request is the right UX.
"""

from __future__ import annotations

import asyncio
import mimetypes
import uuid
from typing import Any

from arq.jobs import Job as ArqJob
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cadx import storage
from cadx.api.deps import enqueue, get_pool, get_session
from cadx.api.routers.versions import _next_number
from cadx.api.schemas import (
    EditorStateRequest,
    EnqueueResponse,
    FemPreviewOut,
    FemRegionRequest,
    MeshEditRequest,
    MeshOpIn,
    MeshPreviewOut,
    MeshPreviewRequest,
)
from cadx.cad.mesh_ops import OPS_REGISTRY, MeshPostParams, merge_post_params
from cadx.db.models import Job, JobType, Version, VersionKind, VersionStatus

router = APIRouter(tags=["editor"])

_CAD_QUEUE = "cad"
_SCRATCH_SUFFIXES = {".glb", ".png", ".json"}
_PREVIEW_TIMEOUT_S = 180.0


async def _run_cad(func: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Enqueue on the 'cad' queue and poll until complete (QueueExecutor pattern)."""
    pool = await get_pool()
    job_id = f"{func}-{uuid.uuid4()}"
    await pool.enqueue_job(func, payload, _queue_name=_CAD_QUEUE, _job_id=job_id)
    job = ArqJob(job_id, pool)
    waited = 0.0
    while waited < timeout_s:
        status = await job.status()
        if status == ArqJobStatus.complete:
            try:
                result = await job.result()
            except Exception as e:
                return {"ok": False, "error": f"executor task failed: {e}"}
            return result if isinstance(result, dict) else {"ok": False, "error": "bad result"}
        await asyncio.sleep(1.0)
        waited += 1.0
    return {"ok": False, "error": f"executor task '{func}' timed out after {timeout_s}s"}


def _validate_ops(ops: list[MeshOpIn]) -> list[dict[str, Any]]:
    unknown = [o.op for o in ops if o.op not in OPS_REGISTRY]
    if unknown:
        raise HTTPException(
            422,
            f"unknown mesh op(s): {', '.join(sorted(set(unknown)))} — "
            f"available: {', '.join(sorted(OPS_REGISTRY))}",
        )
    return [o.model_dump() for o in ops]


def _edit_source(v: Version) -> tuple[str, bool]:
    """(mesh_rel, ops_only) for preview/edit on a version.

    mesh versions replay baseline+ops on the raw generator mesh; every other
    mode applies ops directly to the finished STL (already in the mm frame).
    """
    if (v.resolved_mode or "") == "mesh":
        raw = storage.version_dir(v.project_id, v.id) / "mesh" / "mesh_raw.glb"
        if raw.exists():
            return storage.relpath(raw), False
    rel = storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1)
    if rel is None:
        raise HTTPException(409, "version has no mesh artifact to edit")
    return rel, True


def _scratch_url(version_id: uuid.UUID, relpath: str) -> str:
    """Map a storage relpath (.../_scratch/<sub>) to the scratch route URL."""
    marker = "_scratch/"
    sub = relpath.split(marker, 1)[-1] if marker in relpath else relpath.rsplit("/", 1)[-1]
    return f"/api/versions/{version_id}/scratch/{sub}"


@router.post("/versions/{version_id}/mesh-preview", response_model=MeshPreviewOut)
async def mesh_preview(
    version_id: uuid.UUID, body: MeshPreviewRequest, db: AsyncSession = Depends(get_session)
):
    """Apply the draft ops stack and return a preview GLB (no version created)."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    ops = _validate_ops(body.ops)
    mesh_rel, ops_only = _edit_source(v)

    if ops_only:
        post = MeshPostParams(ops=ops)
    else:
        post = merge_post_params(MeshPostParams(**(v.params or {})), body.post or {})
        post = post.model_copy(update={"ops": ops})

    result = await _run_cad(
        "process_mesh",
        {
            "job_id": f"preview-{uuid.uuid4()}",
            "project_id": str(v.project_id),
            "version_id": str(v.id),
            "iteration": 1,
            "mode": "ops_preview",
            "mesh_rel": mesh_rel,
            "post": post.model_dump(),
            "ops": ops,
            "ops_only": ops_only,
            "spec": v.spec,
        },
        _PREVIEW_TIMEOUT_S,
    )
    glb = result.get("glb")
    return MeshPreviewOut(
        ok=bool(result.get("ok")),
        glb_url=_scratch_url(v.id, glb) if glb else None,
        stats=result.get("stats") or {},
        error=result.get("error"),
    )


@router.get("/versions/{version_id}/scratch/{path:path}")
async def scratch_file(version_id: uuid.UUID, path: str, db: AsyncSession = Depends(get_session)):
    """Serve transient GUI preview files from versions/<vid>/_scratch/."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    base = storage.scratch_dir(v.project_id, v.id).resolve()
    p = (base / path).resolve()
    if not str(p).startswith(str(base)) or not p.is_file():
        raise HTTPException(404, "not found")
    if p.suffix.lower() not in _SCRATCH_SUFFIXES:
        raise HTTPException(403, "file type not allowed")
    media = (
        "model/gltf-binary"
        if p.suffix == ".glb"
        else mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    )
    return FileResponse(p, media_type=media, headers={"Cache-Control": "no-store"})


@router.post("/versions/{version_id}/mesh-edit", response_model=EnqueueResponse)
async def mesh_edit(
    version_id: uuid.UUID, body: MeshEditRequest, db: AsyncSession = Depends(get_session)
):
    """Apply the ops stack for real — creates a tessellated child version."""
    parent = await db.get(Version, version_id)
    if not parent:
        raise HTTPException(404, "version not found")
    if parent.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot edit version in status {parent.status}")
    ops = _validate_ops(body.ops)
    _edit_source(parent)  # 409 when no mesh artifact exists

    v = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=await _next_number(db, parent.project_id),
        kind=VersionKind.mesh_edit,
        prompt=(body.note or f"visual edit of v{parent.number}")[:500],
        gen_mode=parent.gen_mode,
        options=parent.options or {},
    )
    db.add(v)
    await db.flush()
    job = Job(version_id=v.id, type=JobType.mesh_edit)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_mesh_edit_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "ops": ops,
            "post": body.post or {},
        },
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.post("/versions/{version_id}/fem-preview", response_model=FemPreviewOut)
async def fem_preview(
    version_id: uuid.UUID, body: FemRegionRequest, db: AsyncSession = Depends(get_session)
):
    """Resolve declared regions on the voxel domain — returns per-case hit
    counts, warnings and overlay PNGs (fixed=blue load=red keep=green)."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    mesh_rel = storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1)
    if mesh_rel is None:
        raise HTTPException(409, "version has no mesh artifact to analyse")
    if not body.load_cases:
        raise HTTPException(422, "at least one load case is required")

    result = await _run_cad(
        "fem_analyze",
        {
            "job_id": f"fem-preview-{uuid.uuid4()}",
            "project_id": str(v.project_id),
            "version_id": str(v.id),
            "iteration": v.iterations or 1,
            "mesh_rel": mesh_rel,
            "load_cases": body.load_cases,
            "keep_regions": body.keep_regions,
            "anchors": (v.analysis or {}).get("anchors", {}).get("holes"),
            "preview_only": True,
            "scratch": True,
        },
        _PREVIEW_TIMEOUT_S,
    )
    prev = result.get("preview") or {}
    pngs = [_scratch_url(v.id, p) for p in prev.get("regions_png") or []]
    return FemPreviewOut(
        ok=bool(result.get("ok")),
        pitch_mm=prev.get("pitch_mm"),
        dims=prev.get("dims") or [],
        solid_elements=prev.get("solid_elements") or 0,
        cases=prev.get("cases") or [],
        pngs=pngs,
        error=result.get("error"),
    )


@router.post("/versions/{version_id}/fem", response_model=EnqueueResponse)
async def fem_check(
    version_id: uuid.UUID, body: FemRegionRequest, db: AsyncSession = Depends(get_session)
):
    """Standalone FEM run on the SAME version — writes version.fem + stress
    renders, flags needs_review when the worst safety factor drops below 1."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    if v.status not in (VersionStatus.succeeded, VersionStatus.needs_review):
        raise HTTPException(409, f"cannot analyse version in status {v.status}")
    if storage.best_mesh_relpath(v.project_id, v.id, v.iterations or 1) is None:
        raise HTTPException(409, "version has no mesh artifact to analyse")
    if not body.load_cases:
        raise HTTPException(422, "at least one load case is required")

    job = Job(version_id=v.id, type=JobType.fem)
    db.add(job)
    await db.commit()
    await enqueue(
        "run_fem_job",
        {
            "job_id": str(job.id),
            "version_id": str(v.id),
            "load_cases": body.load_cases,
            "keep_regions": body.keep_regions,
            "material": body.material,
        },
    )
    return EnqueueResponse(version_id=v.id, job_id=job.id)


@router.patch("/versions/{version_id}/editor-state")
async def editor_state(
    version_id: uuid.UUID, body: EditorStateRequest, db: AsyncSession = Depends(get_session)
):
    """Persist GUI editor state (painted regions/ops draft) on the version —
    stored under options['ui_editor']; 'ui_' keys never reach run settings."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    v.options = {**(v.options or {}), "ui_editor": body.ui_editor}
    await db.commit()
    return {"ok": True}
