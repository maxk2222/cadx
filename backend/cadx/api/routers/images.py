"""Local text-to-image generation via the mesh-server sidecar.

POST /projects/{id}/images generates images synchronously (FLUX-4-step is
seconds; nginx proxy_read_timeout covers the worst case), persists them as
kind=generated_image assets with provenance, and returns the created assets.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.agent.image_assets import persist_generated_images
from cadx.agent.image_client import ImageGenError, ImageGenRequest, make_image_client
from cadx.api.deps import get_session
from cadx.api.routers.assets import _out
from cadx.api.schemas import AssetOut, ImageGenerateIn, MeshModelOut
from cadx.db.models import Project
from cadx.logging import get_logger
from cadx.runtime_config import load_runtime_settings

log = get_logger(__name__)

router = APIRouter(tags=["images"])


@router.get("/images/models", response_model=list[MeshModelOut])
async def image_models(db: AsyncSession = Depends(get_session)):
    settings = await load_runtime_settings(db)
    client = make_image_client(runtime_settings=settings)
    if client is None:
        return []
    return [MeshModelOut(**m.model_dump()) for m in await client.models()]


@router.post("/projects/{project_id}/images", response_model=list[AssetOut], status_code=201)
async def generate_images(
    project_id: uuid.UUID,
    body: ImageGenerateIn,
    db: AsyncSession = Depends(get_session),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    settings = await load_runtime_settings(db)
    client = make_image_client(runtime_settings=settings)
    if client is None:
        raise HTTPException(503, "image generation not configured (MESH_BASE_URL)")

    # health() can be ok with zero image adapters — check models() explicitly
    models = [m for m in await client.models() if m.available and not m.last_error]
    if not models:
        raise HTTPException(503, "no image adapter available on the mesh server")

    # "auto" (or absent) means the runtime default — an explicit adapter id
    # overrides it; settings.image_model itself defaults to "auto".
    model = body.model if body.model and body.model != "auto" else settings.image_model
    tmp_dir = tempfile.mkdtemp(prefix="cadx-imggen-")
    req = ImageGenRequest(
        prompt=body.prompt,
        negative=body.negative,
        model=model,
        seed=body.seed,
        count=min(body.count, settings.image_max_count),
        width=min(body.width, settings.image_max_side),
        height=min(body.height, settings.image_max_side),
        steps=body.steps if body.steps is not None else settings.image_steps,
        guidance=body.guidance if body.guidance is not None else settings.image_guidance,
        out_dir=Path(tmp_dir),
    )
    try:
        result = await client.generate(req, on_progress=None)
    except ImageGenError as e:
        raise HTTPException(502, str(e)) from e
    except Exception as e:
        log.warning("image_gen.failed", error=str(e))
        raise HTTPException(502, f"image generation failed: {e}") from e

    try:
        # copy bytes into the asset dir BEFORE the temp dir is removed
        assets = persist_generated_images(
            db,
            project_id,
            result.image_paths,
            prompt=body.prompt,
            caption=body.caption,
            gen_params={
                "prompt": body.prompt,
                "negative": body.negative,
                "model": result.model,
                "backend": result.backend,
                "seed": result.seed,
                "steps": result.steps,
                "guidance": result.guidance,
                "width": req.width,
                "height": req.height,
                "duration_s": result.duration_s,
                "requested_model": model,
            },
        )
        await db.commit()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info(
        "image_gen.done",
        project=str(project_id),
        model=result.model,
        n=len(assets),
        duration_s=result.duration_s,
    )
    return [_out(a) for a in assets]
