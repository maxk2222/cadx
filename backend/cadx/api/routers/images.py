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

from cadx import storage
from cadx.agent.image_assets import persist_generated_images
from cadx.agent.image_client import ImageGenError, ImageGenRequest, make_image_client
from cadx.api.deps import get_session
from cadx.api.routers.assets import _out
from cadx.api.schemas import AssetOut, ImageGenerateIn, MeshModelOut
from cadx.db.models import IMAGE_ASSET_KINDS, Asset, Project
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

    # health() can be ok with zero image adapters — check models() explicitly.
    # Edit requests (reference image) can only run on supports_edit adapters.
    init_image: Path | None = None
    ref_asset: Asset | None = None
    if body.reference_asset_id is not None:
        ref_asset = await db.get(Asset, body.reference_asset_id)
        if ref_asset is None or ref_asset.project_id != project_id:
            raise HTTPException(404, "reference asset not found")
        if ref_asset.kind not in IMAGE_ASSET_KINDS:
            raise HTTPException(422, "reference asset must be an image")
        ref_path = storage.resolve(ref_asset.path)
        if not ref_path.exists():
            raise HTTPException(422, "reference asset file missing")
        init_image = ref_path

    edit = init_image is not None
    models = [
        m
        for m in await client.models()
        if m.available and not m.last_error and (m.supports_edit or not edit)
    ]
    if not models:
        raise HTTPException(
            503,
            "no edit-capable image adapter on the mesh server"
            if edit
            else "no image adapter available on the mesh server",
        )

    # "auto" (or absent) means the runtime default — an explicit adapter id
    # overrides it; edit jobs fall back to image_edit_model instead.
    default_model = settings.image_edit_model if edit else settings.image_model
    model = body.model if body.model and body.model != "auto" else default_model
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
        init_image=init_image,
        strength=body.strength
        if body.strength is not None
        else (settings.image_default_strength if edit else None),
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
            derived_from=ref_asset.id if ref_asset is not None else None,
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
                **(
                    {
                        "reference_asset_id": str(ref_asset.id),
                        "strength": req.strength,
                        "edit": True,
                    }
                    if ref_asset is not None
                    else {}
                ),
                **({"role_hint": "sketch"} if body.mark_sketch else {}),
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
