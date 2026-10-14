"""Shared concept-image generation: mesh-server image call + asset persistence.

Used by the `generate_concept_image` tool (agent-driven) and the orchestrator's
deterministic bootstrap for mesh/hybrid runs that have no input images at all —
image->3D generators need at least one reference, so a product-photo-style
concept image is synthesised from the spec when none was supplied.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path

from cadx import storage
from cadx.agent.types import AgentCtx, InputAsset
from cadx.logging import get_logger

log = get_logger(__name__)


async def generate_concept_images(
    ctx: AgentCtx,
    prompt: str,
    *,
    count: int = 1,
    seed: int | None = None,
    ref_asset: InputAsset | None = None,
    strength: float | None = None,
    mark_sketch: bool = False,
    source: str = "concept_tool",
) -> list[InputAsset]:
    """Generate `count` images via the mesh-server image adapter, persist them
    as generated_image assets and append them to `ctx.assets` so downstream
    steps (plan mesh_source_images, _mesh_source_images) can pick them up.

    Raises on generation failure — callers decide how to report it.
    """
    from cadx.agent.image_assets import persist_generated_images
    from cadx.agent.image_client import ImageGenRequest
    from cadx.db.session import session_scope

    s = ctx.runtime_settings
    tmp = Path(tempfile.mkdtemp(prefix="cadx-concept-"))
    side = min(768, s.image_max_side)
    if seed is None:
        # deterministic per job+prompt so retries stay reproducible
        h = hashlib.sha256(f"{ctx.job_id}:{prompt}".encode()).digest()
        seed = int.from_bytes(h[:4]) % 100_000
    init_image = ref_asset.path if ref_asset is not None else None
    req = ImageGenRequest(
        prompt=prompt,
        model=s.image_edit_model if init_image is not None else s.image_model,
        seed=int(seed),
        count=max(1, min(2, count)),
        width=side,
        height=side,
        steps=s.image_steps,
        guidance=s.image_guidance,
        init_image=init_image,
        strength=(
            strength
            if strength is not None
            else (s.image_default_strength if init_image is not None else None)
        ),
        out_dir=tmp,
    )
    try:
        result = await ctx.image.generate(req, None, ctx.check_cancel)
        async with session_scope() as db:
            assets = persist_generated_images(
                db,
                ctx.project_id,
                result.image_paths,
                prompt=prompt,
                caption=f"concept — {prompt[:120]}",
                derived_from=uuid.UUID(ref_asset.asset_id) if ref_asset is not None else None,
                gen_params={
                    "prompt": prompt,
                    "model": result.model,
                    "backend": result.backend,
                    "seed": result.seed,
                    "steps": result.steps,
                    "guidance": result.guidance,
                    "width": side,
                    "height": side,
                    "duration_s": result.duration_s,
                    "source": source,
                    "job_id": ctx.job_id,
                    **(
                        {
                            "reference_asset_id": ref_asset.asset_id,
                            "strength": req.strength,
                            "edit": True,
                        }
                        if ref_asset is not None
                        else {}
                    ),
                    **({"role_hint": "sketch"} if mark_sketch else {}),
                },
            )
            await db.commit()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out: list[InputAsset] = []
    for asset in assets:
        a = InputAsset(
            asset_id=str(asset.id),
            kind="generated_image",
            path=storage.resolve(asset.path),
            caption=asset.caption,
            filename=asset.filename,
            role_hint="sketch" if mark_sketch else "",
        )
        ctx.assets.append(a)
        ctx.generated_asset_ids.append(str(asset.id))
        out.append(a)
    return out
