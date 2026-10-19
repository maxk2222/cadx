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
from typing import Any

from cadx import storage
from cadx.agent.types import AgentCtx, InputAsset
from cadx.llm.schemas import PartSpec
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
    spec: PartSpec | None = None,
    steps: int | None = None,
) -> list[InputAsset]:
    """Generate `count` images via the mesh-server image adapter, persist them
    as generated_image assets and append them to `ctx.assets` so downstream
    steps (plan mesh_source_images, _mesh_source_images) can pick them up.

    When `spec` is given the first generated image is vision-verified against
    it (`image_verify_retries` extra attempts) and only the winning attempt is
    persisted — a rejected image must never land in ctx.assets, it would
    poison mesh conditioning. Raises on generation failure — callers decide
    how to report it.
    """
    from cadx.agent.image_assets import persist_generated_images
    from cadx.agent.image_check import ImageVerdict, generate_with_verification
    from cadx.agent.image_client import ImageGenRequest
    from cadx.db.session import session_scope

    s = ctx.runtime_settings
    side = min(768, s.image_max_side)
    init_image = ref_asset.path if ref_asset is not None else None
    tmp_dirs: list[Path] = []
    last: dict[str, Any] = {}

    async def _gen(p: str) -> list[InputAsset]:
        tmp = Path(tempfile.mkdtemp(prefix="cadx-concept-"))
        tmp_dirs.append(tmp)
        # deterministic per job+prompt so retries stay reproducible; the
        # corrective retry prompt changes the seed naturally
        if seed is None:
            h = hashlib.sha256(f"{ctx.job_id}:{p}".encode()).digest()
            seed_i = int.from_bytes(h[:4]) % 100_000
        else:
            seed_i = seed + (len(tmp_dirs) - 1) * 977
        req = ImageGenRequest(
            prompt=p,
            model=s.image_edit_model if init_image is not None else s.image_model,
            seed=seed_i,
            count=max(1, min(2, count)),
            width=side,
            height=side,
            steps=steps if steps is not None else s.image_steps,
            guidance=s.image_guidance,
            init_image=init_image,
            strength=(
                strength
                if strength is not None
                else (s.image_default_strength if init_image is not None else None)
            ),
            out_dir=tmp,
        )
        result = await ctx.image.generate(req, None, ctx.check_cancel)
        last["result"] = result
        last["strength"] = req.strength
        return [
            InputAsset(asset_id="", kind="generated_image", path=pp, caption="", filename=pp.name)
            for pp in result.image_paths
        ]

    try:
        if spec is not None:
            gen_assets, verdict = await generate_with_verification(
                ctx,
                prompt=prompt,
                spec=spec,
                gen_fn=_gen,
                retries=int(getattr(s, "image_verify_retries", 0) or 0),
            )
        else:
            gen_assets = await _gen(prompt)
            verdict = ImageVerdict(severity="unverifiable")
        result = last["result"]
        async with session_scope() as db:
            assets = persist_generated_images(
                db,
                ctx.project_id,
                [a.path for a in gen_assets],
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
                    "verify": verdict.model_dump(),
                    "attempts": len(tmp_dirs),
                    **(
                        {
                            "reference_asset_id": ref_asset.asset_id,
                            "strength": last.get("strength"),
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
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
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
