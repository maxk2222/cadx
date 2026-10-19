"""Vision verification for synthesised reference images (hero/concept).

The generated photo is the ONLY conditioning the image-to-3D pipeline gets —
a mismatched image produces a wrong part. `verify_concept_image` compares one
image against the part spec via the vision model; `generate_with_verification`
retries the generator with a corrective prompt suffix when the verdict is
`major`. Every failure degrades to `severity="unverifiable"` (accept + warn)
so a broken vision path can never block generation.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cadx.agent.context import system_prompt
from cadx.agent.image_prompt import visual_features
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx, InputAsset
from cadx.llm.images import image_part
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger

log = get_logger(__name__)


class ImageVerdict(BaseModel):
    match: bool = True
    missing: list[str] = Field(default_factory=list)
    wrong: list[str] = Field(default_factory=list)
    severity: Literal["ok", "minor", "major", "unverifiable"] = "ok"


def _dim_text(dim: Any) -> str:
    unit = getattr(dim, "unit", "") or "mm"
    return f"{getattr(dim, 'value', 0):g} {unit}"


def _spec_summary(ctx: AgentCtx, spec: PartSpec | None) -> str:
    """Text block injected into image_verify.md — name, overall dims and the
    visual feature list. Without a spec the raw user prompt is the fallback."""
    if spec is None:
        return ctx.prompt[:200]
    lines = [
        f"Name: {spec.name}",
        "Dimensions (L x W x H): "
        + " x ".join(_dim_text(d) for d in (spec.length_x, spec.width_y, spec.height_z)),
    ]
    if spec.summary:
        lines.append(f"Summary: {spec.summary}")
    feats = visual_features(spec)
    if feats:
        lines.append(f"Visual features: {feats}")
    return "\n".join(lines)


async def verify_concept_image(
    ctx: AgentCtx, image_path: Path, spec: PartSpec | None
) -> ImageVerdict:
    """Vision call: does the generated image match the spec?

    Any failure (exception, missing .parsed, unparseable payload) degrades to
    `severity="unverifiable"` — the image is accepted rather than blocking
    the run on a flaky verifier.
    """
    try:
        summary = _spec_summary(ctx, spec)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt("image_verify.md").format(spec_summary=summary),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Does this generated product photo match the part specification?",
                    },
                    image_part(image_path),
                ],
            },
        ]
        resp = await ctx.llm.chat("vision", messages, json_schema=ImageVerdict, temperature=0.0)
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            verdict = ImageVerdict(match=True, severity="unverifiable")
        elif isinstance(parsed, ImageVerdict):
            verdict = parsed
        else:
            verdict = ImageVerdict.model_validate(parsed)
    except Exception as e:
        log.warning("image_check.failed", error=str(e)[:200])
        verdict = ImageVerdict(match=True, severity="unverifiable")
    log.info(
        "image_check.verdict",
        severity=verdict.severity,
        missing=verdict.missing,
        wrong=verdict.wrong,
    )
    return verdict


GenFn = Callable[[str], Awaitable[list[InputAsset]] | list[InputAsset]]


async def generate_with_verification(
    ctx: AgentCtx,
    *,
    prompt: str,
    spec: PartSpec | None,
    gen_fn: GenFn,
    retries: int,
) -> tuple[list[InputAsset], ImageVerdict]:
    """Call gen_fn(prompt) and verify the first returned image.

    `major` verdict → append "The photo MUST clearly show: <missing>." to the
    prompt and re-generate, up to `retries` extra attempts (total calls =
    retries + 1). `ok`/`minor`/`unverifiable` are accepted immediately —
    minor deviations are stylistic, unverifiable must not block the run.
    Returns (last assets, last verdict).
    """
    attempt = 0
    while True:
        out = gen_fn(prompt)
        assets = await out if inspect.isawaitable(out) else out
        target = next(
            (a for a in assets if a.kind in IMAGE_INPUT_KINDS),
            assets[0] if assets else None,
        )
        verdict = (
            await verify_concept_image(ctx, target.path, spec)
            if target is not None
            else ImageVerdict(match=True, severity="unverifiable")
        )
        if verdict.severity != "major" or attempt >= retries:
            return assets, verdict
        attempt += 1
        prompt += " The photo MUST clearly show: " + ", ".join(verdict.missing) + "."
