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


def _silhouette_aspect(image_path: Path) -> float | None:
    """Pixel w/h of the subject silhouette — deterministic proportion evidence
    the vision model does not measure (it passed a tall-arch hero for a
    4:1-flat handle spec because 'a handle IS a tall arch' to it).

    Studio-background assumption: the border median is the backdrop; the
    subject is the largest component above a colour-distance threshold.
    """
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage

        img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        bg = np.median(border, axis=0)
        mask = np.abs(img - bg).sum(axis=2) > 60
        lab, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        subject = lab == int(np.argmax(sizes)) + 1
        ys, xs = np.where(subject)
        ph, pw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        return float(pw / ph) if ph else None
    except Exception:
        return None


def aspect_mismatch(image_path: Path, spec: PartSpec | None) -> str | None:
    """Mismatch text when the silhouette aspect is far off the spec's L:H
    ratio — None when it fits or is unmeasurable. Only enforced for
    clearly non-square specs where the view direction cannot explain the
    gap (3/4 views land between the ratio and its square root)."""
    if spec is None:
        return None
    try:
        lx = float(spec.length_x.value or 0)
        hz = float(spec.height_z.value or 0)
    except (AttributeError, TypeError, ValueError):
        return None
    if lx <= 0 or hz <= 0:
        return None
    expected = lx / hz
    if 0.55 < expected < 1.8:
        return None  # near-square silhouette — view-robust, nothing to check
    measured = _silhouette_aspect(image_path)
    if measured is None:
        return None
    if expected >= 1.8 and measured < expected * 0.5:
        return (
            f"the declared proportions: the photo shows a much TALLER shape "
            f"(pixel aspect {measured:.2g}) than the spec's {expected:.2g}:1 "
            f"length-to-height ratio — render the part low and elongated"
        )
    if expected <= 0.55 and measured > expected * 2.8:
        return (
            f"the declared proportions: the photo shows a much FLATTER shape "
            f"(pixel aspect {measured:.2g}) than the spec's tall "
            f"{expected:.2g}:1 length-to-height ratio"
        )
    return None


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
        # deterministic proportion gate — the vision model accepts a wrong
        # aspect whenever the shape is plausible for the part category
        aspect_issue = aspect_mismatch(image_path, spec)
        if aspect_issue and verdict.severity in ("ok", "minor"):
            verdict = ImageVerdict(
                match=False,
                missing=[aspect_issue],
                wrong=verdict.wrong,
                severity="major",
            )
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
        # missing alone produced blind retries that repeated the same wrong
        # shape — the corrective suffix must name the defect too (observed:
        # 3x U-channel for an L-bracket spec, each retry only told
        # "MUST show: L-shape" and never "NOT a U-channel")
        fix = []
        if verdict.missing:
            fix.append("The photo MUST clearly show: " + ", ".join(verdict.missing))
        if verdict.wrong:
            fix.append("Correct these errors: " + ", ".join(verdict.wrong))
        prompt += " " + ". ".join(fix) + "."
