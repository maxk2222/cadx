"""CRITIQUE — compare generated model with input images + spec (vision model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadagent.agent.context import (
    dfm_block,
    images_block,
    parent_block,
    renders_block,
    spec_block,
    system_prompt,
)
from cadagent.agent.types import AgentCtx
from cadagent.llm.schemas import Critique, PartSpec

STEP = "critique"


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dim_issues: list[str],
    dfm: dict[str, Any] | None,
    *,
    iteration: int,
) -> Critique:
    numeric = (
        "Numeric dimension check: " + "; ".join(dim_issues)
        if dim_issues
        else "Numeric dimension check: all stated dimensions match within tolerance."
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": "USER INPUT IMAGES (for reference):"}]
    parts += images_block(ctx.assets)
    if ctx.mode == "edit" and ctx.parent.contact_sheet is not None:
        parts += parent_block(None, None, ctx.parent.contact_sheet)
        parts.append(
            {
                "type": "text",
                "text": "Only the requested changes should differ from the previous version.",
            }
        )
    if ctx.resolved_mode in ("mesh", "hybrid"):
        parts.append(
            {
                "type": "text",
                "text": (
                    f"GENERATION MODE: {ctx.resolved_mode} — the shape comes from a "
                    "generative mesh model. Judge silhouette, proportions and the requested "
                    "features; ignore faceting/shading artifacts inherent to triangle meshes."
                ),
            }
        )
    parts.append({"type": "text", "text": spec_block(spec)})
    parts += renders_block(contact_sheet, analysis)
    parts.append({"type": "text", "text": numeric})
    dfm_txt = dfm_block(dfm)
    if dfm_txt:
        parts.append({"type": "text", "text": dfm_txt})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("critique_system.md")},
        {"role": "user", "content": parts},
    ]
    resp = await ctx.llm.chat("vision", messages, json_schema=Critique, temperature=0.0)
    assert isinstance(resp.parsed, Critique)
    await ctx.trace.record(
        step=STEP,
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=resp.parsed.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp.parsed
