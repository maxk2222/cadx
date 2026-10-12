"""INTAKE — extract a PartSpec from images + text (vision model)."""

from __future__ import annotations

from typing import Any

from cadx.agent.context import (
    images_block,
    parent_block,
    references_block,
    system_prompt,
    user_block,
)
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import PartSpec

STEP = "intake"


async def run(ctx: AgentCtx) -> PartSpec:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_block(ctx.prompt, ctx.settings, ctx.mode)}
    ]
    ref = references_block(ctx.assets)
    if ref:
        user_content.append({"type": "text", "text": ref})
    if ctx.mode == "edit" and (ctx.parent.spec or ctx.parent.code):
        user_content += parent_block(ctx.parent.code, ctx.parent.spec, ctx.parent.contact_sheet)
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)

    use_tools = tools_enabled(ctx, "vision")
    system = system_prompt("intake_system.md")
    if use_tools:
        system += "\n\n" + system_prompt("tools_info.md")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="info")
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=PartSpec,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, PartSpec):
            from cadx.llm.structured import LLMFormatError

            raise LLMFormatError("intake session produced no spec")
        spec = res.response.parsed
        if not spec.process:
            spec.process = ctx.settings.process
        ctx.spec = spec
        return spec
    resp = await ctx.llm.chat("vision", messages, json_schema=PartSpec, temperature=0.0)
    assert isinstance(resp.parsed, PartSpec)
    await ctx.trace.record(
        step=STEP,
        iteration=0,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=resp.parsed.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    spec = resp.parsed
    if not spec.process:
        spec.process = ctx.settings.process
    ctx.spec = spec
    return spec
