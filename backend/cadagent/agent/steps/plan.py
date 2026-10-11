"""PLAN — spec -> ordered modeling plan + generation approach (code model)."""

from __future__ import annotations

from typing import Any

from cadagent.agent.context import (
    catalog_listing_block,
    references_block,
    spec_block,
    system_prompt,
)
from cadagent.agent.toolloop import run_session, tools_enabled
from cadagent.agent.toolspecs import ToolSession
from cadagent.agent.types import AgentCtx
from cadagent.llm.schemas import ModelingPlan, PartSpec

STEP = "plan"


async def run(ctx: AgentCtx, spec: PartSpec) -> ModelingPlan:
    ref = references_block(ctx.assets)
    forced = ctx.gen_mode if ctx.gen_mode in ("cad", "mesh", "hybrid") else "auto"
    images_n = sum(1 for a in ctx.assets if a.kind == "image")
    mode_ctx = (
        f"GENERATION CONTEXT: requested mode = {forced}"
        f" | mesh server available = {ctx.mesh_available}"
        f" | reference images = {images_n}"
    )
    if forced != "auto":
        mode_ctx += (
            "\nFORCED MODE: the user explicitly selected this generation mode — "
            f'set `approach` to "{forced}".'
        )
    if not ctx.mesh_available:
        mode_ctx += '\nNOTE: no mesh server available — `approach` must be "cad".'
    if images_n == 0:
        mode_ctx += '\nNOTE: no reference images — `approach` must be "cad".'

    user = spec_block(spec) + ("\n\n" + ref if ref else "") + "\n\n" + mode_ctx
    listing = catalog_listing_block()
    if listing:
        user += "\n\n" + listing
    use_tools = tools_enabled(ctx, "code")
    system = system_prompt("plan_system.md")
    if use_tools:
        system += "\n\n" + system_prompt("tools_info.md")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="info")
        res = await run_session(
            ctx,
            "code",
            messages,
            session=session,
            json_schema=ModelingPlan,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, ModelingPlan):
            from cadagent.llm.structured import LLMFormatError

            raise LLMFormatError("plan session produced no plan")
        ctx.plan = res.response.parsed
        return res.response.parsed
    resp = await ctx.llm.chat("code", messages, json_schema=ModelingPlan, temperature=0.0)
    assert isinstance(resp.parsed, ModelingPlan)
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
    ctx.plan = resp.parsed
    return resp.parsed
