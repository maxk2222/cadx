"""PLAN — spec -> ordered modeling plan + generation approach (code model)."""

from __future__ import annotations

from typing import Any

from cadx.agent.context import (
    catalog_listing_block,
    measurements_block,
    references_block,
    request_block,
    spec_block,
    system_prompt,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from cadx.llm.schemas import ModelingPlan, PartSpec

STEP = "plan"


def _resolve_plan_mode(ctx: AgentCtx) -> str:
    """`agent_plan_mode`: "full" (catalog-aware tool session, default),
    "lite" (one short no-tools call — steps+params only, approach=cad),
    "auto" (lite for text-only cad jobs where the planner has no real
    choice to make anyway)."""
    mode = ctx.runtime_settings.agent_plan_mode
    if mode != "auto":
        return mode
    if any(a.kind in IMAGE_INPUT_KINDS for a in ctx.assets):
        return "full"  # image-driven jobs need the full planner's routing
    if ctx.gen_mode == "cad":
        return "lite"
    if ctx.gen_mode == "auto" and not ctx.mesh_available and not ctx.image_available:
        return "lite"  # nothing but cad is reachable anyway
    return "full"


async def _run_lite(ctx: AgentCtx, spec: PartSpec, forced: str) -> ModelingPlan:
    """Cheap plan for a simple, image-free CAD job: no tools, no catalog
    listing — the planner's only real output for these parts is the ordered
    step list and the PARAMS table."""
    user = (
        (request_block(ctx.prompt) + "\n\n" if request_block(ctx.prompt) else "")
        + spec_block(spec)
        + f"\n\nGENERATION CONTEXT: requested mode = {forced}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("plan_lite_system.md")},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    resp = await ctx.llm.chat(
        "code",
        messages,
        json_schema=ModelingPlan,
        temperature=0.0,
        extra_body=thinking_body(ctx, STEP),
    )
    assert isinstance(resp.parsed, ModelingPlan)
    await ctx.trace.record(
        step=f"{STEP}_lite",
        iteration=0,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=resp.parsed.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    plan = resp.parsed
    plan.approach = "cad"
    plan.strategy = ["cad"]
    plan.approach_reason = (plan.approach_reason or "").rstrip(".") + " (plan-lite)"
    ctx.plan = plan
    return plan


async def run(ctx: AgentCtx, spec: PartSpec) -> ModelingPlan:
    ref = references_block(ctx.assets)
    forced = ctx.gen_mode if ctx.gen_mode in ("cad", "mesh", "hybrid") else "auto"
    if _resolve_plan_mode(ctx) == "lite":
        return await _run_lite(ctx, spec, forced)
    images_n = sum(1 for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS)
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
        if getattr(ctx, "image_available", False):
            mode_ctx += (
                "\nNOTE: no reference images attached — for organic/sculptural parts "
                "you may still choose mesh/hybrid: a concept reference image is "
                "generated automatically, or call `generate_concept_image` first to "
                "control it yourself (its asset_id goes into `mesh_source_images`)."
            )
        else:
            mode_ctx += '\nNOTE: no reference images — `approach` must be "cad".'

    user = (
        (request_block(ctx.prompt) + "\n\n" if request_block(ctx.prompt) else "")
        + spec_block(spec)
        + ("\n\n" + ref if ref else "")
        + "\n\n"
        + mode_ctx
    )
    mm = measurements_block(ctx.measurements)
    if mm:
        user += "\n\n" + mm
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
            from cadx.llm.structured import LLMFormatError

            raise LLMFormatError("plan session produced no plan")
        plan = res.response.parsed
        _finalize(plan, forced)
        ctx.plan = plan
        return plan
    resp = await ctx.llm.chat(
        "code",
        messages,
        json_schema=ModelingPlan,
        temperature=0.0,
        extra_body=thinking_body(ctx, STEP),
    )
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
    plan = resp.parsed
    _finalize(plan, forced)
    ctx.plan = plan
    return plan


def _finalize(plan: ModelingPlan, forced: str) -> None:
    """Fill strategy defaults and keep `approach` = first strategy entry."""
    if forced != "auto":
        plan.strategy = [forced]  # type: ignore[list-item]
    elif not plan.strategy:
        plan.strategy = [plan.approach]
    plan.approach = plan.strategy[0]


# plan-step wording that only makes sense for mesh/hybrid generation — a CAD
# fallback must never see them (the codegen prompt echoes `steps` verbatim and
# the model then calls load_mesh_solid() against a nonexistent base)
_FALLBACK_DROP = ("mesh", "load_mesh_solid", "image-to-3d", "hybrid", "organisch")


def cad_fallback_plan(plan: ModelingPlan | None) -> ModelingPlan | None:
    """Strip mesh/hybrid instructions so a CAD fallback never codes against
    load_mesh_solid().

    Deterministic rewrite (no LLM call): the original plan object is left
    untouched — a deep copy is re-approached as cad, mesh-pipeline steps are
    dropped, `hybrid_cad_features` are appended as regular steps, and the
    mesh-routing fields are cleared.
    """
    if plan is None:
        return None
    p = plan.model_copy(deep=True)
    p.approach = "cad"
    p.strategy = ["cad"]
    p.steps = [
        "0. CAD fallback: build every body parametrically from the spec features (no mesh)",
        *[s for s in p.steps if not any(k in str(s).lower() for k in _FALLBACK_DROP)],
        *[str(f) for f in p.hybrid_cad_features],
    ]
    for f in ("mesh_subject", "mesh_source_images", "mesh_up_hint"):
        if f in type(p).model_fields:
            setattr(p, f, [] if f == "mesh_source_images" else "")
    p.approach_reason = (p.approach_reason + " (cad fallback)").strip()
    return p
