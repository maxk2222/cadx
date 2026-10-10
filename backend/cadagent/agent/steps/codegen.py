"""CODEGEN / REPAIR — produce build123d code (code model)."""

from __future__ import annotations

from typing import Any

from cadagent.agent.context import (
    codegen_system,
    error_block,
    plan_block,
    references_block,
    spec_block,
)
from cadagent.agent.types import AgentCtx
from cadagent.llm.schemas import CodeResponse, ModelingPlan, PartSpec

STEP = "codegen"


def _user_parts(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    feedback: str | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": spec_block(spec)}]
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    if ctx.hybrid_solid is not None:
        bbox = ctx.hybrid_mesh_info.get("bbox_size") or "unknown"
        tris = ctx.hybrid_mesh_info.get("triangles") or "?"
        feats = (
            "\nHybrid CAD features to implement: " + "; ".join(plan.hybrid_cad_features)
            if plan and plan.hybrid_cad_features
            else ""
        )
        parts.append(
            {
                "type": "text",
                "text": (
                    "HYBRID MODE: a generated organic mesh is already converted to a B-Rep "
                    f"solid and available via `load_mesh_solid()` (faceted solid, ~{tris} "
                    f"triangles). It is oriented and scaled: rests on the XY plane, centred "
                    f"at origin, bbox LxWxH = {bbox} mm (also in `MESH_BBOX`). "
                    "Use it as the base body and add the exact CAD features via boolean "
                    "ops — the mesh supplies the organic shape, your code supplies the "
                    "precision features." + feats
                ),
            }
        )
    ref = references_block(ctx.assets)
    if ref:
        parts.append({"type": "text", "text": ref})
    if ctx.mode == "edit" and ctx.parent.code:
        parts.append(
            {
                "type": "text",
                "text": (
                    "PREVIOUS VERSION code — modify it minimally; keep PARAMS names "
                    "stable unless the change requires otherwise:\n"
                    f"```python\n{ctx.parent.code}\n```"
                ),
            }
        )
    if feedback:
        parts.append(
            {
                "type": "text",
                "text": ("PREVIOUS ATTEMPT — issues to fix, ALL of them:\n" + feedback),
            }
        )
    return parts


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    iteration: int,
    feedback: str | None = None,
) -> CodeResponse:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": codegen_system()},
        {"role": "user", "content": _user_parts(ctx, spec, plan, feedback)},
    ]
    resp = await ctx.llm.chat("code", messages, json_schema=CodeResponse)
    assert isinstance(resp.parsed, CodeResponse)
    await ctx.trace.record(
        step=STEP,
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "expected_bbox": resp.parsed.expected_bbox,
            "params": [p.model_dump() for p in resp.parsed.params],
            "code": "<code>",
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp.parsed


async def repair(
    ctx: AgentCtx,
    prev: CodeResponse,
    error: str | None,
    traceback: str | None,
    stdout: str | None,
    *,
    iteration: int,
    repair_n: int,
) -> CodeResponse:
    from cadagent.agent.context import system_prompt

    user = (
        "Fix this code. Return the full corrected file as CodeResponse JSON.\n\n"
        "PREVIOUS CODE:\n```python\n"
        + prev.code
        + "\n```\n\n"
        + error_block(error, traceback, stdout)
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt("repair_system.md") + "\n\n" + codegen_system(),
        },
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    resp = await ctx.llm.chat("code", messages, json_schema=CodeResponse, temperature=0.1)
    assert isinstance(resp.parsed, CodeResponse)
    await ctx.trace.record(
        step=f"repair_{repair_n}",
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "expected_bbox": resp.parsed.expected_bbox,
            "params": [p.model_dump() for p in resp.parsed.params],
            "code": "<code>",
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp.parsed
