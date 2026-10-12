"""CODEGEN / REPAIR — produce build123d code (code model)."""

from __future__ import annotations

from typing import Any

from cadx.agent.context import (
    anchors_block,
    catalog_block,
    codegen_system,
    library_block,
    measurements_block,
    numbered_code_block,
    plan_block,
    references_block,
    spec_block,
)
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession, VerifyFn
from cadx.agent.types import AgentCtx
from cadx.cad.errordiag import diag_block, diagnose
from cadx.llm.schemas import (
    CodeResponse,
    ModelingPlan,
    PartSpec,
    PatchResponse,
)

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
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
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
    lib = library_block(ctx.library_templates)
    if lib:
        parts.append({"type": "text", "text": lib})
    mm = measurements_block(ctx.measurements)
    if mm:
        parts.append({"type": "text", "text": mm})
    ab = anchors_block(ctx.anchors)
    if ab:
        parts.append({"type": "text", "text": ab})
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
    if tools_enabled(ctx, "code"):
        return await _codegen_session(ctx, spec, plan, iteration=iteration, feedback=feedback)
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
    prev_code: str,
    spec: PartSpec,
    plan: ModelingPlan | None,
    error: str | None,
    traceback: str | None,
    stdout: str | None,
    *,
    iteration: int,
    repair_n: int,
    verify_fn: VerifyFn | None = None,
) -> PatchResponse:
    """Ask the model to fix broken code — tool session when the provider
    supports it, else one-shot PatchResponse (edits preferred, full_code
    fallback)."""
    from cadx.agent.context import system_prompt

    diag = diagnose(error, traceback, prev_code)
    if tools_enabled(ctx, "code"):
        return await _repair_session(
            ctx,
            prev_code,
            spec,
            plan,
            error,
            traceback,
            stdout,
            diag=diag,
            iteration=iteration,
            repair_n=repair_n,
            verify_fn=verify_fn,
            max_calls=ctx.runtime_settings.agent_tool_max_calls,
        )
    user = (
        "Fix this code. Return a PatchResponse JSON object — prefer `edits` for "
        "localised fixes, `full_code` only when the fix is structural.\n\n"
        "PREVIOUS CODE:\n```python\n"
        + prev_code
        + "\n```\n\n"
        + diag_block(diag, error, traceback, stdout)
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
    if spec is not None:
        parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt("repair_system.md") + "\n\n" + codegen_system(),
        },
        {"role": "user", "content": parts},
    ]
    resp = await ctx.llm.chat("code", messages, json_schema=PatchResponse, temperature=0.1)
    assert isinstance(resp.parsed, PatchResponse)
    await ctx.trace.record(
        step=f"repair_{repair_n}",
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "diag": diag.category,
            "n_edits": len(resp.parsed.edits),
            "has_full_code": bool(resp.parsed.full_code),
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp.parsed


async def _repair_session(
    ctx: AgentCtx,
    prev_code: str,
    spec: PartSpec | None,
    plan: ModelingPlan | None,
    error: str | None,
    traceback: str | None,
    stdout: str | None,
    *,
    diag: Any,
    iteration: int,
    repair_n: int,
    verify_fn: VerifyFn | None,
    max_calls: int,
) -> PatchResponse:
    from cadx.agent.context import system_prompt

    session = ToolSession(ctx, kind="edit", code=prev_code, verify_fn=verify_fn)
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "Fix this code using the tools — minimal edit only."},
        {"type": "text", "text": numbered_code_block(prev_code, "PREVIOUS CODE")},
        {"type": "text", "text": diag_block(diag, error, traceback, stdout)},
    ]
    if spec is not None:
        parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    cat = catalog_block(plan, spec)
    if cat:
        parts.append({"type": "text", "text": cat})
    system = "\n\n".join(
        [
            system_prompt("repair_tools.md"),
            system_prompt("tools_protocol.md"),
            codegen_system(),
        ]
    )
    res = await run_session(
        ctx,
        "code",
        [
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ],
        session=session,
        max_calls=max_calls,
        step=f"repair_{repair_n}",
        iteration=iteration,
        temperature=0.1,
    )
    if res.code and res.code != prev_code:
        sub = res.submitted or {}
        return PatchResponse(
            reasoning=str(sub.get("reasoning", "")),
            expected_bbox=list(sub.get("expected_bbox") or []),
            full_code=res.code,
        )
    return PatchResponse(reasoning="tool session produced no changes")


async def _codegen_session(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    iteration: int,
    feedback: str | None,
) -> CodeResponse:
    """Fresh generation via tool session: the model may look up catalog parts,
    check its draft with check_snippet, then submit_code. A plain content
    answer is still accepted (CodeResponse schema)."""
    from cadx.agent.context import codegen_tools_system
    from cadx.cad.allowlist import check_code
    from cadx.llm.schemas import ParamDef
    from cadx.llm.structured import LLMFormatError

    session = ToolSession(ctx, kind="codegen")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": codegen_tools_system()},
        {"role": "user", "content": _user_parts(ctx, spec, plan, feedback)},
    ]
    res = await run_session(
        ctx,
        "code",
        messages,
        session=session,
        json_schema=CodeResponse,
        max_calls=ctx.runtime_settings.agent_tool_max_calls,
        step=STEP,
        iteration=iteration,
    )
    if res.submitted and res.submitted.get("code"):
        code = str(res.submitted["code"])
        chk = check_code(code)
        # PARAMS literals in the code are authoritative; the model's `params`
        # arg only supplies metadata (unit/min/max/description)
        meta = {
            str(p.get("name")): p
            for p in (res.submitted.get("params") or [])
            if isinstance(p, dict) and p.get("name")
        }

        def _num(x: Any) -> float | None:
            return float(x) if isinstance(x, int | float) else None

        params = [
            ParamDef(
                name=k,
                value=v,
                unit=str(meta.get(k, {}).get("unit") or "mm"),
                min=_num(meta.get(k, {}).get("min")),
                max=_num(meta.get(k, {}).get("max")),
                description=str(meta.get(k, {}).get("description") or ""),
            )
            for k, v in (chk.params or {}).items()
            if isinstance(v, str | int | float | bool)
        ]
        return CodeResponse(
            reasoning=str(res.submitted.get("reasoning", "")),
            params=params,
            expected_bbox=list(res.submitted.get("expected_bbox") or []),
            code=code,
        )
    if res.response is not None and isinstance(res.response.parsed, CodeResponse):
        return res.response.parsed
    raise LLMFormatError("codegen session produced no code")
