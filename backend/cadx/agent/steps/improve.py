"""IMPROVE — targeted fix of the best code so far (code model).

Two protocols, chosen by provider capability:
- tool session (preferred): the model edits the code buffer interactively
  (read/search/edit/check/verify/done); the buffer IS the result.
- legacy PatchResponse: one-shot find/replace edits or full_code.
"""

from __future__ import annotations

from typing import Any

from cadx.agent.context import (
    anchors_block,
    bindings_block,
    codegen_system,
    issues_block,
    measured_features_block,
    measurements_block,
    numbered_code_block,
    plan_block,
    references_block,
    request_block,
    spec_block,
    system_prompt,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession, VerifyFn
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import ModelingPlan, PartSpec, PatchResponse
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "improve"
# minor-issue rounds (feature findings, lint minors — mostly "apply the bound
# snippet") get their own step name so they stay OUT of the hard-steps set:
# step_is_hard only matches numeric attempt suffixes, so "improve_minor*"
# never consumes the per-job thinking budget (observed: the 8-call budget
# fully spent by early minor improves, leaving later hard steps unthunked)
STEP_MINOR = "improve_minor"

_HARD_SOURCES = ("dims", "critique", "verify_question", "dfm", "solids", "exec")


def _hard_issues(issues: list[dict[str, Any]]) -> bool:
    """Any major issue from a source that needs real exploration — minor
    rounds just patch snippets/fix lint, no vision or deep reasoning."""
    return any(i.get("severity") == "major" and i.get("source") in _HARD_SOURCES for i in issues)


def _step_name(hard: bool, attempt: int) -> str:
    base = STEP if hard else STEP_MINOR
    return base if attempt == 1 else f"{base}_{attempt}"


def _user_parts(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    base_code: str,
    issues: list[dict[str, Any]],
    features: dict[str, Any] | None,
    patch_errors: list[str] | None,
    *,
    numbered: bool = False,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    parts.append({"type": "text", "text": spec_block(spec)})
    if plan is not None:
        parts.append({"type": "text", "text": plan_block(plan)})
    bb = bindings_block(spec)
    if bb:
        parts.append({"type": "text", "text": bb})
    if ctx.hybrid_solid is not None:
        parts.append(
            {
                "type": "text",
                "text": (
                    "HYBRID MODE rules:\n"
                    "- keep `mesh = load_mesh_solid()` — never replace, rotate or "
                    "translate it\n"
                    "- add features only via boolean ops with fresh primitives "
                    "(cylinders, boxes)\n"
                    "- place features on `spine_y(x)` / `top_z(x,y)` helpers, never "
                    "by guessed constants\n"
                    "- `result` must contain the mesh solid"
                ),
            }
        )
    ref = references_block(ctx.assets)
    if ref:
        parts.append({"type": "text", "text": ref})
    # APC ordering: measurements/anchors are job-stable — they precede the
    # per-iteration tail (current code, measured geometry, issues) so the
    # shared prefix reaches as far as possible across improve calls
    mm = measurements_block(ctx.measurements)
    if mm:
        parts.append({"type": "text", "text": mm})
    ab = anchors_block(ctx.anchors)
    if ab:
        parts.append({"type": "text", "text": ab})
    if numbered:
        parts.append({"type": "text", "text": numbered_code_block(base_code)})
    else:
        parts.append(
            {
                "type": "text",
                "text": "CURRENT CODE (working unless noted):\n```python\n" + base_code + "\n```",
            }
        )
    mf = measured_features_block(features)
    if mf:
        parts.append({"type": "text", "text": mf})
    parts.append({"type": "text", "text": issues_block(issues)})
    if patch_errors:
        parts.append(
            {
                "type": "text",
                "text": (
                    "YOUR PREVIOUS PATCH DID NOT APPLY — localised edits have "
                    "already failed on this code, so return the COMPLETE corrected "
                    "source in `full_code` (not `edits`). Errors to fix:\n- "
                    + "\n- ".join(patch_errors)
                ),
            }
        )
    return parts


def _session_system(ctx: AgentCtx, spec: PartSpec, plan: ModelingPlan | None) -> str:
    return "\n\n".join(
        [
            system_prompt("improve_tools.md"),
            system_prompt("tools_protocol.md"),
            codegen_system(plan=plan, spec=spec, fresh=False, hybrid=ctx.hybrid_solid is not None),
        ]
    )


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    base_code: str,
    issues: list[dict[str, Any]],
    features: dict[str, Any] | None = None,
    patch_errors: list[str] | None = None,
    iteration: int,
    attempt: int = 1,
    verify_fn: VerifyFn | None = None,
    render_fn: VerifyFn | None = None,
) -> PatchResponse:
    hard = _hard_issues(issues)
    step = _step_name(hard, attempt)
    if tools_enabled(ctx, "code") and ctx.edit_sessions_exhausted < 1:
        # feature-only improve (missing hole/slot/rib findings + lint minors)
        # is mostly "apply the bound snippet" — a full exploration budget
        # just buys churn (observed: 12-call sessions at ~16k ptok each), and
        # vision renders add nothing a light verify can't tell it
        s = ctx.runtime_settings
        mc = (
            s.agent_tool_max_calls
            if hard
            else min(s.agent_tool_max_calls, s.agent_tool_max_calls_improve)
        )
        return await _run_session(
            ctx,
            spec,
            plan,
            base_code=base_code,
            issues=issues,
            features=features,
            iteration=iteration,
            attempt=attempt,
            verify_fn=verify_fn,
            render_fn=render_fn if hard else None,
            max_calls=mc,
            step=step,
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt("improve_system.md")
            + "\n\n"
            + codegen_system(
                plan=plan, spec=spec, fresh=False, hybrid=ctx.hybrid_solid is not None
            ),
        },
        {
            "role": "user",
            "content": _user_parts(ctx, spec, plan, base_code, issues, features, patch_errors),
        },
    ]
    resp = await ctx.llm.chat(
        "code",
        messages,
        json_schema=PatchResponse,
        temperature=0.1,
        extra_body=thinking_body(ctx, step),
    )
    assert isinstance(resp.parsed, PatchResponse)
    await ctx.trace.record(
        step=step,
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed={
            "reasoning": resp.parsed.reasoning,
            "n_edits": len(resp.parsed.edits),
            "has_full_code": bool(resp.parsed.full_code),
        },
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp.parsed


async def _run_session(
    ctx: AgentCtx,
    spec: PartSpec,
    plan: ModelingPlan | None,
    *,
    base_code: str,
    issues: list[dict[str, Any]],
    features: dict[str, Any] | None,
    iteration: int,
    attempt: int,
    verify_fn: VerifyFn | None,
    render_fn: VerifyFn | None = None,
    max_calls: int,
    step: str = STEP,
) -> PatchResponse:
    session = ToolSession(
        ctx, kind="edit", code=base_code, verify_fn=verify_fn, render_fn=render_fn
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _session_system(ctx, spec, plan)},
        {
            "role": "user",
            "content": _user_parts(
                ctx, spec, plan, base_code, issues, features, None, numbered=True
            ),
        },
    ]
    res = await run_session(
        ctx,
        "code",
        messages,
        session=session,
        max_calls=max_calls,
        step=step,
        iteration=iteration,
        temperature=0.1,
    )
    if res.exhausted:
        ctx.edit_sessions_exhausted += 1
        log.info("edit.oneshot_fallback", exhausted=ctx.edit_sessions_exhausted)
    if res.code and res.code != base_code:
        sub = res.submitted or {}
        return PatchResponse(
            reasoning=str(sub.get("reasoning", "")),
            expected_bbox=list(sub.get("expected_bbox") or []),
            full_code=res.code,
        )
    # session produced nothing usable — signal patch failure so the
    # orchestrator falls back to fresh codegen
    return PatchResponse(reasoning="tool session produced no changes")
