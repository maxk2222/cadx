"""CRITIQUE — compare generated model with input images + spec (vision model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadagent.agent.context import (
    dfm_block,
    images_block,
    measured_features_block,
    parent_block,
    renders_block,
    spec_block,
    system_prompt,
)
from cadagent.agent.toolloop import run_session, tools_enabled
from cadagent.agent.toolspecs import ToolSession
from cadagent.agent.types import AgentCtx
from cadagent.llm.schemas import Critique, Discrepancy, PartSpec

STEP = "critique"


def _parts(
    ctx: AgentCtx,
    spec: PartSpec,
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    features: dict[str, Any] | None,
    dfm: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    numeric = (
        "Numeric dimension check: " + "; ".join(dim_issues)
        if dim_issues
        else "Numeric dimension check: all stated dimensions match within tolerance."
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": "USER INPUT IMAGES (for reference):"}]
    parts += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
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
    mf = measured_features_block(features)
    if mf:
        parts.append({"type": "text", "text": mf})
    if feature_findings:
        lines = ["GEOMETRIC FEATURE CHECK (measured, may be approximate):"]
        for f in feature_findings:
            lines.append(
                f"- [{f.get('severity', 'minor')}] {f.get('feature')}: expected "
                f"{f.get('expected')}, actual {f.get('actual')}"
            )
        parts.append({"type": "text", "text": "\n".join(lines)})
    dfm_txt = dfm_block(dfm)
    if dfm_txt:
        parts.append({"type": "text", "text": dfm_txt})
    return parts


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dim_issues: list[str],
    dfm: dict[str, Any] | None,
    *,
    iteration: int,
    feature_findings: list[dict[str, Any]] | None = None,
    features: dict[str, Any] | None = None,
    code: str = "",
) -> Critique:
    parts = _parts(
        ctx, spec, contact_sheet, analysis, dim_issues, feature_findings or [], features, dfm
    )

    system = system_prompt("critique_system.md")
    use_tools = tools_enabled(ctx, "vision")
    if use_tools:
        if code:
            system += "\n\n" + system_prompt("tools_inspect.md")
        else:
            system += (
                "\n\nNo source code is available in this generation mode — "
                "judge the renders and measured facts only. Catalog lookups "
                "via tools are still allowed."
            )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": parts},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="inspect", code=code)
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=Critique,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            iteration=iteration,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, Critique):
            from cadagent.llm.structured import LLMFormatError

            raise LLMFormatError("critique session produced no verdict")
        critique = res.response.parsed
        resp = res.response
    else:
        resp = await ctx.llm.chat("vision", messages, json_schema=Critique, temperature=0.0)
        assert isinstance(resp.parsed, Critique)
        critique = resp.parsed
    await ctx.trace.record(
        step=STEP,
        iteration=iteration,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=critique.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )

    # empty-discrepancy guard: a failing score with no issues listed gives the
    # loop nothing to fix — re-ask once for concrete problems.
    if (
        not critique.passed
        and not critique.discrepancies
        and ctx.runtime_settings.agent_critique_reask
    ):
        messages.append({"role": "assistant", "content": resp.text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"You scored {critique.score}/100 with passed=false but listed "
                    "NO discrepancies — that gives nothing to fix. List every "
                    "concrete problem now (what is wrong, where, and a fix_hint "
                    "with a target value). Respond with the complete Critique JSON."
                ),
            }
        )
        resp2 = await ctx.llm.chat("vision", messages, json_schema=Critique, temperature=0.0)
        assert isinstance(resp2.parsed, Critique)
        await ctx.trace.record(
            step=f"{STEP}_reask",
            iteration=iteration,
            model=resp2.model,
            messages=messages,
            response_text=resp2.text,
            parsed=resp2.parsed.model_dump(),
            prompt_tokens=resp2.prompt_tokens,
            completion_tokens=resp2.completion_tokens,
            latency_ms=resp2.latency_ms,
        )
        if resp2.parsed.discrepancies or resp2.parsed.score < critique.score:
            critique = resp2.parsed

    # synthesize discrepancies from measured findings if still empty
    if not critique.passed and not critique.discrepancies:
        for i in dim_issues:
            critique.discrepancies.append(
                Discrepancy(what=f"dimension mismatch — {i}", severity="major")
            )
        for f in feature_findings or []:
            critique.discrepancies.append(
                Discrepancy(
                    what=f"{f.get('feature')}: expected {f.get('expected')}, "
                    f"actual {f.get('actual')}",
                    severity=f.get("severity", "minor"),
                    fix_hint=f.get("fix_hint", ""),
                )
            )
    return critique
