"""CRITIQUE — compare generated model with input images + spec (vision model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadx.agent.context import (
    anchors_block,
    dfm_block,
    images_block,
    measured_features_block,
    measurements_block,
    parent_block,
    renders_block,
    request_block,
    spec_block,
    system_prompt,
)
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import Critique, Discrepancy, PartSpec

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
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    parts.append({"type": "text", "text": spec_block(spec)})
    parts += renders_block(contact_sheet, analysis)
    parts.append({"type": "text", "text": numeric})
    mf = measured_features_block(features)
    if mf:
        parts.append({"type": "text", "text": mf})
    mm = measurements_block(ctx.measurements)
    if mm:
        parts.append({"type": "text", "text": mm})
    ab = anchors_block(ctx.anchors)
    if ab:
        parts.append({"type": "text", "text": ab})
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
            from cadx.llm.structured import LLMFormatError

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

    return _finalize_critique(critique, ctx, spec, dim_issues, feature_findings or [], features)


def _finalize_critique(
    critique: Critique,
    ctx: AgentCtx,
    spec: PartSpec,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    features: dict[str, Any] | None,
) -> Critique:
    """Post-LLM calibration: synthesize missing discrepancies, enforce the
    feature hard rule, then bound the score by measured facts."""
    # the floor below protects against UNSOURCED complaints — it only applies
    # when the model itself listed them; a bare failed verdict (discrepancies
    # synthesized here) is the model's own call and must stay low
    had_llm_discrepancies = bool(critique.discrepancies)
    # synthesize discrepancies from measured findings if still empty
    if not critique.passed and not critique.discrepancies:
        for i in dim_issues:
            critique.discrepancies.append(
                Discrepancy(what=f"dimension mismatch — {i}", severity="major", source="spec")
            )
        for f in feature_findings:
            critique.discrepancies.append(
                Discrepancy(
                    what=f"{f.get('feature')}: expected {f.get('expected')}, "
                    f"actual {f.get('actual')}",
                    severity=f.get("severity", "minor"),
                    fix_hint=f.get("fix_hint", ""),
                    source="spec",
                )
            )
        # the model's own feature verdicts are actionable too — a 'missing' or
        # 'wrong' verdict without a discrepancy is a self-contradiction
        for fv in critique.feature_verdicts:
            if fv.status in ("missing", "wrong"):
                critique.discrepancies.append(
                    Discrepancy(
                        what=f"feature {fv.status}: {fv.name}"
                        + (f" — {fv.note}" if fv.note else ""),
                        severity="major" if fv.status == "missing" else "minor",
                        fix_hint=f"make the '{fv.name}' feature match the spec"
                        + (f" ({fv.note})" if fv.note else ""),
                        source="spec",
                    )
                )
        # last resort: a failing verdict with literally nothing actionable —
        # give the improve step the honest generic hint instead of silence
        if not critique.discrepancies:
            critique.discrepancies.append(
                Discrepancy(
                    what="overall shape does not match the reference images",
                    severity="major",
                    fix_hint="compare silhouette and proportions against the "
                    "reference views and reshape the body",
                )
            )

    # hard rule: the request names features (holes/slots/ribs/fillets/...) but
    # the measured geometry has none and the spec lists none → the model must
    # not pass (observed: bracket prompt -> plain box scored 95)
    from cadx.agent.steps.intake import _FEATURE_WORDS

    measured_n = (
        int((features or {}).get("total_holes") or 0)
        + len((features or {}).get("rounds") or [])
        + int((features or {}).get("chamfer_faces") or 0)
        + int((features or {}).get("countersinks") or 0)
        + int((features or {}).get("counterbores") or 0)
    )
    if _FEATURE_WORDS.search(ctx.prompt or "") and measured_n == 0 and not spec.features:
        critique.discrepancies.append(
            Discrepancy(
                what="the request names features (holes/slots/ribs/fillets) but "
                "the geometry has none",
                where="whole part",
                severity="major",
                source="prompt",
                fix_hint="implement every feature named in the USER REQUEST",
            )
        )
        critique.passed = False
        critique.score = min(critique.score, 40)

    # fact-based calibration: measured facts bound the LLM's score so a
    # confident 95 on a broken part — or a paranoid 40 on a clean one — both
    # get pulled toward reality (T3.1). Runs AFTER normalize_score so its
    # empty-discrepancy floor cannot undo the measured ceiling.
    critique = normalize_score(critique)
    ceiling = fact_ceiling(dim_issues, feature_findings)
    if critique.score > ceiling:
        critique.score = ceiling
    floor = (
        fact_floor(critique, dim_issues, feature_findings) if had_llm_discrepancies else 0
    )
    if floor:
        for d in critique.discrepancies:
            if d.source in ("other", "dfm") and d.severity == "major":
                d.severity = "minor"
        critique.score = max(critique.score, floor)
    return critique


def fact_ceiling(dim_issues: list[str], feature_findings: list[dict[str, Any]]) -> int:
    """Upper bound from measured facts only — dims + major feature findings."""
    majors = sum(1 for f in feature_findings if f.get("severity") == "major")
    if dim_issues or majors:
        return max(0, 70 - 15 * (len(dim_issues) + majors))
    return 100


def fact_floor(
    critique: Critique,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
) -> int:
    """Lower bound when every measured fact passes and only unsourced
    remarks remain — invented flaws must not tank a clean part."""
    if dim_issues or any(f.get("severity") == "major" for f in feature_findings):
        return 0
    if critique.discrepancies and all(d.source in ("other", "dfm") for d in critique.discrepancies):
        return 60
    return 0


def normalize_score(critique: Critique) -> Critique:
    """Score consistency clamp: the LLM's number is untrusted when it
    contradicts its own discrepancy list (observed: '0/100 — 0 major,
    0 minor'). Deterministic ceiling so gates see an explainable score."""
    if critique.discrepancies:
        n_major = sum(1 for d in critique.discrepancies if d.severity == "major")
        n_minor = len(critique.discrepancies) - n_major
        ceiling = max(0, 100 - 25 * n_major - 10 * n_minor)
        if critique.score > ceiling:
            critique.score = ceiling
    elif critique.score < 60:
        # a low score with zero listed issues is meaningless — normalize so
        # measured facts (dim/feature findings, passed flag) decide instead
        critique.score = 60
    return critique
