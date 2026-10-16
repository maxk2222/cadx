"""INTAKE — extract a PartSpec from images + text (vision model)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from cadx.agent.context import (
    images_block,
    parent_block,
    references_block,
    system_prompt,
    user_block,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from cadx.llm.schemas import Feature, LoadCase, OptimizeSpec, PartSpec
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "intake"


class _LoadCasesOut(BaseModel):
    load_cases: list[LoadCase] = Field(default_factory=list)
    optimize: OptimizeSpec | None = None


def sanitize_load_cases(spec: PartSpec) -> None:
    """Drop degenerate cases (no supports and no loads) a weak model emits as
    `{"name":"","fixed":[],"loads":[]}` — truthy lists of those would skip the
    extraction fallback and produce unsolvable FEM runs. Also clamps bogus
    negative case weights back to neutral."""
    kept: list[LoadCase] = []
    for c in spec.load_cases:
        if not c.fixed and not c.loads:
            continue
        if c.weight <= 0:
            c.weight = 1.0
        kept.append(c)
    spec.load_cases = kept


_MM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\b", re.IGNORECASE)
_DIMSET_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*mm",
    re.IGNORECASE,
)


def _prompt_mm_values(prompt: str) -> list[float]:
    """Explicit millimetre values mentioned in the user prompt, largest first."""
    vals: set[float] = set()
    for m in _MM_RE.finditer(prompt):
        vals.add(float(m.group(1).replace(",", ".")))
    for m in _DIMSET_RE.finditer(prompt):
        vals.update(float(g.replace(",", ".")) for g in m.groups())
    return sorted(vals, reverse=True)


def sanitize_dims(spec: PartSpec, prompt: str) -> list[str]:
    """Recover an overall dimension the model dropped or misread.

    Weak models routinely lose an explicitly stated size ("approx 130 mm
    long") or substitute a feature size — downstream, scale_to_spec then fits
    the generated mesh to the wrong magnitude (a 130 mm grip becomes a 40 mm
    nugget). When the largest explicit mm value in the prompt exceeds every
    stated dim by a wide margin and length_x is not itself stated, re-inject
    it as the stated length."""
    notes: list[str] = []
    vals = _prompt_mm_values(prompt)
    if not vals:
        return notes
    lx = spec.length_x
    if lx.source == "stated" and lx.value > 0:
        return notes  # the model stated an overall length — trust it
    stated_max = max(
        (
            float(d.value)
            for d in (spec.length_x, spec.width_y, spec.height_z)
            if d is not None and d.source == "stated" and d.value > 0
        ),
        default=0.0,
    )
    if stated_max >= vals[0] * 0.5:
        return notes  # stated dims already cover the part's overall size
    notes.append(
        f"prompt states ~{vals[0]:g} mm but intake had length_x="
        f"{lx.value:g} {lx.source} — re-stated"
    )
    spec.length_x = lx.model_copy(update={"value": vals[0], "source": "stated"})
    return notes


# --------------------------------------------------------------------------- #
# degenerate-spec gate                                                          #
# --------------------------------------------------------------------------- #

_TOOL_NAMES = {
    "search_catalog",
    "lookup_part",
    "cadlib_lookup",
    "search_library",
    "library_part",
    "attach_library_part",
    "view_image",
    "generate_concept_image",
}
_FEATURE_WORDS = re.compile(
    r"\b(loch|löcher|loecher|bohrung|bohrungen|hole|holes|bore|bores|langloch|"
    r"slot|slots|nut|groove|tasche|pocket|rippe|rippen|rib|ribs|steg|web|"
    r"radius|radien|fillet|verrund|verrundung|fase|chamfer|senk|senkung|"
    r"countersunk|countersink|counter|gewinde|thread|insert|boss|zapfen|"
    r"ausschnitt|cutout|aussparung|recess)\w*",
    re.I,
)


def spec_gaps(spec: PartSpec, prompt: str) -> list[str]:
    """Deterministic plausibility check. Returns reasons (empty = ok)."""
    gaps: list[str] = []
    if spec.name.strip().lower() in _TOOL_NAMES or "catalog search" in spec.summary.lower():
        gaps.append("spec name/summary is a tool echo")
    words = sorted({m.group(0).lower() for m in _FEATURE_WORDS.finditer(prompt or "")})
    if words and not spec.features:
        gaps.append(f"prompt names features ({', '.join(words[:8])}) but spec.features is empty")
    return gaps


class _FeaturesOut(BaseModel):
    """Focused re-ask payload: every feature the prompt names, plus identity."""

    name: str = ""
    summary: str = ""
    features: list[Feature] = Field(default_factory=list)


async def extract_features(ctx: AgentCtx, spec: PartSpec) -> _FeaturesOut:
    """Second-chance extraction when the intake spec dropped named features.

    Small one-shot call (prompt + current spec as JSON + images) — cheaper
    than re-running the full intake, same pattern as extract_load_cases."""
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "current_spec": spec.model_dump(
                include={"name", "summary", "length_x", "width_y", "height_z", "features"}
            ),
        },
        ensure_ascii=False,
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_features.md")},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_FeaturesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _FeaturesOut) else _FeaturesOut()
        await ctx.trace.record(
            step="intake_features",
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed
    except Exception:
        log.warning("intake.features_retry_failed", exc_info=True)
        return _FeaturesOut()


async def extract_load_cases(
    ctx: AgentCtx, spec: PartSpec
) -> tuple[list[LoadCase], OptimizeSpec | None]:
    """Focused second-chance extraction: the main intake emitted no usable
    `load_cases` although the user described forces/optimization. Small
    one-shot call on prompt+spec — cheaper than re-running full intake.
    Also recovers the `optimize` block when intake left it empty."""
    user_text = json.dumps(
        {"prompt": ctx.prompt, "spec": spec.model_dump(exclude={"load_cases", "optimize"})},
        ensure_ascii=False,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_loads.md")},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_LoadCasesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _LoadCasesOut) else _LoadCasesOut()
        await ctx.trace.record(
            step=STEP,
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed.load_cases, parsed.optimize
    except Exception:
        log.warning("intake.load_cases_retry_failed", exc_info=True)
        return [], None


def _repair_image_roles(spec: PartSpec, ctx: AgentCtx) -> None:
    """Bind blank image_roles entries to assets positionally.

    Local vision models regularly emit one entry per image but leave
    `asset_id` (and sometimes `role`) blank. Entries are one-per-image in
    image order, so a count match lets us recover the ids the model meant.
    """
    imgs = [a for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS]
    shown = imgs[: ctx.runtime_settings.agent_max_input_images]
    if not shown or len(spec.image_roles) != len(shown):
        return
    if all(not r.asset_id for r in spec.image_roles):
        for r, a in zip(spec.image_roles, shown, strict=False):
            r.asset_id = a.asset_id


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
        _repair_image_roles(spec, ctx)
        sanitize_load_cases(spec)
        sanitize_dims(spec, ctx.prompt or "")
        if not spec.process:
            spec.process = ctx.settings.process
        ctx.spec = spec
        return spec
    resp = await ctx.llm.chat(
        "vision",
        messages,
        json_schema=PartSpec,
        temperature=0.0,
        extra_body=thinking_body(ctx, STEP),
    )
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
    _repair_image_roles(spec, ctx)
    sanitize_load_cases(spec)
    sanitize_dims(spec, ctx.prompt or "")
    if not spec.process:
        spec.process = ctx.settings.process
    ctx.spec = spec
    return spec
