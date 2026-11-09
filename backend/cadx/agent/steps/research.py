"""research — optional pre-intake step: ground referenced products/norms.

When the prompt names an external product or standard ("heatsink for
i7-14700K", "NEMA 23 mount", "DIN 6885 keyway"), the part dims are not
design decisions — they are lookup facts the model routinely
hallucinates. This step runs a bounded tool session (web_search +
fetch_url via the shared info surface) that returns a ResearchBrief:
sourced facts, design constraints, open questions. Intake injects the
brief and lands every fact in spec.assumptions with its source URL —
auditable, never model-invented.

Gate: runtime_settings.agent_research = off|auto|on. auto fires on
product/norm tokens in the prompt; off/intake-unaffected when tools are
unavailable or the brief comes back empty.
"""

from __future__ import annotations

import re
from typing import Any

from cadx.agent.context import system_prompt
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import ResearchBrief
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "research"

# product/norm tokens worth a lookup — product model numbers, mechanical
# norms and mounting standards where dims are published facts
_NEEDS_RESEARCH = re.compile(
    r"(?:\bf(?:ür|or)\s+\w*\d|"
    r"\bi[3579]-\d{4,5}[a-z]*\b|"
    r"\bryzen\s*\d|"
    r"\bnema\s*\d+|"
    r"\bdin\s*\d+|"
    r"\biso\s*\d{3,}|"
    r"\blga\s*\d+|"
    r"\bam[45]\b|"
    r"\bsocket\b|\bsockel\b|"
    r"\bmount(?:ing)?\s+(?:for|für)\s+\w*\d|"
    r"\bthread(?:ed)?\s+m\d+|"
    r"\b\d{2,4}\s*(?:mm\s*)?(?:fan|lüfter)|"
    r"\bvitamix\b|\bnerf\b|\bgarmin\b)",
    re.I,
)


def needs_research(prompt: str) -> bool:
    """auto-gate heuristic: the prompt references a product/norm whose
    dims are published facts, not design decisions."""
    return bool(_NEEDS_RESEARCH.search(prompt or ""))


async def run(ctx: AgentCtx) -> ResearchBrief | None:
    settings = ctx.runtime_settings
    mode = str(getattr(settings, "agent_research", "off") or "off")
    if mode == "off":
        return None
    prompt = ctx.prompt or ""
    if mode == "auto" and not needs_research(prompt):
        return None
    if not tools_enabled(ctx, "vision"):
        log.info("research.skipped", reason="tools disabled")
        return None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("research_system.md")},
        {
            "role": "user",
            "content": (
                "Part request to ground:\n\n" + prompt + "\n\nResearch ONLY the "
                "referenced products/norms whose dimensions are published "
                "facts (socket, mount pattern, bore/thread sizes, TDP, norm "
                "numbers). Return a ResearchBrief; an empty brief when the "
                "request needs no external facts."
            ),
        },
    ]
    session = ToolSession(ctx, kind="info")
    try:
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=ResearchBrief,
            max_calls=int(getattr(settings, "agent_research_max_calls", 8) or 8),
            step=STEP,
            temperature=0.0,
        )
    except Exception as e:
        log.info("research.failed", error=str(e)[:200])
        return None
    brief = res.response.parsed if res.response else None
    if not isinstance(brief, ResearchBrief):
        return None
    if not (brief.facts or brief.constraints):
        return None
    await ctx.emit(
        STEP,
        0,
        f"researched {brief.product or 'referenced specs'}: {len(brief.facts)} sourced facts",
    )
    return brief
