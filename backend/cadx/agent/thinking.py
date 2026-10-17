"""Adaptive per-call reasoning ("thinking") control.

Reasoning tokens are expensive: the policy keeps them OFF for cheap steps
and ON for hard steps (``repair,improve`` by default) plus every call once
the orchestrator latches ``ctx.thinking_escalated``. Provider dialects stay
config-driven — ``llm_thinking_body_on`` / ``llm_thinking_body_off`` are
JSON dicts merged verbatim into the request payload by the client
(Qwen3-style ``{"chat_template_kwargs": {"enable_thinking": false}}``,
OpenAI-style ``{"reasoning_effort": "low"}``); nothing is hardcoded here.

The policy is pure and never raises — a malformed body setting logs one
warning and degrades to "no override" rather than failing the LLM call.
"""

from __future__ import annotations

import json
from typing import Any

from cadx.agent.types import AgentCtx
from cadx.logging import get_logger

log = get_logger(__name__)

# malformed body settings warn once per direction — the policy runs on
# EVERY llm call and must not spam the log
_warned: set[str] = set()


def step_is_hard(step: str, csv: str) -> bool:
    """Prefix match against the hard-steps CSV: "repair_2" matches "repair".

    The match is boundary-aware ("repairman" does NOT match "repair") so a
    step named like a hard step's prefix doesn't escalate by accident.
    """
    hard = [s.strip() for s in csv.split(",") if s.strip()]
    return any(step == h or step.startswith(h + "_") for h in hard)


def _body(raw: str, which: str) -> dict[str, Any] | None:
    """Parse a thinking-body setting into a payload dict (None = no override)."""
    if not raw.strip():
        return None
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        if which not in _warned:
            _warned.add(which)
            log.warning("thinking.body.invalid_json", which=which)
        return None
    if not isinstance(parsed, dict):
        if which not in _warned:
            _warned.add(which)
            log.warning("thinking.body.not_object", which=which)
        return None
    return parsed


def _policy_on(ctx: AgentCtx, step: str) -> bool:
    s = ctx.runtime_settings
    if s.llm_thinking == "always":
        return True
    if s.llm_thinking == "off":
        return False
    # auto: hard steps and the stall latch escalate to thinking-on
    return ctx.thinking_escalated or step_is_hard(step, s.llm_thinking_hard_steps)


def thinking_on(ctx: AgentCtx, step: str) -> bool:
    """Whether reasoning is enabled for this call under the configured policy.

    The per-job cost cap ``llm_thinking_max_calls`` wins over every mode —
    once the counted thinking calls are spent this resolves False regardless
    of escalation, and the first suppressed call latches
    ``ctx.thinking_budget_exhausted`` + logs one ``thinking_budget_exhausted``
    event."""
    if not _policy_on(ctx, step):
        return False
    used = getattr(ctx, "thinking_calls_used", 0)
    if used >= ctx.runtime_settings.llm_thinking_max_calls:
        if not getattr(ctx, "thinking_budget_exhausted", False):
            ctx.thinking_budget_exhausted = True
            log.warning(
                "thinking_budget_exhausted",
                step=step,
                calls_used=used,
                max_calls=ctx.runtime_settings.llm_thinking_max_calls,
            )
        return False
    return True


def thinking_body(ctx: AgentCtx, step: str) -> dict[str, Any] | None:
    """Resolve the per-call request-body override for `step`.

    Returns the parsed ``llm_thinking_body_on`` dict when the policy resolves
    to thinking-ON, ``llm_thinking_body_off`` when OFF, or None when the
    selected body is empty/unparseable (provider default — nothing merged).

    Call sites invoke this exactly once per chat call, so the ON count is
    also the per-job thinking budget ledger (``ctx.thinking_calls_used``)."""
    s = ctx.runtime_settings
    on = thinking_on(ctx, step)
    if on:
        ctx.thinking_calls_used = getattr(ctx, "thinking_calls_used", 0) + 1
    return _body(s.llm_thinking_body_on if on else s.llm_thinking_body_off, "on" if on else "off")
