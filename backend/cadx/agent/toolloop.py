"""Turn loop for LLM tool-calling sessions.

`run_session` sends the seeded messages plus the session's tool schemas,
executes each tool call the model makes, feeds results back, and repeats
until the model answers without tool calls (final answer), submits via a
done/submit tool, or the call budget is exhausted.

Protocol notes:
- native tool_calls: the assistant message is echoed back WITH its
  tool_calls array and each call answered by a `role:"tool"` message with
  matching `tool_call_id` (required by OpenAI/vLLM).
- content-fallback calls (provider can't emit tool_calls but writes a JSON
  call in content): we keep a plain text conversation instead — assistant
  text echoed, result returned as a user message, because role:"tool" is
  invalid without tool_calls on that provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.client import LLMResponse
from cadx.llm.images import image_part
from cadx.llm.tools import ToolCall
from cadx.logging import get_logger

log = get_logger(__name__)


@dataclass
class SessionResult:
    """What a session produced.

    - `response`: the final non-tool LLM answer (schema-parsed as usual)
    - `submitted`: args captured by a done/submit_code tool call
    - `code`: the session buffer at the end (edits applied or None)
    - `exhausted`: budget ran out without done/final answer
    """

    response: LLMResponse | None = None
    submitted: dict[str, Any] | None = None
    code: str | None = None
    exhausted: bool = False
    cancelled: bool = False
    tool_calls_made: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)


def tools_enabled(ctx: AgentCtx, role: str) -> bool:
    """Whether this step should use a tool session.

    Probe state is per role: a 400-strip on the vision model must not disable
    tools for the code model (and vice versa).
    """
    llm = ctx.llm
    if getattr(llm, "tools_mode", "off") == "off":
        return False
    ok = getattr(llm, "tools_ok", {})
    if isinstance(ok, dict):
        return ok.get(role) is not False
    return ok is not False  # legacy bool/None shape


# tools that validate progress — a turn without any of these counts as churn
_CHECK_TOOLS = {
    "check_code",
    "verify",
    "render_model",
    "done",
    "submit_code",
    "mesh_preview",
    "mesh_measure",
}


def _trim_tool_results(msgs: list[dict[str, Any]], keep_last: int = 6) -> None:
    """Replace bodies of old read-only tool results with a stub (in place)."""
    idx = [
        i
        for i, m in enumerate(msgs)
        if m.get("role") == "tool"
        or (m.get("role") == "user" and str(m.get("content", "")).startswith("TOOL RESULT for"))
    ]
    for i in idx[:-keep_last]:
        m = msgs[i]
        body = m.get("content")
        if (
            isinstance(body, str)
            and len(body) > 200
            and ('"lines"' in body or "read_lines" in body or "search_code" in body)
        ):
            m["content"] = body[:120] + " …[elided — call read_lines again if needed]"


def _assistant_echo(resp: LLMResponse, native: bool) -> dict[str, Any]:
    if native:
        return {
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.raw_arguments or "{}"},
                }
                for tc in resp.tool_calls or []
            ],
        }
    return {"role": "assistant", "content": resp.text}


def _result_messages(tc: ToolCall, result: dict[str, Any], native: bool) -> list[dict[str, Any]]:
    """Serialise a tool result into message(s). A result may carry `"_images"`
    (list of image file paths) — OpenAI tool messages can't hold images, so
    they go out as a following user message with image content parts."""
    images = result.get("_images") or []
    body = json.dumps({k: v for k, v in result.items() if k != "_images"}, default=str)
    out: list[dict[str, Any]] = []
    if native:
        out.append({"role": "tool", "tool_call_id": tc.id, "content": body})
    else:
        out.append({"role": "user", "content": f"TOOL RESULT for {tc.name}:\n{body}"})
    if images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"{tc.name} output image(s):"}]
        for p in images:
            try:
                parts.append(image_part(Path(p)))
            except Exception:
                continue
        if len(parts) > 1:
            out.append({"role": "user", "content": parts})
    return out


async def run_session(
    ctx: AgentCtx,
    role: str,
    messages: list[dict[str, Any]],
    *,
    session: ToolSession,
    json_schema: type[BaseModel] | None = None,
    max_calls: int,
    step: str,
    iteration: int = 0,
    temperature: float | None = None,
) -> SessionResult:
    """Run the tool-call turn loop. Returns when the model stops calling tools."""
    msgs = list(messages)
    calls = 0
    turns = 0
    error_streak = 0
    last_sig = ""
    last_text = ""
    repeat_streak = 0
    edits_since_check = 0
    nudged = False
    churn_exceeded = False
    native = True  # per-response: all calls in one response share the mode

    while True:
        turns += 1
        # cancellation is polled once per turn — a single tool session can
        # span dozens of LLM calls and must not outlive a cancelled job
        if await ctx.check_cancel():
            log.info("tool_session.cancelled", step=step, calls=calls)
            return SessionResult(
                cancelled=True,
                exhausted=True,
                tool_calls_made=calls,
                tool_log=session.tool_log,
            )
        _trim_tool_results(msgs)
        # fresh each turn: e.g. `verify` drops out once its budget is spent
        tools = session.tools()
        resp = await ctx.llm.chat(
            role,
            msgs,
            json_schema=json_schema,
            tools=tools,
            temperature=temperature,
        )
        tool_calls = resp.tool_calls or []
        await ctx.trace.record(
            step=step,
            iteration=iteration,
            model=resp.model,
            messages=msgs,
            response_text=resp.text,
            parsed={
                "tool_calls": [f"{tc.name}({_short(tc.raw_arguments)})" for tc in tool_calls],
                "has_final": not tool_calls,
            },
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )

        if not tool_calls:
            return SessionResult(
                response=resp,
                submitted=session.submitted,
                code=session.buffer or None,
                tool_calls_made=calls,
                tool_log=session.tool_log,
            )

        native = not all(tc.via_content for tc in tool_calls)
        msgs.append(_assistant_echo(resp, native))
        for tc in tool_calls:
            calls += 1
            if tc.arguments_error:
                result = {"ok": False, "error": tc.arguments_error}
                session.tool_log.append(
                    {"name": tc.name, "args": {"_raw": tc.raw_arguments[:300]}, "result": result}
                )
            else:
                result = await session.execute(tc.name, tc.arguments)
            msgs.extend(_result_messages(tc, result, native))
            error_streak = error_streak + 1 if not result.get("ok", True) else 0
            if session.is_done and tc.name in ("done", "submit_code", "regenerate"):
                return SessionResult(
                    response=resp,
                    submitted=session.submitted,
                    code=session.buffer or None,
                    tool_calls_made=calls,
                    tool_log=session.tool_log,
                )
        # identical-call loop detector: weak models re-emit the same turn
        # forever (e.g. search_catalog({}) on every reply) — after 3 identical
        # turns treat it like budget exhaustion and force the final answer.
        # Args can vary slightly while the prose repeats verbatim ("I'm stuck
        # in a loop"), so also fingerprint the response text.
        sig = json.dumps(
            [[tc.name, tc.arguments] for tc in tool_calls], sort_keys=True, default=str
        )
        # only fingerprint substantive prose — empty/trivial text repeats
        # legitimately across different tool calls and must not count
        text_sig = re.sub(r"\s+", " ", (resp.text or "")[:200]).strip()
        text_hit = len(text_sig) >= 16 and text_sig == last_text
        repeat_streak = repeat_streak + 1 if (sig == last_sig or text_hit) else 0
        last_sig, last_text = sig, text_sig
        # churn detector: weak models burn turns on read/edit pairs without
        # ever validating — nudge once, then end the session
        names = {tc.name for tc in tool_calls}
        edits_since_check = 0 if names & _CHECK_TOOLS else edits_since_check + 1
        if edits_since_check >= ctx.runtime_settings.agent_tool_churn_limit and session.kind in (
            "edit",
            "codegen",
        ):
            if not nudged:
                nudged = True
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "You have made many edits without checking. "
                            "Call check_code now, then done."
                        ),
                    }
                )
                edits_since_check = 0
            else:
                churn_exceeded = True
        if (
            calls >= max_calls
            or turns >= ctx.runtime_settings.agent_tool_max_turns
            or error_streak >= 4
            or repeat_streak >= 2
            or churn_exceeded
        ):
            reason = (
                "budget"
                if calls >= max_calls
                else "turns"
                if turns >= ctx.runtime_settings.agent_tool_max_turns
                else "error_streak"
                if error_streak >= 4
                else "churn"
                if churn_exceeded
                else "repeat_loop"
            )
            log.info("tool_session.end", step=step, calls=calls, reason=reason)
            # rescue turn: schema-final sessions (intake/plan/critique/codegen)
            # still owe an answer — one last call WITHOUT tools lets the model
            # deliver it instead of failing the step outright
            if json_schema is not None and not session.is_done:
                resp = await _grace_turn(ctx, role, msgs, json_schema, step, iteration)
                if resp is not None:
                    return SessionResult(
                        response=resp,
                        submitted=session.submitted,
                        code=session.buffer or None,
                        exhausted=True,
                        tool_calls_made=calls,
                        tool_log=session.tool_log,
                    )
            return SessionResult(
                response=None,
                submitted=session.submitted,
                code=session.buffer or None,
                exhausted=True,
                tool_calls_made=calls,
                tool_log=session.tool_log,
            )


async def _grace_turn(
    ctx: AgentCtx,
    role: str,
    msgs: list[dict[str, Any]],
    json_schema: type[BaseModel],
    step: str,
    iteration: int,
) -> LLMResponse | None:
    """One final tools-off turn so an exhausted session can still deliver its
    schema answer. Returns None when even that fails."""
    try:
        resp: LLMResponse = await ctx.llm.chat(
            role,
            [
                *msgs,
                {
                    "role": "user",
                    "content": (
                        "Tool budget reached. Return the FINAL answer JSON "
                        "now — no tool calls, just the JSON object."
                    ),
                },
            ],
            json_schema=json_schema,
        )
    except Exception as e:
        log.warning("tool_session.grace_failed", step=step, error=str(e))
        return None
    await ctx.trace.record(
        step=f"{step}_final",
        iteration=iteration,
        model=resp.model,
        messages=msgs,
        response_text=resp.text,
        parsed={"grace_turn": True},
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp


def _short(raw: str, n: int = 80) -> str:
    return raw if len(raw) <= n else raw[:n] + "…"
