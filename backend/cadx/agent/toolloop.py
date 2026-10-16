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
from collections.abc import Awaitable, Callable
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
    # why the session exhausted: budget | turns | error_streak | churn |
    # repeat_loop | verify_loop — callers use "verify_loop" to switch to the
    # full-code one-shot path (a session stuck on one error won't converge)
    end_reason: str = ""
    cancelled: bool = False
    tool_calls_made: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    # the full mutated message list — chat sessions persist the tail so the
    # conversation (assistant echoes + tool results) survives across turns
    messages_out: list[dict[str, Any]] = field(default_factory=list)


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
        if not isinstance(body, str) or len(body) <= 200:
            continue
        if '"lines"' in body or "read_lines" in body or "search_code" in body:
            m["content"] = body[:120] + " …[elided — call read_lines again if needed]"
        elif len(body) > 2000 and '"category"' not in body:
            # large one-shot JSON payloads (mesh_measure, feature dumps) are
            # stale once a newer measurement exists — a stub keeps the shape
            m["content"] = body[:120] + " …[elided — superseded tool result]"
    # preview/section/overlay images are the heaviest history items — the
    # latest render makes older ones worthless, keep only the last 2
    img_idx = [
        i
        for i, m in enumerate(msgs)
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
    ]
    for i in img_idx[:-2]:
        msgs[i] = {
            **msgs[i],
            "content": [{"type": "text", "text": "[earlier preview image removed — superseded]"}],
        }
    # failed-verify results carry the full excerpt+traceback and repeat the
    # same blob turn after turn — keep the last 2 intact, compact older ones
    v_idx = [
        i
        for i, m in enumerate(msgs)
        if m.get("role") in ("tool", "user")
        and isinstance(m.get("content"), str)
        and '"category"' in m["content"]
        and '"excerpt"' in m["content"]
    ]
    for i in v_idx[:-2]:
        try:
            body = msgs[i]["content"]
            raw = body.split(":\n", 1)[-1] if body.startswith("TOOL RESULT") else body
            d = json.loads(raw)
            stub = json.dumps(
                {
                    "ok": False,
                    "category": d.get("category"),
                    "line": d.get("line"),
                    "error": str(d.get("error", ""))[:150],
                    "elided": True,
                }
            )
            msgs[i]["content"] = (
                "TOOL RESULT for verify (elided):\n" + stub
                if body.startswith("TOOL RESULT")
                else stub
            )
        except Exception:
            continue


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


async def run_session(  # noqa: C901
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
    drain_fn: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    on_delta: Callable[[str], None] | None = None,
) -> SessionResult:
    """Run the tool-call turn loop. Returns when the model stops calling tools.

    ``drain_fn`` (chat sessions) is awaited once per turn and its returned
    messages appended — newly submitted user text joins mid-turn.
    ``on_delta`` is forwarded to ``llm.chat`` for incremental streaming."""
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
    verify_loop = False
    # per-error-class tally of FAILED verify results — survives interleaved
    # successful edits (the observed non-converging repair did
    # find_replace/verify ping-pong 24 turns on one fillet error)
    verify_fails: dict[str, int] = {}
    verify_nudged = False
    # edit/codegen sessions pay the full sandbox pipeline per verify — they
    # get a tighter turn cap than read-only info/inspect sessions
    max_turns = (
        ctx.runtime_settings.agent_tool_max_turns_edit
        if session.kind in ("edit", "codegen")
        else ctx.runtime_settings.agent_tool_max_turns
    )
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
                messages_out=msgs,
            )
        if drain_fn is not None:
            msgs.extend(await drain_fn())
        _trim_tool_results(msgs)
        # fresh each turn: e.g. `verify` drops out once its budget is spent
        tools = session.tools()
        resp = await ctx.llm.chat(
            role,
            msgs,
            json_schema=json_schema,
            tools=tools,
            temperature=temperature,
            on_delta=on_delta,
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
                messages_out=msgs,
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
                # durable pre-execution record — if the tool kills the
                # process (mesh ops OOMed the eval twice), the JSONL still
                # names the exact call that ran last
                await ctx.trace.event(
                    "tool_call",
                    step=step,
                    iteration=iteration,
                    tool=tc.name,
                    args=_short(tc.raw_arguments, 400),
                )
                result = await session.execute(tc.name, tc.arguments)
            msgs.extend(_result_messages(tc, result, native))
            error_streak = error_streak + 1 if not result.get("ok", True) else 0
            # same failure signature again? A verify loop patching around
            # ONE error won't converge — nudge once, kill the session at 3
            if (
                tc.name in ("verify", "render_model")
                and result.get("ok") is False
                and result.get("category")
            ):
                vsig = f"{result.get('category')}:{result.get('line')}"
                verify_fails[vsig] = verify_fails.get(vsig, 0) + 1
                if verify_fails[vsig] == 2 and not verify_nudged:
                    verify_nudged = True
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"verify failed again with the SAME error "
                                f"({result.get('category')} at line "
                                f"{result.get('line')}) — tuning that line won't "
                                "fix it, the cause is EARLIER. Check "
                                "geometry_at_failure for where the bodies "
                                "actually are (a part at the origin whose code "
                                "says otherwise had its placement discarded by "
                                "a bare `Pos * X(...)` — use "
                                "`with Locations(Pos(...)):` or `add(...)`). "
                                "Fix the construction, not the selector; if the "
                                "shape is structurally wrong, call done."
                            ),
                        }
                    )
                elif verify_fails[vsig] >= 3:
                    verify_loop = True
            if session.is_done and tc.name in ("done", "submit_code", "regenerate"):
                return SessionResult(
                    response=resp,
                    submitted=session.submitted,
                    code=session.buffer or None,
                    tool_calls_made=calls,
                    tool_log=session.tool_log,
                    messages_out=msgs,
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
            or turns >= max_turns
            or error_streak >= 4
            or repeat_streak >= 2
            or churn_exceeded
            or verify_loop
        ):
            reason = (
                "budget"
                if calls >= max_calls
                else "turns"
                if turns >= max_turns
                else "error_streak"
                if error_streak >= 4
                else "churn"
                if churn_exceeded
                else "verify_loop"
                if verify_loop
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
                        end_reason=reason,
                        tool_calls_made=calls,
                        tool_log=session.tool_log,
                        messages_out=msgs,
                    )
            return SessionResult(
                response=None,
                submitted=session.submitted,
                code=session.buffer or None,
                exhausted=True,
                end_reason=reason,
                tool_calls_made=calls,
                tool_log=session.tool_log,
                messages_out=msgs,
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
