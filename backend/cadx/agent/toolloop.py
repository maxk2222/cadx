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

from cadx.agent.thinking import thinking_body, thinking_on
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


# --------------------------------------------------------------------------- #
# deterministic history condensation (OpenHands-condenser pattern)             #
#                                                                             #
# Long tool sessions resend the whole history every turn; the resend grows     #
# unboundedly and APC can only reuse the stable head. Once the exchange count  #
# passes `agent_session_condense_turns` the middle of the message list is      #
# replaced by ONE digest message: [system + first user seed] + [digest] +      #
# [last ~half of the exchanges verbatim]. The digest is rebuilt from           #
# session.tool_log every time, so repeated condenses stay consistent.          #

# seed head that always survives verbatim: msgs[0] system prompt + msgs[1]
# first user message (spec/request) — every caller seeds exactly this pair
_CONDENSE_HEAD = 2
_DIGEST_MARKER = "[SESSION DIGEST — earlier exchanges condensed]"


def _exchange_units(msgs: list[dict[str, Any]], start: int) -> list[list[int]]:
    """Segment msgs[start:] into turn units (index lists). Each unit is one
    assistant message plus everything up to the next assistant message — all
    its tool results and any image/nudge follow-ups — or a run of
    non-assistant messages (drained user text, a previous digest). Units are
    the atomic condense granule: a tool_call/tool_result pair can never be
    split across them."""
    units: list[list[int]] = []
    cur: list[int] = []
    for i in range(start, len(msgs)):
        if msgs[i].get("role") == "assistant":
            if cur:
                units.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        units.append(cur)
    return units


def _unit_tool_calls(msgs: list[dict[str, Any]], unit: list[int]) -> int:
    """Tool-result messages inside one unit — one per executed call, matching
    session.tool_log entries 1:1 (native `role:"tool"` messages and
    content-fallback `TOOL RESULT for ...` user messages)."""
    n = 0
    for i in unit:
        m = msgs[i]
        if m.get("role") == "tool" or (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("TOOL RESULT for")
        ):
            n += 1
    return n


def _one_line(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _digest_call(name: str, args: dict[str, Any]) -> str:
    """`add_block(base)`-style short rendering of one logged call."""
    for key in ("name", "query", "pattern"):
        v = args.get(key)
        if isinstance(v, str | int | float) and str(v):
            return f"{name}({_one_line(str(v), 24)})"
    for k, v in args.items():
        if k in ("code", "_raw") or not isinstance(v, str | int | float | bool):
            continue
        return f"{name}({k}={_one_line(str(v), 24)})"
    return name


_METRIC_KEYS = (
    "volfrac_achieved",
    "volfrac_target",
    "safety_factor",
    "worst_safety_factor",
    "sf_export",
    "sf_mask",
    "bbox_size",
    "volume_mm3",
    "watertight",
    "measured_dia_mm",
    "dia_mm",
    "faces",
    "error",
)


def _metrics_hint(res: dict[str, Any]) -> str:
    """Compact `k=v` tail for digested tool results — optimize/mesh sessions
    must not lose the numbers `done(run_index)`/issue fixes compare on.
    Scans the result plus its `topopt`/`fem`/`analysis` sub-dicts for the
    known metric keys (first 3 wins)."""
    found: list[str] = []
    scopes: list[dict[str, Any]] = [res] + [
        res.get(k) or {} for k in ("topopt", "fem", "analysis", "report")
    ]
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        for k in _METRIC_KEYS:
            v = scope.get(k)
            if v is None or isinstance(v, dict | list) or f"{k}=" in ";".join(found):
                continue
            found.append(f"{k}={_one_line(str(v), 40)}")
            if len(found) >= 3:
                return "; ".join(found)
    return "; ".join(found)


def _digest_line(entry: dict[str, Any]) -> str:
    """One `call(args) -> ok(hint)` / `-> fail: <category> <one-line error>` line."""
    call = _digest_call(str(entry.get("name") or "?"), entry.get("args") or {})
    res = entry.get("result")
    if isinstance(res, str):
        # _truncate() degraded a huge result to a string — classify by its head
        if '"ok": false' not in res[:60]:
            return f"{call} -> ok"
        return f"{call} -> fail: {_one_line(res, 140)}"
    if not isinstance(res, dict) or res.get("ok", True):
        hint = _metrics_hint(res) if isinstance(res, dict) else ""
        return f"{call} -> ok" + (f" ({hint})" if hint else "")
    tail = " ".join(
        p
        for p in (
            _one_line(str(res.get("category") or ""), 40),
            _one_line(str(res.get("error") or ""), 140),
        )
        if p
    )
    return f"{call} -> fail: {tail}".rstrip()


async def _condense_history(
    ctx: AgentCtx, session: ToolSession, msgs: list[dict[str, Any]], step: str
) -> None:
    """Replace the middle of `msgs` with a one-message digest, in place.

    Fires once the middle holds more than `agent_session_condense_turns`
    exchange units; keeps the system+seed head and the last ~half of the
    exchanges verbatim. The digest covers every tool call NOT in the kept
    tail (from session.tool_log — append-only, so re-condenses rebuild a
    complete digest) plus the build session's current block list, so the
    model never loses track of what exists."""
    threshold = int(ctx.runtime_settings.agent_session_condense_turns or 0)
    if threshold <= 0 or len(msgs) <= _CONDENSE_HEAD:
        return
    units = _exchange_units(msgs, _CONDENSE_HEAD)
    keep = max(4, threshold // 2)
    # only fire when something would actually be dropped — tiny thresholds
    # (<8) keep more than they trigger at
    if len(units) <= threshold or len(units) <= keep:
        return
    tail = units[-keep:]
    tail_calls = sum(_unit_tool_calls(msgs, u) for u in tail)
    # tool_log is append-only and matches result messages 1:1 — everything
    # not still verbatim in the tail belongs in the digest
    entries = session.tool_log[: max(0, len(session.tool_log) - tail_calls)]
    lines = [
        _DIGEST_MARKER,
        "The middle of this session was condensed to keep context small — the "
        "full tool results below are gone, only the outcome per call remains:",
        *(_digest_line(e) for e in entries),
    ]
    if session.kind == "build":
        # the model must never lose track of which feature blocks exist
        blocks = session._block_names()
        lines.append("BLOCKS IN BUFFER NOW: " + (", ".join(blocks) if blocks else "(none)"))
    lines.append(
        "Live session state (code buffer, checkpoints) is unchanged — use "
        "read_lines/search_code to inspect it rather than dropped results."
    )
    tail_msgs = [msgs[i] for u in tail for i in u]
    before_len = len(msgs)
    msgs[_CONDENSE_HEAD:] = [{"role": "user", "content": "\n".join(lines)}, *tail_msgs]
    log.info(
        "tool_session.condensed",
        step=step,
        before_len=before_len,
        after_len=len(msgs),
    )
    await ctx.trace.event(
        "tool_session.condensed", step=step, before_len=before_len, after_len=len(msgs)
    )


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
        if drain_fn is None:
            # OpenHands-style condensation bounds the resend on long sessions.
            # Chat sessions (drain_fn set) skip it: they persist
            # messages_out[n0:] positionally, so rewriting the middle here
            # would corrupt that slice — _trim_history bounds them instead.
            await _condense_history(ctx, session, msgs, step)
        # fresh each turn: e.g. `verify` drops out once its budget is spent
        tools = session.tools()
        # resolved per turn — the escalation latch may flip mid-session
        thinking = thinking_body(ctx, step)
        resp = await ctx.llm.chat(
            role,
            msgs,
            json_schema=json_schema,
            tools=tools,
            temperature=temperature,
            on_delta=on_delta,
            extra_body=thinking,
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
                "thinking": thinking_on(ctx, step),
                # vLLM APC: how much of the prompt prefix was already cached —
                # only recorded on a hit so trace rows stay clean
                **({"cached_tokens": resp.cached_tokens} if resp.cached_tokens else {}),
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
            extra_body=thinking_body(ctx, step),
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
        parsed={"grace_turn": True, "thinking": thinking_on(ctx, step)},
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    return resp


def _short(raw: str, n: int = 80) -> str:
    return raw if len(raw) <= n else raw[:n] + "…"
