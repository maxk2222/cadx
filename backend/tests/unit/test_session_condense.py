"""M9 session history condensation + M7 prompt-ordering / cache-hit checks.

Condense: a scripted FakeLLM session that exceeds
`agent_session_condense_turns` must rewrite its history to
[system + first user seed] + [one digest message] + [verbatim tail] — never
splitting a tool_call/tool_result pair, keeping the build block list visible,
and rebuilding a complete digest on repeated condenses. 0 disables.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolloop import run_session
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()

SEED = '''"""seed scaffold."""

from build123d import *

PARAMS = {}
'''

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]


def _ctx(llm: FakeLLM, **settings_kw: Any) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="test",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(**settings_kw),
    )


def _call(tool: str, **args) -> dict:
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _digest_msgs(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m
        for m in msgs
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("[SESSION DIGEST")
    ]


def _assert_no_orphaned_tool_results(msgs: list[dict[str, Any]]) -> None:
    """Every role:"tool" message must answer a tool_call of the assistant
    message directly preceding its result block — condensing must never drop
    an assistant call while keeping one of its results."""
    open_ids: set[str] = set()
    for m in msgs:
        if m.get("role") == "assistant":
            open_ids = {tc["id"] for tc in (m.get("tool_calls") or [])}
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in open_ids, f"orphaned tool result {m['tool_call_id']}"
            open_ids.discard(m["tool_call_id"])


async def test_condense_replaces_middle_with_digest():
    """Build session past the threshold: system+seed stay byte-identical, the
    middle becomes ONE digest built from tool_log, the tail stays verbatim,
    and the digest carries the live block list."""
    llm = FakeLLM(
        [
            _call("add_block", name="base", code="result = Box(10, 10, 5)"),
            _call("verify"),
            _call(
                "add_block",
                name="boss",
                code="result = result + Pos(0, 0, 5) * Box(4, 4, 3)",
            ),
            _call("verify"),
            _call("add_block", name="holes", code="# holes placeholder"),
            _call("verify"),
            _call("done", reasoning="all blocks in"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_session_condense_turns=4)
    n = {"n": 0}

    async def verify_fn(code: str):
        n["n"] += 1
        if n["n"] == 1:
            return {
                "ok": False,
                "error": "ValueError: boom",
                "traceback": 'File "model.py", line 2, in <module>',
            }
        return {"ok": True, "analysis": {"bbox_size": [10, 10, 8], "is_valid": True}}

    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=verify_fn)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="codegen")
    assert res.submitted is not None

    # condense fired and each rewrite shrank the list
    events = [r for r in ctx.trace.records if r.get("event") == "tool_session.condensed"]
    assert events
    assert all(e["before_len"] > e["after_len"] for e in events)
    assert all(e["step"] == "codegen" for e in events)

    last = llm.calls[-1]["messages"]
    # the seed head is untouched
    assert last[0] == MESSAGES[0] and last[1] == MESSAGES[1]
    # exactly one digest message — a re-condense rebuilds it, never stacks
    digests = _digest_msgs(last)
    assert len(digests) == 1
    text = digests[0]["content"]
    # dropped exchanges show up as one-line outcomes from the tool_log
    assert "add_block(base) -> ok" in text
    assert "verify -> fail:" in text
    assert "ValueError: boom" in text
    # the model never loses track of what the buffer already contains
    assert "BLOCKS IN BUFFER NOW: base, boss, holes" in text
    # the tail is verbatim exchanges: real assistant/tool pairs still there
    assert any(m.get("role") == "assistant" for m in last[2:])
    assert any(m.get("role") == "tool" for m in last[2:])
    # no call ever saw a tool_call/result pair split by the rewrite
    for c in llm.calls:
        _assert_no_orphaned_tool_results(c["messages"])


async def test_condense_digest_carries_state_lines():
    """The digest footer synthesizes STATE: last verify outcome, accepted
    submissions, per-kind progress — so the post-condense model continues
    instead of re-exploring (observed: a build session re-ran
    search_library and re-added an existing block)."""
    from cadx.agent.toolloop import _digest_state_lines

    ctx = _ctx(FakeLLM([{"content": "x"}]))
    session = ToolSession(ctx, kind="build", code=SEED)
    entries = [
        {"name": "add_block", "args": {"name": "base"}, "result": {"ok": True}},
        {
            "name": "verify",
            "args": {},
            "result": {
                "ok": True,
                "analysis": {
                    "bbox_size": [10, 10, 8],
                    "watertight": True,
                    "features": {"total_holes": 2},
                },
            },
        },
        {"name": "done", "args": {}, "result": {"ok": True}},
    ]
    lines = _digest_state_lines(session, entries)
    blob = "\n".join(lines)
    assert "STATE last verify: OK" in blob
    assert "watertight=True" in blob and "10x10x8" in blob and "holes=2" in blob
    assert "already ACCEPTED" in blob

    # optimize sessions summarize runs for done(run_index)
    session2 = ToolSession(ctx, kind="optimize")
    session2.struct_runs.append(
        {
            "index": 0,
            "res": {"topopt": {"sf_export": 2.4, "volfrac_achieved": 0.41}},
            "params": {},
        }
    )
    lines2 = _digest_state_lines(session2, [])
    assert any("run0" in ln and "sf_export=2.4" in ln for ln in lines2)

    # a failed verify surfaces the error, not a stale OK
    bad = [{"name": "verify", "args": {}, "result": {"ok": False, "category": "boom"}}]
    lines3 = _digest_state_lines(ToolSession(ctx, kind="edit", code=SEED), bad)
    assert any("FAILED" in ln and "boom" in ln for ln in lines3)


async def test_condense_disabled_at_zero():
    """agent_session_condense_turns=0 keeps the full growing history."""
    llm = FakeLLM(
        [_call("read_lines", start=i * 5 + 1, end=i * 5 + 4) for i in range(8)]
        + [{"content": "done reading"}]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_session_condense_turns=0)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=20, step="improve")
    assert res.response is not None
    last = llm.calls[-1]["messages"]
    assert not _digest_msgs(last)
    assert len(last) > 12  # full history still resent
    assert not [r for r in ctx.trace.records if r.get("event") == "tool_session.condensed"]


async def test_condense_below_threshold_is_noop():
    """A session inside the turn budget never condenses."""
    llm = FakeLLM([_call("read_lines", start=1, end=4), _call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_session_condense_turns=12)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="improve")
    for c in llm.calls:
        assert not _digest_msgs(c["messages"])
    assert not [r for r in ctx.trace.records if r.get("event") == "tool_session.condensed"]


async def test_cached_tokens_reach_response_and_trace():
    """M7: usage.prompt_tokens_details.cached_tokens lands on LLMResponse and
    into the step trace record (only when nonzero)."""
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {"name": "set_param", "arguments": {"name": "fillet_r", "value": 2.0}}
                ],
                "cached_tokens": 17,
            },
            {"tool_calls": [{"name": "done", "arguments": {}}]},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=3, step="improve")
    assert res.submitted is not None
    rec = next(r for r in ctx.trace.records if r.get("step") == "improve" and "parsed" in r)
    assert rec["parsed"]["cached_tokens"] == 17

    # a cache miss leaves no field — the trace row stays clean
    llm2 = FakeLLM([{"content": "plain answer"}])
    ctx2 = _ctx(llm2)
    session2 = ToolSession(ctx2, kind="edit", code=BOX_CODE)
    await run_session(ctx2, "code", MESSAGES, session=session2, max_calls=3, step="improve")
    rec2 = next(r for r in ctx2.trace.records if r.get("step") == "improve" and "parsed" in r)
    assert "cached_tokens" not in rec2["parsed"]


async def test_repair_prompt_keeps_dynamic_content_last():
    """M7 APC ordering: the per-attempt code + diagnosis must be the LAST
    part of the repair user message — request/spec/context stay byte-stable
    at the head across repair_1..N so prefix caching can reuse them."""
    from cadx.agent.steps import codegen as codegen_step
    from cadx.llm.schemas import Dim, PartSpec

    dim = Dim(value=60.0)
    spec = PartSpec(name="box", length_x=dim, width_y=dim, height_z=dim)
    llm = FakeLLM([{"reasoning": "r", "edits": [{"find": "a", "replace": "b"}]}])
    ctx = _ctx(llm)
    await codegen_step.repair(
        ctx,
        "result = Box(1, 1, 1)",
        spec,
        None,
        "RuntimeError: boom",
        None,
        None,
        iteration=1,
        repair_n=1,
    )
    parts = llm.calls[0]["messages"][1]["content"]
    texts = [p["text"] for p in parts]
    assert "PREVIOUS CODE" in texts[-1]
    assert "boom" in texts[-1]  # the diagnosis rides in the dynamic tail too
    assert any("PART SPEC" in t for t in texts[:-1])
