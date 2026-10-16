"""Turn-cap / churn-guard / context-trimming for tool sessions (T2.1, T2.2)."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolloop import _trim_tool_results, run_session
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()


def _ctx(llm: FakeLLM, **settings_kw) -> AgentCtx:
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


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]


async def test_churn_guard_nudges_then_exhausts():
    """read_lines-only turns: one nudge at the churn limit, then the
    session ends as exhausted when the churn continues."""
    llm = FakeLLM([_call("read_lines", start=i * 10 + 1, end=i * 10 + 5) for i in range(14)])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_churn_limit=4)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=50, step="improve")
    assert res.exhausted
    # nudge went out exactly once (it stays in history), then the guard
    # stopped the session
    nudges = [
        m
        for m in llm.calls[-1]["messages"]
        if m.get("role") == "user" and "without checking" in str(m.get("content", ""))
    ]
    assert len(nudges) == 1


async def test_turn_cap_ends_session():
    """A session that keeps making progress-free calls stops at max_turns
    even though the call budget is far from spent. edit sessions get the
    tighter edit cap (each verify pays the full sandbox pipeline)."""

    async def ok_verify(code: str):
        return {"ok": True}

    # alternate check tools: identical repeated calls would trip the
    # repeat-loop detector before the turn cap is even reached
    llm = FakeLLM([_call("check_code" if i % 2 else "verify") for i in range(40)])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_max_turns=40, agent_tool_max_turns_edit=7, agent_tool_verify_max=20)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=ok_verify)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=50, step="improve")
    assert res.exhausted
    assert res.end_reason == "turns"
    assert len(llm.calls) <= 8  # 7 turns + optional grace


async def test_verify_loop_detector_nudges_then_kills():
    """3 failed verifies with the same (category, line) signature end the
    session — the model was ping-ponging edits around one error."""
    calls_n = {"n": 0}

    async def failing_verify(code: str):
        calls_n["n"] += 1
        # the session re-diagnoses from error+traceback — keep both stable
        return {
            "ok": False,
            "error": "ValueError: Failed creating a fillet with radius of 3.25",
            "traceback": 'File "model.py", line 32, in <module>\n  fillet(...)',
        }

    llm = FakeLLM([_call("verify") for _ in range(8)])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_verify_max=10)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=failing_verify)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=50, step="repair_1")
    assert res.exhausted
    assert res.end_reason == "verify_loop"
    assert calls_n["n"] == 3
    nudges = [
        m
        for m in llm.calls[-1]["messages"]
        if m.get("role") == "user" and "SAME error" in str(m.get("content", ""))
    ]
    assert len(nudges) == 1


async def test_verify_loop_different_errors_do_not_abort():
    """Failures with different signatures must not trip the loop detector.
    (verify results get re-diagnosed from the error text, so the sequence
    needs realistically distinct failure classes.)"""
    seq = [
        {
            "ok": False,
            "error": "AttributeError: x has no attribute 'y'",
            "traceback": 'File "model.py", line 5',
        },
        {
            "ok": False,
            "error": "BRep_API: command not done",
            "traceback": 'File "model.py", line 9',
        },
        {"ok": False, "error": "NameError: name 'q' is not defined", "traceback": None},
        {
            "ok": False,
            "error": "TimeoutError: execution timed out",
            "traceback": 'File "model.py", line 20',
        },
    ]
    it = iter(seq)

    async def varying_verify(code: str):
        return next(it, {"ok": True})

    # interleave reads so identical verify() calls don't trip repeat_loop
    script = [c for _ in range(4) for c in (_call("verify"), _call("read_lines", start=1, end=3))]
    llm = FakeLLM([*script, {"text": "done"}])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_verify_max=10)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE, verify_fn=varying_verify)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=50, step="repair_1")
    assert not res.exhausted
    assert res.response is not None


def test_trim_tool_results_compacts_old_verify_failures():
    import json as _json

    body = _json.dumps(
        {
            "ok": False,
            "category": "occ-fillet-empty",
            "line": 32,
            "excerpt": ">>> " + "x" * 500,
            "error": "e" * 300,
        }
    )
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for _ in range(5):
        msgs.append({"role": "tool", "tool_call_id": "c", "content": body})
    _trim_tool_results(msgs, keep_last=10)
    elided = [m for m in msgs if m.get("role") == "tool" and '"elided": true' in m["content"]]
    full = [m for m in msgs if m.get("role") == "tool" and "excerpt" in m["content"]]
    assert len(elided) == 3
    assert len(full) == 2


def test_trim_tool_results_elides_old_reads():
    big = '{"lines": ["' + "x" * 500 + '"]}'
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
    ]
    for _ in range(10):
        msgs.append({"role": "tool", "tool_call_id": "c", "content": big})
    _trim_tool_results(msgs, keep_last=6)
    full = [m for m in msgs if m.get("role") == "tool" and len(m["content"]) > 200]
    elided = [m for m in msgs if m.get("role") == "tool" and "elided" in m["content"]]
    assert len(full) == 6
    assert len(elided) == 4
    assert msgs[0]["content"] == "sys" and msgs[1]["content"] == "go"


async def test_trim_runs_during_session():
    """After many read_lines turns the older results are stubbed before the
    next LLM call — the model never sees the full history grow unbounded."""
    llm = FakeLLM(
        [_call("read_lines", start=i * 10 + 1, end=i * 10 + 50) for i in range(10)]
        + [{"text": "done"}]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    await run_session(ctx, "code", MESSAGES, session=session, max_calls=20, step="improve")
    last_msgs = llm.calls[-1]["messages"]
    big = [m for m in last_msgs if m.get("role") == "tool" and len(str(m.get("content", ""))) > 400]
    assert len(big) <= 6
