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
    even though the call budget is far from spent."""
    llm = FakeLLM([_call("check_code")] * 40)  # check tools reset churn
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_max_turns=7)
    session = ToolSession(ctx, kind="edit", code=BOX_CODE)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=50, step="improve")
    assert res.exhausted
    assert len(llm.calls) <= 8  # 7 turns + optional grace


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
