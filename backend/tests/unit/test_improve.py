"""improve step: attempt-memory block, patch-error variants, exhaustion latch."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.steps import improve
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import PartSpec
from cadx.runtime_config import RuntimeSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()

SPEC = PartSpec(
    name="test block",
    summary="40x20x10 block",
    units="mm",
    length_x={"value": 40.0, "unit": "mm", "source": "stated"},
    width_y={"value": 20.0, "unit": "mm", "source": "stated"},
    height_z={"value": 10.0, "unit": "mm", "source": "stated"},
    features=[],
    open_questions=[],
)

ISSUES = [{"source": "critique", "severity": "major", "what": "missing fillet"}]


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
        prompt="box",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(**settings_kw),
    )


def _call(tool: str, **args) -> dict:
    return {"tool_calls": [{"name": tool, "arguments": args}]}


def _text(parts: list[dict]) -> str:
    return json.dumps(parts)


def test_user_parts_attempts_block():
    """The attempt memory is injected so identical issue+code prompts cannot
    deterministically regenerate the same failed patch."""
    parts = improve._user_parts(
        _ctx(FakeLLM([])),
        SPEC,
        None,
        BOX_CODE,
        ISSUES,
        None,
        None,
        attempts=["it1 improve: line 5: 'r=2' -> 'r=3'"],
    )
    text = _text(parts)
    assert "PREVIOUS ATTEMPTS" in text
    assert "do NOT repeat" in text
    assert "r=3" in text


def test_user_parts_patch_errors_oneshot_asks_full_code():
    parts = improve._user_parts(
        _ctx(FakeLLM([])), SPEC, None, BOX_CODE, ISSUES, None, ["find 'x' not found"]
    )
    assert "full_code" in _text(parts)


def test_user_parts_patch_errors_session_warns_not_full_code():
    """Inside a tool session 'return full_code' is wrong advice — the warning
    must tell the model not to repeat the missed edits instead."""
    parts = improve._user_parts(
        _ctx(FakeLLM([])),
        SPEC,
        None,
        BOX_CODE,
        ISSUES,
        None,
        ["find 'x' not found"],
        numbered=True,
    )
    text = _text(parts)
    assert "do not repeat" in text
    assert "full_code" not in text


async def test_session_exhaustion_budget_does_not_latch():
    """A session that merely ran out of budget still produced a buffer — it
    must NOT blind every subsequent repair to the one-shot path."""
    llm = FakeLLM([_call("read_lines", start=i * 10 + 1, end=i * 10 + 5) for i in range(14)])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_churn_limit=4)
    presp = await improve.run(
        ctx,
        SPEC,
        None,
        base_code=BOX_CODE,
        issues=ISSUES,
        iteration=2,
        verify_fn=None,
    )
    assert presp is not None
    assert ctx.edit_sessions_exhausted == 0


async def test_session_exhaustion_verify_loop_latches():
    """verify_loop means the session interaction pattern is broken — THAT is
    the case the blind one-shot fallback exists for."""
    async def failing_verify(code: str):
        return {
            "ok": False,
            "error": "ValueError: Failed creating a fillet with radius of 3.25",
            "traceback": 'File "model.py", line 32, in <module>\n  fillet(...)',
        }

    llm = FakeLLM([_call("verify") for _ in range(8)])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_tool_verify_max=10)
    await improve.run(
        ctx,
        SPEC,
        None,
        base_code=BOX_CODE,
        issues=ISSUES,
        iteration=2,
        verify_fn=failing_verify,
    )
    assert ctx.edit_sessions_exhausted == 1


async def test_attempts_reach_session_prompt():
    """End-to-end: the attempts list lands inside the tool-session user
    message, not just the one-shot path."""
    llm = FakeLLM([_call("done", reasoning="no change")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    await improve.run(
        ctx,
        SPEC,
        None,
        base_code=BOX_CODE,
        issues=ISSUES,
        attempts=["it1 improve: line 9: 'x=1' -> 'x=2'"],
        iteration=2,
        verify_fn=None,
    )
    msgs = llm.calls[0]["messages"]
    assert "PREVIOUS ATTEMPTS" in json.dumps(msgs)
