"""Staged build sessions (kind="build"): add_block / checkpoint / rollback,
the build verify budget, block attribution and done -> submitted code."""

from __future__ import annotations

import uuid

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolloop import run_session
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

SEED = '''"""seed scaffold."""

from build123d import *

PARAMS = {}
'''


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


async def _ok_verify(code: str):
    return {"ok": True, "analysis": {"bbox_size": [10, 10, 5], "is_valid": True}}


def test_build_tools_schema():
    """build kind offers the code suite + build tools; submit_code is gone —
    the buffer itself is the submission."""
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=_ok_verify)
    names = session.allowed_names()
    for t in ("add_block", "checkpoint", "rollback", "verify", "render_model", "done"):
        assert t in names
    for t in (
        "read_lines",
        "search_code",
        "replace_lines",
        "insert_after",
        "insert_before",
        "find_replace",
        "set_param",
        "check_snippet",
        "search_catalog",
        "cadlib_lookup",
        "search_library",
        "view_image",
    ):
        assert t in names
    assert "submit_code" not in names
    assert "check_code" not in names  # snippet check only — buffer check is done's job

    # no verify funding -> verify/render drop out of the advertised schema
    empty = ToolSession(ctx, kind="build", code=SEED)
    names2 = empty.allowed_names()
    assert "verify" not in names2 and "render_model" not in names2
    assert "add_block" in names2


def test_build_verify_budget_uses_build_max():
    """kind='build' spends agent_build_verify_max, not agent_tool_verify_max."""
    ctx = _ctx(FakeLLM([]), agent_tool_verify_max=2, agent_build_verify_max=7)
    build = ToolSession(ctx, kind="build", code=SEED, verify_fn=_ok_verify)
    edit = ToolSession(ctx, kind="edit", code=SEED, verify_fn=_ok_verify)
    assert build.verify_left == 7
    assert edit.verify_left == 2

    # a zero build budget removes verify even with verify_fn wired
    ctx0 = _ctx(FakeLLM([]), agent_build_verify_max=0)
    s0 = ToolSession(ctx0, kind="build", code=SEED, verify_fn=_ok_verify)
    assert s0.verify_left == 0
    assert "verify" not in s0.allowed_names()


async def test_add_block_appends_and_rejects_duplicates():
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="build", code=SEED)

    res = await session.execute("add_block", {"name": "base", "code": "result = Box(10, 10, 5)"})
    assert res["ok"] and res["block"] == "base"
    assert session.edits_applied == 1
    assert "## block: base" in session.buffer
    assert session.buffer.rstrip().endswith("result = Box(10, 10, 5)")

    res = await session.execute("add_block", {"name": "base", "code": "result = Box(1, 1, 1)"})
    assert res["ok"] is False and "already exists" in res["error"]
    assert session.edits_applied == 1  # rejected — no edit counted

    res = await session.execute("add_block", {"name": "", "code": "x = 1"})
    assert res["ok"] is False
    res = await session.execute("add_block", {"name": "holes", "code": "   "})
    assert res["ok"] is False
    assert session.edits_applied == 1


async def test_checkpoint_and_rollback_restore_buffer():
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="build", code=SEED)
    await session.execute("add_block", {"name": "base", "code": "result = Box(10, 10, 5)"})
    res = await session.execute("checkpoint", {"name": "base_ok"})
    assert res["ok"] and res["checkpoint"] == "base_ok"
    saved = session.buffer

    res = await session.execute("checkpoint", {"name": "base_ok"})
    assert res["ok"] is False and "already exists" in res["error"]

    await session.execute("add_block", {"name": "boss", "code": "result = result + Box(4, 4, 3)"})
    assert session.buffer != saved

    res = await session.execute("rollback", {"name": "base_ok"})
    assert res["ok"] and res["restored"] == "base_ok"
    assert session.buffer == saved
    assert session.edits_applied == 3  # two add_blocks + the rollback itself

    res = await session.execute("rollback", {"name": "nope"})
    assert res["ok"] is False and "unknown checkpoint" in res["error"]


async def test_checkpoint_budget_bounded():
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(ctx, kind="build", code=SEED)
    for i in range(8):
        res = await session.execute("checkpoint", {"name": f"cp{i}"})
        assert res["ok"]
    res = await session.execute("checkpoint", {"name": "one_too_many"})
    assert res["ok"] is False and "limit" in res["error"]


async def test_verify_failure_names_the_block():
    """exec failures map the failing line back to its `## block:` section."""
    ctx = _ctx(FakeLLM([]), agent_build_verify_max=4)

    async def failing(code: str):
        n = next(i for i, ln in enumerate(code.splitlines(), 1) if "Cylinder" in ln)
        return {
            "ok": False,
            "error": "ValueError: boom",
            "traceback": f'File "model.py", line {n}, in <module>\n  x',
        }

    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=failing)
    await session.execute("add_block", {"name": "base", "code": "result = Box(10, 10, 5)"})
    await session.execute(
        "add_block",
        {
            "name": "holes",
            "code": "result = result - Pos(0, 0, 2) * Cylinder(radius=1, height=6)",
        },
    )
    res = await session.execute("verify", {})
    assert res["ok"] is False
    assert res["block"] == "holes"
    assert session.verify_left == 3


async def test_scripted_build_flow_submits_buffer():
    """add_block -> verify -> checkpoint -> add_block -> verify -> done."""
    llm = FakeLLM(
        [
            _call("add_block", name="base", code="result = Box(10, 10, 5)"),
            _call("verify"),
            _call("checkpoint", name="base_ok"),
            _call("add_block", name="boss", code="result = result + Pos(0, 0, 5) * Box(4, 4, 3)"),
            _call("verify"),
            _call("done", reasoning="base + boss verified", expected_bbox=[10, 10, 8]),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_build_verify_max=4)
    seen: list[str] = []

    async def verify_fn(code: str):
        seen.append(code)
        return {"ok": True, "analysis": {"bbox_size": [10, 10, 8], "is_valid": True}}

    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=verify_fn)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="codegen")
    assert res.submitted is not None
    code = res.submitted["code"]
    assert "## block: base" in code and "## block: boss" in code
    assert "result = result + Pos(0, 0, 5) * Box(4, 4, 3)" in code
    assert len(seen) == 2
    assert session.verify_left == 2


async def test_done_submits_buffer_to_submitted_code():
    """done without verify still submits the buffer (contract may warn)."""
    llm = FakeLLM(
        [
            _call("add_block", name="base", code="result = Box(10, 10, 5)"),
            _call("done", reasoning="looks right"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    session = ToolSession(ctx, kind="build", code=SEED)
    res = await run_session(ctx, "code", MESSAGES, session=session, max_calls=10, step="codegen")
    assert res.submitted is not None
    assert res.submitted["code"] == session.buffer
    assert "## block: base" in res.submitted["code"]


async def test_codegen_run_uses_build_session_when_staged():
    """codegen.run with agent_staged_build routes to a build session and the
    done-submitted buffer becomes the CodeResponse code."""
    from cadx.agent.steps import codegen as codegen_step
    from cadx.llm.schemas import Dim, PartSpec

    llm = FakeLLM(
        [
            _call("add_block", name="base", code="result = Box(10, 10, 5)"),
            _call("done", reasoning="single box", expected_bbox=[10, 10, 5]),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_staged_build="on")
    spec = PartSpec(
        name="box",
        length_x=Dim(value=10),
        width_y=Dim(value=10),
        height_z=Dim(value=5),
    )
    resp = await codegen_step.run(ctx, spec, None, iteration=1)
    assert "## block: base" in resp.code
    assert "result = Box(10, 10, 5)" in resp.code
    tool_names = llm.calls[0]["tool_names"]
    assert "add_block" in tool_names
    assert "submit_code" not in tool_names


async def test_codegen_run_staged_off_keeps_submit_code():
    """agent_staged_build='off' keeps the classic one-shot codegen session."""
    from cadx.agent.steps import codegen as codegen_step
    from cadx.llm.schemas import Dim, PartSpec

    llm = FakeLLM(
        [
            _call(
                "submit_code",
                code="from build123d import *\nPARAMS = {}\nresult = Box(1, 1, 1)",
                reasoning="one-shot",
            ),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_staged_build="off")
    spec = PartSpec(
        name="box",
        length_x=Dim(value=10),
        width_y=Dim(value=10),
        height_z=Dim(value=5),
    )
    resp = await codegen_step.run(ctx, spec, None, iteration=1)
    assert "result = Box(1, 1, 1)" in resp.code
    assert "submit_code" in llm.calls[0]["tool_names"]
    assert "add_block" not in llm.calls[0]["tool_names"]
