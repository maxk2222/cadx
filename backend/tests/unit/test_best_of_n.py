"""Best-of-N first codegen (agent_codegen_samples > 1): N parallel candidate
sessions, in-session verify evidence reuse, scoring and non-fatal failures.

Under FakeLLM nothing ever truly suspends, so asyncio.gather runs each
candidate session to completion in order — the script is consumed
candidate-by-candidate, making deterministic scripting possible.
"""

from __future__ import annotations

import uuid

import pytest

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.steps import codegen as codegen_step
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Dim, PartSpec
from cadx.llm.structured import LLMFormatError
from cadx.runtime_config import RuntimeSettings

GOOD = "from build123d import *\nPARAMS = {}\nresult = Box(10, 10, 5)"
GOOD_LONG = GOOD + "\n\n# deliberate padding to lose the shorter-code tiebreak\n" + ("# x\n" * 20)
# contract-broken: no `result` assignment — accepted only after the done
# gate's two re-asks are spent
BAD = "from build123d import *\nPARAMS = {}\nx = Box(10, 10, 5)"


def _ctx(llm: FakeLLM, emitted: list[str] | None = None, **settings_kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        if emitted is not None:
            emitted.append(msg)

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


def _spec() -> PartSpec:
    return PartSpec(
        name="box",
        length_x=Dim(value=10),
        width_y=Dim(value=10),
        height_z=Dim(value=5),
    )


def _best_of_n_events(ctx: AgentCtx) -> list[dict]:
    return [r for r in ctx.trace.records if r.get("event") == "codegen.best_of_n"]


async def test_best_of_n_picks_good_candidate():
    """n=3, staged off: candidate 0 submits contract-broken code (kept after
    the submit gate's re-asks), candidates 1+2 submit clean code — the
    verified clean one wins, shorter code breaks the tie."""
    llm = FakeLLM(
        [
            # candidate 0: broken contract, two re-asks then accepted anyway
            _call("submit_code", code=BAD, reasoning="broken"),
            _call("submit_code", code=BAD, reasoning="still broken"),
            _call("submit_code", code=BAD, reasoning="gives up"),
            # candidate 1: clean + short
            _call("submit_code", code=GOOD, reasoning="good"),
            # candidate 2: clean but longer — loses the tiebreak
            _call("submit_code", code=GOOD_LONG, reasoning="long good"),
        ]
    )
    llm.tools_mode = "auto"
    emitted: list[str] = []
    ctx = _ctx(llm, emitted, agent_codegen_samples=3, agent_staged_build="off")

    async def verify_fn(code: str):
        return {"ok": "result = Box(10, 10, 5)" in code}

    resp = await codegen_step.run(ctx, _spec(), None, iteration=1, verify_fn=verify_fn)
    assert resp.code == GOOD

    events = _best_of_n_events(ctx)
    assert len(events) == 1
    ev = events[0]
    assert ev["n"] == 3 and ev["winner"] == 1
    scores = {c["i"]: c for c in ev["candidates"]}
    assert scores[0]["verified"] is False  # broken code failed the light exec
    assert scores[0]["contract_errors"] > 0
    assert scores[1]["verified"] is True
    assert scores[2]["verified"] is True
    assert any("best of 3" in m for m in emitted)


async def test_single_sample_is_unchanged():
    """n=1 (and default) keeps the plain single-session path — no tournament
    trace event, one chat call, submitted code straight through."""
    llm = FakeLLM([_call("submit_code", code=GOOD, reasoning="one-shot")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_codegen_samples=1, agent_staged_build="off")
    resp = await codegen_step.run(ctx, _spec(), None, iteration=1)
    assert resp.code == GOOD
    assert len(llm.calls) == 1
    assert _best_of_n_events(ctx) == []


async def test_best_of_n_only_first_iteration():
    """iteration>1 stays single-shot even with samples>1 — repairs/improves
    and follow-up attempts are conditioned on a specific failure."""
    llm = FakeLLM([_call("submit_code", code=GOOD, reasoning="retry")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_codegen_samples=4, agent_staged_build="off")
    resp = await codegen_step.run(
        ctx, _spec(), None, iteration=2, feedback="previous attempt failed"
    )
    assert resp.code == GOOD
    assert len(llm.calls) == 1
    assert _best_of_n_events(ctx) == []


async def test_failing_candidate_is_skipped():
    """A candidate whose session raises (LLMFormatError on a non-JSON final
    answer) is skipped — the surviving good candidate still wins."""
    llm = FakeLLM(
        [
            {"content": "this is not JSON and cannot parse as CodeResponse"},
            _call("submit_code", code=GOOD, reasoning="good"),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_codegen_samples=2, agent_staged_build="off")
    resp = await codegen_step.run(ctx, _spec(), None, iteration=1)
    assert resp.code == GOOD
    ev = _best_of_n_events(ctx)[0]
    # only the surviving candidate is scored
    assert [c["i"] for c in ev["candidates"]] == [1]


async def test_all_candidates_failing_raises():
    """When EVERY candidate session fails the first error is surfaced —
    identical to what a single attempt would have raised."""
    llm = FakeLLM(
        [
            {"content": "garbage one"},
            {"content": "garbage two"},
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_codegen_samples=2, agent_staged_build="off")
    with pytest.raises(LLMFormatError):
        await codegen_step.run(ctx, _spec(), None, iteration=1)


async def test_in_session_verify_is_reused_and_preferred():
    """Staged build candidates: A grows a longer buffer and done()s without
    verify; B verifies (PASS) then done()s. B's in-session verdict is reused
    — it is NOT re-executed post-hoc — and B wins over A's longer code even
    though A's code is contract-clean."""
    llm = FakeLLM(
        [
            # candidate 0: two blocks, no verify, done — longer final code
            _call("add_block", name="base", code="result = Box(10, 10, 5)"),
            _call("add_block", name="boss", code="result = result + Pos(0, 0, 5) * Box(4, 4, 3)"),
            _call("done", reasoning="no verify", expected_bbox=[10, 10, 8]),
            # candidate 1: one block, verify, done — verified PASS in-session
            _call("add_block", name="base", code="result = Box(8, 8, 4)"),
            _call("verify"),
            _call("done", reasoning="verified", expected_bbox=[8, 8, 4]),
        ]
    )
    llm.tools_mode = "auto"
    ctx = _ctx(llm, agent_codegen_samples=2, agent_staged_build="on")
    seen: list[str] = []

    async def verify_fn(code: str):
        seen.append(code)
        # only B's box executes cleanly
        return {"ok": "Box(8, 8, 4)" in code}

    resp = await codegen_step.run(ctx, _spec(), None, iteration=1, verify_fn=verify_fn)
    assert "Box(8, 8, 4)" in resp.code
    assert "## block: base" in resp.code

    # verify_fn ran exactly twice: B inside its own session (during gather),
    # then A as the post-hoc light check (A never verified in-session). B's
    # PASS was reused — no second call for its code.
    assert len(seen) == 2
    assert "Box(8, 8, 4)" in seen[0]  # in-session, during candidate 1's session
    assert "Box(10, 10, 5)" in seen[1]  # post-hoc on candidate 0's code

    ev = _best_of_n_events(ctx)[0]
    scores = {c["i"]: c for c in ev["candidates"]}
    assert scores[1]["verified"] is True and scores[1]["verify_source"] == "session"
    assert scores[0]["verified"] is False and scores[0]["verify_source"] == "exec"
    assert ev["winner"] == 1
