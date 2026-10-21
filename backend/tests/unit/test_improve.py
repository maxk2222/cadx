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
from cadx.llm.schemas import Feature, PartSpec
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


def test_feature_audit_flags_probe_stub():
    """The observed stub: load_mesh_solid + prints only, spec declares lugs
    and bores — the audit must reject it before critique burns iterations."""
    from cadx.agent.steps.codegen import _feature_audit

    stub = (
        "from build123d import *\nimport cadlib\n"
        "PARAMS = {'hole_diameter': 5.5}\n"
        "mesh = load_mesh_solid()\nprint(measure(mesh))\nresult = mesh\n"
    )
    spec = PartSpec(
        name="handle",
        length_x={"value": 120.0},
        width_y={"value": 40.0},
        height_z={"value": 30.0},
        features=[
            Feature(name="arch", type="arch", dims={}),
            Feature(name="lug -x", type="pad", dims={}),
            Feature(name="lug +x", type="pad", dims={}),
            Feature(name="bore -x", type="through_hole", dims={"diameter": 5.5}),
        ],
    )
    assert _feature_audit(stub, spec)


def test_feature_audit_accepts_real_build():
    from cadx.agent.steps.codegen import _feature_audit

    code = (
        "mesh = load_mesh_solid()\n"
        "lug = Pos(-45,0,3) * Box(30,20,6)\n"
        "body = cadlib.mesh_fuse(mesh, lug)\n"
        "body = cadlib.mesh_cut(body, Pos(-45,0,-1) * Cylinder(radius=2.75, height=12))\n"
        "result = body\n"
    )
    spec = PartSpec(
        name="handle",
        length_x={"value": 120.0},
        width_y={"value": 40.0},
        height_z={"value": 30.0},
        features=[
            Feature(name="lug", type="pad", dims={}),
            Feature(name="bore", type="through_hole", dims={}),
        ],
    )
    assert _feature_audit(code, spec) == []


async def test_artifact_and_infra_issues_spawn_no_session():
    """Artifact-flagged repeats and infra errors carry no code change —
    when nothing actionable remains, improve must not spawn a session at
    all (no LLM call, no tool budget spent)."""
    llm = FakeLLM([_call("done", reasoning="x")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    presp = await improve.run(
        ctx,
        SPEC,
        None,
        base_code=BOX_CODE,
        issues=[
            {
                "source": "critique",
                "severity": "minor",
                "what": "edge looks off",
                "artifact": True,
            },
            {
                "source": "infra",
                "severity": "major",
                "what": "sandbox infrastructure error — not a code problem",
            },
        ],
        iteration=2,
    )
    assert "no actionable" in presp.reasoning
    assert llm.calls == []


async def test_mixed_issues_still_reach_session():
    """Gating drops the noise but keeps real issues — a mixed list still
    drives a session on the actionable remainder."""
    llm = FakeLLM([_call("done", reasoning="fixed the fillet")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    presp = await improve.run(
        ctx,
        SPEC,
        None,
        base_code=BOX_CODE,
        issues=[
            {"source": "critique", "severity": "major", "what": "missing fillet"},
            {"source": "infra", "severity": "major", "what": "sandbox schema skew"},
        ],
        iteration=2,
    )
    assert presp is not None
    assert llm.calls  # a session actually spawned
    # the session prompt only carries the actionable issue
    assert "missing fillet" in json.dumps(llm.calls[0]["messages"])
    assert "schema skew" not in json.dumps(llm.calls[0]["messages"])


async def test_last_good_rescues_broken_head_buffer(monkeypatch):
    """P6: a session that exhausted its budget mid-edit leaves a broken
    head buffer — the last verified-good buffer is the candidate."""
    from cadx.agent.toolloop import SessionResult

    good = BOX_CODE + "\n# verified-safe edit\n"
    res = SessionResult(code="result = broken_call(", exhausted=True, end_reason="budget")
    res.last_good = good  # contract field lands on SessionResult separately

    async def fake_session(*_a, **_kw):
        return res

    monkeypatch.setattr(improve, "run_session", fake_session)
    llm = FakeLLM([_call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    presp = await improve.run(ctx, SPEC, None, base_code=BOX_CODE, issues=ISSUES, iteration=2)
    assert presp.full_code == good


async def test_exhausted_session_with_no_better_buffer_yields_nothing(monkeypatch):
    """P6: exhausted + last_good identical to base (or absent) means the
    session produced no changes — do not submit a 'patch'."""
    from cadx.agent.toolloop import SessionResult

    res = SessionResult(code="result = broken_call(", exhausted=True, end_reason="budget")
    res.last_good = BOX_CODE  # the only good state IS the base — nothing new

    async def fake_session(*_a, **_kw):
        return res

    monkeypatch.setattr(improve, "run_session", fake_session)
    llm = FakeLLM([_call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    presp = await improve.run(ctx, SPEC, None, base_code=BOX_CODE, issues=ISSUES, iteration=2)
    assert "no changes" in presp.reasoning
    assert not presp.full_code


async def test_infra_error_latches_one_shot_fallback(monkeypatch):
    """An executor-infra session abort is a broken interaction pattern —
    it counts toward the blind one-shot fallback like verify_loop."""
    from cadx.agent.toolloop import SessionResult

    res = SessionResult(code=None, exhausted=True, end_reason="infra_error")

    async def fake_session(*_a, **_kw):
        return res

    monkeypatch.setattr(improve, "run_session", fake_session)
    llm = FakeLLM([_call("done")])
    llm.tools_mode = "auto"
    ctx = _ctx(llm)
    await improve.run(ctx, SPEC, None, base_code=BOX_CODE, issues=ISSUES, iteration=2)
    assert ctx.edit_sessions_exhausted == 1
