"""Agent orchestrator tests with FakeLLM + InProcessExecutor (real CAD pipeline)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from cadagent import storage
from cadagent.agent.executor_client import InProcessExecutor
from cadagent.agent.orchestrator import build_feedback, compare_dims, run_generation
from cadagent.agent.trace import MemoryTrace
from cadagent.agent.types import AgentCtx, ParentContext
from cadagent.db.models import ProjectSettings
from cadagent.llm.fake import FakeLLM
from cadagent.llm.schemas import PartSpec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOX_CODE = (FIXTURES / "simple_box.py").read_text()

SPEC = {
    "name": "test block",
    "summary": "40x20x10 rounded block",
    "units": "mm",
    "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
    "width_y": {"value": 20.0, "unit": "mm", "source": "stated"},
    "height_z": {"value": 10.0, "unit": "mm", "source": "stated"},
    "features": [{"name": "edge fillet", "type": "fillet", "dims": {"radius": 1.5}}],
    "open_questions": [],
}

PLAN = {"steps": ["box + z-edge fillet"], "params": [], "standard_parts": [], "notes": ""}

CODE = {
    "reasoning": "simple box",
    "params": [
        {"name": "length", "value": 40.0, "unit": "mm"},
        {"name": "width", "value": 20.0, "unit": "mm"},
        {"name": "height", "value": 10.0, "unit": "mm"},
        {"name": "fillet_r", "value": 1.5, "unit": "mm"},
    ],
    "expected_bbox": [40.0, 20.0, 10.0],
    "code": BOX_CODE,
}

PASS_CRITIQUE = {
    "score": 92,
    "passed": True,
    "discrepancies": [],
    "matches": ["box"],
    "confidence": 0.9,
}
FAIL_CRITIQUE = {
    "score": 55,
    "passed": False,
    "discrepancies": [{"what": "missing fillet", "where": "top edges", "severity": "major"}],
    "matches": [],
    "confidence": 0.7,
}


def _ctx(llm: FakeLLM, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume"))
    kw.setdefault("mode", "generate")
    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="make a 40x20x10 block",
        **kw,
    )


def _llm(*responses) -> FakeLLM:
    return FakeLLM(list(responses))


# --------------------------------------------------------------------------- #


def test_compare_dims_ok():
    spec = PartSpec(**SPEC)
    assert compare_dims(spec, {"bbox_size": [40.0, 20.0, 10.0]}, 2.0, 0.5) == []


def test_compare_dims_mismatch_and_swap():
    spec = PartSpec(**SPEC)
    issues = compare_dims(spec, {"bbox_size": [40.0, 10.0, 20.0]}, 2.0, 0.5)
    assert any("width/Y" in i for i in issues)
    assert any("axis" in i for i in issues)


def test_build_feedback_includes_major_and_dims():
    crit_json = dict(FAIL_CRITIQUE)
    from cadagent.llm.schemas import Critique

    fb = build_feedback(Critique(**crit_json), ["length/X: spec 40 vs 41"], None)
    assert "length/X" in fb
    assert "missing fillet" in fb


# --------------------------------------------------------------------------- #


@pytest.mark.slow
async def test_happy_path_one_iteration():
    llm = _llm(SPEC, PLAN, CODE, PASS_CRITIQUE)
    ctx = _ctx(llm)
    outcome = await run_generation(ctx)

    assert outcome.status == "succeeded"
    assert outcome.code == BOX_CODE
    assert outcome.score == 92
    assert outcome.iterations == 1
    assert outcome.analysis is not None
    assert outcome.analysis["bbox_size"] == pytest.approx([40.0, 20.0, 10.0], abs=2.0)
    # exports registered
    exported = {k.split(":")[1] for k in outcome.artifacts if k.startswith("export:")}
    assert {"step", "stl", "glb"} <= exported
    # artifacts actually copied to version root
    v_dir = storage.version_dir(ctx.project_id, ctx.version_id)
    assert (v_dir / "contact_sheet.png").exists()
    assert (v_dir / "renders" / "iso.png").exists()
    assert (v_dir / "dfm.json").exists()


@pytest.mark.slow
async def test_waiting_for_input_on_blocking_questions():
    spec = dict(SPEC)
    spec["open_questions"] = [
        {"question": "which wall thickness?", "blocking": True, "default_assumption": "2mm"}
    ]
    llm = _llm(spec)
    ctx = _ctx(llm, settings=ProjectSettings(clarification_mode="ask"))
    outcome = await run_generation(ctx)

    assert outcome.status == "waiting_for_input"
    assert outcome.questions and outcome.questions[0]["blocking"]
    assert outcome.resume_state["spec"]["name"] == "test block"


@pytest.mark.slow
async def test_repair_then_succeed():
    bad_code = dict(CODE, code="this is not python")
    llm = _llm(SPEC, PLAN, bad_code, CODE, PASS_CRITIQUE)
    ctx = _ctx(llm)
    outcome = await run_generation(ctx)

    assert outcome.status == "succeeded"
    assert outcome.iterations == 1


@pytest.mark.slow
async def test_needs_review_when_critique_fails():
    llm = _llm(SPEC, PLAN, CODE, FAIL_CRITIQUE)
    ctx = _ctx(llm, settings=ProjectSettings(clarification_mode="assume", max_iterations=1))
    outcome = await run_generation(ctx)

    assert outcome.status == "needs_review"
    assert outcome.score == 55


@pytest.mark.slow
async def test_failed_when_code_never_executes():
    bad = dict(CODE, code="import os\nos.system('x')")
    llm = _llm(SPEC, PLAN, bad)  # last repeats forever -> codegen + repairs all fail
    ctx = _ctx(
        llm, settings=ProjectSettings(clarification_mode="assume", max_iterations=1, max_repairs=1)
    )
    outcome = await run_generation(ctx)

    assert outcome.status == "failed"
    assert outcome.error


@pytest.mark.slow
async def test_resume_skips_intake():
    """ctx.spec pre-set (resume) -> first LLM call is plan, not intake."""
    llm = _llm(PLAN, CODE, PASS_CRITIQUE)
    ctx = _ctx(llm)
    ctx.spec = PartSpec(**SPEC)
    outcome = await run_generation(ctx)

    assert outcome.status == "succeeded"
    # first call must be role 'code' (plan), vision never called for intake
    assert llm.calls[0]["role"] == "code"


@pytest.mark.slow
async def test_edit_mode_includes_parent_code():
    llm = _llm(SPEC, PLAN, CODE, PASS_CRITIQUE)
    ctx = _ctx(
        llm,
        mode="edit",
        parent=ParentContext(code="# parent code\nresult = None", number=1),
    )
    outcome = await run_generation(ctx)

    assert outcome.status == "succeeded"
    codegen_msgs = [c for c in llm.calls if c["role"] == "code"]
    assert any("parent code" in json.dumps(m["messages"]) for m in codegen_msgs)
