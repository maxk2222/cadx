"""Strategy chaining tests: ordered approaches, guards, plateau switching."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import run_generation
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _ctx(llm: FakeLLM, emits: list | None = None, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        if emits is not None:
            emits.append((step, it, msg))

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume"))
    kw.setdefault("mode", "generate")
    kw.setdefault("prompt", "x")
    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        **kw,
    )


def _img() -> InputAsset:
    return InputAsset(
        asset_id=str(uuid.uuid4()),
        kind="image",
        path=FIXTURES / "images" / "knob.png",
        filename="knob.png",
    )


BOX_SPEC = {
    "name": "box",
    "summary": "40x20x10 block",
    "units": "mm",
    "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
    "width_y": {"value": 20.0, "unit": "mm", "source": "stated"},
    "height_z": {"value": 10.0, "unit": "mm", "source": "stated"},
    "features": [],
    "open_questions": [],
}
PASS_CRITIQUE = {
    "score": 90,
    "passed": True,
    "discrepancies": [],
    "matches": ["box"],
    "confidence": 0.9,
}
FAIL_CRITIQUE = {
    "score": 30,
    "passed": False,
    "discrepancies": [],
    "matches": [],
    "confidence": 0.8,
}


async def test_strategy_guard_drops_unviable_hybrid():
    """strategy=[hybrid,cad] without a mesh server runs cad only."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    plan = {
        "approach": "hybrid",
        "approach_reason": "organic",
        "strategy": ["hybrid", "cad"],
        "steps": [],
        "params": [],
    }
    llm = FakeLLM(
        [
            BOX_SPEC,
            plan,
            {"reasoning": "box", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            PASS_CRITIQUE,
        ]
    )
    emits: list = []
    ctx = _ctx(
        llm,
        emits,
        gen_mode="auto",
        assets=[_img()],
        mesh_available=False,
        prompt="40x20x10 box",
    )
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode == "cad"
    assert outcome.status in ("succeeded", "needs_review")
    assert "strategy" in outcome.mode_note


async def test_strategy_plateau_switches_to_second_approach():
    """A cad approach stuck below the switch score after 4 iterations yields
    to the next strategy entry, which then succeeds."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    code_resp = {
        "reasoning": "box",
        "params": [],
        "expected_bbox": [40, 20, 10],
        "code": box_code,
    }
    plan = {
        "approach": "cad",
        "approach_reason": "technical",
        "strategy": ["cad", "cad"],  # retry the approach fresh on plateau
        "steps": [],
        "params": [],
    }
    llm = FakeLLM(
        [
            BOX_SPEC,
            plan,
            # approach 1: codegen + failing critique at it=1 and it=4 (the
            # failing critique always yields issues; with patch_retries=0 the
            # improve path consumes no LLM calls on it=2/3)
            code_resp,
            FAIL_CRITIQUE,
            code_resp,
            FAIL_CRITIQUE,
            # approach 2: passes on the first iteration
            code_resp,
            PASS_CRITIQUE,
        ]
    )
    emits: list = []
    ctx = _ctx(
        llm,
        emits,
        gen_mode="auto",
        assets=[_img()],
        mesh_available=False,
        prompt="40x20x10 box",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=4, pass_score=80),
        explicit_max_iterations=True,
        runtime_settings=RuntimeSettings(agent_critique_reask=False, agent_patch_retries=0),
    )
    outcome = await run_generation(ctx)
    assert outcome.score == 90
    assert outcome.iterations >= 5  # iteration counter continued across approaches
    assert any("strategy 2/2" in m for _, _, m in emits)
    assert any("plateau" in m or "switching" in m for _, _, m in emits)
