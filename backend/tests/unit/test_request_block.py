"""USER REQUEST block must reach plan/codegen/critique (T1.1)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from cadx.agent.context import request_block
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import run_generation
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

_SPEC = {
    "name": "block",
    "summary": "40x20x10 block",
    "units": "mm",
    "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
    "width_y": {"value": 20.0, "unit": "mm", "source": "stated"},
    "height_z": {"value": 10.0, "unit": "mm", "source": "stated"},
    "features": [],
    "open_questions": [],
}
_PLAN = {"approach": "cad", "approach_reason": "prismatic", "steps": [], "params": []}


def test_request_block_content() -> None:
    assert "build a 40mm cube" in request_block("build a 40mm cube")
    assert "USER REQUEST" in request_block("x")
    assert request_block("") == ""
    assert request_block("   ") == ""


def _text_of(messages: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [p.get("text", "") for p in c if isinstance(p, dict)]
    return "\n".join(out)


async def test_prompt_reaches_downstream_steps():
    box_code = (FIXTURES / "simple_box.py").read_text()
    marker = "PROMPTMARKER-ACME"
    llm = FakeLLM(
        [
            _SPEC,
            _PLAN,
            {"reasoning": "box", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            {"score": 90, "passed": True, "discrepancies": [], "matches": [], "confidence": 0.9},
        ]
    )

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    trace = MemoryTrace()
    ctx = AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=trace,
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt=f"40x20x10 block {marker}",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        gen_mode="cad",
        assets=[],
    )
    outcome = await run_generation(ctx)
    assert outcome.status in ("succeeded", "needs_review")
    steps = {r["step"]: r for r in trace.records}
    for step in ("plan", "codegen", "critique"):
        assert step in steps, f"missing trace for {step}: {sorted(steps)}"
        assert marker in _text_of(steps[step]["messages"]), f"prompt missing in {step}"
