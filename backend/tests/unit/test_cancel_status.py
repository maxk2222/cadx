"""User cancellation must persist as `cancelled`, not `failed` (T0.2)."""

from __future__ import annotations

import uuid
from pathlib import Path

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
_PLAN = {
    "approach": "cad",
    "approach_reason": "prismatic",
    "steps": [],
    "params": [],
    "standard_parts": [],
    "notes": "",
}


def _ctx(llm: FakeLLM, cancel_after: int, **kw) -> AgentCtx:
    """check_cancel returns False for the first `cancel_after` calls, then True."""
    calls = {"n": 0}

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def maybe_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > cancel_after

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=5))
    kw.setdefault("mode", "generate")
    kw.setdefault("prompt", "40x20x10 block")
    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=maybe_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        **kw,
    )


async def test_cancel_during_cad_loop_is_cancelled_not_failed():
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            _SPEC,
            _PLAN,
            {"reasoning": "box", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            {"score": 30, "passed": False, "discrepancies": [], "matches": [], "confidence": 0.5},
        ]
        * 3
    )
    ctx = _ctx(llm, cancel_after=6, gen_mode="cad", assets=[])
    outcome = await run_generation(ctx)
    assert outcome.status == "cancelled"
    assert outcome.error is None
