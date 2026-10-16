"""M6: plan-lite — one cheap no-tools call for simple image-free cad jobs."""

from __future__ import annotations

import json
import uuid

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.steps import plan as plan_step
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Dim, PartSpec
from cadx.runtime_config import RuntimeSettings

PLAN_JSON = {
    "approach": "cad",
    "approach_reason": "simple parametric plate.",
    "strategy": ["cad"],
    "optimize": False,
    "steps": ["1. base plate Box(80,50,6)", "2. 2x2 hole grid d=5 on top face"],
    "params": [],
}


def _ctx(llm: FakeLLM, gen_mode: str = "cad", **kw) -> AgentCtx:
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
        prompt="plate 80x50x6 with 4 holes",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(**kw),
        gen_mode=gen_mode,
    )


def _spec() -> PartSpec:
    return PartSpec(
        name="plate",
        summary="flat plate with hole grid",
        length_x=Dim(value=80, source="stated"),
        width_y=Dim(value=50, source="stated"),
        height_z=Dim(value=6, source="stated"),
        features=[],
        image_roles=[],
    )


async def test_plan_lite_auto_for_text_cad():
    """auto + gen_mode=cad + no images -> lite path: single call, no tools,
    approach forced to cad."""
    llm = FakeLLM([{"content": json.dumps(PLAN_JSON)}])
    ctx = _ctx(llm, agent_plan_mode="auto")
    plan = await plan_step.run(ctx, _spec())
    assert plan.approach == "cad" and plan.strategy == ["cad"]
    assert "plan-lite" in plan.approach_reason
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is False


async def test_plan_lite_forced_mode():
    """agent_plan_mode=lite takes the lite path regardless of mode."""
    llm = FakeLLM([{"content": json.dumps(PLAN_JSON)}])
    ctx = _ctx(llm, gen_mode="hybrid", agent_plan_mode="lite")
    plan = await plan_step.run(ctx, _spec())
    assert plan.approach == "cad"  # lite is cad-only
    assert len(llm.calls) == 1


async def test_plan_full_default_still_uses_full_path():
    """Default agent_plan_mode=full keeps the existing plan prompt
    (plan_system.md + tools_info when tools are on)."""
    llm = FakeLLM([{"content": json.dumps(PLAN_JSON)}])
    ctx = _ctx(llm)  # default: full
    plan = await plan_step.run(ctx, _spec())
    assert plan.approach == "cad"
    sys_msg = llm.calls[0]["messages"][0]["content"]
    assert "strategy" in sys_msg  # full prompt mentions strategy/fallback rules


async def test_plan_lite_still_validates_schema():
    """A malformed lite response must not silently produce an empty plan —
    FakeLLM returns unparseable text and the step must raise."""
    import pytest

    from cadx.llm.structured import LLMFormatError

    llm = FakeLLM([{"content": "not json at all"}])
    ctx = _ctx(llm, agent_plan_mode="lite")
    with pytest.raises((LLMFormatError, AssertionError)):
        await plan_step.run(ctx, _spec())
