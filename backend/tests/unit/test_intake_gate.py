"""Degenerate-spec gate: tool-echo names + dropped prompt features (T1.2)."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import run_generation
from cadx.agent.steps.intake import spec_gaps
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import PartSpec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _spec(name: str = "plate", features: list | None = None) -> PartSpec:
    return PartSpec(
        name=name,
        summary="a plate" if name == "plate" else "",
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 4.0, "source": "stated"},
        features=features or [],
        image_roles=[],
    )


def test_spec_gaps_tool_echo_name() -> None:
    gaps = spec_gaps(_spec(name="search_catalog", features=[]), "plain block")
    assert any("tool echo" in g for g in gaps)


def test_spec_gaps_catalog_summary() -> None:
    s = _spec()
    s.summary = "catalog search for parts"
    assert any("tool echo" in g for g in spec_gaps(s, "plain block"))


def test_spec_gaps_prompt_features_dropped() -> None:
    gaps = spec_gaps(_spec(features=[]), "Platte mit 2 Bohrungen und einer Fase")
    assert any("spec.features is empty" in g for g in gaps)


def test_spec_gaps_clean() -> None:
    assert spec_gaps(_spec(), "a simple smooth box") == []


def test_spec_gaps_features_present_ok() -> None:
    s = _spec(features=[{"name": "hole", "type": "hole"}])
    assert spec_gaps(s, "Platte mit Bohrung") == []


def _ctx(llm: FakeLLM, prompt: str, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=1))
    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt=prompt,
        gen_mode="cad",
        assets=[],
        **kw,
    )


async def test_degenerate_intake_triggers_feature_reask():
    """Intake emits a tool-echo spec with no features; the gate re-asks and
    recovers them before planning."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            # 1) intake: degenerate spec (tool echo, no features)
            {
                "name": "search_catalog",
                "summary": "catalog search",
                "units": "mm",
                "length_x": {"value": 60.0, "source": "stated"},
                "width_y": {"value": 40.0, "source": "stated"},
                "height_z": {"value": 4.0, "source": "stated"},
                "features": [],
                "open_questions": [],
            },
            # 2) intake_features re-ask: recover the holes
            {
                "name": "bracket plate",
                "summary": "plate with two holes",
                "features": [
                    {
                        "name": "holes",
                        "type": "hole",
                        "description": "two through holes",
                        "dims": {"diameter": 5.5},
                        "position": "in foot",
                        "count": 2,
                        "source": "stated",
                    }
                ],
            },
            # 3) plan
            {"approach": "cad", "approach_reason": "r", "steps": [], "params": []},
            # 4) codegen
            {
                "reasoning": "box",
                "params": [],
                "expected_bbox": [60, 40, 4],
                "code": box_code,
            },
            # 5) critique
            {
                "score": 30,
                "passed": False,
                "discrepancies": [
                    {"what": "missing holes", "severity": "major", "fix_hint": "add holes"}
                ],
                "matches": [],
                "confidence": 0.8,
            },
        ]
    )
    emits: list[tuple[str, str]] = []

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        emits.append((step, msg))

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt="Platte 60x40x4 mit 2 Bohrungen",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        gen_mode="cad",
        assets=[],
    )
    outcome = await run_generation(ctx)
    spec = outcome.spec or {}
    assert len(spec.get("features") or []) == 1
    assert spec.get("features", [{}])[0]["type"] == "hole"
    assert spec.get("name") == "bracket plate"
    assert any("recovered" in m for _s, m in emits)
