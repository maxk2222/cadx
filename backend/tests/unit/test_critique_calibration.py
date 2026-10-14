"""Critique hard rules + fact-based score calibration (T1.3, T3.1, T3.2)."""

from __future__ import annotations

import uuid

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.steps import critique as critique_step
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Feature, PartSpec


def _ctx(llm: FakeLLM, prompt: str = "x", **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
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
        prompt=prompt,
        assets=[],
        **kw,
    )


def _spec() -> PartSpec:
    return PartSpec(
        name="part",
        summary="s",
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 4.0, "source": "stated"},
        features=[],
        image_roles=[],
    )


def _llm_critique(score: int, passed: bool, discrepancies: list | None = None) -> dict:
    return {
        "score": score,
        "passed": passed,
        "discrepancies": discrepancies or [],
        "matches": [],
        "confidence": 0.9,
    }


async def test_feature_request_without_geometry_capped():
    """Prompt names holes but the spec has no features and the measured
    geometry has none → score capped at 40 and forced fail."""
    llm = FakeLLM([_llm_critique(95, True)])
    ctx = _ctx(llm, prompt="Winkel mit 2 Bohrungen")
    c = await critique_step.run(
        ctx,
        _spec(),
        None,
        {"bbox_size": [60, 40, 4], "watertight": True},
        [],
        None,
        iteration=1,
        features={"total_holes": 0, "rounds": [], "chamfer_faces": 0},
    )
    assert c.score <= 40
    assert c.passed is False
    assert any(d.severity == "major" for d in c.discrepancies)


async def test_feature_request_with_geometry_passes_rule():
    """Same prompt but the part actually has holes → no hard fail."""
    llm = FakeLLM([_llm_critique(85, True)])
    ctx = _ctx(llm, prompt="Winkel mit 2 Bohrungen")
    c = await critique_step.run(
        ctx,
        _spec(),
        None,
        {"bbox_size": [60, 40, 4], "watertight": True},
        [],
        None,
        iteration=1,
        features={"total_holes": 2, "rounds": [], "chamfer_faces": 0},
    )
    assert c.score == 85


async def test_fact_ceiling_caps_score_on_dim_issues():
    """LLM gave 95 but a dimension mismatch exists → ceiling applies (T3.1)."""
    llm = FakeLLM([_llm_critique(95, True)])
    ctx = _ctx(llm, prompt="simple box")
    c = await critique_step.run(
        ctx,
        _spec(),
        None,
        {"bbox_size": [60, 40, 4]},
        ["length_x: expected 60.0, measured 50.0"],
        None,
        iteration=1,
        features={"total_holes": 1},
    )
    assert c.score <= 55


async def test_fact_floor_lifts_unsourced_remarks():
    """All facts pass, only unsourced 'other' discrepancies → floor 60 and
    those discrepancies demoted to minor (T3.1)."""
    llm = FakeLLM(
        [
            _llm_critique(
                40,
                False,
                [
                    {"what": "surface finish could be nicer", "severity": "major"},
                    {"what": "proportions slightly off", "severity": "minor"},
                ],
            )
        ]
    )
    ctx = _ctx(llm, prompt="simple box")
    c = await critique_step.run(
        ctx,
        _spec(),
        None,
        {"bbox_size": [60, 40, 4]},
        [],
        None,
        iteration=1,
        features={"total_holes": 1},
    )
    assert c.score >= 60
    assert not any(d.severity == "major" for d in c.discrepancies)


async def test_sourced_major_discrepancy_stays_major():
    """A major discrepancy sourced from the spec keeps its weight."""
    llm = FakeLLM(
        [
            _llm_critique(
                30,
                False,
                [
                    {
                        "what": "hole missing",
                        "severity": "major",
                        "source": "spec",
                        "fix_hint": "add the hole",
                    }
                ],
            )
        ]
    )
    ctx = _ctx(llm, prompt="plate with hole")
    spec = _spec()
    spec.features = [Feature(name="h", type="hole", count=1)]
    c = await critique_step.run(
        ctx,
        spec,
        None,
        {"bbox_size": [60, 40, 4]},
        [],
        None,
        iteration=1,
        features={"total_holes": 0},
    )
    assert any(d.severity == "major" for d in c.discrepancies)
    assert c.passed is False
