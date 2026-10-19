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


async def test_high_silhouette_downgrades_image_only_majors():
    """Mesh mode + measured silhouette IoU >= 0.8: majors sourced purely from
    image judgement are texture noise (observed: knob skirt->dome step scored
    as a flaw) — downgrade to minor. Spec-sourced majors stand."""
    llm = FakeLLM(
        [
            _llm_critique(
                45,
                False,
                [
                    {
                        "what": "step between skirt and dome looks wrong",
                        "severity": "major",
                        "source": "image",
                    },
                    {
                        "what": "through-bore missing",
                        "severity": "major",
                        "source": "spec",
                        "fix_hint": "drill the bore",
                    },
                ],
            )
        ]
    )
    ctx = _ctx(
        llm,
        prompt="knob",
        resolved_mode="mesh",
        mesh_info={"_silhouette_iou": 0.86},
    )
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
    by_what = {d.what.split(" (")[0]: d for d in c.discrepancies}
    img = by_what["step between skirt and dome looks wrong"]
    assert img.severity == "minor"
    assert "silhouette" in img.what
    assert by_what["through-bore missing"].severity == "major"


async def test_low_silhouette_keeps_image_majors():
    """Below the IoU threshold the visual complaint is probably real."""
    llm = FakeLLM(
        [
            _llm_critique(
                45,
                False,
                [
                    {
                        "what": "outline is the wrong shape",
                        "severity": "major",
                        "source": "image",
                    },
                ],
            )
        ]
    )
    ctx = _ctx(
        llm,
        prompt="knob",
        resolved_mode="mesh",
        mesh_info={"_silhouette_iou": 0.42},
    )
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
    assert any(d.severity == "major" for d in c.discrepancies)


async def test_verify_question_majors_weigh_less():
    """M6 duplicates: a 'no' verification answer usually repeats a discrepancy
    the critic already listed. 2 spec majors + 2 VQ majors used to clamp the
    score to 0 (4x25) — now the VQ copies weigh 10 each -> ceiling 30."""
    spec = _spec()
    spec.verification_questions = ["are both holes through?", "is the rib centred?"]
    llm = FakeLLM(
        [
            _llm_critique(
                95,
                True,
                [
                    {
                        "what": "arm holes are blind, not through",
                        "severity": "major",
                        "source": "spec",
                        "fix_hint": "drill through",
                    },
                    {
                        "what": "strut is a filled gusset",
                        "severity": "major",
                        "source": "spec",
                        "fix_hint": "rebuild as a bar",
                    },
                ],
            )
        ]
    )
    ctx = _ctx(llm, prompt="bracket with holes and a rib")
    c = await critique_step.run(
        ctx,
        spec,
        None,
        {"bbox_size": [60, 40, 4], "watertight": True},
        [],
        None,
        iteration=1,
        features={"total_holes": 2},
    )
    # without verification answers only the 2 spec majors exist
    assert c.score == 50


async def test_verify_question_duplicates_do_not_saturate_score():
    """Same as above but the critic also answers 'no' on both questions —
    the added verify_question majors must not drive the score to 0."""
    spec = _spec()
    spec.verification_questions = ["are both holes through?", "is the rib centred?"]
    resp = _llm_critique(
        95,
        True,
        [
            {
                "what": "arm holes are blind, not through",
                "severity": "major",
                "source": "spec",
                "fix_hint": "drill through",
            },
            {
                "what": "strut is a filled gusset",
                "severity": "major",
                "source": "spec",
                "fix_hint": "rebuild as a bar",
            },
        ],
    )
    resp["verification_answers"] = [
        {"question": "are both holes through?", "answer": "no", "evidence": "blind bores"},
        {"question": "is the rib centred?", "answer": "no", "evidence": "offset rib"},
    ]
    llm = FakeLLM([resp])
    ctx = _ctx(llm, prompt="bracket with holes and a rib")
    c = await critique_step.run(
        ctx,
        spec,
        None,
        {"bbox_size": [60, 40, 4], "watertight": True},
        [],
        None,
        iteration=1,
        features={"total_holes": 2},
    )
    vq = [d for d in c.discrepancies if d.source == "verify_question"]
    assert len(vq) == 2 and all(d.severity == "major" for d in vq)
    assert c.score == 30  # 100 - 25*2 - 10*2
    assert c.passed is False


def test_normalize_score_vq_weighting():
    """Direct unit check of the ceiling formula."""
    from cadx.agent.steps.critique import normalize_score
    from cadx.llm.schemas import Critique, Discrepancy

    def disc(src: str) -> Discrepancy:
        return Discrepancy(what=src, severity="major", source=src)  # type: ignore[arg-type]

    c = Critique(
        score=95,
        passed=True,
        discrepancies=[
            disc("spec"),
            disc("spec"),
            disc("verify_question"),
            disc("verify_question"),
        ],
    )
    assert normalize_score(c).score == 30
    # four non-VQ majors still saturate to 0
    c2 = Critique(
        score=95,
        passed=True,
        discrepancies=[disc("spec"), disc("spec"), disc("prompt"), disc("image")],
    )
    assert normalize_score(c2).score == 0
