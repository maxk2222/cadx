"""M6 (CADCodeVerify): spec-derived verification questions answered by the
critic — a "no"/"unclear"+evidence answer becomes a deterministic
verify_question issue even when the render looks plausible enough for a
high score."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import build_issues, run_generation
from cadx.agent.steps import critique as critique_step
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Critique, PartSpec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

QUESTIONS = [
    "Is the slot vertical on the upright leg?",
    "Do all ribs stay within the stated 40 mm width?",
]


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


def _spec(questions: list[str] | None = None) -> PartSpec:
    return PartSpec(
        name="bracket",
        summary="L-bracket with slot and ribs",
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 20.0, "source": "stated"},
        features=[{"name": "slot", "type": "slot", "position": "upright leg"}],
        image_roles=[],
        verification_questions=questions or [],
    )


def _llm_critique(
    score: int,
    passed: bool,
    discrepancies: list | None = None,
    answers: list | None = None,
) -> dict:
    return {
        "score": score,
        "passed": passed,
        "discrepancies": discrepancies or [],
        "matches": [],
        "verification_answers": answers or [],
        "confidence": 0.9,
    }


def _run_kwargs(spec: PartSpec) -> dict:
    return {
        "iteration": 1,
        "features": {"total_holes": 0, "rounds": [], "chamfer_faces": 0},
    }


async def _critique(llm: FakeLLM, spec: PartSpec, prompt: str = "L-bracket") -> Critique:
    return await critique_step.run(
        _ctx(llm, prompt=prompt),
        spec,
        None,
        {"bbox_size": [60, 40, 20], "watertight": True},
        [],
        None,
        **_run_kwargs(spec),
    )


# --------------------------------------------------------------------------- #
# schema                                                                        #
# --------------------------------------------------------------------------- #


def test_partspec_verification_questions_roundtrip() -> None:
    spec = _spec(QUESTIONS)
    assert spec.verification_questions == QUESTIONS
    assert spec.model_dump()["verification_questions"] == QUESTIONS
    assert (
        PartSpec(
            length_x={"value": 1.0},
            width_y={"value": 1.0},
            height_z={"value": 1.0},
        ).verification_questions
        == []
    )


def test_partspec_schema_pins_questions() -> None:
    """Required-field pinning convention: guided decoding must emit the key."""
    schema = PartSpec.model_json_schema()
    assert "verification_questions" in schema["properties"]
    assert "verification_questions" in schema["required"]


def test_critique_answers_optional_in_schema() -> None:
    """Zero-cost when the spec has no questions: answers must not be pinned."""
    schema = Critique.model_json_schema()
    assert "verification_answers" in schema["properties"]
    assert "verification_answers" not in schema["required"]
    assert Critique().verification_answers == []


def test_partspec_questions_deduped_and_capped() -> None:
    """Over-emitting models must not sink the whole spec parse."""
    spec = PartSpec(
        length_x={"value": 1.0},
        width_y={"value": 1.0},
        height_z={"value": 1.0},
        verification_questions=["Q1?", "", "  ", "q1?", *{f"Q{i}?" for i in range(2, 9)}],
    )
    assert spec.verification_questions[0] == "Q1?"
    assert "q1?" not in spec.verification_questions  # casefold dedupe
    assert len(spec.verification_questions) == 6


def test_critique_answers_capped() -> None:
    c = Critique(verification_answers=[{"question": f"Q{i}?", "answer": "yes"} for i in range(20)])
    assert len(c.verification_answers) == 12


# --------------------------------------------------------------------------- #
# prompt injection                                                              #
# --------------------------------------------------------------------------- #


async def test_questions_listed_in_critique_prompt() -> None:
    llm = FakeLLM([_llm_critique(90, True)])
    await _critique(llm, _spec(QUESTIONS))
    user_parts = llm.calls[0]["messages"][1]["content"]
    text = "\n".join(p.get("text", "") for p in user_parts if p.get("type") == "text")
    assert "SPEC VERIFICATION QUESTIONS" in text
    assert f"1. {QUESTIONS[0]}" in text
    assert f"2. {QUESTIONS[1]}" in text


async def test_no_questions_no_prompt_section() -> None:
    llm = FakeLLM([_llm_critique(90, True)])
    await _critique(llm, _spec([]))
    user_parts = llm.calls[0]["messages"][1]["content"]
    text = "\n".join(p.get("text", "") for p in user_parts if p.get("type") == "text")
    assert "SPEC VERIFICATION QUESTIONS" not in text


# --------------------------------------------------------------------------- #
# answer -> issue wiring                                                        #
# --------------------------------------------------------------------------- #


async def test_no_answer_becomes_major_verify_issue() -> None:
    """High-scoring plausible render + one "no" -> deterministic fail."""
    llm = FakeLLM(
        [
            _llm_critique(
                95,
                True,
                answers=[
                    {
                        "question": QUESTIONS[0],
                        "answer": "no",
                        "evidence": "front view shows the slot horizontal",
                    },
                    {"question": QUESTIONS[1], "answer": "yes", "evidence": "ribs inside"},
                ],
            )
        ]
    )
    c = await _critique(llm, _spec(QUESTIONS))
    vq = [d for d in c.discrepancies if d.source == "verify_question"]
    assert len(vq) == 1
    assert vq[0].severity == "major"
    assert vq[0].what == QUESTIONS[0]
    assert "horizontal" in vq[0].fix_hint
    assert c.passed is False
    assert c.score <= 90  # one verify_question major -> ceiling (vq weigh 10)


async def test_unclear_with_evidence_becomes_issue() -> None:
    llm = FakeLLM(
        [
            _llm_critique(
                90,
                True,
                answers=[
                    {
                        "question": QUESTIONS[1],
                        "answer": "unclear",
                        "evidence": "right view shows a rib flush with the edge",
                    }
                ],
            )
        ]
    )
    c = await _critique(llm, _spec(QUESTIONS))
    assert any(d.source == "verify_question" and d.severity == "major" for d in c.discrepancies)
    assert c.passed is False


async def test_yes_and_bare_unclear_add_nothing() -> None:
    llm = FakeLLM(
        [
            _llm_critique(
                90,
                True,
                answers=[
                    {"question": QUESTIONS[0], "answer": "yes", "evidence": "looks right"},
                    {"question": QUESTIONS[1], "answer": "unclear", "evidence": ""},
                ],
            )
        ]
    )
    c = await _critique(llm, _spec(QUESTIONS))
    assert not any(d.source == "verify_question" for d in c.discrepancies)
    assert c.passed is True
    assert c.score == 90


async def test_verify_issue_not_duplicated_when_already_listed() -> None:
    """A "no" the critic already described as a discrepancy is not listed twice."""
    llm = FakeLLM(
        [
            _llm_critique(
                40,
                False,
                discrepancies=[
                    {
                        "what": f"{QUESTIONS[0]} — it runs horizontal",
                        "severity": "major",
                        "source": "spec",
                    }
                ],
                answers=[{"question": QUESTIONS[0], "answer": "no", "evidence": "same"}],
            )
        ]
    )
    c = await _critique(llm, _spec(QUESTIONS))
    assert sum(1 for d in c.discrepancies if QUESTIONS[0] in d.what) == 1


def test_build_issues_preserves_verify_source() -> None:
    critique = Critique(
        score=50,
        passed=False,
        discrepancies=[
            {
                "what": QUESTIONS[0],
                "severity": "major",
                "fix_hint": "slot horizontal",
                "source": "verify_question",
            },
            {"what": "shape off", "severity": "minor", "source": "image"},
        ],
    )
    issues = build_issues(critique, [], [], None)
    by_what = {i["what"]: i for i in issues}
    assert by_what[QUESTIONS[0]]["source"] == "verify_question"
    assert by_what[QUESTIONS[0]]["severity"] == "major"
    assert by_what["shape off"]["source"] == "critique"


# --------------------------------------------------------------------------- #
# end-to-end-ish (FakeLLM through the full pipeline)                            #
# --------------------------------------------------------------------------- #


async def test_pipeline_verify_no_lands_in_outcome_issues() -> None:
    """intake emits questions -> critique answers "no" -> outcome.issues
    carries a verify_question entry the improve step can act on."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            # 1) intake: spec carries the verification questions
            {
                "name": "bracket",
                "summary": "L-bracket",
                "units": "mm",
                "length_x": {"value": 60.0, "source": "stated"},
                "width_y": {"value": 40.0, "source": "stated"},
                "height_z": {"value": 4.0, "source": "stated"},
                "features": [{"name": "slot", "type": "slot"}],
                "image_roles": [],
                "verification_questions": QUESTIONS,
            },
            # 2) plan
            {"approach": "cad", "approach_reason": "r", "steps": [], "params": []},
            # 3) codegen
            {"reasoning": "box", "params": [], "expected_bbox": [60, 40, 4], "code": box_code},
            # 4) critique: plausible-looking pass, but a "no" on the slot check
            _llm_critique(
                95,
                True,
                answers=[
                    {
                        "question": QUESTIONS[0],
                        "answer": "no",
                        "evidence": "slot runs horizontal in the front view",
                    },
                    {"question": QUESTIONS[1], "answer": "yes", "evidence": "contained"},
                ],
            ),
        ]
    )

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

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
        prompt="L-Winkel 60x40x4 mit Langloch",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        gen_mode="cad",
        assets=[],
    )
    outcome = await run_generation(ctx)
    vq = [i for i in outcome.issues if i.get("source") == "verify_question"]
    assert vq, f"expected a verify_question issue, got {outcome.issues}"
    assert vq[0]["severity"] == "major"
    assert vq[0]["what"] == QUESTIONS[0]
