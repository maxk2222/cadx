"""reasoning overshoot: guided decoders may ignore maxLength — an over-long
comment must clamp, not fail the whole response (wasted iteration observed
at PatchResponse.reasoning > 1200 chars)."""

from cadx.llm.schemas import CodeResponse, PatchResponse


def test_patch_reasoning_clamped_not_rejected() -> None:
    p = PatchResponse(reasoning="x" * 5000, full_code="result = None")
    assert len(p.reasoning) == 1200


def test_code_reasoning_clamped_not_rejected() -> None:
    c = CodeResponse(reasoning="y" * 3000, code="result = None")
    assert len(c.reasoning) == 1200


def test_short_reasoning_unchanged() -> None:
    assert PatchResponse(reasoning="ok").reasoning == "ok"
