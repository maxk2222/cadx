"""Unit tests for the vision verify step (image_check.py) — FakeLLM only."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from PIL import Image

from cadx.agent.image_check import (
    ImageVerdict,
    generate_with_verification,
    verify_concept_image,
)
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.client import LLMResponse
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import Feature, PartSpec

_DIM = {"value": 80.0, "unit": "mm", "source": "stated"}


class _Exec:
    async def execute(self, payload):  # pragma: no cover - unused here
        return {"ok": True}


class _Trace:
    async def record(self, **kw):  # pragma: no cover
        pass


def _ctx(llm) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=llm,
        executor=_Exec(),
        trace=_Trace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="imgcheck",
        project_id=None,
        version_id=uuid.uuid4(),
        mode="generate",
        prompt="a wall hook with two mounting tabs",
        settings=ProjectSettings(),
    )


def _spec() -> PartSpec:
    return PartSpec(
        name="mounting plate",
        length_x=dict(_DIM),
        width_y=dict(_DIM),
        height_z=dict(_DIM),
        features=[
            Feature(
                name="mounting_tab_left",
                type="other",
                description="mounting tab with rounded edge",
            ),
            Feature(
                name="corner_holes",
                type="through_hole",
                description="mounting holes",
                count=2,
                axis="z",
            ),
        ],
    )


@pytest.fixture
def png(tmp_path) -> Path:
    p = tmp_path / "hero.png"
    Image.new("RGB", (32, 32), (120, 120, 120)).save(p)
    return p


def _asset(png: Path) -> list[InputAsset]:
    return [InputAsset(asset_id="a1", kind="generated_image", path=png)]


async def test_verify_major_verdict(png):
    llm = FakeLLM(
        [
            {
                "match": False,
                "missing": ["mounting tabs"],
                "wrong": [],
                "severity": "major",
            }
        ]
    )
    ctx = _ctx(llm)
    v = await verify_concept_image(ctx, png, _spec())
    assert v.severity == "major"
    assert v.missing == ["mounting tabs"]
    assert not v.match
    call = llm.calls[0]
    assert call["role"] == "vision"
    sys_msg = call["messages"][0]["content"]
    assert "mounting plate" in sys_msg  # spec summary injected
    assert "through-holes" in sys_msg  # visual features listed


async def test_verify_parsed_none_is_unverifiable(png):
    class NoneLLM(FakeLLM):
        async def chat(self, *a, **kw):
            return LLMResponse(text="no verdict", parsed=None, model="fake-vision")

    v = await verify_concept_image(_ctx(NoneLLM([])), png, _spec())
    assert v.severity == "unverifiable"
    assert v.match  # accepted — must not block the run


async def test_verify_unparseable_text_is_unverifiable(png):
    llm = FakeLLM(["totally not json"])
    v = await verify_concept_image(_ctx(llm), png, _spec())
    assert v.severity == "unverifiable"
    assert v.match


async def test_verify_chat_exception_is_unverifiable(png):
    class FailLLM(FakeLLM):
        async def chat(self, *a, **kw):
            raise RuntimeError("vision backend down")

    v = await verify_concept_image(_ctx(FailLLM([])), png, _spec())
    assert v.severity == "unverifiable"
    assert v.match


async def test_verify_spec_none_falls_back_to_prompt(png):
    llm = FakeLLM([{"match": True, "severity": "ok"}])
    ctx = _ctx(llm)
    v = await verify_concept_image(ctx, png, None)
    assert v.match
    sys_msg = llm.calls[0]["messages"][0]["content"]
    assert "wall hook" in sys_msg  # ctx.prompt[:200] as summary


async def test_gen_verify_retries_on_major_then_accepts(png):
    llm = FakeLLM(
        [
            {
                "match": False,
                "missing": ["mounting tabs"],
                "wrong": [],
                "severity": "major",
            },
            {"match": True, "missing": [], "wrong": [], "severity": "ok"},
        ]
    )
    calls: list[str] = []

    async def gen(prompt: str) -> list[InputAsset]:
        calls.append(prompt)
        return _asset(png)

    assets, verdict = await generate_with_verification(
        _ctx(llm), prompt="hero prompt", spec=_spec(), gen_fn=gen, retries=2
    )
    assert len(calls) == 2
    assert calls[0] == "hero prompt"
    assert "The photo MUST clearly show:" in calls[1]
    assert "mounting tabs" in calls[1]
    assert verdict.severity == "ok"
    assert len(assets) == 1


async def test_gen_verify_respects_retry_budget(png):
    llm = FakeLLM(
        [
            {
                "match": False,
                "missing": ["holes"],
                "wrong": [],
                "severity": "major",
            }
        ]
    )
    calls: list[str] = []

    async def gen(prompt: str) -> list[InputAsset]:
        calls.append(prompt)
        return _asset(png)

    assets, verdict = await generate_with_verification(
        _ctx(llm), prompt="p", spec=_spec(), gen_fn=gen, retries=2
    )
    assert len(calls) == 3  # 1 initial + 2 retries
    assert verdict.severity == "major"
    assert len(assets) == 1


async def test_gen_verify_zero_retries_single_call(png):
    llm = FakeLLM([{"match": False, "missing": ["holes"], "severity": "major"}])
    calls: list[str] = []

    async def gen(prompt: str) -> list[InputAsset]:
        calls.append(prompt)
        return _asset(png)

    _assets, verdict = await generate_with_verification(
        _ctx(llm), prompt="p", spec=_spec(), gen_fn=gen, retries=0
    )
    assert len(calls) == 1
    assert verdict.severity == "major"


async def test_gen_verify_unverifiable_accepts_immediately(png):
    llm = FakeLLM(["unparseable text"])
    calls: list[str] = []

    async def gen(prompt: str) -> list[InputAsset]:
        calls.append(prompt)
        return _asset(png)

    _assets, verdict = await generate_with_verification(
        _ctx(llm), prompt="p", spec=_spec(), gen_fn=gen, retries=3
    )
    assert len(calls) == 1
    assert verdict.severity == "unverifiable"


async def test_gen_verify_accepts_sync_gen_fn(png):
    llm = FakeLLM([{"match": True, "severity": "ok"}])

    def gen(prompt: str) -> list[InputAsset]:
        return _asset(png)

    assets, verdict = await generate_with_verification(
        _ctx(llm), prompt="p", spec=None, gen_fn=gen, retries=2
    )
    assert len(assets) == 1
    assert verdict.severity == "ok"


def test_image_verdict_defaults():
    v = ImageVerdict()
    assert v.match
    assert v.severity == "ok"
    assert v.missing == []
    assert v.wrong == []


def _flat_spec() -> PartSpec:
    return PartSpec(
        name="flat handle",
        length_x={"value": 120.0, "unit": "mm", "source": "stated"},
        width_y={"value": 40.0, "unit": "mm", "source": "stated"},
        height_z={"value": 30.0, "unit": "mm", "source": "stated"},
    )


def _studio_png(tmp_path, w: int, h: int) -> Path:
    """White studio backdrop + one centered dark subject blob w x h px."""
    p = tmp_path / f"subj_{w}x{h}.png"
    img = Image.new("RGB", (512, 512), (245, 245, 245))
    for x in range(256 - w // 2, 256 + w // 2):
        for y in range(256 - h // 2, 256 + h // 2):
            img.putpixel((x, y), (80, 80, 80))
    img.save(p)
    return p


async def test_verify_flags_tall_image_for_flat_spec(tmp_path):
    """Live failure: z-image rendered a tall U-arch (aspect ~1.4) for the
    4:1 flat handle spec and the vision verdict passed it — the
    silhouette gate must escalate to major."""
    png = _studio_png(tmp_path, w=200, h=280)  # ~0.7 aspect — far too tall
    llm = FakeLLM([{"match": True, "severity": "ok"}])
    v = await verify_concept_image(_ctx(llm), png, _flat_spec())
    assert v.severity == "major"
    assert any("proportions" in m for m in v.missing)


async def test_verify_accepts_flat_image_for_flat_spec(tmp_path):
    png = _studio_png(tmp_path, w=360, h=110)  # ~3.3 aspect — fits 4:1 spec
    llm = FakeLLM([{"match": True, "severity": "ok"}])
    v = await verify_concept_image(_ctx(llm), png, _flat_spec())
    assert v.severity == "ok"


async def test_verify_aspect_gate_skips_square_spec(tmp_path):
    """Near-square specs (L:H ~1) give no view-robust proportion signal —
    the gate must stay silent even on odd silhouettes."""
    png = _studio_png(tmp_path, w=200, h=280)
    llm = FakeLLM([{"match": True, "severity": "ok"}])
    v = await verify_concept_image(_ctx(llm), png, _spec())  # 80x80x80
    assert v.severity == "ok"
