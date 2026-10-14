"""Measurement authority: only explicit plan views may drive geometry (T1.4)."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.orchestrator import _measured_plan_outline, _measurement_authoritative
from cadx.agent.steps import measure as measure_step
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import PartSpec
from cadx.vision.measure import ImageMeasurement


def _ctx(prompt: str = "plate 60x40", assets: list | None = None, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume"))
    kw.setdefault("mode", "generate")
    return AgentCtx(
        llm=FakeLLM([]),
        executor=None,  # type: ignore[arg-type]
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt=prompt,
        assets=assets or [],
        **kw,
    )


def _spec(role: str | None = None, asset_id: str = "a1") -> PartSpec:
    return PartSpec(
        name="plate",
        summary="s",
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 4.0, "source": "stated"},
        features=[],
        image_roles=[{"asset_id": asset_id, "role": role}] if role else [],
    )


def _alpha_image(tmp_path: Path, w: int = 300, h: int = 200) -> Path:
    """Rectangle silhouette with an alpha channel — skips the rembg path."""
    from PIL import Image, ImageDraw

    p = tmp_path / "top.png"
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, w - 10, h - 10], fill=(0, 0, 0, 255))
    img.save(p)
    return p


def test_image_measurement_default_not_authoritative() -> None:
    m = ImageMeasurement(px_per_mm=5.0, bbox_mm=[60, 40], outline_mm=[[0, 0], [60, 0], [60, 40]])
    assert m.authoritative is False


async def test_measure_marks_explicit_top_view_authoritative(tmp_path):
    p = _alpha_image(tmp_path)
    asset = InputAsset(asset_id="a1", kind="image", path=p, filename="top.png")
    ctx = _ctx(assets=[asset])
    spec = _spec(role="top", asset_id="a1")
    out = await measure_step.run(ctx, spec)
    assert out["a1"]["authoritative"] is True


async def test_measure_fallback_role_not_authoritative(tmp_path):
    """No image_roles → aspect fallback may measure, but never authoritative."""
    p = _alpha_image(tmp_path)
    asset = InputAsset(asset_id="a1", kind="image", path=p, filename="shot.png")
    ctx = _ctx(assets=[asset])
    spec = _spec()  # no roles at all
    out = await measure_step.run(ctx, spec)
    if "a1" in out:  # aspect fallback may or may not trigger
        assert out["a1"]["authoritative"] is False


async def test_measure_width_mismatch_not_authoritative(tmp_path):
    """A width-mismatch warning downgrades an explicit top view too."""
    p = _alpha_image(tmp_path, w=300, h=60)  # 5:1 silhouette vs stated 1.5:1
    asset = InputAsset(asset_id="a1", kind="image", path=p, filename="top.png")
    ctx = _ctx(assets=[asset])
    spec = _spec(role="top", asset_id="a1")
    out = await measure_step.run(ctx, spec)
    assert "a1" in out
    assert out["a1"]["authoritative"] is False


def test_measured_plan_outline_requires_authority():
    ctx = _ctx()
    outline = [[0, 0], [60, 0], [60, 40], [0, 40]]
    ctx.measurements = {
        "a1": {
            "confidence": 0.7,
            "outline_mm": outline,
            "authoritative": False,
        }
    }
    ctx.plan = None
    assert _measured_plan_outline(ctx) is None

    ctx.measurements["a1"]["authoritative"] = True
    assert _measured_plan_outline(ctx) == outline


def test_measured_plan_outline_use_image_profile_override():
    class _Plan:
        use_image_profile = True

    ctx = _ctx()
    outline = [[0, 0], [60, 0], [60, 40], [0, 40]]
    ctx.measurements = {"a1": {"confidence": 0.7, "outline_mm": outline, "authoritative": False}}
    ctx.plan = _Plan()
    assert _measured_plan_outline(ctx) == outline


def test_measurement_authoritative_helper():
    ctx = _ctx()
    assert _measurement_authoritative(ctx) is False
    ctx.measurements = {"a1": {"authoritative": True}}
    assert _measurement_authoritative(ctx) is True
