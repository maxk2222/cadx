"""Tests for cadx.vision.measure — deterministic plan-view measurement."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from cadx.vision.measure import measure_plan_view


def _synth(path, *, angle: float = 0.0, hole_ys=None) -> None:
    """600x200 RGBA: rounded rect 480x160 px centred, 3 clear holes Ø20 px
    at x = 120/300/480 (= -45/0/+45 mm at 4 px/mm)."""
    img = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
    obj = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(obj)
    d.rounded_rectangle([60, 20, 540, 180], radius=30, fill=(120, 120, 120, 255))
    ys = hole_ys or [100, 100, 100]
    for x, y in zip([120, 300, 480], ys, strict=True):
        d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(0, 0, 0, 0))
    obj.putalpha(Image.fromarray(np.asarray(obj)[..., 3]))
    if angle:
        obj = obj.rotate(angle, resample=Image.BICUBIC, center=(300, 100))
    img.alpha_composite(obj)
    img.save(path)


def test_measure_plan_view_basic(tmp_path):
    p = tmp_path / "top.png"
    _synth(p)
    m = measure_plan_view(p, length_mm=120.0, width_mm=40.0)
    assert m.bbox_mm[0] == pytest.approx(120.0, abs=0.6)
    assert m.bbox_mm[1] == pytest.approx(40.0, abs=2.5)
    assert len(m.holes) == 3
    xs = [h.x for h in m.holes]
    assert xs == sorted(xs)
    assert xs[0] == pytest.approx(-45, abs=1.0)
    assert xs[2] == pytest.approx(45, abs=1.0)
    for h in m.holes:
        assert h.dia == pytest.approx(5.0, abs=0.5)
        assert h.y == pytest.approx(0.0, abs=1.5)
    assert m.outline_mm[0] == m.outline_mm[-1]
    assert len(m.outline_mm) <= 200


def test_measure_plan_view_rotated(tmp_path):
    p = tmp_path / "rot.png"
    _synth(p, angle=7.0)
    m = measure_plan_view(p, length_mm=120.0)
    assert m.bbox_mm[0] == pytest.approx(120.0, abs=1.0)
    assert len(m.holes) == 3


def test_measure_plan_view_bottom_mirrors_y(tmp_path):
    p1, p2 = tmp_path / "t.png", tmp_path / "b.png"
    _synth(p1, hole_ys=[100, 100, 100])
    # asymmetric y positions to detect the mirror
    _synth(p2, hole_ys=[60, 100, 140])
    top = measure_plan_view(p2, length_mm=120.0, role="top")
    bot = measure_plan_view(p2, length_mm=120.0, role="bottom")
    assert len(top.holes) == len(bot.holes) == 3
    for ht, hb in zip(top.holes, bot.holes, strict=True):
        assert ht.y == pytest.approx(-hb.y, abs=0.4)
