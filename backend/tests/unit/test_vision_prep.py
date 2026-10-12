"""vision.prep — mesh-generator input preparation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cadx.vision.prep import prepare_for_meshgen


def _synthetic(path, *, margin_side: str = "right") -> None:
    """RGBA: opaque blob pushed against one edge, transparent elsewhere."""
    im = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    a = np.zeros((300, 400), dtype=np.uint8)
    if margin_side == "right":
        a[100:200, 300:398] = 255
    else:
        a[100:200, 2:100] = 255
    arr = np.asarray(im).copy()
    arr[..., 3] = a
    arr[..., :3] = 120
    Image.fromarray(arr).save(path)


def test_prepare_for_meshgen_centers_and_squares(tmp_path):
    src = tmp_path / "src.png"
    out = tmp_path / "prepped.png"
    _synthetic(src)
    res = prepare_for_meshgen(src, out)
    assert res == out and out.exists()
    im = Image.open(out)
    assert im.width == im.height
    a = np.asarray(im)[..., 3]
    # object bbox centred within ±3 % of the frame
    cols = np.flatnonzero((a > 8).any(axis=0))
    rows = np.flatnonzero((a > 8).any(axis=1))
    cx = (cols[0] + cols[-1]) / 2 / im.width
    cy = (rows[0] + rows[-1]) / 2 / im.height
    assert cx == pytest.approx(0.5, abs=0.03)
    assert cy == pytest.approx(0.5, abs=0.03)
    # borders are transparent
    assert a[0].max() <= 8 and a[-1].max() <= 8
    assert a[:, 0].max() <= 8 and a[:, -1].max() <= 8


def test_prepare_for_meshgen_bad_input_returns_source(tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"not an image")
    assert prepare_for_meshgen(src, tmp_path / "o.png") == src
