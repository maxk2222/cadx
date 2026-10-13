"""bench_adapters script — smoke test against the fake adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_adapters  # noqa: E402


def test_footprint_iou_rect_mesh():
    import trimesh

    box = trimesh.creation.box(extents=[2.0, 1.0, 0.2])
    # corpus outline covers the left half only → IoU ~0.5
    outline = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0]]
    iou = bench_adapters._footprint_iou(box, outline)
    assert iou == pytest.approx(0.5, abs=0.08)


def test_render_corpus_entry():
    entry = json.loads(
        (bench_adapters.CORPUS_DIR / "flat_plate.json").read_text(encoding="utf-8")
    )
    img = bench_adapters._render(entry)
    assert img.size == (512, 512)


def test_bench_main_fake(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["bench_adapters.py", "--fake", "--adapters", "fake",
         "--only", "flat_plate", "--out",
         str(tmp_path / "mesh_bench.json")],
    )
    assert bench_adapters.main() == 0
    report = json.loads((tmp_path / "mesh_bench.json").read_text())
    assert "fake" in report["models"]
    st = report["models"]["fake"]["flat"]
    assert st["n"] == 1 and st["success"] == 1.0
