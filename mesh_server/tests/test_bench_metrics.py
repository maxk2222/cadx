"""bench_adapters — extended metrics: side_iou, circularity, wall_min."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_adapters  # noqa: E402


def _entry(name: str) -> dict:
    return json.loads(
        (bench_adapters.CORPUS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )


def test_grip_waist_corpus_schema():
    entry = _entry("grip_waist")
    assert entry["class"] == "organic"
    assert len(entry["outline"]) >= 8
    assert len(entry["side_outline"]) >= 8
    # soft waist: top edge (v < 0.5) dips lowest at the mid-point
    top_mid = min((p for p in entry["side_outline"] if abs(p[0] - 0.5) < 0.05 and p[1] < 0.5),
                  key=lambda p: abs(p[0] - 0.5))
    assert 0.05 < top_mid[1] < 0.25  # radius dips 16 -> 12 mm of 32 mm height
    assert entry["holes"]


def test_new_corpus_entries_load():
    names = {p.stem for p in bench_adapters.CORPUS_DIR.glob("*.json")}
    assert {"grip_waist", "dome_knob", "thin_wall"} <= names
    loaded = bench_adapters._load_corpus({"dome_knob", "thin_wall"})
    assert {d["name"] for d in loaded} == {"dome_knob", "thin_wall"}
    assert _entry("dome_knob")["side_outline"]
    # thin_wall is a shell probe — no side_outline required
    assert "side_outline" not in _entry("thin_wall")


def test_cross_circularity_cylinder():
    import trimesh

    cyl = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=64)
    circ = bench_adapters._cross_circularity(cyl)
    assert circ == pytest.approx(1.0, abs=0.05)


def test_cross_circularity_box():
    import trimesh

    box = trimesh.creation.box(extents=[4.0, 1.0, 1.0])
    # square cross-section: 4*pi*A/P^2 = pi/4
    assert bench_adapters._cross_circularity(box) == pytest.approx(0.785, abs=0.05)


def test_side_iou_cylinder_rectangle():
    import trimesh

    cyl = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=32)
    rect = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    assert bench_adapters._side_iou(cyl, rect) == pytest.approx(1.0, abs=0.05)
    waist = _entry("grip_waist")["side_outline"]
    assert 0.0 <= bench_adapters._side_iou(cyl, waist) <= 1.0


def test_wall_min_thin_shell():
    import trimesh

    tube = trimesh.creation.annulus(r_min=9.0, r_max=10.0, height=20.0)
    w = bench_adapters._wall_min(tube)
    assert w is not None
    assert w == pytest.approx(1.0, abs=0.5)


def test_wall_min_solid_box():
    import trimesh

    box = trimesh.creation.box(extents=[4.0, 1.0, 1.0])
    w = bench_adapters._wall_min(box)
    # solid: chord to the nearest oppositely-facing surface ~ smallest extent
    assert w is None or w > 0.0


def test_metrics_emits_new_keys():
    import trimesh

    cyl = trimesh.creation.cylinder(radius=0.5, height=4.0, sections=32)
    row = bench_adapters._metrics(cyl, _entry("grip_waist"), 0.1)
    assert "circularity" in row and "side_iou" in row
    assert 0.0 <= row["side_iou"] <= 1.0
    assert row["circularity"] == pytest.approx(1.0, abs=0.1)
    row2 = bench_adapters._metrics(cyl, _entry("flat_plate"), 0.1)
    assert "circularity" in row2
    assert "side_iou" not in row2  # no side_outline in this entry
