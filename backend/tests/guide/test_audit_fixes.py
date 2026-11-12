"""Regression tests for bugs found in the full-implementation audit.

Each test names the failure it would have caught:
- GuidePart.status/.error missing fields (pydantic setattr crash)
- part.part_id vs part.id (AttributeError in warning paths)
- group_vectors (unique, inverse) misused as flat list
- manifold3d has no ``*`` operator — every collision check failed
- review verdict never reopened "review" → infinite repair loop
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadx.guide import cadmeta, solver
from cadx.guide import scene as gscene
from cadx.guide.guide_runner import _review_verdict
from cadx.guide.state import GuidePart, GuideState, PortInfo, ReviewFinding


def _box_mesh(path: Path, size=(20.0, 10.0, 5.0)) -> Path:
    import trimesh

    m = trimesh.creation.box(extents=list(size))
    m.export(str(path))
    return path


def test_guidepart_status_fields_exist() -> None:
    """cadmeta sets part.status/.error — they must be model fields."""
    p = GuidePart(id="x", name="y")
    p.status = "analyzed"  # raised ValueError before the fields existed
    p.error = "e"
    assert p.status == "analyzed"


def test_mesh_port_extraction_finds_planar_ports(tmp_path: Path) -> None:
    """group_vectors returns (unique, inverse) — the flat-list misuse left
    every mesh part with zero ports."""
    mesh = _box_mesh(tmp_path / "plate.stl")
    part = GuidePart(id="plate", name="Plate", mesh_path=str(mesh))
    part = cadmeta.extract_ports_worker(part)
    assert part.status == "analyzed", part.error
    assert part.ports, "no planar ports detected on a box mesh"
    assert all(p.kind == "planar" for p in part.ports)
    assert part.bbox_max[0] == pytest.approx(10.0, abs=0.5)


def test_collision_report_real_meshes(tmp_path: Path) -> None:
    """manifold3d batch_boolean — ``a * b`` raised TypeError, flagging every
    candidate as colliding."""
    a_path = _box_mesh(tmp_path / "a.stl")
    sc = gscene.Scene()
    sc.add("a", a_path, solver.identity_transform())
    # far away — no collision
    far = np.eye(4)
    far[0, 3] = 100.0
    rep = sc.collision_report("b", a_path, far.tolist())
    assert rep["ok"], rep
    assert not rep["collisions"]
    # overlapping — real collision (same location)
    rep = sc.collision_report("b", a_path, solver.identity_transform())
    assert not rep["ok"], rep
    assert rep["collisions"][0]["volume_mm3"] > 500.0


def test_review_verdict_reopens_review_mission(tmp_path: Path) -> None:
    """Without reopening 'review' the verdict re-reads the stale report
    forever — the repair loop never terminates."""
    state = GuideState(output_dir=str(tmp_path))
    state.missions_done = ["doc", "steps", "parts", "assembly", "review", "annotate"]
    state.review.iteration = 1
    state.review.score = 5.0
    state.review.findings = [
        ReviewFinding(severity="blocker", mission="assembly", text="part floats")
    ]
    ship, notes = _review_verdict(state, tmp_path, score_target=8.0, max_iter=3)
    assert not ship and notes
    assert "review" not in state.missions_done, "review must re-run for a fresh verdict"
    assert "assembly" not in state.missions_done
    assert "annotate" not in state.missions_done
    assert "doc" in state.missions_done  # untouched


def test_free_placement_front_is_positive_y() -> None:
    """CADX convention (AGENTS.md): +Y is front."""
    part = GuidePart(
        id="p", name="P", bbox_min=[-5, -5, 0], bbox_max=[5, 5, 10]
    )
    placed = [{"bbox_min_world": [-5, -5, 0], "bbox_max_world": [5, 5, 10]}]
    t = np.asarray(solver.solve_free_placement(part, placed, "front")).reshape(4, 4)
    assert t[1, 3] > 0, "front placement must move toward +Y"
    t = np.asarray(solver.solve_free_placement(part, placed, "back")).reshape(4, 4)
    assert t[1, 3] < 0, "back placement must move toward -Y"


def test_env_hint_word_boundaries() -> None:
    """'laufen'/'verkauf' must not trigger the 'auf' hint."""
    part = GuidePart(id="p", name="P", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    placed = [{"bbox_min_world": [0, 0, 0], "bbox_max_world": [10, 10, 10]}]
    t_top = np.asarray(solver.solve_env_placement(part, placed, "auf die Platte"))
    t_none = np.asarray(solver.solve_env_placement(part, placed, "laufen Sie es ein"))
    assert t_top[2, 3] > t_none[2, 3]


def test_evaluate_mate_antiparallel() -> None:
    mp = PortInfo(name="m", pos=[0, 0, 0], axis=[0, 0, 1])
    fp = PortInfo(name="f", pos=[0, 0, 0], axis=[0, 0, 1])
    t_move = np.eye(4)
    t_move[0:3, 0:3] = solver.axis_angle_rotation_matrix(np.array([1.0, 0, 0]), 180.0)
    gap, ndot = solver.evaluate_mate(mp, t_move, fp, np.eye(4))
    assert gap == pytest.approx(0.0, abs=1e-9)
    assert ndot == pytest.approx(-1.0, abs=1e-6)
