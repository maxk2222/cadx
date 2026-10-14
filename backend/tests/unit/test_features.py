from __future__ import annotations

from pathlib import Path

import pytest
from build123d import Align, Box, Cone, Cylinder, Pos, export_stl, import_stl

from cadx.cad.features import _mesh_through_holes, extract_features


@pytest.fixture()
def bored_stl(tmp_path: Path) -> Path:
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for x in (-12.0, 0.0, 12.0):
        part -= Pos(x, 0, 4) * Cylinder(radius=2, height=20)
    path = tmp_path / "bored.stl"
    export_stl(part, str(path))
    return path


def test_mesh_through_holes_detects_bores(bored_stl: Path) -> None:
    groups, csk, slots = _mesh_through_holes(bored_stl)
    assert slots == 0
    z_groups = [g for g in groups if g.axis == "+z"]
    assert len(z_groups) == 3
    for g in z_groups:
        assert 3.0 < g.diameter < 5.5
        assert g.depth_class == "through"
    assert csk == 0


def test_mesh_through_holes_detects_countersink(tmp_path: Path) -> None:
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(0, 0, 4) * Cylinder(radius=2, height=20)
    part -= Pos(0, 0, 7) * Cone(bottom_radius=2, top_radius=4.25, height=2)
    path = tmp_path / "csk.stl"
    export_stl(part, str(path))
    groups, csk, _ = _mesh_through_holes(path)
    assert len([g for g in groups if g.axis == "+z"]) == 1
    assert csk == 1


def test_mesh_through_holes_ignores_solid(tmp_path: Path) -> None:
    path = tmp_path / "solid.stl"
    export_stl(Box(40, 20, 8), str(path))
    groups, csk, _ = _mesh_through_holes(path)
    assert groups == []
    assert csk == 0


def test_mesh_through_holes_ignores_open_slot(tmp_path: Path) -> None:
    # slot open to the silhouette edge is a bay, not a tunnel — must not
    # report a hole even though rays pass through it unobstructed
    part = Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(0, 14, -1) * Box(10, 12, 10)
    path = tmp_path / "slot.stl"
    export_stl(part, str(path))
    groups, _, _ = _mesh_through_holes(path)
    assert [g for g in groups if g.axis == "+z"] == []


def test_mesh_through_holes_detects_enclosed_slot(tmp_path: Path) -> None:
    # 20x6 mm enclosed through-void: elongated → reported as a slot and
    # flagged on its group, not counted as a round bore
    part = Box(70, 30, 5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cutter = Box(14, 6, 10, align=(Align.CENTER,) * 3)
    cutter += Pos(-7, 0, 0) * Cylinder(3, 10) + Pos(7, 0, 0) * Cylinder(3, 10)
    part -= Pos(0, 0, 2.5) * cutter  # cutter protrudes both faces — no membrane
    path = tmp_path / "slot.stl"
    export_stl(part, str(path))
    groups, _, slots = _mesh_through_holes(path)
    assert slots >= 1
    z_groups = [g for g in groups if g.axis == "+z"]
    assert z_groups and all(g.elongated for g in z_groups)


def test_extract_features_mesh_fallback(bored_stl: Path) -> None:
    faceted = import_stl(str(bored_stl))
    no_stl = extract_features(faceted)
    assert no_stl.total_holes == 0
    with_stl = extract_features(faceted, bored_stl)
    assert with_stl.total_holes == 3
    assert with_stl.approximate is True


def test_mesh_profile_reports_spine(tmp_path: Path) -> None:
    # stepped body: a tab shifts the y-extent at one end — the profile must
    # show the drift that a bbox alone would hide
    from build123d import export_brep

    from cadx.cad.runner_entry import _mesh_profile

    part = Box(40, 10, 8) + Pos(25, 8, 0) * Box(10, 10, 8)
    path = tmp_path / "stepped.brep"
    export_brep(part, str(path))
    rows = _mesh_profile(path)["x"]
    assert len(rows) >= 2
    for r in rows:
        assert r[1] <= r[2] and r[3] <= r[4]
        assert r[3] >= -4.1 and r[4] <= 4.1  # z stays within ±4
    left = [r for r in rows if r[0] < -10]
    right = [r for r in rows if r[0] > 20]
    assert left and right
    assert all(abs(r[1] + 5) < 0.6 and abs(r[2] - 5) < 0.6 for r in left)
    assert all(r[2] > 10 for r in right)  # tab pulls ymax out to ~13
