"""T4.1 — watertight repair of mesh exports (STL/3MF/GLB)."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from cadx.cad.exporters import repair_mesh_file


def _leaky_box(path: Path) -> None:
    """Watertight box minus one face → open mesh on disk."""
    m = trimesh.creation.box([20, 20, 20])
    # drop the last face deterministically
    keep = list(range(len(m.faces) - 1))
    m = m.submesh([keep], append=True)
    assert not m.is_watertight
    m.export(str(path))


def test_repair_mesh_file_fixes_open_stl(tmp_path):
    stl = tmp_path / "leaky.stl"
    _leaky_box(stl)
    assert repair_mesh_file(stl) is True
    m = trimesh.load(str(stl), force="mesh")
    assert m.is_watertight and m.is_winding_consistent


def test_repair_mesh_file_already_watertight(tmp_path):
    stl = tmp_path / "box.stl"
    trimesh.creation.box([10, 10, 10]).export(str(stl))
    assert repair_mesh_file(stl) is True


def test_repair_mesh_file_rejects_drifting_voxel_rebuild(tmp_path, monkeypatch):
    """Voxel rebuild that balloons bbox/volume (fills a lattice's voids) must not
    be written back — ship the lightly repaired mesh instead."""
    stl = tmp_path / "leaky.stl"
    _leaky_box(stl)
    # force the voxel-rebuild path — fill_holes alone could seal the leaky box
    monkeypatch.setattr(trimesh.Trimesh, "fill_holes", lambda _m: None)

    def _inflating(_m: trimesh.Trimesh) -> trimesh.Trimesh:
        return trimesh.creation.box([30, 30, 30])

    monkeypatch.setattr("cadx.cad.mesh_ops._ensure_watertight", _inflating)
    assert repair_mesh_file(stl) is False
    m = trimesh.load(str(stl), force="mesh")
    assert max(m.extents) <= 21.0  # original 20mm box, not the 30mm rebuild


def test_repair_mesh_file_rejects_interior_fill_same_bbox(tmp_path, monkeypatch):
    """Voxel fill of enclosed voids leaves the bbox untouched — the volume gate
    must still reject it (the topology-optimised lattice case)."""
    stl = tmp_path / "hollow.stl"
    outer = trimesh.creation.box([20, 20, 20])
    inner = trimesh.creation.box([16, 16, 16])
    shell = outer.difference(inner)  # enclosed cavity — the design
    leaky = shell.submesh([list(range(len(shell.faces) - 1))], append=True)
    assert not leaky.is_watertight
    leaky.export(str(stl))
    monkeypatch.setattr(trimesh.Trimesh, "fill_holes", lambda _m: None)
    monkeypatch.setattr(
        "cadx.cad.mesh_ops._ensure_watertight",
        lambda _m: trimesh.creation.box([20, 20, 20]),  # same bbox, cavity filled
    )
    assert repair_mesh_file(stl) is False
    m = trimesh.load(str(stl), force="mesh")
    assert abs(m.volume) < 6000.0  # shell kept, not the solid 8000mm³ rebuild


def test_repair_mesh_file_accepts_faithful_voxel_rebuild(tmp_path, monkeypatch):
    stl = tmp_path / "leaky.stl"
    _leaky_box(stl)
    monkeypatch.setattr(trimesh.Trimesh, "fill_holes", lambda _m: None)
    monkeypatch.setattr(
        "cadx.cad.mesh_ops._ensure_watertight",
        lambda _m: trimesh.creation.box([20.5, 20.5, 20.5]),
    )
    assert repair_mesh_file(stl) is True
    m = trimesh.load(str(stl), force="mesh")
    assert m.is_watertight and max(m.extents) <= 21.0


def test_run_pipeline_reports_watertight_after_repair(tmp_path):
    pytest.importorskip("build123d")
    from cadx.cad.engine import run_pipeline
    from cadx.db.models import ProjectSettings

    code = "PARAMS = {}\nfrom build123d import *\nresult = Pos(0, 0, 0) * Box(30, 20, 10)\n"
    res = run_pipeline(
        code=code,
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        full_exports=True,
    )
    assert res.ok, res.error
    assert res.analysis is not None
    assert res.analysis.watertight_after_repair is True
    for fmt in ("stl", "3mf", "glb"):
        fname = res.exports.get(fmt)
        assert fname, fmt
        m = trimesh.load(str(tmp_path / "it1" / "exports" / fname), force="mesh")
        m.merge_vertices()  # glb splits verts per face normal — merge like a slicer does
        assert m.is_watertight, fmt
