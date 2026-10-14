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
