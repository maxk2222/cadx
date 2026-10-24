"""Evalkit local driver — final-mesh artifact selection and spec imports.

Mesh gates must measure the FINAL committed export, not the first STL an
iteration happened to register: an early topopt extract can carry more
bodies or degraded bores than the version-level export the run committed.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

from cadx.evalkit.local import _mesh_artifact_files


def _version(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    v_dir = tmp_path / "projects" / "p1" / "versions" / "v1"
    v_dir.mkdir(parents=True)
    return SimpleNamespace(project_id=uuid.uuid4(), id=uuid.uuid4()), v_dir


def _patch_storage(monkeypatch, v_dir: Path):
    from cadx import storage

    monkeypatch.setattr(storage, "version_dir", lambda pid, vid: v_dir)
    monkeypatch.setattr(storage, "resolve", lambda rel: v_dir / rel)


def test_final_export_wins_over_iteration_artifacts(monkeypatch, tmp_path):
    """The version-level export is authoritative even when artifact rows
    point at older iteration STLs."""
    v, v_dir = _version(tmp_path)
    _patch_storage(monkeypatch, v_dir)

    (v_dir / "iterations" / "1" / "exports").mkdir(parents=True)
    (v_dir / "iterations" / "1" / "exports" / "model.stl").write_bytes(b"iter1")
    (v_dir / "iterations" / "2" / "exports").mkdir(parents=True)
    (v_dir / "iterations" / "2" / "exports" / "model.stl").write_bytes(b"iter2")
    (v_dir / "exports").mkdir()
    final = v_dir / "exports" / "model.stl"
    final.write_bytes(b"final")

    arts = [SimpleNamespace(kind="stl", path="iterations/1/exports/model.stl")]
    files = _mesh_artifact_files(v, arts)
    assert files == [str(final)]


def test_highest_iteration_wins_when_no_version_export(monkeypatch, tmp_path):
    v, v_dir = _version(tmp_path)
    _patch_storage(monkeypatch, v_dir)
    (v_dir / "iterations" / "4" / "exports").mkdir(parents=True)
    (v_dir / "iterations" / "4" / "exports" / "model.stl").write_bytes(b"i4")
    (v_dir / "iterations" / "5" / "exports").mkdir(parents=True)
    i5 = v_dir / "iterations" / "5" / "exports" / "model.stl"
    i5.write_bytes(b"i5")

    files = _mesh_artifact_files(v, [])
    assert files == [str(i5)]


def test_artifact_rows_last_resort(monkeypatch, tmp_path):
    v, v_dir = _version(tmp_path)
    _patch_storage(monkeypatch, v_dir)
    stl = v_dir / "mesh" / "raw.stl"
    stl.parent.mkdir(parents=True)
    stl.write_bytes(b"row")
    arts = [SimpleNamespace(kind="stl", path="mesh/raw.stl")]
    assert _mesh_artifact_files(v, arts) == [str(stl)]
    # non-mesh kinds never qualify
    assert _mesh_artifact_files(v, [SimpleNamespace(kind="png", path="x.png")]) == []


def test_mesh_clean_fallback(monkeypatch, tmp_path):
    v, v_dir = _version(tmp_path)
    _patch_storage(monkeypatch, v_dir)
    clean = v_dir / "mesh" / "mesh_clean.stl"
    clean.parent.mkdir(parents=True)
    clean.write_bytes(b"clean")
    assert _mesh_artifact_files(v, []) == [str(clean)]
