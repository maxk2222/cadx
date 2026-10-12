"""Filesystem layout under DATA_DIR. ALL file paths go through this module.

Layout:
    data/projects/<pid>/assets/<aid>__<safe_filename>
    data/projects/<pid>/versions/<vid>/model.py
    data/projects/<pid>/versions/<vid>/params.json
    data/projects/<pid>/versions/<vid>/iterations/<n>/...
    data/projects/<pid>/versions/<vid>/exports/model.<ext>
    data/projects/<pid>/versions/<vid>/renders/<view>.png, contact_sheet.png, thumbnail.png
    data/projects/<pid>/versions/<vid>/dfm.json
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from cadx.config import get_settings

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

RENDER_VIEWS = ["front", "back", "left", "right", "top", "bottom", "iso"]


def data_root() -> Path:
    return get_settings().data_path


def resolve(relpath: str | Path) -> Path:
    """Resolve a stored relative path safely inside DATA_DIR."""
    p = (data_root() / relpath).resolve()
    if not str(p).startswith(str(data_root())):
        raise ValueError(f"path escapes data dir: {relpath}")
    return p


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(data_root()))


def safe_filename(name: str) -> str:
    return _SAFE_NAME.sub("_", Path(name).name).strip("._") or "file"


# --- projects / assets -------------------------------------------------------


def project_dir(project_id: uuid.UUID | str) -> Path:
    return data_root() / "projects" / str(project_id)


def asset_dir(project_id: uuid.UUID | str) -> Path:
    return project_dir(project_id) / "assets"


def asset_path(project_id: uuid.UUID | str, asset_id: uuid.UUID | str, filename: str) -> Path:
    return asset_dir(project_id) / f"{asset_id}__{safe_filename(filename)}"


# --- versions ----------------------------------------------------------------


def version_dir(project_id: uuid.UUID | str, version_id: uuid.UUID | str) -> Path:
    return project_dir(project_id) / "versions" / str(version_id)


def iteration_dir(project_id: uuid.UUID | str, version_id: uuid.UUID | str, iteration: int) -> Path:
    return version_dir(project_id, version_id) / "iterations" / str(iteration)


def scratch_dir(project_id: uuid.UUID | str, version_id: uuid.UUID | str) -> Path:
    """Transient GUI preview output (ops previews, region renders). Not part of
    the version's artifact set — served via the /scratch API route only."""
    return version_dir(project_id, version_id) / "_scratch"


def render_dir(
    project_id: uuid.UUID | str, version_id: uuid.UUID | str, iteration: int | None = None
) -> Path:
    if iteration is None:
        return version_dir(project_id, version_id) / "renders"
    return iteration_dir(project_id, version_id, iteration) / "renders"


def export_dir(
    project_id: uuid.UUID | str, version_id: uuid.UUID | str, iteration: int | None = None
) -> Path:
    if iteration is None:
        return version_dir(project_id, version_id) / "exports"
    return iteration_dir(project_id, version_id, iteration) / "exports"


def best_mesh_relpath(
    project_id: uuid.UUID | str, version_id: uuid.UUID | str, iterations: int = 1
) -> str | None:
    """Best available triangle-mesh file of a version (relpath), for mesh-level
    operations on any version kind (edit, FEM, optimize, exports)."""
    v_dir = version_dir(project_id, version_id)
    for cand in (
        v_dir / "exports" / "model.stl",
        v_dir / "mesh" / "mesh_clean.stl",
    ):
        if cand.exists():
            return relpath(cand)
    # last iteration's working STL (cad: renders/mesh.stl, mesh: _run/model.stl)
    it_dir = iteration_dir(project_id, version_id, iterations or 1)
    for cand in (
        it_dir / "_run" / "model.stl",
        it_dir / "renders" / "mesh.stl",
        it_dir / "_run" / "mesh.stl",
    ):
        if cand.exists():
            return relpath(cand)
    return None


def library_dir() -> Path:
    return data_root() / "library"


def library_part_path(part_id: uuid.UUID | str, filename: str) -> Path:
    return library_dir() / str(part_id) / filename


def delete_library_dir(part_id: uuid.UUID | str) -> None:
    import shutil

    d = library_dir() / str(part_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def tmp_dir() -> Path:
    p = data_root() / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def delete_project_dir(project_id: uuid.UUID | str) -> None:
    p = project_dir(project_id)
    if p.exists():
        shutil.rmtree(p)


def delete_version_dir(project_id: uuid.UUID | str, version_id: uuid.UUID | str) -> None:
    p = version_dir(project_id, version_id)
    if p.exists():
        shutil.rmtree(p)
