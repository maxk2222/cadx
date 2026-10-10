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

from cadagent.config import get_settings

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
