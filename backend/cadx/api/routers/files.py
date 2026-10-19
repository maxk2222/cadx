"""Artifact file downloads + version code."""

from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cadx import storage
from cadx.api.deps import get_session
from cadx.db.models import Artifact, Version

router = APIRouter(tags=["files"])


@router.get("/artifacts/{artifact_id}/file")
async def artifact_file(
    artifact_id: uuid.UUID,
    download: bool = False,
    db: AsyncSession = Depends(get_session),
):
    a = await db.get(Artifact, artifact_id)
    if not a:
        raise HTTPException(404, "artifact not found")
    try:
        p = storage.resolve(a.path)
    except ValueError:
        raise HTTPException(400, "invalid path") from None
    if not p.exists():
        raise HTTPException(404, "file missing on disk")
    media = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    fname = p.name if download else None
    return FileResponse(p, media_type=media, filename=fname)


@router.get("/versions/{version_id}/code", response_class=PlainTextResponse)
async def version_code(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    return PlainTextResponse(v.code or "")


_PREVIEW_PREFER = ("iso", "contact_sheet", "orient_sheet", "thumbnail")
_VIEW_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _pngs(d: Path) -> list[Path]:
    return sorted(d.glob("*.png")) if d.is_dir() else []


def _base_views(base: Path) -> dict[str, Path]:
    """All live-previewable images of one artifact dir: renders/<view>.png,
    the contact/orient sheet and thumbnail at the base level, fem/*.png."""
    views = {p.stem: p for p in _pngs(base / "renders")}
    for name in ("contact_sheet", "orient_sheet", "thumbnail"):
        p = base / f"{name}.png"
        if p.exists():
            views[name] = p
    views.update({f"fem:{p.stem}": p for p in _pngs(base / "fem")})
    return views


def _render_bases(v_dir: Path) -> list[tuple[Path, int | None, dict[str, Path]]]:
    """(base_dir, iteration, views) with previewable images — freshest write
    first. The finalized set at the version root (iteration=None) wins equal
    mtimes: shutil.copy2 keeps the source mtime, so a tie means it IS the
    newest iteration's render. Every exec writes into its own per-attempt
    subdir (a1, a2, parametrize, …) so iteration dirs are scanned one level
    deep — otherwise a running job would look like it produced no renders."""
    out: list[tuple[float, int, Path, int | None, dict[str, Path]]] = []

    def add(base: Path, it: int | None, prio: int) -> None:
        views = _base_views(base)
        if views:
            out.append((max(p.stat().st_mtime for p in views.values()), prio, base, it, views))

    add(v_dir, None, 1)
    it_root = v_dir / "iterations"
    if it_root.is_dir():
        for it in it_root.iterdir():
            if not (it.is_dir() and it.name.isdigit()):
                continue
            add(it, int(it.name), 0)
            for sub in it.iterdir():
                if sub.is_dir():
                    add(sub, int(it.name), 0)
    out.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [(b, it, vws) for _f, _p, b, it, vws in out]


def _png(p: Path) -> FileResponse:
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/versions/{version_id}/live-renders")
async def live_renders(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Which render views exist right now (for the running-job live preview).

    Freshest write wins: finalized renders at the version root once the job
    promotes its winner, otherwise the newest attempt's renders — including
    per-attempt subdirs under each iteration."""
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    v_dir = storage.version_dir(v.project_id, v.id)

    sets = _render_bases(v_dir)
    if not sets:
        return {"iteration": None, "views": [], "mtime": 0.0, "final": False}
    _base, it_no, views_map = sets[0]
    return {
        "iteration": it_no,
        "views": sorted(views_map),
        "mtime": max(p.stat().st_mtime for p in views_map.values()),
        "final": it_no is None,
    }


@router.get("/versions/{version_id}/preview")
async def version_preview(
    version_id: uuid.UUID,
    view: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """Newest render image — provisional while a job is running.

    The executor writes renders under
    versions/<vid>/iterations/<n>/<attempt>/renders/ on the shared DATA_DIR
    volume; artifacts only land at the version root at finalize, so this
    scans the directories directly — freshest write first (_render_bases).
    `view` picks a specific render (stem name, 'contact_sheet',
    'orient_sheet', 'thumbnail', 'fem:<stem>').
    """
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    if view is not None and not _VIEW_RE.fullmatch(view.removeprefix("fem:")):
        raise HTTPException(400, "invalid view name")
    v_dir = storage.version_dir(v.project_id, v.id)
    sets = _render_bases(v_dir)
    if view is not None:
        for _base, _it, views_map in sets:
            p = views_map.get(view)
            if p is not None:
                return _png(p)
        raise HTTPException(404, f"no render for view '{view}'")
    for _base, _it, views_map in sets:
        for name in _PREVIEW_PREFER:
            p = views_map.get(name)
            if p is not None:
                return _png(p)
        newest = max(views_map.values(), key=lambda p: p.stat().st_mtime)
        return _png(newest)
    raise HTTPException(404, "no preview yet")
