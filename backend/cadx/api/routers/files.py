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


_PREVIEW_PREFER = ("iso.png", "contact_sheet.png", "thumbnail.png")
_VIEW_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _render_dirs(v_dir: Path) -> list[tuple[Path, Path]]:
    """(renders_dir, base_dir) pairs, finalized renders first then newest
    iteration. `base_dir` holds contact_sheet.png / fem/ next to renders/."""
    dirs = [(v_dir / "renders", v_dir)]
    it_root = v_dir / "iterations"
    if it_root.is_dir():
        its = sorted(
            (d for d in it_root.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
            reverse=True,
        )
        dirs += [(d / "renders", d) for d in its]
    return dirs


def _view_png(rdir: Path, base: Path, view: str) -> Path:
    """Resolve a view name: render stems -> renders/<view>.png,
    'contact_sheet' -> <iter>/contact_sheet.png, 'fem:<x>' -> <iter>/fem/<x>.png."""
    if view == "contact_sheet":
        return base / "contact_sheet.png"
    if view.startswith("fem:"):
        return base / "fem" / f"{view[4:]}.png"
    return rdir / f"{view}.png"


def _png(p: Path) -> FileResponse:
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/versions/{version_id}/live-renders")
async def live_renders(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Which render views exist right now (for the running-job live preview).

    Finalized renders (version root) win; otherwise the newest iteration dir.
    """
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    v_dir = storage.version_dir(v.project_id, v.id)
    it_root = v_dir / "iterations"

    root_pngs = sorted((v_dir / "renders").glob("*.png")) if (v_dir / "renders").is_dir() else []
    if root_pngs:
        views = sorted(p.stem for p in root_pngs)
        if (v_dir / "contact_sheet.png").exists():
            views.append("contact_sheet")
        return {
            "iteration": None,
            "views": views,
            "mtime": max(p.stat().st_mtime for p in root_pngs),
            "final": True,
        }

    if it_root.is_dir():
        its = sorted(
            (d for d in it_root.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
            reverse=True,
        )
        for d in its:
            rdir = d / "renders"
            pngs = sorted(rdir.glob("*.png")) if rdir.is_dir() else []
            if not pngs:
                continue
            views = sorted(p.stem for p in pngs)
            if (d / "contact_sheet.png").exists():
                views.append("contact_sheet")
            fem_dir = d / "fem"
            if fem_dir.is_dir():
                views += [f"fem:{p.stem}" for p in sorted(fem_dir.glob("*.png"))]
            return {
                "iteration": int(d.name),
                "views": views,
                "mtime": max(p.stat().st_mtime for p in pngs),
                "final": False,
            }
    return {"iteration": None, "views": [], "mtime": 0.0, "final": False}


@router.get("/versions/{version_id}/preview")
async def version_preview(
    version_id: uuid.UUID,
    view: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """Newest render image — provisional while a job is running.

    The executor writes renders under versions/<vid>/iterations/<n>/renders/
    on the shared DATA_DIR volume; artifacts only land at finalize, so this
    scans the directories directly. Finalized renders (version root) win.
    `view` picks a specific render (stem name, 'contact_sheet', 'fem:<stem>').
    """
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    if view is not None and not _VIEW_RE.fullmatch(view.removeprefix("fem:")):
        raise HTTPException(400, "invalid view name")
    v_dir = storage.version_dir(v.project_id, v.id)
    dirs = _render_dirs(v_dir)
    if view is not None:
        for rdir, base in dirs:
            p = _view_png(rdir, base, view)
            if p.exists():
                return _png(p)
        raise HTTPException(404, f"no render for view '{view}'")
    for rdir, _base in dirs:
        for name in _PREVIEW_PREFER:
            p = rdir / name
            if p.exists():
                return _png(p)
        if rdir.is_dir():
            pngs = sorted(rdir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            if pngs:
                return _png(pngs[0])
    raise HTTPException(404, "no preview yet")
