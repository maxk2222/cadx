"""Artifact file downloads + version code."""

from __future__ import annotations

import mimetypes
import uuid

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


@router.get("/versions/{version_id}/preview")
async def version_preview(version_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Newest render image — provisional while a job is running.

    The executor writes renders under versions/<vid>/iterations/<n>/renders/
    on the shared DATA_DIR volume; artifacts only land at finalize, so this
    scans the directories directly. Finalized renders (version root) win.
    """
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")
    v_dir = storage.version_dir(v.project_id, v.id)
    dirs = [v_dir / "renders"]
    it_root = v_dir / "iterations"
    if it_root.is_dir():
        its = sorted(
            (d for d in it_root.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
            reverse=True,
        )
        dirs += [d / "renders" for d in its]
    for d in dirs:
        for name in _PREVIEW_PREFER:
            p = d / name
            if p.exists():
                return FileResponse(
                    p, media_type="image/png", headers={"Cache-Control": "no-store"}
                )
        if d.is_dir():
            pngs = sorted(d.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
            if pngs:
                return FileResponse(
                    pngs[0], media_type="image/png", headers={"Cache-Control": "no-store"}
                )
    raise HTTPException(404, "no preview yet")
