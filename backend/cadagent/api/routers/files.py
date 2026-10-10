"""Artifact file downloads + version code."""

from __future__ import annotations

import mimetypes
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from cadagent import storage
from cadagent.api.deps import get_session
from cadagent.db.models import Artifact, Version

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
