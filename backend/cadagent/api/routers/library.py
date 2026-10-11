"""Reusable part library — project-independent STEP/STL bodies and code templates.

File parts are exposed to generated code via the existing reference mechanism:
selected parts show up as reference assets and can be loaded with
``load_reference(part_id)``. Templates provide their build123d source as
context for the codegen model.
"""

from __future__ import annotations

import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadagent import storage
from cadagent.api.deps import get_session
from cadagent.api.schemas import (
    LibraryFromVersion,
    LibraryPartOut,
    LibraryPartUpdate,
)
from cadagent.db.models import Artifact, ArtifactKind, LibraryPart, Version

router = APIRouter(tags=["library"])

_UPLOAD_EXT = {".step": "step", ".stp": "step", ".stl": "stl"}
MAX_LIBRARY_BYTES = 50 * 1024 * 1024


def _out(p: LibraryPart) -> LibraryPartOut:
    return LibraryPartOut(
        id=p.id,
        name=p.name,
        kind=p.kind,
        filename=p.filename,
        caption=p.caption,
        params=p.params,
        source_version_id=p.source_version_id,
        created_at=p.created_at,
        url=f"/api/library/{p.id}/file" if p.path else "",
    )


@router.post("/library", response_model=list[LibraryPartOut])
async def upload_library_parts(
    files: list[UploadFile],
    name: str = Query(""),
    caption: str = Query(""),
    db: AsyncSession = Depends(get_session),
):
    out: list[LibraryPart] = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_LIBRARY_BYTES:
            raise HTTPException(413, f"{f.filename}: exceeds {MAX_LIBRARY_BYTES // 2**20}MB")
        ext = (
            "." + (f.filename or "").rsplit(".", 1)[-1].lower() if "." in (f.filename or "") else ""
        )
        kind = _UPLOAD_EXT.get(ext)
        if kind is None:
            raise HTTPException(415, f"{f.filename}: only STEP/STL uploads are supported")
        part = LibraryPart(
            name=name or (f.filename or "part").rsplit(".", 1)[0],
            kind=kind,
            filename=storage.safe_filename(f.filename or "part"),
            caption=caption,
        )
        path = storage.library_part_path(part.id, part.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        part.path = storage.relpath(path)
        db.add(part)
        out.append(part)
    await db.commit()
    return [_out(p) for p in out]


@router.post("/library/from-version/{version_id}", response_model=LibraryPartOut)
async def save_version_to_library(
    version_id: uuid.UUID,
    body: LibraryFromVersion,
    db: AsyncSession = Depends(get_session),
):
    v = await db.get(Version, version_id)
    if not v:
        raise HTTPException(404, "version not found")

    if body.as_template:
        if not v.code:
            raise HTTPException(409, "version has no generated code to save as template")
        part = LibraryPart(
            name=body.name,
            kind="template",
            code=v.code,
            params=v.params,
            caption=body.caption,
            source_version_id=v.id,
        )
    else:
        rows = await db.execute(
            select(Artifact).where(
                Artifact.version_id == version_id, Artifact.kind == ArtifactKind.step
            )
        )
        step = rows.scalars().first()
        if step is None:
            raise HTTPException(409, "version has no STEP export to save")
        try:
            src = storage.resolve(step.path)
        except ValueError:
            raise HTTPException(400, "invalid artifact path") from None
        if not src.exists():
            raise HTTPException(404, "STEP artifact file missing")
        part = LibraryPart(
            name=body.name,
            kind="step",
            filename=f"{storage.safe_filename(body.name)}.step",
            caption=body.caption,
            source_version_id=v.id,
        )
        dst = storage.library_part_path(part.id, part.filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        part.path = storage.relpath(dst)

    db.add(part)
    await db.commit()
    await db.refresh(part)
    return _out(part)


@router.get("/library", response_model=list[LibraryPartOut])
async def list_library_parts(db: AsyncSession = Depends(get_session)):
    rows = await db.execute(select(LibraryPart).order_by(col(LibraryPart.created_at)))
    return [_out(p) for p in rows.scalars().all()]


@router.get("/library/{part_id}/file")
async def library_part_file(part_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(LibraryPart, part_id)
    if not p or not p.path:
        raise HTTPException(404, "library part not found")
    try:
        f = storage.resolve(p.path)
    except ValueError:
        raise HTTPException(400, "invalid path") from None
    if not f.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(f, filename=p.filename)


@router.get("/library/{part_id}", response_model=LibraryPartOut)
async def get_library_part(part_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(LibraryPart, part_id)
    if not p:
        raise HTTPException(404, "library part not found")
    return _out(p)


@router.patch("/library/{part_id}", response_model=LibraryPartOut)
async def update_library_part(
    part_id: uuid.UUID, body: LibraryPartUpdate, db: AsyncSession = Depends(get_session)
):
    p = await db.get(LibraryPart, part_id)
    if not p:
        raise HTTPException(404, "library part not found")
    if body.name is not None:
        p.name = body.name
    if body.caption is not None:
        p.caption = body.caption
    await db.commit()
    await db.refresh(p)
    return _out(p)


@router.delete("/library/{part_id}")
async def delete_library_part(part_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(LibraryPart, part_id)
    if not p:
        raise HTTPException(404, "library part not found")
    storage.delete_library_dir(part_id)
    await db.delete(p)
    await db.commit()
    return {"ok": True}
