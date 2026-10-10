"""Asset upload/download (reference images and CAD references)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadagent import storage
from cadagent.api.deps import get_session
from cadagent.api.schemas import AssetOut
from cadagent.db.models import Asset, AssetKind, Project

router = APIRouter(tags=["assets"])

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_REF_EXT = {
    ".step": AssetKind.reference_step,
    ".stp": AssetKind.reference_step,
    ".stl": AssetKind.reference_stl,
    ".dxf": AssetKind.reference_dxf,
}
MAX_ASSET_BYTES = 25 * 1024 * 1024


def _out(a: Asset) -> AssetOut:
    return AssetOut(
        id=a.id,
        project_id=a.project_id,
        kind=a.kind,
        filename=a.filename,
        mime=a.mime,
        width=a.width,
        height=a.height,
        caption=a.caption,
        created_at=a.created_at,
        url=f"/api/assets/{a.id}/file",
    )


def _kind_for(filename: str, kind_hint: str | None) -> AssetKind | None:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _IMAGE_EXT:
        return AssetKind.image
    return _REF_EXT.get(ext)


@router.post("/projects/{project_id}/assets", response_model=list[AssetOut])
async def upload_assets(
    project_id: uuid.UUID,
    files: list[UploadFile],
    caption: str = Query(""),
    db: AsyncSession = Depends(get_session),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    out: list[Asset] = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_ASSET_BYTES:
            raise HTTPException(413, f"{f.filename}: exceeds {MAX_ASSET_BYTES // 2**20}MB")
        kind = _kind_for(f.filename or "", None)
        if kind is None:
            raise HTTPException(415, f"{f.filename}: unsupported type")
        asset = Asset(
            project_id=project_id,
            kind=kind,
            filename=storage.safe_filename(f.filename or "file"),
            path="",  # filled below
            mime=f.content_type or "",
            caption=caption,
        )
        path = storage.asset_path(project_id, asset.id, asset.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        asset.path = storage.relpath(path)
        if kind == AssetKind.image:
            try:
                from PIL import Image

                with Image.open(path) as im:
                    asset.width, asset.height = im.size
            except Exception:
                pass
        db.add(asset)
        out.append(asset)
    await db.commit()
    return [_out(a) for a in out]


@router.get("/projects/{project_id}/assets", response_model=list[AssetOut])
async def list_assets(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Asset).where(Asset.project_id == project_id).order_by(col(Asset.created_at))
    )
    return [_out(a) for a in rows.scalars().all()]


@router.get("/assets/{asset_id}/file")
async def asset_file(asset_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    a = await db.get(Asset, asset_id)
    if not a:
        raise HTTPException(404, "asset not found")
    try:
        p = storage.resolve(a.path)
    except ValueError:
        raise HTTPException(400, "invalid path") from None
    if not p.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(p, media_type=a.mime or None, filename=a.filename)


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    a = await db.get(Asset, asset_id)
    if not a:
        raise HTTPException(404, "asset not found")
    try:
        p = storage.resolve(a.path)
        if p.exists():
            p.unlink()
    except ValueError:
        pass
    await db.delete(a)
    await db.commit()
    return {"ok": True}
