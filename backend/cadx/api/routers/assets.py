"""Asset upload/download (reference images and CAD references)."""

from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx import storage
from cadx.api.deps import get_session
from cadx.api.schemas import AssetOut, AssetUpdate
from cadx.db.models import IMAGE_ASSET_KINDS, Asset, AssetKind, Project
from cadx.llm.images import normalize
from cadx.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["assets"])

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".heif"}
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
        derived_from=a.derived_from,
        gen_params=a.gen_params,
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
        filename = f.filename or "file"
        mime = f.content_type or ""
        width: int | None = None
        height: int | None = None
        if kind == AssetKind.image:
            norm = normalize(data)
            if norm is not None:
                data, width, height = norm
                filename = filename.rsplit(".", 1)[0] + ".jpg"
                mime = "image/jpeg"
            else:
                log.warning("asset.normalize_failed", filename=filename)
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(data)) as im:
                        width, height = im.size
                except Exception:
                    pass
        asset = Asset(
            project_id=project_id,
            kind=kind,
            filename=storage.safe_filename(filename),
            path="",  # filled below
            mime=mime,
            width=width,
            height=height,
            caption=caption,
        )
        path = storage.asset_path(project_id, asset.id, asset.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        asset.path = storage.relpath(path)
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


_rembg_sess = None


def _rembg_session():
    """Lazy shared u2net session (downloads the model on first call)."""
    global _rembg_sess
    if _rembg_sess is None:
        import rembg

        _rembg_sess = rembg.new_session("u2net")
    return _rembg_sess


def _store_derivative(src: Asset, png: bytes, suffix: str) -> Asset | None:
    """Persist an edited image as a new asset linked to `src`.

    Returns None when the payload is not a decodable PNG. Alpha is preserved —
    normalize() would flatten it to JPEG.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as im:
            width, height = im.size
            if im.format != "PNG":
                return None
    except Exception:
        return None
    stem = src.filename.rsplit(".", 1)[0]
    asset = Asset(
        project_id=src.project_id,
        kind=AssetKind.image,
        filename=storage.safe_filename(f"{stem}-{suffix}.png"),
        path="",
        mime="image/png",
        width=width,
        height=height,
        caption=src.caption,
        derived_from=src.id,
    )
    path = storage.asset_path(src.project_id, asset.id, asset.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    asset.path = storage.relpath(path)
    return asset


@router.post("/assets/{asset_id}/derivative", response_model=AssetOut, status_code=201)
async def create_derivative(
    asset_id: uuid.UUID, file: UploadFile, db: AsyncSession = Depends(get_session)
):
    """Store a client-edited PNG (crop/mask result) as a new asset."""
    src = await db.get(Asset, asset_id)
    if not src:
        raise HTTPException(404, "asset not found")
    if src.kind not in IMAGE_ASSET_KINDS:
        raise HTTPException(400, "not an image asset")
    png = await file.read()
    if len(png) > MAX_ASSET_BYTES:
        raise HTTPException(413, f"exceeds {MAX_ASSET_BYTES // 2**20}MB")
    asset = _store_derivative(src, png, "edit")
    if asset is None:
        raise HTTPException(415, "expected a PNG image")
    db.add(asset)
    await db.commit()
    return _out(asset)


@router.post("/assets/{asset_id}/remove-bg")
async def remove_bg(asset_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    """Cut out the subject via rembg. Returns RGBA PNG bytes (not stored yet —
    the editor applies it and persists via /derivative on save)."""
    src = await db.get(Asset, asset_id)
    if not src:
        raise HTTPException(404, "asset not found")
    if src.kind not in IMAGE_ASSET_KINDS:
        raise HTTPException(400, "not an image asset")
    try:
        p = storage.resolve(src.path)
        data = p.read_bytes()
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "asset file missing") from None
    try:
        import rembg
    except ImportError:
        raise HTTPException(
            503, "rembg is not installed — install the backend with rembg support"
        ) from None
    try:
        import asyncio

        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        # u2net (~170 MB) instead of the bria-rmbg default (~1 GB) — keeps RAM
        # and download size sane inside the container.
        out = await asyncio.to_thread(lambda: rembg.remove(img, session=_rembg_session()))
        if isinstance(out, (bytes, bytearray)):
            png = bytes(out)
        else:
            buf = io.BytesIO()
            out.convert("RGBA").save(buf, format="PNG")
            png = buf.getvalue()
    except HTTPException:
        raise
    except Exception as e:
        log.warning("asset.remove_bg_failed", asset=str(asset_id), error=str(e))
        raise HTTPException(500, f"background removal failed: {e}") from e
    return Response(content=png, media_type="image/png")


@router.patch("/assets/{asset_id}", response_model=AssetOut)
async def update_asset(
    asset_id: uuid.UUID, body: AssetUpdate, db: AsyncSession = Depends(get_session)
):
    a = await db.get(Asset, asset_id)
    if not a:
        raise HTTPException(404, "asset not found")
    if body.caption is not None:
        a.caption = body.caption
    await db.commit()
    await db.refresh(a)
    return _out(a)


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
