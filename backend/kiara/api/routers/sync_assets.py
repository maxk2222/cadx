from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from kiara.guides.asset_ops import (
    MediaReferencedError,
    delete_project_media,
    get_project_asset_path,
    list_assets_for_guide,
)
from kiara.guides.asset_resolution import list_guide_model_changes
from kiara.guides.asset_store import (
    AssetConflictError,
    AssetNotFoundError,
    create_store_asset,
    delete_store_asset,
    get_store_asset,
    get_store_model_file,
    guide_titles_using,
    list_store_assets,
    replace_store_asset,
)
from kiara.guides.file_store import resolve_safe_path
from kiara.guides.paths import PROJECT_ASSET_TYPES, safe_guide_id
from kiara.guides.upload_errors import (
    ConverterUnavailableError,
    InvalidUploadError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)


router = APIRouter()


def _validate_project_asset_type(asset_type: str) -> str:
    normalized = str(asset_type or "").strip()
    if normalized not in PROJECT_ASSET_TYPES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown asset type '{asset_type}'.",
        )
    return normalized


def _ensure_existing_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset file not found.",
        )
    return path


def _build_file_response(path: Path) -> FileResponse:
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type=media_type or "application/octet-stream",
        # B-096: a plain <img> load gets no CORS headers; without this the browser may hand that
        # cached copy to the editor's cross-origin loader, which then refuses it.
        headers={"Vary": "Origin"},
    )


@router.get("/sync/assets/{guide_id}", status_code=status.HTTP_200_OK)
@router.get("/api/sync/assets/{guide_id}", status_code=status.HTTP_200_OK)
def list_project_assets_route(guide_id: str):
    return list_assets_for_guide(guide_id)


@router.get("/sync/assets/{guide_id}/{asset_type}", status_code=status.HTTP_200_OK)
@router.get("/api/sync/assets/{guide_id}/{asset_type}", status_code=status.HTTP_200_OK)
def list_project_assets_by_type_route(guide_id: str, asset_type: str):
    normalized_type = _validate_project_asset_type(asset_type)
    items = list_assets_for_guide(guide_id)
    return [item for item in items if str(item.get("assetType") or "") == normalized_type]


@router.get("/sync/assets/{guide_id}/{asset_type}/{file_name:path}", status_code=status.HTTP_200_OK)
@router.get("/api/sync/assets/{guide_id}/{asset_type}/{file_name:path}", status_code=status.HTTP_200_OK)
def get_project_asset_file_route(guide_id: str, asset_type: str, file_name: str):
    normalized_type = _validate_project_asset_type(asset_type)

    try:
        path = get_project_asset_path(guide_id, normalized_type, file_name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset file not found.",
        ) from exc

    return _build_file_response(_ensure_existing_file(path))


# NEU: Route für AR-generierte Assets
@router.get("/api/assets/{guide_id}/{path:path}", status_code=status.HTTP_200_OK)
def get_ar_generated_asset(guide_id: str, path: str):
    """
    Serviert Assets direkt aus uploads/project_data/{guide_id}/
    Beispiel: /api/assets/{guide_id}/output/step_images/.../primary.png
    """
    from kiara.guides.paths import PROJECT_DATA_DIR

    # Basis-Verzeichnis ist uploads/project_data/{guide_id}/
    base_dir = PROJECT_DATA_DIR / safe_guide_id(guide_id)

    # Sicherheitscheck: Pfad muss innerhalb von base_dir bleiben
    try:
        file_resolved = resolve_safe_path(base_dir, path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied.",
        ) from exc

    if not file_resolved.exists() or not file_resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset file not found.",
        )

    return _build_file_response(file_resolved)


# C2 §2l: the shared model store. Replies never hold a server path (A26).

_STORE_ERRORS = (
    AssetConflictError,
    AssetNotFoundError,
    ConverterUnavailableError,
    InvalidUploadError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)


def _store_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AssetConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail)
    if isinstance(exc, AssetNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, UnsupportedMediaTypeError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ConverterUnavailableError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


async def _add_guides_to_replace_offers(exc: Exception) -> None:
    """
    C2 §2q: a `name` conflict or `collisions` entry that offers to replace a model lists who uses it. Here,
    after the store's lock is released, because it reads every guide. A built-in can't be replaced, and a
    refused new name has nothing to replace.
    """
    if not isinstance(exc, AssetConflictError):
        return
    if exc.detail.get("reason") == "name":
        entries = [exc.detail]
    elif exc.detail.get("reason") == "collisions":
        entries = exc.detail["collisions"]
    else:
        return

    for entry in entries:
        existing = entry.get("existing")
        if existing and not existing.get("builtIn"):
            try:
                entry["guides"] = await run_in_threadpool(guide_titles_using, existing["assetId"])
            except Exception:
                # R15-1: the question still goes out. The dialogs print no usage line without `guides`.
                log.exception("Couldn't list the guides using asset '%s'.", existing.get("assetId"))


@router.get("/sync/model-changes", status_code=status.HTTP_200_OK)
@router.get("/api/sync/model-changes", status_code=status.HTTP_200_OK)
def list_model_changes_route():
    return list_guide_model_changes()


@router.get("/sync/store", status_code=status.HTTP_200_OK)
@router.get("/api/sync/store", status_code=status.HTTP_200_OK)
def list_store_assets_route():
    return list_store_assets()


@router.post("/sync/store", status_code=status.HTTP_201_CREATED)
@router.post("/api/sync/store", status_code=status.HTTP_201_CREATED)
async def create_store_asset_route(
    file: UploadFile = File(...),
    display_name: str | None = Form(default=None, alias="displayName"),
    category: str | None = Form(default=None),
):
    try:
        return await create_store_asset(file, display_name, category)
    except _STORE_ERRORS as exc:
        await _add_guides_to_replace_offers(exc)
        raise _store_http_error(exc) from exc


@router.get("/sync/store/{asset_id}", status_code=status.HTTP_200_OK)
@router.get("/api/sync/store/{asset_id}", status_code=status.HTTP_200_OK)
def get_store_asset_route(asset_id: str):
    try:
        return get_store_asset(asset_id)
    except _STORE_ERRORS as exc:
        raise _store_http_error(exc) from exc


@router.get("/sync/store/{asset_id}/model.glb", status_code=status.HTTP_200_OK)
@router.get("/api/sync/store/{asset_id}/model.glb", status_code=status.HTTP_200_OK)
def get_store_model_route(asset_id: str):
    try:
        path, content_hash = get_store_model_file(asset_id)
    except _STORE_ERRORS as exc:
        raise _store_http_error(exc) from exc

    return FileResponse(
        path=str(path),
        media_type="model/gltf-binary",
        headers={"ETag": f'"{content_hash}"', "Cache-Control": "no-cache"},
    )


@router.put("/sync/store/{asset_id}/file", status_code=status.HTTP_200_OK)
@router.put("/api/sync/store/{asset_id}/file", status_code=status.HTTP_200_OK)
async def replace_store_asset_route(
    asset_id: str,
    file: UploadFile = File(...),
    display_name: str | None = Form(default=None, alias="displayName"),
    category: str | None = Form(default=None),
):
    try:
        return await replace_store_asset(asset_id, file, display_name, category)
    except _STORE_ERRORS as exc:
        raise _store_http_error(exc) from exc


@router.delete("/sync/store/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
@router.delete("/api/sync/store/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_store_asset_route(asset_id: str):
    try:
        delete_store_asset(asset_id)
    except _STORE_ERRORS as exc:
        raise _store_http_error(exc) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/sync/media/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
@router.delete("/api/sync/media/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_media_route(media_id: str):
    try:
        delete_project_media(media_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except MediaReferencedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": "referenced", "message": str(exc)},
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
