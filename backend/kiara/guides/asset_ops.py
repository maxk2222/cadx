from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from fastapi import UploadFile

from kiara.guides.file_store import (
    delete_file_if_exists,
    guide_referenced_media_urls,
    reindex_project_assets,
    resolve_project_asset_path,
    save_asset_for_guide,
)
from kiara.guides.media_index import media_repository
from kiara.guides.paths import PROJECT_DATA_DIR
from kiara.guides.paths import guide_dir as storage_guide_dir
from kiara.guides.repository import guide_repository

log = logging.getLogger(__name__)


def touch_guide_updated_at(guide_id: str) -> None:
    """Bump only `updatedAt` — a derived write through the mutation primitive
    (never advances authoring version/hash)."""
    from kiara.guides import mutation

    try:
        mutation.mutate_guide(
            guide_id,
            lambda current: current,
            require_expected_version=False,
            derived_only=True,
        )
    except mutation.GuideNotFoundError:
        return


def count_all_media_entries() -> int:
    total = len(media_repository.find_global_media())

    for guide in guide_repository.find_all():
        guide_id = str(guide.get("guideId") or "").strip()
        if not guide_id:
            continue
        total += len(media_repository.find_project_media_by_guide_id(guide_id))

    return total


def find_project_media_entry(media_id: str) -> tuple[str | None, dict | None]:
    for guide in guide_repository.find_all():
        guide_id = str(guide.get("guideId") or "").strip()
        if not guide_id:
            continue

        for media_doc in media_repository.find_project_media_by_guide_id(guide_id):
            current_id = str(media_doc.get("_id") or media_doc.get("id") or "")
            if current_id == str(media_id):
                return guide_id, media_doc

    return None, None


async def upload_media_bulk(files: list[UploadFile], guide_id: str) -> list[dict[str, Any]]:
    if not files:
        raise ValueError("No files provided.")

    results = []
    for file in files:
        object_id = str(uuid4())
        media_doc = await save_asset_for_guide(guide_id=guide_id, object_id=object_id, file=file)
        results.append(
            {
                "mediaId": str(media_doc["_id"]),
                "objectId": object_id,
                "filename": media_doc["fileName"],
                "fileUrl": media_doc["fileUrl"],
                "status": "uploaded",
            }
        )

    touch_guide_updated_at(guide_id)
    return results


# Basic per-guide upload serialization (upload-vs-upload). V1 limitation
# (documented): in-process only, and delete/reindex are not covered by it.
_upload_locks: dict[str, asyncio.Lock] = {}


def _upload_lock(guide_id: str) -> asyncio.Lock:
    return _upload_locks.setdefault(str(guide_id), asyncio.Lock())


async def upload_project_media(file: UploadFile, guide_id: str, object_id: str) -> dict[str, Any]:
    async with _upload_lock(str(guide_id)):
        media_doc = await save_asset_for_guide(
            guide_id=guide_id,
            object_id=object_id,
            file=file,
        )

    # KEIN touch_guide_updated_at: ein reiner Medien-Upload ist keine
    # Guide-Änderung — ein updatedAt-Bump würde falsche Sync-Konflikte erzeugen.

    return {
        "mediaId": str(media_doc["_id"]),
        "fileUrl": media_doc["fileUrl"],
        "status": "uploaded",
        "objectId": media_doc.get("objectId"),
        "assetType": media_doc.get("assetType"),
    }


def list_assets_for_guide(guide_id: str) -> list[dict[str, Any]]:
    reindex_project_assets(guide_id)
    media_list = media_repository.find_project_media_by_guide_id(guide_id)

    with contextlib.suppress(Exception):
        media_list.sort(key=lambda item: (item.get("assetType", ""), item.get("fileName", "")))

    return media_list


def get_project_asset_path(guide_id: str, asset_type: str, file_name: str) -> Path:
    return resolve_project_asset_path(guide_id, asset_type, file_name)


class MediaReferencedError(Exception):
    """The guide still shows this file. → 409."""


def _guide_references_media_file(guide_id: str, asset_type: str, file_name: str) -> bool:
    """With or without `/api` or a host in front, and the generators' `/api/assets/…` form too."""
    tail = f"{guide_id}/{asset_type}/{file_name}"
    for url in guide_referenced_media_urls(guide_id):
        path = unquote(url.split("?", 1)[0].split("#", 1)[0])
        if path.endswith(f"/sync/assets/{tail}") or path.endswith(f"/api/assets/{tail}"):
            return True
    return False


def _resolve_media_file(guide_id: str, media_doc: dict) -> Path | None:
    """
    C2 §2l (A17): the file comes from guideId, type and name. The row's `filePath` is never used, as
    it's bound to the machine that wrote it (the Demonstrator's rows point at another PC).
    """
    file_name = str(media_doc.get("uniqueFileName") or media_doc.get("fileName") or "")
    if not file_name or "/" in file_name or "\\" in file_name:
        return None
    try:
        return resolve_project_asset_path(guide_id, str(media_doc.get("assetType") or ""), file_name)
    except ValueError:
        return None


def delete_project_media(media_id: str) -> None:
    """
    C2 §2l (A8): refused while the guide references the file, or the library could delete a photo a
    guide still shows. A row with no resolvable file still goes. No `updatedAt` bump, as with
    `upload_project_media`.
    """
    media_id = str(media_id).strip()
    guide_id, media_doc = find_project_media_entry(media_id)
    if not media_doc or not guide_id:
        raise FileNotFoundError(f"Media '{media_id}' not found.")

    path = _resolve_media_file(guide_id, media_doc)
    if path is not None and path.is_file():
        if _guide_references_media_file(guide_id, str(media_doc.get("assetType")), path.name):
            raise MediaReferencedError(f"The guide still uses '{path.name}'.")

        try:
            delete_file_if_exists(path)
        except Exception as exc:
            log.error(f"Could not delete physical file: {exc}", exc_info=True)

    media_repository.delete_project_media(guide_id, media_id)


def delete_guide_directory(guide_id: str) -> None:
    guide_dir_path = storage_guide_dir(guide_id)
    if guide_dir_path.exists() and guide_dir_path.is_dir():
        shutil.rmtree(guide_dir_path)
        log.info(f"✅ Guide directory '{guide_dir_path}' deleted successfully.")


def project_data_storage_path_exists() -> bool:
    return PROJECT_DATA_DIR.exists()
