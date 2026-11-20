"""S2-AP1 (F-040): the headset's in-app updater asks here for the newest version and downloads it.

Contract: planning/SPEC_S2AP1_implementation.md §2 "Server parts". Read-only routes over
APP_RELEASES_DIR (latest.json plus the APKs), written only by backend/scripts/publish_app_release.py.
Every problem with the release data answers 404, never 500: the headset treats any non-200 as
"no update" and starts normally. No login yet (S2-AP2 adds it to every route).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from kiara.guides import paths
from kiara.guides.file_store import resolve_safe_path

log = logging.getLogger(__name__)

router = APIRouter()

PLATFORMS = ("ml2", "quest")
LATEST_FILE_NAME = "latest.json"
APK_MEDIA_TYPE = "application/vnd.android.package-archive"
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _not_found(reason: str) -> HTTPException:
    log.info("app update: %s", reason)
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No app release available.")


def _release_entry(platform: str) -> tuple[dict[str, Any], Path]:
    """The validated latest.json entry for one platform and its APK path, or a 404."""
    if platform not in PLATFORMS:
        raise _not_found(f"unknown platform {platform!r}")

    releases_dir = paths.APP_RELEASES_DIR
    latest_path = releases_dir / LATEST_FILE_NAME
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _not_found("no latest.json") from None
    except (OSError, ValueError) as exc:
        raise _not_found(f"latest.json unreadable: {exc}") from None

    entry = latest.get(platform) if isinstance(latest, dict) else None
    if not isinstance(entry, dict):
        raise _not_found(f"no entry for {platform}")

    version_name = entry.get("versionName")
    version_code = entry.get("versionCode")
    file_name = entry.get("file")
    size = entry.get("size")
    sha256 = entry.get("sha256")
    if not isinstance(version_name, str) or not version_name.strip():
        raise _not_found(f"{platform}: versionName missing")
    if not isinstance(version_code, int) or isinstance(version_code, bool) or version_code <= 0:
        raise _not_found(f"{platform}: versionCode missing or not positive")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise _not_found(f"{platform}: size missing or not positive")
    if not isinstance(sha256, str) or not _SHA256_HEX.match(sha256):
        raise _not_found(f"{platform}: sha256 missing or malformed")
    # One plain file name inside the folder, never a path (contract §2, review 14).
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
        or "/" in file_name
        or "\\" in file_name
        or not file_name.lower().endswith(".apk")
    ):
        raise _not_found(f"{platform}: file name not a single .apk name")

    try:
        apk_path = resolve_safe_path(releases_dir, file_name)
    except ValueError:
        raise _not_found(f"{platform}: file outside the releases folder") from None
    if not apk_path.is_file():
        raise _not_found(f"{platform}: {file_name} missing")
    if apk_path.stat().st_size != size:
        raise _not_found(f"{platform}: {file_name} size differs from latest.json")

    return entry, apk_path


@router.get("/api/app/latest", status_code=status.HTTP_200_OK)
def get_latest_app_release(platform: str = Query(default="")) -> dict[str, Any]:
    # A missing parameter is "no release" (404) like every other problem, not FastAPI's 422.
    entry, _ = _release_entry(platform)
    return {
        "platform": platform,
        "versionName": entry["versionName"],
        "versionCode": entry["versionCode"],
        "size": entry["size"],
        "sha256": entry["sha256"].lower(),
        "url": f"/api/app/download/{platform}",
    }


@router.get("/api/app/download/{platform}", status_code=status.HTTP_200_OK)
def download_app_release(platform: str) -> FileResponse:
    entry, apk_path = _release_entry(platform)
    return FileResponse(path=apk_path, media_type=APK_MEDIA_TYPE, filename=entry["file"])
