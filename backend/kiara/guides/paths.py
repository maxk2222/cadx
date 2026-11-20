"""AR guide filesystem layout under DATA_DIR — ported from the KIARA web suite
(`backend/storage/paths.py`) so the sync/asset routes serve the same tree.

    data/guides/<guide_id>/instruction/instruction.json  (canonical doc, when file-persisted)
    data/guides/<guide_id>/media_index.json              (media registry rows)
    data/guides/<guide_id>/{cad_models,images,videos,audio,hands}/<files>
    data/global_assets/{markers,symbols,tools}/<files>
    data/upload_attempts/<attempt_id>/{manifest.json,media/}
    data/asset_store/<asset_id>/model.glb + .staging
    data/app_releases/latest.json + APKs
"""

from __future__ import annotations

from pathlib import Path

from kiara.config import get_settings

DATA_DIR = get_settings().data_path
INDEX_DIR = DATA_DIR / "index"
#: Guide media tree. The suite calls this `project_data`; the wire URLs only
#: carry `/sync/assets/<guideId>/<type>/<file>` so the disk name is free.
PROJECT_DATA_DIR = DATA_DIR / "guides"
GLOBAL_ASSETS_DIR = DATA_DIR / "global_assets"
#: Attempt staging lives OUTSIDE the committed guide tree, deliberately (§8b):
#: nothing under `guides/<guideId>/` is touched until an attempt commits, and
#: attempt expiry can therefore never walk into a committed guide directory.
UPLOAD_ATTEMPTS_DIR = DATA_DIR / "upload_attempts"
#: One folder per store asset. Staging sits inside it, so a new asset commits
#: with one rename.
ASSET_STORE_DIR = DATA_DIR / "asset_store"
ASSET_STORE_STAGING_DIR = ASSET_STORE_DIR / ".staging"
#: Headset app releases: one APK per platform plus latest.json. Filled by
#: `scripts/publish_app_release.py` only; inside DATA_DIR, so outside Git.
APP_RELEASES_DIR = DATA_DIR / "app_releases"

GLOBAL_ASSET_CATEGORIES = ("markers", "symbols", "tools")
# WICHTIG: "videos"/"audio" PLURAL bzw. exakt wie Unity sie in
# CloudSyncService.BuildServerMediaUrl vorab in die URL bäckt
# (/api/sync/assets/<guide>/<images|videos|audio>/<objectId>_<name>).
PROJECT_ASSET_TYPES = ("instruction", "cad_models", "images", "videos", "audio", "hands")

ALLOWED_MODEL_EXTENSIONS = {".glb", ".gltf", ".stl", ".fbx", ".obj"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}
ALLOWED_AUDIO_EXTENSIONS = {".wav"}
ALLOWED_HAND_EXTENSIONS = {".json"}
# One hand recording measured ~9 MB per take; this leaves room for ~5x that.
MEDIA_MAX_HANDS_BYTES = 50 * 1024 * 1024
EXT_PRIORITY = [".fbx", ".glb", ".gltf", ".obj", ".stl"]
DEFAULT_MODEL_EXTENSION = ".fbx"
FILENAME_DISPLAY_EXTENSION = ".fbx"

PREFAB_EXT_OVERRIDES: dict[str, str] = {}


def classify_project_asset(ext: str) -> str | None:
    """Map a file extension to its project asset folder, or None if not allowed."""
    e = str(ext or "").lower()
    if e in ALLOWED_IMAGE_EXTENSIONS:
        return "images"
    if e in ALLOWED_VIDEO_EXTENSIONS:
        return "videos"
    if e in ALLOWED_AUDIO_EXTENSIONS:
        return "audio"
    if e in ALLOWED_HAND_EXTENSIONS:
        return "hands"
    if e in ALLOWED_MODEL_EXTENSIONS:
        return "cad_models"
    return None


def media_cap_bytes(asset_type: str) -> int:
    """Configured upload size cap for an asset type (env-overridable)."""
    s = get_settings()
    caps = {
        "images": s.media_max_image_bytes,
        "videos": s.media_max_video_bytes,
        "audio": s.media_max_audio_bytes,
        "cad_models": s.media_max_cad_bytes,
        "hands": MEDIA_MAX_HANDS_BYTES,
    }
    return caps[asset_type]


def ensure_storage_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_STORE_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    for category in GLOBAL_ASSET_CATEGORIES:
        (GLOBAL_ASSETS_DIR / category).mkdir(parents=True, exist_ok=True)


class InvalidGuideIdError(Exception):
    """A guideId that cannot be one folder name. The app layer answers HTTP 400."""


def safe_guide_id(guide_id: str) -> str:
    """A guideId must be exactly ONE safe path segment under the guide tree.

    Bewusst KEINE UUID-Pflicht (bestehende Tests nutzen Ids wie `g1`). `:` und ein
    Punkt/Leerzeichen am Ende stehen hier, weil Windows daraus einen anderen Ordner
    macht: `D:x` verlässt project_data, `.. ` wird zu `..`, `x.` zu `x`.
    """
    value = str(guide_id or "")
    if not value or value in (".", ".."):
        raise InvalidGuideIdError("guideId is not a valid folder name")
    if any(c in "/\\:" or ord(c) < 32 for c in value) or value[-1] in ". ":
        raise InvalidGuideIdError("guideId is not a valid folder name")
    return value


def guide_dir(guide_id: str) -> Path:
    return PROJECT_DATA_DIR / safe_guide_id(guide_id)


def guide_instruction_dir(guide_id: str) -> Path:
    return guide_dir(guide_id) / "instruction"


def guide_instruction_path(guide_id: str) -> Path:
    return guide_instruction_dir(guide_id) / "instruction.json"


def guide_media_index_path(guide_id: str) -> Path:
    return guide_dir(guide_id) / "media_index.json"


def guide_asset_dir(guide_id: str, asset_type: str) -> Path:
    return guide_dir(guide_id) / asset_type


# ── upload-attempt staging paths ─────────────────────────────────────────────
# One attempt owns exactly one directory. Abort/expiry delete that directory and
# nothing else, so "clean up an attempt" can never become the recursive "delete
# anything unreferenced" sweep rejected by the sync spec.


def upload_attempt_dir(attempt_id: str) -> Path:
    return UPLOAD_ATTEMPTS_DIR / str(attempt_id)


def upload_attempt_media_dir(attempt_id: str) -> Path:
    return upload_attempt_dir(attempt_id) / "media"


def upload_attempt_manifest_path(attempt_id: str) -> Path:
    return upload_attempt_dir(attempt_id) / "manifest.json"


def global_asset_dir(category: str) -> Path:
    return GLOBAL_ASSETS_DIR / category


def global_asset_path(category: str, file_name: str) -> Path:
    return global_asset_dir(category) / file_name
