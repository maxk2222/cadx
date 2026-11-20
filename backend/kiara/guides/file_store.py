from __future__ import annotations

import logging
import mimetypes
import re
import sys
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from kiara.guides.media_index import media_repository
from kiara.guides.media_validation import upright_image_in_place, validate_media_content
from kiara.guides.paths import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_HAND_EXTENSIONS,
    ALLOWED_MODEL_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_MODEL_EXTENSION,
    EXT_PRIORITY,
    PREFAB_EXT_OVERRIDES,
    PROJECT_ASSET_TYPES,
    PROJECT_DATA_DIR,
    classify_project_asset,
    ensure_storage_dirs,
    guide_asset_dir,
    guide_dir,
    guide_instruction_dir,
    guide_instruction_path,
    media_cap_bytes,
    safe_guide_id,
)
from kiara.guides.persistence import now_iso
from kiara.guides.repository import guide_repository
from kiara.guides.upload_errors import (
    GuideNotFoundError,
    InvalidUploadError,
    PayloadTooLargeError,
    StorageRecoveryError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)

# C2 §2l: inside `cad_models/`, while it's a folder built only for one generator run.
CAD_RUN_MARKER = ".materialised"


def guide_cad_inputs_path(guide_id: str) -> Path:
    """A list of assetIds, the generator's model inputs. Beside the guide, not in its JSON."""
    return guide_dir(guide_id) / "input" / "cad_inputs.json"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename so it can safely be used as a directory-like token."""
    name_without_extension = Path(filename).stem
    sanitized = re.sub(r'[\\/*?:"<>| \t]', "_", name_without_extension)
    return re.sub(r"__+", "_", sanitized)


def get_unique_filepath(target_dir: Path, filename: str) -> Path:
    """Return a unique file path, adding suffixes like _2, _3 if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename
    if not file_path.exists():
        return file_path

    counter = 1
    original_stem = file_path.stem
    original_suffix = file_path.suffix

    while file_path.exists():
        counter += 1
        file_path = target_dir / f"{original_stem}_{counter}{original_suffix}"

    return file_path


_LONG_PATH_PREFIX = "\\\\?\\"
_LONG_PATH_UNC_PREFIX = "\\\\?\\UNC\\"


def _extended_length_windows_str(raw: str) -> str:
    """
    Pure string half of `_long_path_safe` — Windows semantics only, no OS check
    and no filesystem access, so it is testable on any platform.

    Returns `raw` UNCHANGED unless it is an absolute Windows path that is not
    already extended-length prefixed. Callers must pass an already normalized
    path: `\\\\?\\` paths bypass Win32 path parsing almost verbatim, so any
    remaining `.`/`..`/relative segment would no longer be resolved.
    """
    if raw.startswith(_LONG_PATH_PREFIX):  # already `\\?\...` or `\\?\UNC\...`
        return raw
    if not PureWindowsPath(raw).is_absolute():
        return raw
    if raw.startswith("\\\\"):  # UNC: \\server\share\x -> \\?\UNC\server\share\x
        return _LONG_PATH_UNC_PREFIX + raw[2:]
    return _LONG_PATH_PREFIX + raw


def _long_path_safe(path: Path) -> Path:
    """
    Make an already-resolved absolute path usable beyond Windows' classic
    260-character MAX_PATH limit by prefixing it with `\\\\?\\`.

    Windows-only by construction: on every other platform this returns the very
    same object it was given (no copy, no normalization) — POSIX has no MAX_PATH
    equivalent, so there is nothing to fix and nothing to risk there.
    """
    if sys.platform != "win32":
        return path

    raw = str(path)
    extended = _extended_length_windows_str(raw)
    return path if extended == raw else Path(extended)


def resolve_safe_path(base_dir: Path, *parts: str) -> Path:
    """
    Resolve a path under a base directory and prevent path traversal.

    The traversal check runs on the plain resolved path; only the returned value
    is made long-path-safe (Windows-only no-op elsewhere), so every caller —
    `final_path` and the `tmp_path`/backup paths derived from it via
    `Path.with_name()`, which provably keeps the prefix — can exceed MAX_PATH.
    """
    base = base_dir.resolve()
    resolved = base.joinpath(*parts).resolve()

    # is_relative_to statt String-Prefix: "base-evil" würde einen
    # startswith("base")-Check fälschlich passieren.
    if not resolved.is_relative_to(base):
        raise ValueError("Zugriff verweigert.")

    return _long_path_safe(resolved)


# ---------------------------------------------------------------------------
# Upload-Validierung (B-036 upload half / media-upload route)
# ---------------------------------------------------------------------------

# Alle produktiven guideIds/objectIds sind kanonische UUIDs (Unity: Guid.NewGuid,
# Web: uuid()). Das eliminiert Traversal-/Mehrdeutigkeitsklassen komplett.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def validate_uuid_id(value: str, field: str) -> str:
    v = str(value or "").strip()
    if not _UUID_RE.match(v):
        raise InvalidUploadError(f"{field} must be a canonical UUID")
    return v


# ── C1 Slice 3 (§8b) — the reserved content-addressed filename namespace ─────────────────────────
#
# Two upload models coexist in the SAME category directories:
#
#   * LEGACY `/upload_media` (web library, independently uploaded media): `{objectId}_{safeName}`,
#     replace-in-place. Unchanged by Slice 3 — same behaviour, same tests.
#   * ATTEMPT finals (Unity's transactional Push): `{objectId}_sha256-{64 lowercase hex}{ext}`,
#     IMMUTABLE. Identical bytes reuse the identical name, so a retry never overwrites anything.
#
# The two can only ever collide if a legacy upload is handed a filename that happens to produce the
# reserved shape. `save_asset_for_guide` therefore REJECTS that shape outright (§8b): the legacy
# route keeps every other filename it accepts today, and can no longer overwrite an attempt final.

CONTENT_ADDRESSED_MARKER = "sha256-"

_RESERVED_FINAL_NAME_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"_sha256-[0-9a-f]{64}(\.[A-Za-z0-9]+)?$"
)


def build_content_addressed_filename(object_id: str, digest: str, ext: str) -> str:
    """
    §8b — the immutable final name for one verified attempt medium.

    `digest` is the FULL 64-character lowercase SHA-256 hex of the verified bytes, so the name is
    the content: the same bytes always land on the same path, and different bytes never can.
    """
    oid = validate_uuid_id(object_id, "object_id")
    d = str(digest or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", d):
        raise InvalidUploadError("content hash must be 64 lowercase hex characters")

    suffix = str(ext or "").strip().lower()
    if suffix and not re.fullmatch(r"\.[a-z0-9]+", suffix):
        raise InvalidUploadError(f"unusable file extension '{ext}'")

    return f"{oid}_{CONTENT_ADDRESSED_MARKER}{d}{suffix}"


def is_reserved_content_addressed_filename(file_name: str | None) -> bool:
    """TRUE for a name in the reserved attempt-final namespace (§8b)."""
    return bool(_RESERVED_FINAL_NAME_RE.match(str(file_name or "")))


def guide_referenced_media_urls(guide_id: str) -> set[str]:
    """
    §8b — every media URL the CANONICAL guide JSON currently references.

    This is what makes reindex reference-aware (review finding F12): an attempt final that the
    committed guide no longer points at is an older immutable file left on disk for safe cleanup
    later, and reindex must not resurrect it as a current media row.

    It reads the guide through the repository, never by inferring from the filesystem.
    """
    doc = guide_repository.find_by_guide_id(str(guide_id or "").strip()) or {}
    urls: set[str] = set()

    for step in doc.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for obj in step.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            for field in ("photoLocation", "videoLocation", "voiceLocation", "modelLocation", "handDataLocation"):
                value = obj.get(field)
                if isinstance(value, str) and value.strip():
                    urls.add(value.strip())

    return urls


def _may_index_discovered_file(file_name: str, file_url: str, referenced: set[str] | None) -> bool:
    """
    §8b — legacy filenames keep the existing reindex behaviour unchanged. A reserved attempt final
    is indexed ONLY while the canonical guide JSON references that exact URL.
    """
    if not is_reserved_content_addressed_filename(file_name):
        return True
    return file_url in (referenced if referenced is not None else set())


# Windows-Gerätenamen gelten auch mit Endung: `CON.fbx` öffnet das Gerät, nicht eine Datei.
_WINDOWS_RESERVED_STEM_RE = re.compile(r"^(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])$", re.IGNORECASE)


def validate_upload_filename(raw_name: str | None) -> str:
    """
    Validate the RAW client filename BEFORE any stripping (reject, never
    sanitize): Unity prebuilds its media URL from this exact name, so a
    silently altered name would break the later download. A name containing
    path separators, dot segments or control characters is rejected outright,
    and so are the Windows forms (C2 Slice 0b, B-087): `:` (drive-relative names,
    NTFS streams), `*?"<>|`, a trailing dot or space, and reserved device names.
    """
    r = str(raw_name or "")
    if not r or len(r) > 255:
        raise InvalidUploadError("filename is empty or too long")
    if "/" in r or "\\" in r:
        raise InvalidUploadError("filename must not contain path separators")
    if r in (".", "..") or ".." in r:
        raise InvalidUploadError("filename must not contain dot segments")
    if any(ord(c) < 32 for c in r):
        raise InvalidUploadError("filename must not contain control characters")
    if any(c in r for c in ':*?"<>|'):
        raise InvalidUploadError('filename must not contain any of : * ? " < > |')
    if r[-1] in ". ":
        raise InvalidUploadError("filename must not end with a dot or space")
    if _WINDOWS_RESERVED_STEM_RE.match(r.split(".", 1)[0].rstrip(" ")):
        raise InvalidUploadError("filename uses a reserved device name")
    return r


def resolve_project_asset_path(guide_id: str, asset_type: str, file_name: str) -> Path:
    if asset_type not in PROJECT_ASSET_TYPES:
        raise ValueError("Ungültiger Asset-Typ.")

    return resolve_safe_path(PROJECT_DATA_DIR / safe_guide_id(guide_id), asset_type, file_name)


def delete_file_if_exists(path: str | Path | None) -> bool:
    if not path:
        return False

    target = Path(path)
    if not target.exists() or not target.is_file():
        return False

    target.unlink()
    return True


async def save_asset_for_guide(guide_id: str, object_id: str, file: UploadFile) -> dict:
    """
    Save an asset file for a guide under:
      project_data/<guideId>/(images|videos|audio|hands)/<objectId>_<originalName>

    Hardened upload path (media-upload route):
    - guide_id/object_id must be canonical UUIDs; the RAW filename is validated
      (rejected, never sanitized) because Unity prebuilds its URL from it.
    - Extension allowlist decides the (plural, Unity-compatible) asset folder;
      the file CONTENT is verified (Pillow / magic bytes) — client MIME ignored.
    - Streamed to a temp file with a size cap, then promoted atomically.
    - Deterministic filename {objectId}_{name} — retries REPLACE, never _2/_3.
    - A file already at the final path (indexed or not) is backed up before
      replacement and restored if promote/index fails. Ordinary caught failures
      never destroy the previous media; a failed restore preserves the backup
      and raises StorageRecoveryError.
    """
    ensure_storage_dirs()

    guide_id = validate_uuid_id(guide_id, "guide_id")
    object_id = validate_uuid_id(object_id, "object_id")

    instruction_doc = guide_repository.find_by_guide_id(guide_id)
    if not instruction_doc:
        log.error(f"Guide '{guide_id}' not found while storing asset.")
        raise GuideNotFoundError(f"Anleitung mit ID {guide_id} nicht gefunden")

    safe_filename = validate_upload_filename(file.filename)
    file_extension = Path(safe_filename).suffix.lower()

    asset_type = classify_project_asset(file_extension)
    if asset_type is None:
        raise UnsupportedMediaTypeError(
            f"file type '{file_extension or 'unknown'}' is not allowed"
        )
    # C2 §2l: no new per-guide copies of a model.
    if asset_type == "cad_models":
        raise UnsupportedMediaTypeError("a model file belongs in the model store: POST /sync/store")

    unique_filename = f"{object_id}_{safe_filename}"

    # C1 Slice 3 (§8b): the ONE new rule on this legacy route. Its replace-in-place model is
    # otherwise untouched — but it may never produce a name in the immutable attempt-final
    # namespace, or a web library upload could silently overwrite a committed attempt medium.
    # Rejected BEFORE any byte is written, so nothing on disk is disturbed.
    if is_reserved_content_addressed_filename(unique_filename):
        raise InvalidUploadError(
            "filename uses the reserved content-addressed form "
            "'sha256-<64 hex>'; that namespace belongs to transactional guide uploads"
        )

    target_dir = guide_asset_dir(guide_id, asset_type)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = resolve_safe_path(guide_dir(guide_id), asset_type, unique_filename)

    # --- Stream in eine Temp-Datei (Größenlimit greift während des Streams) ---
    tmp_path = final_path.with_name(f".{unique_filename}.tmp-{uuid4().hex}")
    cap = media_cap_bytes(asset_type)
    file_size = 0
    try:
        with tmp_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > cap:
                    raise PayloadTooLargeError(
                        f"upload exceeds the {cap}-byte limit for {asset_type}"
                    )
                buffer.write(chunk)
        if file_size == 0:
            raise InvalidUploadError("empty upload")

        # C2 §2k: off the loop. A hand file is parsed whole, and that would stall every other request.
        verified_content_type = await run_in_threadpool(
            validate_media_content, tmp_path, asset_type, file_extension
        )
        # C2 §2n: the headset's image loader ignores an EXIF orientation.
        if asset_type == "images":
            await run_in_threadpool(upright_image_in_place, tmp_path)
            file_size = tmp_path.stat().st_size
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    file_url = f"/api/sync/assets/{guide_id}/{asset_type}/{unique_filename}"
    media_doc = {
        "guideId": guide_id,
        "projectName": guide_id,
        "scope": "project",
        "objectId": object_id,
        "assetType": asset_type,
        "fileName": safe_filename,
        "uniqueFileName": unique_filename,
        "filePath": str(final_path),
        "fileUrl": file_url,
        "fileSize": file_size,
        "contentType": verified_content_type,
        "uploadDate": now_iso(),
    }

    return _promote_and_index_project_media(
        guide_id, object_id, asset_type, final_path, tmp_path, media_doc
    )


def _promote_and_index_project_media(
    guide_id: str,
    object_id: str,
    asset_type: str,
    final_path: Path,
    tmp_path: Path,
    media_doc: dict,
) -> dict:
    """
    Phased replace: (A) back up whatever file exists at final_path (filesystem
    truth — protects unindexed/legacy files too), (B) promote the temp file
    atomically, (C) commit the index row. After the commit, cleanup is
    best-effort and NEVER rolls back the committed upload. V1 limitation
    (documented): an abrupt process crash between phases can leave a .bak/orphan.
    """
    prior = media_repository.find_project_media_by_object(
        guide_id, object_id, asset_type, media_doc["uniqueFileName"]
    )
    # Normalized through the SAME helper as final_path: an index row written
    # before long-path support stores the plain form of the very file we are
    # about to promote. Comparing plain-vs-prefixed would report "different old
    # file" and the post-commit cleanup below would delete the fresh upload.
    old_path = (
        _long_path_safe(Path(str(prior["filePath"])))
        if prior and prior.get("filePath")
        else None
    )

    backup: Path | None = None
    promoted = False
    committed = False
    try:
        if final_path.is_file():  # (A) Dateisystem-Wahrheit, nicht Index-Zustand
            backup = final_path.with_name(final_path.name + f".bak-{uuid4().hex}")
            final_path.replace(backup)

        tmp_path.replace(final_path)  # (B) atomare Promotion
        promoted = True

        row = media_repository.upsert_project_media_by_object(  # (C) Index-Commit
            guide_id, object_id, asset_type, media_doc
        )
        committed = True
    except BaseException:
        if not committed:
            _rollback_promotion(final_path, backup, promoted)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)

    # --- Post-Commit-Cleanup: best-effort, rollt NIE zurück -----------------
    if backup is not None and backup.exists():
        try:
            backup.unlink()
        except OSError as exc:
            log.warning(
                "post-commit: leaving orphan backup %s (harmless, URL intact): %s",
                backup, exc,
            )
    if old_path is not None and old_path != final_path:
        try:
            delete_file_if_exists(str(old_path))
        except OSError as exc:
            log.warning(
                "post-commit: leaving unindexed old file %s (URL intact): %s",
                old_path, exc,
            )

    log.info(f"✅ Guide asset stored: {final_path}")
    return row


def _rollback_promotion(final_path: Path, backup: Path | None, promoted: bool) -> None:
    """
    Restore the previous state after a failed promote/index step.
    If restoring from backup fails, the backup is PRESERVED and a
    StorageRecoveryError is raised — never delete the only remaining copy.
    """
    if backup is not None and backup.exists():
        try:
            backup.replace(final_path)
        except OSError as exc:
            log.error(
                "CATASTROPHIC: could not restore %s from backup %s — backup PRESERVED: %s",
                final_path, backup, exc,
            )
            raise StorageRecoveryError(
                f"restore failed; previous media preserved at {backup}"
            ) from exc
    elif promoted:
        # Vorher existierte nichts an final_path (kein Backup) → neue Datei entfernen.
        final_path.unlink(missing_ok=True)


def save_instruction_for_guide(instruction_doc: dict) -> Path:
    """
    Persist the guide document to:
      project_data/<guideId>/instruction/instruction.json

    C1 Slice 2 (§7b) — this used to be a SECOND writer of the canonical instruction path: it called
    `guide_repository.save_guide()` (which already writes that exact file) and then wrote the same
    file again with `atomic_write_json`, both outside any lock. Two writes, no critical section, no
    version/hash policy. §0c.1 confirmed there is no separate mirror to keep in sync — it is one
    canonical file.

    It is now a thin wrapper BENEATH the one mutation primitive: exactly one locked, hash-stamped,
    atomic write. Do not add a direct write back here.
    """
    ensure_storage_dirs()

    if not instruction_doc or not isinstance(instruction_doc, dict):
        raise ValueError("instruction_doc muss ein Dict sein.")

    guide_id = str(instruction_doc.get("guideId") or "").strip()
    if not guide_id:
        raise ValueError("instruction_doc benötigt eine guideId.")

    # Imported here: guide_mutation_service imports the repository, which imports core.persistence;
    # a module-level import would make storage <-> services a cycle.
    from kiara.guides.mutation import replace_guide_document

    guide_instruction_dir(guide_id).mkdir(parents=True, exist_ok=True)
    replace_guide_document(instruction_doc)

    instruction_path = guide_instruction_path(guide_id)
    log.info(f"✅ instruction.json written for guide '{guide_id}': {instruction_path}")
    return instruction_path


def scan_project_cad_files(guide_id: str) -> list[Path]:
    directory = guide_asset_dir(guide_id, "cad_models")
    if not directory.exists():
        return []

    return [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_MODEL_EXTENSIONS
    ]


def pick_best_match(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None

    by_ext: dict[str, list[Path]] = {}
    for candidate in candidates:
        by_ext.setdefault(candidate.suffix.lower(), []).append(candidate)

    for ext in EXT_PRIORITY:
        if by_ext.get(ext):
            return sorted(by_ext[ext])[0]

    return sorted(candidates)[0]


def ensure_project_media_doc_for_file(guide_id: str, file_path: Path, file_url: str) -> dict:
    content_type, _ = mimetypes.guess_type(str(file_path))

    media_doc = {
        "guideId": guide_id,
        "projectName": guide_id,
        "scope": "project",
        "assetType": "cad_models",
        "fileName": file_path.name,
        "uniqueFileName": file_path.name,
        "filePath": str(file_path),
        "fileUrl": file_url,
        "fileSize": file_path.stat().st_size if file_path.exists() else None,
        "contentType": content_type or "application/octet-stream",
        "uploadDate": now_iso(),
    }

    ensure_method = getattr(media_repository, "ensure_project_media_entry", None)
    if callable(ensure_method):
        return ensure_method(
            guide_id,
            media_doc,
            match_keys=("filePath", "uniqueFileName", "fileUrl"),
        )

    existing_items = media_repository.find_project_media_by_guide_id(guide_id)
    for item in existing_items:
        if str(item.get("filePath") or "") == str(file_path):
            return item

    return media_repository.insert_project_media(guide_id, media_doc)


def _is_reindexable_file(file_path: Path) -> bool:
    """Skip hidden/temp/backup artifacts from interrupted uploads."""
    name = file_path.name
    if name.startswith("."):
        return False
    if ".tmp-" in name or ".bak-" in name:
        return False
    return file_path.is_file()


def _object_id_from_filename(file_name: str) -> str | None:
    """Recover the objectId from a deterministic '{uuid}_{name}' upload filename."""
    prefix = file_name.split("_", 1)[0] if "_" in file_name else ""
    if _UUID_RE.match(prefix):
        return prefix
    return None


def _reindex_media_dir(
    guide_id: str,
    asset_type: str,
    allowed_extensions: set[str],
    referenced: set[str] | None = None,
) -> None:
    directory = guide_asset_dir(guide_id, asset_type)
    if not directory.exists():
        return

    if referenced is None:
        referenced = guide_referenced_media_urls(guide_id)

    for file_path in directory.iterdir():
        if not _is_reindexable_file(file_path):
            continue
        if file_path.suffix.lower() not in allowed_extensions:
            continue

        file_url = f"/api/sync/assets/{guide_id}/{asset_type}/{file_path.name}"
        # §8b/F12: a reserved attempt final is current media only while the canonical guide JSON
        # says so. Legacy names are unaffected.
        if not _may_index_discovered_file(file_path.name, file_url, referenced):
            continue

        content_type, _ = mimetypes.guess_type(str(file_path))
        media_doc = {
            "guideId": guide_id,
            "projectName": guide_id,
            "scope": "project",
            "assetType": asset_type,
            "fileName": file_path.name,
            "uniqueFileName": file_path.name,
            "filePath": str(file_path),
            "fileUrl": file_url,
            "fileSize": file_path.stat().st_size if file_path.exists() else None,
            "contentType": content_type or "application/octet-stream",
            "uploadDate": now_iso(),
        }
        object_id = _object_id_from_filename(file_path.name)
        if object_id:
            media_doc["objectId"] = object_id

        ensure_method = getattr(media_repository, "ensure_project_media_entry", None)
        if callable(ensure_method):
            ensure_method(
                guide_id,
                media_doc,
                match_keys=("filePath", "uniqueFileName", "fileUrl"),
            )
        else:
            existing_items = media_repository.find_project_media_by_guide_id(guide_id)
            exists = any(str(item.get("filePath") or "") == str(file_path) for item in existing_items)
            if not exists:
                media_repository.insert_project_media(guide_id, media_doc)


def reindex_project_assets(guide_id: str) -> None:
    """
    Reindex CAD models, images, videos and audio from the filesystem into
    media metadata. Temp/backup artifacts (.tmp-*/.bak-*) are skipped.

    C1 Slice 3 (§8b, review finding F12): reindex recognises the reserved content-addressed
    attempt-final name and its UUID prefix, but indexes such a file ONLY when the canonical guide
    JSON currently references that exact URL. Older immutable finals may legitimately remain on
    disk for safe cleanup later; they must not be resurrected as current media rows. Legacy
    `{objectId}_{name}` files keep the behaviour they have always had.
    """
    referenced = guide_referenced_media_urls(guide_id)

    # C2 §2l: a marked folder exists only for a generator run. Its files are about to vanish.
    cad_dir = guide_asset_dir(guide_id, "cad_models")
    if cad_dir.exists() and not (cad_dir / CAD_RUN_MARKER).exists():
        for file_path in cad_dir.iterdir():
            if not _is_reindexable_file(file_path):
                continue
            if file_path.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
                continue

            file_url = f"/api/sync/assets/{guide_id}/cad_models/{file_path.name}"
            if not _may_index_discovered_file(file_path.name, file_url, referenced):
                continue
            ensure_project_media_doc_for_file(guide_id, file_path, file_url)

    # Bilder: wie bisher über den Bild-MIME-Typ erkannt (deckt auch Alt-Dateien
    # mit unüblichen Bild-Endungen ab), zusätzlich Temp/Backup-Skip.
    img_dir = guide_asset_dir(guide_id, "images")
    if img_dir.exists():
        for file_path in img_dir.iterdir():
            if not _is_reindexable_file(file_path):
                continue

            content_type, _ = mimetypes.guess_type(str(file_path))
            if not (content_type or "").startswith("image/"):
                continue

            image_url = f"/api/sync/assets/{guide_id}/images/{file_path.name}"
            if not _may_index_discovered_file(file_path.name, image_url, referenced):
                continue

            media_doc = {
                "guideId": guide_id,
                "projectName": guide_id,
                "scope": "project",
                "assetType": "images",
                "fileName": file_path.name,
                "uniqueFileName": file_path.name,
                "filePath": str(file_path),
                "fileUrl": image_url,
                "fileSize": file_path.stat().st_size if file_path.exists() else None,
                "contentType": content_type or "application/octet-stream",
                "uploadDate": now_iso(),
            }
            object_id = _object_id_from_filename(file_path.name)
            if object_id:
                media_doc["objectId"] = object_id

            ensure_method = getattr(media_repository, "ensure_project_media_entry", None)
            if callable(ensure_method):
                ensure_method(
                    guide_id,
                    media_doc,
                    match_keys=("filePath", "uniqueFileName", "fileUrl"),
                )
            else:
                existing_items = media_repository.find_project_media_by_guide_id(guide_id)
                exists = any(str(item.get("filePath") or "") == str(file_path) for item in existing_items)
                if not exists:
                    media_repository.insert_project_media(guide_id, media_doc)

    # Videos + Audio: neue Upload-Kategorien (Unity-kompatible Plural-Ordner).
    _reindex_media_dir(guide_id, "videos", ALLOWED_VIDEO_EXTENSIONS, referenced)
    _reindex_media_dir(guide_id, "audio", ALLOWED_AUDIO_EXTENSIONS, referenced)
    _reindex_media_dir(guide_id, "hands", ALLOWED_HAND_EXTENSIONS, referenced)


def storage_model_extension_for_prefab(base_name: str) -> str:
    return PREFAB_EXT_OVERRIDES.get(base_name.lower(), DEFAULT_MODEL_EXTENSION)
