from __future__ import annotations

import logging
import mimetypes
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from kiara.guides import mutation
from kiara.guides.asset_ops import (
    count_all_media_entries,
    delete_guide_directory,
    project_data_storage_path_exists,
)
from kiara.guides.asset_resolution import (
    dedupe_step_objects_by_lineage,
    hydrate_missing_model_paths,
    prefab_without_clone,
)
from kiara.guides.file_store import resolve_project_asset_path, resolve_safe_path
from kiara.guides.media_index import media_repository
from kiara.guides.paths import PROJECT_DATA_DIR
from kiara.guides.persistence import now_iso
from kiara.guides.repository import guide_repository
from kiara.guides.schemas import GuideSchema, GuideSchemaForDB
from kiara.guides.sync_schemas import ClientSyncRequest, GuideMetaResponse, SyncConflictResponse

log = logging.getLogger(__name__)


def serialize_guide(doc: dict | None) -> dict | None:
    if not doc:
        return doc

    serialized = dict(doc)

    if "_id" in serialized and not isinstance(serialized["_id"], str):
        serialized["_id"] = str(serialized["_id"])

    if not serialized.get("databaseId") and serialized.get("_id"):
        serialized["databaseId"] = serialized["_id"]

    return serialized


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=UTC)


def get_health_payload() -> dict:
    return {
        "status": "healthy",
        "timestamp": now_iso(),
        "database": {
            "instructions_count": len(guide_repository.find_all()),
            "media_files_count": count_all_media_entries(),
        },
        "project_data_storage_path_exists": project_data_storage_path_exists(),
    }


def check_sync_conflict(payload: GuideSchema) -> SyncConflictResponse:
    """
    §7c.5 — a UX PREFLIGHT only. The upload compare-and-swap is the authority; this can never
    authorize a write.

    §7f.8: no timestamp participates in the decision any more. It compares the client's declared
    base version against the stored version, exactly like the CAS the client is about to attempt.
    """
    server_doc = guide_repository.find_by_guide_id(payload.guideId)

    if not server_doc:
        return SyncConflictResponse(hasConflict=False, conflictType="none", recommendedAction="upload")

    try:
        server_version = mutation.current_version(server_doc)
        client_base = payload.baseVersion if payload.baseVersion is not None else payload.version

        if client_base is not None and int(client_base) == server_version:
            return SyncConflictResponse(hasConflict=False, conflictType="none", recommendedAction="upload")

        return SyncConflictResponse(
            hasConflict=True,
            conflictType="version_conflict",
            serverVersion=GuideSchemaForDB.model_validate(serialize_guide(server_doc)),
            recommendedAction="download_before_upload",
        )
    except Exception as exc:
        log.error(f"❌ Conflict check failed: {exc}", exc_info=True)
        return SyncConflictResponse(
            hasConflict=True,
            conflictType="unknown_error",
            serverVersion=GuideSchemaForDB.model_validate(serialize_guide(server_doc)),
            recommendedAction="download_before_upload",
        )


def _incoming_document(payload: GuideSchema, guide_id: str) -> dict:
    """
    The client's guide as a plain dict, with the server-owned metadata stripped.

    `extra="allow"` means every unknown extension a newer client sent is still in here (§7d) —
    only the fields the SERVER owns are removed, because a client may not choose its own revision
    (§1c "no client-computed revision number trusted by the server").
    """
    incoming = payload.model_dump(by_alias=True, exclude_unset=True)

    server_owned = {"version", "baseVersion", "contentHash", "_id", "databaseId", "createdAt", "updatedAt"}
    doc = {key: value for key, value in incoming.items() if key not in server_owned}
    doc["guideId"] = guide_id

    # C2 §2l: the store lookup isn't here. This has no stored document, and it belongs after the stored
    # identity is filled in (`merge_authoring_document`).
    return dedupe_step_objects_by_lineage(doc)


def incoming_authoring_document(payload: GuideSchema) -> dict:
    """
    C1 Slice 3 (§8a.4) — the SAME normalisation `sync_guide_cas` applies to an incoming guide,
    exposed for the upload-attempt commit path.

    Shared rather than reimplemented on purpose: an attempt commit is the same authoring write as
    the CAS upload, only with media published alongside it. If it built its own document it could
    drift into accepting a client-computed version or reverting a server-side translation — the two
    things §7a/§1c exist to prevent.
    """
    doc = _incoming_document(payload, payload.guideId)
    doc.pop("translations", None)
    return doc


def _with_stored_identity(current: dict, incoming: dict) -> dict:
    """
    C2 §2k — a client that doesn't hold assetId/authoredHash yet sends them empty, and that isn't a
    change. Each empty value is filled from the stored object with the same objectId (in document
    order when an id repeats), so the merged content and its hash stay the same. A hash is only
    taken over for the same asset. `incoming` isn't changed.

    §2l R3: only for the same model, by prefabName without `(Clone)`. An object that changed model gets
    its new identity from the store (`hydrate_missing_model_paths`).
    """
    stored_by_id: dict[Any, list[dict]] = {}
    for step in current.get("steps") or []:
        for obj in (step or {}).get("objects") or []:
            if isinstance(obj, dict) and obj.get("objectId"):
                stored_by_id.setdefault(obj["objectId"], []).append(obj)

    filled_steps = []
    for step in incoming["steps"]:
        if not isinstance(step, dict) or not isinstance(step.get("objects"), list):
            filled_steps.append(step)
            continue

        filled_objects = []
        for obj in step["objects"]:
            matches = stored_by_id.get(obj.get("objectId")) if isinstance(obj, dict) else None
            stored = matches.pop(0) if matches else None
            if stored is not None and prefab_without_clone(obj.get("prefabName")) == prefab_without_clone(
                stored.get("prefabName")
            ):
                obj = dict(obj)
                if not obj.get("assetId") and stored.get("assetId"):
                    obj["assetId"] = stored["assetId"]
                if (
                    not obj.get("authoredHash")
                    and stored.get("authoredHash")
                    and (obj.get("assetId") or "") == (stored.get("assetId") or "")
                ):
                    obj["authoredHash"] = stored["authoredHash"]
            filled_objects.append(obj)
        filled_steps.append({**step, "objects": filled_objects})

    return {**incoming, "steps": filled_steps}


def merge_authoring_document(current: dict, incoming: dict) -> dict:
    """
    §7a.1 — merge an incoming authoring document over the freshly re-read stored one, keeping the
    server-derived `translations` subtree the client never authors. Identical to `sync_guide_cas`'s
    own `_apply`, shared for the attempt commit path.

    C2 §2l: every path that stores a client's document passes here, a new guide with `current` empty.
    The order is load-bearing, see `hydrate_missing_model_paths`. `incoming` isn't changed.
    """
    merged = dict(current or {})
    if current and isinstance((incoming or {}).get("steps"), list):
        incoming = _with_stored_identity(current, incoming)
    if isinstance((incoming or {}).get("steps"), list):
        incoming = hydrate_missing_model_paths(deepcopy(incoming))
    merged.update(incoming or {})
    if "translations" in (current or {}):
        merged["translations"] = current["translations"]
    return merged


def sync_guide_cas(
    payload: GuideSchema,
    expected_base_version: int | None,
) -> tuple[GuideSchemaForDB, dict | None, bool]:
    """
    §7a/§7c.1 — the compare-and-swap guide upload.

    Returns:
        (authoritative guide, previous snapshot or None, content actually changed)

    Raises the mutation service's precondition errors, which the route turns into 409/428. A stale
    write NEVER returns success and never mutates anything (§7a.6, §1c).

    The client's `translations` subtree is deliberately NOT accepted here: translations are
    server-derived (§7a.1). Dropping the client's copy is what stops an authoring save from
    silently reverting a translation that landed while the client was editing (F7's other half).
    """
    guide_id = payload.guideId
    doc_to_set = _incoming_document(payload, guide_id)
    doc_to_set.pop("translations", None)

    existing_guide = guide_repository.find_by_guide_id(guide_id)

    if existing_guide is None:
        # §7a.5 / §7c.2: creation is allowed ONLY with no expected version. A client that presents
        # a base for a guide the server does not have is not creating anything — its guide was
        # deleted, or it is talking to the wrong server. Silently creating it would hide that, so
        # it is a 404 and the client re-enters its own workflow.
        if expected_base_version is not None:
            raise mutation.GuideNotFoundError(
                f"Guide with guideId '{guide_id}' not found; it cannot be updated against "
                f"expected base version {expected_base_version}."
            )

        created = mutation.create_guide(merge_authoring_document({}, doc_to_set))
        return GuideSchemaForDB.model_validate(serialize_guide(created)), None, True

    old_guide_snapshot = dict(existing_guide)

    def _apply(current: dict) -> dict:
        # Derived subtrees stay whatever the server currently holds — the client never authors them.
        return merge_authoring_document(current, doc_to_set)

    final_doc, changed = mutation.mutate_guide(
        guide_id,
        _apply,
        expected_base_version=expected_base_version,
        require_expected_version=True,
    )

    return (
        GuideSchemaForDB.model_validate(serialize_guide(final_doc)),
        old_guide_snapshot if changed else None,
        changed,
    )


def get_guide_meta(guide_id: str) -> GuideMetaResponse:
    doc = guide_repository.find_by_guide_id(guide_id)
    if not doc:
        raise FileNotFoundError("Guide not found")

    meta = mutation.guide_meta(doc)
    return GuideMetaResponse(
        guideId=meta["guideId"],
        updatedAt=meta["updatedAt"],
        version=meta["version"],
        baseVersion=meta["baseVersion"],
        contentHash=meta["contentHash"],
    )


def get_single_guide(guide_id: str) -> GuideSchemaForDB:
    guide_doc = guide_repository.find_by_guide_id(guide_id)
    if not guide_doc:
        raise FileNotFoundError(f"Anleitung mit guideId '{guide_id}' nicht gefunden.")

    return GuideSchemaForDB.model_validate(serialize_guide(guide_doc))


def get_guides_to_pull(request: ClientSyncRequest) -> list[GuideSchemaForDB]:
    """
    Pull SELECTION — which server guides to offer the client for download.

    This is not a conflict decision (§7f.8): the conflict is decided by the CAS on push and by the
    clean/dirty dialog on pull (§7e). When the client reports the version it holds, that decides;
    `updatedAt` survives only as a fallback for a legacy client that reports no version.
    """
    client_guides_map = {guide.guideId: guide for guide in request.localGuides}
    guides_to_send: list[GuideSchemaForDB] = []

    for server_guide_doc in guide_repository.find_all():
        server_guide = GuideSchemaForDB.model_validate(serialize_guide(server_guide_doc))

        client_info = client_guides_map.get(server_guide.guideId)
        if client_info is None:
            guides_to_send.append(server_guide)
            continue

        if client_info.version is not None:
            if mutation.current_version(server_guide_doc) > int(client_info.version):
                guides_to_send.append(server_guide)
            continue

        try:
            client_dt = _parse_iso_datetime(client_info.updatedAt)
            server_dt = _parse_iso_datetime(server_guide.updatedAt)

            if server_dt > client_dt:
                guides_to_send.append(server_guide)
        except (ValueError, TypeError) as exc:
            log.warning(
                f"Could not compare timestamps for guide '{server_guide.title}': {exc}. "
                f"Sending guide as fallback."
            )
            guides_to_send.append(server_guide)

    return guides_to_send


def delete_guide(guide_id: str, expected_base_version: int | None = None) -> None:
    """
    §7b — a user-driven delete is a guide mutation and runs inside the same per-guide critical
    section, so it cannot interleave with a CAS write. An expected base is enforced when supplied.
    """
    existing_guide = guide_repository.find_by_guide_id(guide_id)
    if not existing_guide:
        return

    deleted = mutation.delete_guide(guide_id, expected_base_version=expected_base_version)
    if not deleted:
        return

    try:
        delete_guide_directory(guide_id)
    except Exception as exc:
        log.error(f"Error deleting guide directory {guide_id}: {exc}", exc_info=True)


# =========================================================================
#  C1 Slice 4 (§9c.11, B-081) — the web conflict copy
# =========================================================================
#
# "Save mine as a copy": a NEW guide built from the editor's visible document, with its own
# guideId, its own databaseId and its OWN copies of every media file the document references.
# Web media are per guide and deleting a guide `rmtree`s its folder (S4-4), so a copy that only
# shared the original's URLs would break the moment the original is deleted.
#
# Order, locks and failure (R2-2), exactly:
#   (1) read the source document + its media_index.json under the SOURCE lock only, briefly;
#   (2) copy the media into the fresh target folder with NO lock held (the new GUID has no
#       contender);
#   (3) write the target's media_index.json rows;
#   (4) `create_guide` LAST — the target's instruction/instruction.json is the visibility signal,
#       so the copy is listed only once it is complete.
# The source lock is never taken while the target lock is held. Any failure removes the target
# folder: no half-created guide, no orphaned files.

#: R2-3 (amended 4A-1) — identity/lineage fields stripped before creation, so the copy gets its own
#: database identity, starts a new lineage and is never selected by PDF reprocessing
#: (`find_by_source_file_id` returns the first match).
CONFLICT_COPY_STRIPPED_FIELDS = (
    "databaseId",
    "_id",
    "version",
    "baseVersion",
    "contentHash",
    "createdAt",
    "translations",
    "source_file_id",
)

#: The per-guide media folders a document can reference. `instruction` is not media.
_CONFLICT_COPY_MEDIA_TYPES = ("cad_models", "images", "videos", "audio", "hands")


class ConflictCopySourceMissingError(Exception):
    """The source guide, or a media file its document references, does not exist. → 404."""


def conflict_copy_title(title: str | None, when: datetime | None = None) -> str:
    """'<title> (Konfliktkopie <date time>)' — visibly distinct in the guide list (§9c.11)."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M")
    return f"{title or 'Unbenannte Anleitung'} (Konfliktkopie {stamp})"


def _source_media_reference_pattern(source_guide_id: str) -> re.Pattern[str]:
    """
    A reference into the SOURCE guide's own media folder, with or without the `/api` prefix and
    with or without a host in front of it. The generators' `/api/assets/…` form counts too (C2 §2k,
    A7), but a file missing behind only that form doesn't stop the copy (review R5). References to
    other guides or to global assets do not match and are left untouched.
    """
    return re.compile(
        r"(/sync/assets/|/api/assets/)"
        + re.escape(source_guide_id)
        + r"/("
        + "|".join(_CONFLICT_COPY_MEDIA_TYPES)
        + r")/([^?#\s\"']+)"
    )


def _rewrite_source_media_references(
    value: Any,
    pattern: re.Pattern[str],
    new_guide_id: str,
    found: dict[tuple[str, str], bool],
) -> Any:
    """
    Rewrite every source-media URL in the document to the new guide and collect each file, with
    whether the copy needs it: true unless only `/api/assets/…` refers to it (C2 §2k, review R5).
    """
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            key = (match.group(2), match.group(3))
            found[key] = found.get(key, False) or match.group(1) != "/api/assets/"
            return f"{match.group(1)}{new_guide_id}/{match.group(2)}/{match.group(3)}"

        return pattern.sub(_replace, value)
    if isinstance(value, dict):
        return {
            key: _rewrite_source_media_references(item, pattern, new_guide_id, found)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_source_media_references(item, pattern, new_guide_id, found) for item in value]
    return value


def _conflict_copy_media_row(
    source_rows: list[dict[str, Any]],
    new_guide_id: str,
    asset_type: str,
    file_name: str,
    url_name: str,
    target_path: Any,
) -> dict[str, Any]:
    """
    The target's index row for one copied file: the source row when there is one (so objectId,
    contentHash and the rest survive), otherwise a plain row of the shape reindex writes. Either way
    the row gets a FRESH media id, so a later delete of one guide's medium cannot hit the other.
    """
    match = next(
        (
            row
            for row in source_rows
            if str(row.get("assetType") or "") == asset_type
            and str(row.get("uniqueFileName") or "") == file_name
        ),
        None,
    )

    if match is not None:
        row = deepcopy(match)
    else:
        content_type, _ = mimetypes.guess_type(file_name)
        row = {
            "assetType": asset_type,
            "fileName": file_name,
            "uniqueFileName": file_name,
            "contentType": content_type or "application/octet-stream",
        }

    row.pop("_id", None)
    row.pop("id", None)
    row.update(
        {
            "guideId": new_guide_id,
            "projectName": new_guide_id,
            "scope": "project",
            "filePath": str(target_path),
            "fileUrl": f"/api/sync/assets/{new_guide_id}/{asset_type}/{url_name}",
            "fileSize": target_path.stat().st_size,
            "uploadDate": now_iso(),
        }
    )
    return row


def _remove_conflict_copy_target(target_root: Any) -> None:
    try:
        if target_root.exists():
            shutil.rmtree(target_root)
    except Exception as exc:
        log.error(f"Konfliktkopie: Zielordner {target_root} konnte nicht entfernt werden: {exc}", exc_info=True)


def create_conflict_copy(source_guide_id: str, document: dict[str, Any]) -> dict:
    """
    §9c.11 — create the conflict copy and return the stored copy document.

    Raises `ConflictCopySourceMissingError` when the source guide or a referenced media file is
    gone — including when the source is deleted between steps (1) and (2).
    """
    source_id = str(source_guide_id or "").strip()
    if not source_id:
        raise ValueError("Conflict copy requires the source guideId.")

    # (1) Source lock only, briefly. No file is copied while it is held.
    source_doc, source_rows = mutation.read_guide_for_copy(source_id)
    if source_doc is None:
        raise ConflictCopySourceMissingError(
            f"Die Originalanleitung '{source_id}' existiert nicht mehr; die Kopie wurde nicht angelegt."
        )

    new_guide_id = str(uuid4())

    copy_doc = {
        key: deepcopy(value)
        for key, value in (document or {}).items()
        if key not in CONFLICT_COPY_STRIPPED_FIELDS
    }
    referenced: dict[tuple[str, str], bool] = {}
    copy_doc = _rewrite_source_media_references(
        copy_doc, _source_media_reference_pattern(source_id), new_guide_id, referenced
    )
    copy_doc["guideId"] = new_guide_id
    copy_doc["title"] = conflict_copy_title(str(copy_doc.get("title") or ""))
    # C2 §2k: a copy keeps its source's format, also when that is none.
    copy_doc["formatVersion"] = source_doc.get("formatVersion")

    target_root = resolve_safe_path(PROJECT_DATA_DIR, new_guide_id)

    try:
        # (2) Media copy with NO lock held.
        copied: list[tuple[str, str, str, Any]] = []
        for (asset_type, url_name), required in referenced.items():
            file_name = unquote(url_name)
            source_path = resolve_project_asset_path(source_id, asset_type, file_name)
            target_path = resolve_project_asset_path(new_guide_id, asset_type, file_name)
            if not source_path.is_file():
                if not required:
                    log.warning(
                        "Konfliktkopie aus %s: generierte Datei '%s/%s' fehlt und wird übersprungen.",
                        source_id, asset_type, file_name,
                    )
                    continue
                raise ConflictCopySourceMissingError(
                    f"Mediendatei '{asset_type}/{file_name}' der Originalanleitung fehlt; "
                    "die Kopie wurde nicht angelegt."
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source_path, target_path)
            except FileNotFoundError as exc:
                if not required:
                    log.warning(
                        "Konfliktkopie aus %s: generierte Datei '%s/%s' fehlt und wird übersprungen.",
                        source_id, asset_type, file_name,
                    )
                    continue
                raise ConflictCopySourceMissingError(
                    f"Mediendatei '{asset_type}/{file_name}' der Originalanleitung ist während des "
                    "Kopierens verschwunden; die Kopie wurde nicht angelegt."
                ) from exc
            copied.append((asset_type, file_name, url_name, target_path))

        # (3) The target's media_index.json rows.
        for asset_type, file_name, url_name, target_path in copied:
            media_repository.insert_project_media(
                new_guide_id,
                _conflict_copy_media_row(
                    source_rows, new_guide_id, asset_type, file_name, url_name, target_path
                ),
            )

        # (4) LAST: the guide JSON — the visibility signal. Takes only the TARGET lock.
        created = mutation.create_guide(copy_doc)
    except BaseException:
        _remove_conflict_copy_target(target_root)
        raise

    log.info(
        "Konfliktkopie %s aus %s angelegt (%d Mediendatei(en) kopiert).",
        new_guide_id, source_id, len(copied),
    )
    return created


# =========================================================================
#  Auto-Retranslation: nur geänderte Texte neu übersetzen
# =========================================================================

def _extract_text_fields(guide: dict | None) -> dict[str, str]:
    """
    Extrahiert alle Textfelder aus einem Guide als flaches Dict.
    Key: '{step_idx}:{obj_index}' -> Value: text
    """
    texts: dict[str, str] = {}
    if not guide:
        return texts

    for step in guide.get("steps") or []:
        step_idx = str(step.get("step", 0))
        for obj_idx, obj in enumerate(step.get("objects") or []):
            if obj.get("isTextField"):
                texts[f"{step_idx}:{obj_idx}"] = str(obj.get("text") or "")

    return texts


def _detect_text_changes(
    old_guide: dict | None,
    new_guide: dict | None,
) -> tuple[list[str], bool]:
    """
    Vergleicht Textfelder zwischen alter und neuer Anleitung.

    Returns:
        (changed_texts, title_changed)
        changed_texts: Liste der neuen Texte, die sich geändert haben (in Reihenfolge).
        title_changed: True wenn sich der Titel geändert hat.
    """
    old_texts = _extract_text_fields(old_guide)
    new_texts = _extract_text_fields(new_guide)

    changed_texts: list[str] = []

    for key, new_text in new_texts.items():
        old_text = old_texts.get(key, "")
        if old_text != new_text:
            changed_texts.append(new_text)

    old_title = str((old_guide or {}).get("title") or "")
    new_title = str((new_guide or {}).get("title") or "")
    title_changed = old_title != new_title

    return changed_texts, title_changed


def retranslate_changed_texts_task(
    guide_id: str,
    old_guide: dict | None,
    new_guide: dict | None,
) -> None:
    """
    Background-Task: Übersetzt nur geänderte Texte für alle bestehenden Übersetzungen neu.
    Läuft asynchron nach dem Speichern - blockiert den Client nicht.
    """
    if not old_guide or not new_guide:
        return

    translations = new_guide.get("translations") or {}
    if not translations:
        return

    changed_texts, title_changed = _detect_text_changes(old_guide, new_guide)
    if not changed_texts and not title_changed:
        return

    log.info(
        "[AUTO-RETRANSLATE] Guide %s: %d geänderte Text(e), title_changed=%s, %d Sprache(n)",
        guide_id, len(changed_texts), title_changed, len(translations),
    )

    from kiara.guides.translate import GuideTranslator as OllamaClient
    ollama = OllamaClient()

    old_texts_map = _extract_text_fields(old_guide)
    new_texts_map = _extract_text_fields(new_guide)

    for lang, snapshot in translations.items():
        try:
            _retranslate_single_language(
                guide_id, lang, snapshot,
                old_texts_map, new_texts_map,
                changed_texts, title_changed,
                ollama,
            )
        except Exception as exc:
            log.warning(
                "[AUTO-RETRANSLATE] Fehler beim Neu-Übersetzen von %s für Guide %s: %s",
                lang, guide_id, exc, exc_info=True,
            )


def _retranslate_single_language(
    guide_id: str,
    language: str,
    snapshot: dict[str, Any],
    old_texts_map: dict[str, str],
    new_texts_map: dict[str, str],
    changed_texts: list[str],
    title_changed: bool,
    ollama,
) -> None:
    """
    Aktualisiert eine einzelne Übersetzung: nur geänderte Texte neu übersetzen,
    unveränderte Texte im Snapshot erhalten.
    """
    # 1. Collect texts to translate for this language
    texts_to_translate: list[str] = []
    text_keys_to_update: list[str] = []

    for key, new_text in new_texts_map.items():
        old_text = old_texts_map.get(key, "")
        if old_text != new_text:
            texts_to_translate.append(new_text)
            text_keys_to_update.append(key)

    # 2. Translate changed texts via LLM
    translated_texts: list[str] = []
    if texts_to_translate:
        translated_texts = ollama.translate_texts_batch(texts_to_translate, language)

    # 3. Translate title if changed
    translated_title = None
    if title_changed:
        new_title = str(new_texts_map.get("__title__") or "")
        # Use the new guide title directly
        from kiara.guides.repository import guide_repository as _repo
        current_guide = _repo.find_by_guide_id(guide_id)
        new_title = str((current_guide or {}).get("title") or "")
        if new_title:
            title_translations = ollama.translate_texts_batch([new_title], language)
            translated_title = title_translations[0] if title_translations else None

    # 4. Merge: update snapshot steps with new translated texts
    snapshot_steps = snapshot.get("steps") or []
    updated_steps = []
    text_update_map = dict(zip(text_keys_to_update, translated_texts, strict=False))

    for step in snapshot_steps:
        step_idx = str(step.get("step", 0))
        updated_objects = []
        for obj_idx, obj in enumerate(step.get("objects") or []):
            if obj.get("isTextField"):
                key = f"{step_idx}:{obj_idx}"
                if key in text_update_map:
                    obj = {**obj, "text": text_update_map[key]}
            updated_objects.append(obj)
        updated_steps.append({**step, "objects": updated_objects})

    snapshot["steps"] = updated_steps
    snapshot["translatedAt"] = now_iso()

    if translated_title is not None:
        snapshot["title"] = translated_title

    # 5. Derived Unity-guide output FIRST — it is file work and must stay OUTSIDE the lock (§7b).
    #    Doing it here rather than after the JSON commit is the sequencing change §7 requires: the
    #    resulting unityGuidePath becomes part of the SAME single derived write below, so the stored
    #    guide never advertises a translation whose Unity output has not been produced yet.
    try:
        from kiara.guides.translate import translate_unity_guide
        unity_path = translate_unity_guide(guide_id, language, snapshot)
        if unity_path:
            snapshot["unityGuidePath"] = unity_path
    except Exception as exc:
        log.warning(
            "[AUTO-RETRANSLATE] Unity-Guide-Update für %s/%s fehlgeschlagen: %s",
            guide_id, language, exc,
        )

    # 6. F7 FIX — the short derived commit.
    #
    #    Was: `_repo.update_guide(guide_id, {"translations": all_translations, "updatedAt": ...})`
    #    read the guide, merged and wrote with NO lock, so it could race a concurrent authoring
    #    save and drop it, and the stored version/contentHash could stop describing the stored
    #    content. It also carried its own stale `all_translations` snapshot across the LLM call.
    #
    #    Now: the same per-guide critical section every other writer uses. It RE-READS the current
    #    guide inside the lock and merges only THIS language entry — a concurrent authoring change
    #    to steps/title made during the LLM call survives, because this writer never replaces them
    #    from its own earlier snapshot (§7b). `derived_only=True` preserves the authoring
    #    version/contentHash, so the client's outstanding base does not go stale behind its back
    #    and its next N-based save still succeeds (§7f.9).
    def _merge_language(current: dict) -> dict:
        merged = dict(current)
        translations_now = dict(merged.get("translations") or {})
        translations_now[language] = snapshot
        merged["translations"] = translations_now
        return merged

    try:
        mutation.mutate_guide(guide_id, _merge_language, derived_only=True)
    except mutation.GuideNotFoundError:
        return

    log.info(
        "[AUTO-RETRANSLATE] Guide %s: Übersetzung '%s' aktualisiert (%d Text(e), title=%s)",
        guide_id, language, len(translated_texts), title_changed,
    )
