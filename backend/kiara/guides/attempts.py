"""
C1 Slice 3 (§8) — the guide upload ATTEMPT state machine.

WHAT THIS REPLACES. Until Slice 3 a Unity Push was: POST the whole guide document, then loop over
its media and upload each file separately. Two independent consequences followed, both observed on
device: the guide could land on the server while its media did not (the web editor then 404s every
picture), and the media route's own `{objectId}_{name}` replace-in-place model meant a partial retry
could overwrite bytes that a still-referenced guide pointed at.

THE PROTOCOL (§8a). One Push is one attempt, with exactly ONE visibility point:

    1. POST   /api/sync/guide-upload-attempts                       -> opaque attemptId
    2. PUT    /api/sync/guide-upload-attempts/{id}/media/{mediaKey}  -> one medium, streamed
    3. POST   /api/sync/guide-upload-attempts/{id}/commit            -> THE commit point
    4. DELETE /api/sync/guide-upload-attempts/{id}                   -> abort

Nothing an attempt has received is "on the server" until step 3 succeeds. Steps 1, 2 and 4 touch
ONLY `DATA_DIR/upload_attempts/<attemptId>/`, which is outside every committed guide tree.

THE COMMIT (§8a.4). Under the SAME per-guide lock the compare-and-swap upload uses, and in this
order: recheck the expected version -> verify every declared medium actually landed -> promote the
verified `.part` files to immutable content-addressed finals -> update `media_index.json` -> write
the guide JSON once, atomically. The guide write is the visibility point; if it fails, the finals
this attempt created are removed and the previous index is restored, so prior committed state
survives byte-for-byte (§8b).

WHY PROMOTION RUNS INSIDE THE MUTATION CALLBACK. `guide_mutation_service` forbids expensive work
under its lock — no await, no network, no LLM, no media streaming. Promotion here is none of those:
the megabytes were already streamed and verified during step 2, OUTSIDE the lock. What happens
under the lock is a handful of `os.replace` calls between two directories plus one index write —
the same order of work as the atomic guide write the primitive itself performs. Doing it anywhere
else would put media publication outside the critical section that §8a.4 requires it to be in.

DEPLOYMENT LIMIT: inherited verbatim from `guide_mutation_service`. The lock is process-local; this
module does not make multiple Uvicorn workers safe.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool

from kiara.guides import mutation as mutation
from kiara.guides import sync_ops
from kiara.guides.file_store import (
    build_content_addressed_filename,
    guide_referenced_media_urls,
    resolve_safe_path,
    validate_upload_filename,
    validate_uuid_id,
)
from kiara.guides.media_index import media_repository
from kiara.guides.media_validation import validate_media_content
from kiara.guides.paths import (
    UPLOAD_ATTEMPTS_DIR,
    classify_project_asset,
    ensure_storage_dirs,
    guide_asset_dir,
    guide_dir,
    media_cap_bytes,
    safe_guide_id,
    upload_attempt_dir,
    upload_attempt_manifest_path,
    upload_attempt_media_dir,
)
from kiara.guides.persistence import atomic_write_json, now_iso, read_json_file
from kiara.guides.repository import guide_repository
from kiara.guides.upload_errors import (
    InvalidUploadError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────────────────────────

#: §8a.6 — how long an abandoned attempt directory may sit before cleanup removes it. Generous on
#: purpose: a slow video upload over a headset's Wi-Fi must never have its own staging expire
#: underneath it.
ATTEMPT_MAX_AGE_SECONDS = 6 * 60 * 60

#: §8a.6 "BOUNDED age-based cleanup". One sweep inspects at most this many attempt directories, so
#: cleanup can never turn into an unbounded filesystem walk on a busy server.
ATTEMPT_CLEANUP_SCAN_LIMIT = 200

#: A sweep runs at most this often per process, so a burst of attempt creations does not re-scan
#: the staging root for each one.
ATTEMPT_CLEANUP_MIN_INTERVAL_SECONDS = 60.0

#: Which guide-JSON field an asset type publishes into (§8a.4 "rewrites/returns authoritative
#: URLs"). One medium of each kind per object, exactly as Unity's ObjectInfo models it.
ASSET_TYPE_TO_GUIDE_FIELD = {
    "images": "photoLocation",
    "videos": "videoLocation",
    "audio": "voiceLocation",
    "cad_models": "modelLocation",
    "hands": "handDataLocation",
}

_ATTEMPT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
#: A media key is a client-chosen label. It is validated to a strict, separator-free charset so it
#: can never express a path, on top of `resolve_safe_path`'s containment check.
_MEDIA_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_STATE_DECLARED = "declared"
_STATE_UPLOADED = "uploaded"

_last_cleanup_at = 0.0


# ── Errors ───────────────────────────────────────────────────────────────────────────────────────


class AttemptError(Exception):
    """Base class for every attempt-protocol failure."""


class AttemptNotFoundError(AttemptError):
    """No such attempt (or it already committed/aborted/expired). → HTTP 404."""


class AttemptMediaNotDeclaredError(AttemptError):
    """The media key is not part of this attempt's manifest. → HTTP 404."""


class AttemptIncompleteError(AttemptError):
    """
    §8a.4 — commit was requested while a declared REQUIRED medium had not landed and been
    verified. Nothing is promoted and nothing is committed. → HTTP 409.
    """


class AttemptGuideMismatchError(AttemptError):
    """The committed document is not the guide this attempt was opened for. → HTTP 400."""


# ── Manifest I/O ─────────────────────────────────────────────────────────────────────────────────


def _validate_attempt_id(attempt_id: str) -> str:
    value = str(attempt_id or "").strip().lower()
    if not _ATTEMPT_ID_RE.match(value):
        raise AttemptNotFoundError(f"'{attempt_id}' is not a valid attempt id")
    return value


def _validate_media_key(media_key: str) -> str:
    value = str(media_key or "").strip()
    if not _MEDIA_KEY_RE.match(value):
        raise InvalidUploadError("mediaKey contains characters that are not allowed")
    return value


def _read_manifest(attempt_id: str) -> dict[str, Any]:
    path = upload_attempt_manifest_path(attempt_id)
    if not path.is_file():
        raise AttemptNotFoundError(f"upload attempt '{attempt_id}' does not exist")

    manifest = read_json_file(path, default=None)
    if not isinstance(manifest, dict):
        raise AttemptNotFoundError(f"upload attempt '{attempt_id}' has an unreadable manifest")
    return manifest


def _write_manifest(attempt_id: str, manifest: dict[str, Any]) -> None:
    atomic_write_json(upload_attempt_manifest_path(attempt_id), manifest)


def _part_path(attempt_id: str, media_key: str) -> Path:
    return resolve_safe_path(upload_attempt_media_dir(attempt_id), f"{media_key}.part")


# ── 1. Create (§8a.1) ────────────────────────────────────────────────────────────────────────────


def _normalize_descriptor(raw: dict[str, Any], seen_keys: set[str]) -> dict[str, Any]:
    """
    Validate ONE declared medium at declaration time (§8a.2: metadata comes from the manifest, not
    from an untrusted filesystem path later).
    """
    media_key = _validate_media_key(raw.get("mediaKey"))
    if media_key in seen_keys:
        raise InvalidUploadError(f"mediaKey '{media_key}' is declared twice")
    seen_keys.add(media_key)

    object_id = validate_uuid_id(raw.get("objectId"), "objectId")
    file_name = validate_upload_filename(raw.get("fileName"))
    ext = Path(file_name).suffix.lower()

    derived_type = classify_project_asset(ext)
    if derived_type is None:
        raise UnsupportedMediaTypeError(f"file type '{ext or 'unknown'}' is not allowed")

    declared_type = str(raw.get("assetType") or "").strip() or derived_type
    if declared_type != derived_type:
        raise InvalidUploadError(
            f"declared assetType '{declared_type}' does not match the '{ext}' file extension"
        )

    length = raw.get("length")
    if length is not None:
        try:
            length = int(length)
        except (TypeError, ValueError) as exc:
            raise InvalidUploadError("declared length must be an integer") from exc
        if length <= 0:
            raise InvalidUploadError("declared length must be positive")
        cap = media_cap_bytes(derived_type)
        if length > cap:
            # Fail before a single byte crosses the wire.
            raise PayloadTooLargeError(
                f"declared length {length} exceeds the {cap}-byte limit for {derived_type}"
            )

    digest = raw.get("sha256")
    if digest is not None:
        digest = str(digest).strip().lower()
        if not _SHA256_RE.match(digest):
            raise InvalidUploadError("declared sha256 must be 64 lowercase hex characters")

    return {
        "mediaKey": media_key,
        "objectId": object_id,
        "assetType": derived_type,
        "fileName": file_name,
        "ext": ext,
        "declaredLength": length,
        "declaredSha256": digest,
        "required": bool(raw.get("required", True)),
        "state": _STATE_DECLARED,
        "receivedLength": None,
        "sha256": None,
        "contentType": None,
        "finalFileName": None,
        "fileUrl": None,
    }


def create_attempt(
    guide_id: str,
    expected_base_version: int | None,
    media: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    §8a.1 — open an attempt and return its opaque id.

    This validates and reserves; it publishes nothing. The expected base version is recorded here
    and rechecked under the lock at commit — recording it now is not a precondition check.

    An attempt with NO declared media is valid (§8a): a guide edit that touched no media still goes
    through the very same CAS commit path.
    """
    ensure_storage_dirs()
    _maybe_expire_attempts()

    guide = str(guide_id or "").strip()
    if not guide:
        raise InvalidUploadError("guideId is required")
    # C2 Slice 0b: opening an attempt touches no guide folder, so nothing else would check this.
    safe_guide_id(guide)

    base: int | None = None
    if expected_base_version is not None:
        try:
            base = int(expected_base_version)
        except (TypeError, ValueError) as exc:
            raise InvalidUploadError("expectedBaseVersion must be an integer") from exc

    seen: set[str] = set()
    descriptors = [_normalize_descriptor(item or {}, seen) for item in (media or [])]

    attempt_id = uuid4().hex
    upload_attempt_media_dir(attempt_id).mkdir(parents=True, exist_ok=True)

    created = datetime.now(UTC)
    manifest = {
        "attemptId": attempt_id,
        "guideId": guide,
        "expectedBaseVersion": base,
        "createdAt": created.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "expiresAt": (created + timedelta(seconds=ATTEMPT_MAX_AGE_SECONDS))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "media": {item["mediaKey"]: item for item in descriptors},
    }
    _write_manifest(attempt_id, manifest)

    log.info(
        "C1 §8a.1: opened upload attempt %s for guide %s (expectedBaseVersion=%s, %d media).",
        attempt_id, guide, base, len(descriptors),
    )
    return manifest


# ── 2. Stream one medium (§8a.2/§8a.3) ───────────────────────────────────────────────────────────


async def store_attempt_media(
    attempt_id: str,
    media_key: str,
    chunks: AsyncIterator[bytes],
) -> dict[str, Any]:
    """
    §8a.2/§8a.3 — stream a RAW request body into this attempt's own `.part`.

    The bytes are hashed and counted AS THEY ARRIVE and the size cap is enforced mid-stream, so
    nothing is ever fully buffered: peak memory is one chunk, whatever the file's size (§1c "no
    whole-video ReadAllBytes upload", §8d.6).

    Every fact used here — object id, asset type, extension, caps — comes from the manifest entry
    written at declaration time, never from the request path or a client-supplied final path
    (§8a.2). The content itself is then verified against the declared length/hash and its own magic
    bytes before the medium counts as complete.
    """
    aid = _validate_attempt_id(attempt_id)
    key = _validate_media_key(media_key)

    manifest = _read_manifest(aid)
    entry = (manifest.get("media") or {}).get(key)
    if entry is None:
        raise AttemptMediaNotDeclaredError(
            f"media '{key}' is not declared by upload attempt '{aid}'"
        )

    asset_type = str(entry["assetType"])
    cap = media_cap_bytes(asset_type)
    part = _part_path(aid, key)
    part.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    received = 0
    try:
        with part.open("wb") as buffer:
            async for chunk in chunks:
                if not chunk:
                    continue
                received += len(chunk)
                if received > cap:
                    raise PayloadTooLargeError(
                        f"upload exceeds the {cap}-byte limit for {asset_type}"
                    )
                digest.update(chunk)
                buffer.write(chunk)

        if received == 0:
            raise InvalidUploadError("empty upload")

        actual = digest.hexdigest()

        declared_length = entry.get("declaredLength")
        if declared_length is not None and int(declared_length) != received:
            raise InvalidUploadError(
                f"declared length {declared_length} does not match the {received} bytes received"
            )

        declared_sha = entry.get("declaredSha256")
        if declared_sha and declared_sha != actual:
            raise InvalidUploadError(
                "the uploaded bytes do not match the declared sha256"
            )

        # C2 §2k: off the loop. A hand file is parsed whole, and that would stall every other request.
        content_type = await run_in_threadpool(
            validate_media_content, part, asset_type, str(entry.get("ext") or "")
        )
    except BaseException:
        # A failed medium leaves no residue and stays 'declared', so a retry of this one key is a
        # clean overwrite and commit still refuses until it really lands.
        part.unlink(missing_ok=True)
        raise

    entry["state"] = _STATE_UPLOADED
    entry["receivedLength"] = received
    entry["sha256"] = actual
    entry["contentType"] = content_type
    entry["finalFileName"] = build_content_addressed_filename(
        entry["objectId"], actual, str(entry.get("ext") or "")
    )
    manifest["media"][key] = entry
    _write_manifest(aid, manifest)

    log.info(
        "C1 §8a.3: attempt %s staged media '%s' (%d bytes, sha256=%s).",
        aid, key, received, actual,
    )
    return entry


# ── 3. Commit (§8a.4/§8b) — the ONE visibility point ─────────────────────────────────────────────


def _apply_authoritative_urls(doc: dict[str, Any], urls: dict[tuple[str, str], str]) -> None:
    """
    §8a.4 — rewrite the incoming guide document so it points at the server's own final URLs.

    The client's predicted URL is never trusted as the stored one: the server just wrote those
    files and knows their content-addressed names, so it is the server's mapping that reaches disk.
    """
    for step in doc.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for obj in step.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            object_id = str(obj.get("objectId") or "")
            if not object_id:
                continue
            for asset_type, field in ASSET_TYPE_TO_GUIDE_FIELD.items():
                url = urls.get((object_id, asset_type))
                if url:
                    obj[field] = url


def _promote_and_index(
    attempt_id: str,
    manifest: dict[str, Any],
    journal: dict[str, Any],
) -> dict[tuple[str, str], str]:
    """
    §8b — promote every verified medium to its IMMUTABLE content-addressed final and record it in
    `media_index.json`. Runs under the guide lock; see the module docstring for why that is correct.

    A final that already exists is REUSED, never overwritten: the name is the content hash, so an
    existing file at that path already holds exactly these bytes. That is what makes a retry with
    identical media produce one file and one stable URL (§8d.4), and it is also why the rollback
    journal records only finals this attempt actually created (§8b).
    """
    guide_id = str(manifest["guideId"])
    journal["index_snapshot"] = media_repository.snapshot_project_media_index(guide_id)

    urls: dict[tuple[str, str], str] = {}

    for key, entry in sorted((manifest.get("media") or {}).items()):
        if entry.get("state") != _STATE_UPLOADED:
            continue

        asset_type = str(entry["assetType"])
        final_name = str(entry["finalFileName"])
        guide_asset_dir(guide_id, asset_type).mkdir(parents=True, exist_ok=True)
        final_path = resolve_safe_path(guide_dir(guide_id), asset_type, final_name)
        part = _part_path(attempt_id, key)

        if final_path.is_file():
            # Same bytes already published — reuse, do NOT overwrite a prior committed file (§8b).
            log.info("C1 §8b: attempt %s reuses existing final %s.", attempt_id, final_name)
        else:
            if not part.is_file():
                raise AttemptIncompleteError(
                    f"staged media '{key}' disappeared before it could be published"
                )
            part.replace(final_path)
            journal["created_finals"].append(final_path)

        file_url = f"/api/sync/assets/{guide_id}/{asset_type}/{final_name}"
        entry["fileUrl"] = file_url
        urls[(str(entry["objectId"]), asset_type)] = file_url

        media_repository.upsert_project_media_for_attempt(
            guide_id,
            str(entry["objectId"]),
            asset_type,
            {
                "guideId": guide_id,
                "projectName": guide_id,
                "scope": "project",
                "objectId": str(entry["objectId"]),
                "assetType": asset_type,
                "fileName": str(entry["fileName"]),
                "uniqueFileName": final_name,
                "filePath": str(final_path),
                "fileUrl": file_url,
                "fileSize": entry.get("receivedLength"),
                "contentType": entry.get("contentType"),
                "contentHash": str(entry["sha256"]),
                "uploadDate": now_iso(),
            },
        )

    return urls


def _rollback(guide_id: str, journal: dict[str, Any]) -> None:
    """
    §8b — undo a commit that did not reach its visibility point.

    Two rules, both narrow on purpose:
      * only finals THIS attempt created are removed — a reused final belongs to an earlier commit;
      * and even then, never one the canonical guide JSON currently references. That last guard
        matters when two pushes carry byte-identical media: the loser must not delete a file the
        winner's committed guide now points at. §1c's rejected "delete anything unreferenced" sweep
        is the opposite of this — nothing is deleted by inference, only by this attempt's journal.
    """
    referenced = set()
    try:
        referenced = guide_referenced_media_urls(guide_id)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("C1 §8b: could not read referenced media while rolling back: %s", exc)

    for final_path in journal.get("created_finals") or []:
        try:
            url = f"/api/sync/assets/{guide_id}/{final_path.parent.name}/{final_path.name}"
            if url in referenced:
                log.warning(
                    "C1 §8b: keeping %s during rollback — the committed guide references it.",
                    final_path,
                )
                continue
            Path(final_path).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("C1 §8b: could not remove attempt final %s: %s", final_path, exc)

    snapshot = journal.get("index_snapshot")
    if snapshot is not None:
        try:
            media_repository.restore_project_media_index(guide_id, snapshot)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("C1 §8b: could not restore the media index for %s: %s", guide_id, exc)


def commit_attempt(
    attempt_id: str, guide_doc: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """
    §8a.4 — THE commit point. Everything before this published nothing.

    Order, all of it under the same per-guide lock the compare-and-swap upload takes:
      1. verify every declared REQUIRED medium landed and was verified;
      2. recheck the expected base version (the mutation primitive does this before the callback,
         re-reading the stored document inside the lock);
      3. promote verified media to immutable content-addressed finals and update the media index;
      4. rewrite the guide document to the server's authoritative URLs;
      5. write the guide JSON once, atomically.

    If anything after step 2 fails, step 3's effects are undone and prior state survives
    byte-for-byte (§8b). A stale base fails at step 2, so nothing was promoted at all.
    """
    aid = _validate_attempt_id(attempt_id)
    manifest = _read_manifest(aid)
    guide_id = str(manifest["guideId"])

    doc = dict(guide_doc or {})
    incoming_id = str(doc.get("guideId") or "").strip()
    if incoming_id and incoming_id != guide_id:
        raise AttemptGuideMismatchError(
            f"attempt '{aid}' was opened for guide '{guide_id}', not '{incoming_id}'"
        )
    doc["guideId"] = guide_id

    # (1) §8a.4 — "verifies every declared medium". A required medium that never arrived means the
    # guide would commit pointing at a file that is not there, which is exactly what Slice 3 exists
    # to prevent.
    missing = sorted(
        key
        for key, entry in (manifest.get("media") or {}).items()
        if entry.get("required", True) and entry.get("state") != _STATE_UPLOADED
    )
    if missing:
        raise AttemptIncompleteError(
            f"upload attempt '{aid}' cannot commit: {len(missing)} declared medium/media never "
            f"arrived ({', '.join(missing)})"
        )

    expected_base = manifest.get("expectedBaseVersion")
    journal: dict[str, Any] = {"created_finals": [], "index_snapshot": None}

    existing = guide_repository.find_by_guide_id(guide_id)
    old_snapshot = dict(existing) if existing is not None else None

    def _mutate(current: dict[str, Any]) -> dict[str, Any]:
        # Reached ONLY after the primitive re-read the stored document inside the lock and the
        # precondition passed (§7a.6). A stale base never gets here, so it never promotes anything.
        urls = _promote_and_index(aid, manifest, journal)
        merged = sync_ops.merge_authoring_document(current, doc)
        _apply_authoritative_urls(merged, urls)
        return merged

    try:
        if expected_base is None and existing is None:
            # CREATION. `create_guide` asserts absence and takes the same lock, but it has no
            # mutation callback, so promotion happens immediately before it rather than inside it.
            # That window is safe by construction: finals are content-addressed, so a concurrent
            # creation of the same guideId with the same bytes reuses the identical path (and is
            # therefore not journalled for deletion), while different bytes produce different
            # paths that cannot interfere. If `create_guide` then loses the race, the rollback
            # below removes only what this attempt created and never a file the winner's committed
            # guide references.
            urls = _promote_and_index(aid, manifest, journal)
            merged = sync_ops.merge_authoring_document({}, doc)
            _apply_authoritative_urls(merged, urls)
            stored = mutation.create_guide(merged)
            changed = True
        else:
            stored, changed = mutation.mutate_guide(
                guide_id,
                _mutate,
                expected_base_version=expected_base,
                require_expected_version=True,
            )
    except BaseException:
        _rollback(guide_id, journal)
        raise

    # The guide is committed. Everything below is bookkeeping that cannot un-commit it.
    _write_manifest(aid, manifest)
    _delete_attempt_dir(aid)

    log.info(
        "C1 §8a.4: attempt %s committed guide %s at version %s (changed=%s, %d media published).",
        aid, guide_id, stored.get("version"), changed,
        sum(1 for e in (manifest.get("media") or {}).values() if e.get("state") == _STATE_UPLOADED),
    )
    # Same shape `sync_guide_cas` returns, so the route can schedule the SAME background
    # retranslation it already schedules for a CAS upload. Slice 3 replaces the transport Unity
    # uses; it must not silently drop a behaviour that transport had.
    return stored, (old_snapshot if changed else None), changed


# ── 4. Abort (§8a.5) ─────────────────────────────────────────────────────────────────────────────


def _delete_attempt_dir(attempt_id: str) -> bool:
    directory = upload_attempt_dir(attempt_id)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return True


def abort_attempt(attempt_id: str) -> bool:
    """
    §8a.5 — abandon an attempt and delete ONLY paths in that attempt's own directory.

    It cannot touch a committed guide or its media: an attempt directory lives under
    `DATA_DIR/upload_attempts/`, never inside `project_data/<guideId>/`.
    """
    aid = _validate_attempt_id(attempt_id)
    removed = _delete_attempt_dir(aid)
    if removed:
        log.info("C1 §8a.5: upload attempt %s aborted; its staging was deleted.", aid)
    return removed


# ── 5. Expiry (§8a.6) ────────────────────────────────────────────────────────────────────────────


def _attempt_age_seconds(directory: Path, reference: datetime) -> float:
    """Age from the manifest's own `createdAt`, falling back to the directory mtime."""
    manifest = read_json_file(directory / "manifest.json", default=None)
    created = (manifest or {}).get("createdAt") if isinstance(manifest, dict) else None

    if isinstance(created, str) and created:
        try:
            parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return (reference - parsed).total_seconds()
        except ValueError:
            pass

    try:
        return reference.timestamp() - directory.stat().st_mtime
    except OSError:
        return 0.0


def expire_attempts(max_age_seconds: float | None = None, scan_limit: int | None = None) -> int:
    """
    §8a.6 — remove abandoned attempt directories, bounded by age and by how many are inspected.

    It iterates `DATA_DIR/upload_attempts/` and nothing else. It never scans a guide directory and
    never decides a committed file is deletable because it "looks unreferenced" (§1c).
    """
    if not UPLOAD_ATTEMPTS_DIR.is_dir():
        return 0

    cutoff = ATTEMPT_MAX_AGE_SECONDS if max_age_seconds is None else float(max_age_seconds)
    limit = ATTEMPT_CLEANUP_SCAN_LIMIT if scan_limit is None else int(scan_limit)
    reference = datetime.now(UTC)

    removed = 0
    inspected = 0
    for directory in UPLOAD_ATTEMPTS_DIR.iterdir():
        if inspected >= limit:
            break
        if not directory.is_dir():
            continue
        inspected += 1

        if _attempt_age_seconds(directory, reference) < cutoff:
            continue

        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
        log.info("C1 §8a.6: removed abandoned upload attempt %s.", directory.name)

    return removed


def _maybe_expire_attempts() -> None:
    """
    Opportunistic, throttled cleanup hooked to attempt creation.

    HONEST LIMIT: §8a.6 also asks for a sweep at process startup. A startup hook belongs in
    `backend/main.py`, which is outside this campaign's permitted file set, so it is deliberately
    NOT added here. `expire_attempts()` is public and side-effect-bounded precisely so that hook
    can call it in one line once main.py is in scope.
    """
    global _last_cleanup_at

    now = time.monotonic()
    if now - _last_cleanup_at < ATTEMPT_CLEANUP_MIN_INTERVAL_SECONDS:
        return
    _last_cleanup_at = now

    try:
        expire_attempts()
    except Exception as exc:  # pragma: no cover - cleanup must never fail a real upload
        log.warning("C1 §8a.6: attempt cleanup failed (harmless): %s", exc)
