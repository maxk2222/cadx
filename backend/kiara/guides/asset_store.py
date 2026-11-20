"""C2 §2l: the shared model store, `uploads/asset_store/<assetId>/`.

One folder per model, holding `asset.json`, `model-<hash>.glb` and `source/<sourceHash>/<original name>`.
There's no shared index and no stored path: the list is a scan of the folders. A new asset commits with
one folder rename; a replacement copies its files in, swaps `asset.json` and deletes nothing, so a reader
holding the old record still finds its file. The next start removes what the record doesn't name.

Single-process only (§2l): the lock doesn't reach a second worker. Conversion and the check run before
it, off the event loop; the lock covers only the final name and content check and the commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from kiara.guides.file_store import resolve_safe_path, validate_upload_filename, validate_uuid_id
from kiara.guides.glb_tools import CanonicalGlb, canonicalize_glb
from kiara.guides.model_conversion import convert_upload_to_glb
from kiara.guides.paths import (
    ASSET_STORE_DIR,
    ASSET_STORE_STAGING_DIR,
    GLOBAL_ASSET_CATEGORIES,
    media_cap_bytes,
)
from kiara.guides.persistence import atomic_write_json, now_iso, read_json_file
from kiara.guides.repository import guide_repository
from kiara.guides.upload_errors import (
    InvalidUploadError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)

_RECORD_FILE = "asset.json"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_UPLOAD_EXTENSIONS = (".glb", ".fbx", ".stl", ".obj")
# `FunctionalPrefabs` in `BuiltInModelExporter.cs`. The headset knows them, and they have no record here.
_FUNCTIONAL_PREFABS = (
    "P_MovableTextFieldWithButton",
    "P_Photograph",
    "P_Videograph",
    "P_Voicegraph",
    "P_LeftHand",
    "P_RightHand",
)

_store_lock = threading.Lock()


class AssetNotFoundError(Exception):
    """No readable record for this assetId. → 404."""


class AssetConflictError(Exception):
    """A refusal the author has to resolve. → 409, with `detail` as the reply."""

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__(str(detail.get("reason")))
        self.detail = detail


def _asset_path(asset_id: str, *parts: str) -> Path:
    return resolve_safe_path(ASSET_STORE_DIR, asset_id, *parts)


def _staging_path(staging_id: str, *parts: str) -> Path:
    return resolve_safe_path(ASSET_STORE_STAGING_DIR, staging_id, *parts)


def _model_file_name(content_hash: str) -> str:
    return f"model-{content_hash}.glb"


def _checked_asset_id(asset_id: str) -> str:
    # Lower case because NTFS would find the same folder under an upper-case id anyway.
    return validate_uuid_id(asset_id, "assetId").lower()


def _read_record(asset_id: str) -> dict[str, Any] | None:
    """The record, or None with a log line if it's missing or can't be used."""
    try:
        record = read_json_file(_asset_path(asset_id, _RECORD_FILE), default=None)
    except (OSError, ValueError) as exc:
        log.warning(f"Asset store: skipping '{asset_id}', its record can't be read: {exc}")
        return None

    if (
        not isinstance(record, dict)
        or str(record.get("assetId") or "").lower() != asset_id.lower()
        or not _HASH_RE.match(str(record.get("currentHash") or ""))
        or not isinstance(record.get("displayName"), str)
    ):
        log.warning(f"Asset store: skipping '{asset_id}', it has no usable {_RECORD_FILE}.")
        return None
    return record


def _asset_folder_names() -> list[str]:
    if not ASSET_STORE_DIR.exists():
        return []
    # `.staging` isn't a model, half-written or not.
    return sorted(
        path.name for path in ASSET_STORE_DIR.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def _read_all_records() -> list[dict[str, Any]]:
    records = [_read_record(name) for name in _asset_folder_names()]
    return [record for record in records if record is not None]


def list_store_assets() -> list[dict[str, Any]]:
    return sorted(_read_all_records(), key=lambda record: record["displayName"].casefold())


def _require_record(asset_id: str) -> dict[str, Any]:
    record = _read_record(asset_id) if _asset_path(asset_id).is_dir() else None
    if record is None:
        raise AssetNotFoundError(f"Asset '{asset_id}' not found.")
    return record


def get_store_asset(asset_id: str) -> dict[str, Any]:
    return _require_record(_checked_asset_id(asset_id))


def get_store_model_file(asset_id: str) -> tuple[Path, str]:
    """The current canonical file and its hash."""
    record = get_store_asset(asset_id)
    path = _asset_path(record["assetId"], _model_file_name(record["currentHash"]))
    if not path.is_file():
        log.error(f"Asset store: '{record['assetId']}' names a model file that isn't there.")
        raise AssetNotFoundError(f"Asset '{asset_id}' has no model file.")
    return path, record["currentHash"]


def store_model_url(asset_id: str) -> str:
    """What a guide holds as `modelLocation`. It names no version: the same URL serves a replacement."""
    return f"/api/sync/store/{asset_id}/model.glb"


def get_store_source_file(record: dict[str, Any]) -> Path | None:
    """The upload as received, or None if there is none: a built-in has no source."""
    source_name = str(record.get("sourceFileName") or "")
    source_hash = str(record.get("sourceHash") or "")
    if not source_name or not source_hash:
        return None
    path = _asset_path(record["assetId"], "source", source_hash, source_name)
    return path if path.is_file() else None


# ── Checks on what's sent ────────────────────────────────────────────────────────────────────────


def _checked_upload_name(file: UploadFile) -> str:
    upload_name = validate_upload_filename(file.filename)
    ext = Path(upload_name).suffix.lower()
    if ext == ".gltf":
        raise UnsupportedMediaTypeError("A .gltf file can't be stored on its own: upload as .glb.")
    if ext not in _UPLOAD_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"This file type can't be uploaded to the model store. Allowed: {', '.join(_UPLOAD_EXTENSIONS)}"
        )
    return upload_name


def _checked_category(category: str) -> str:
    if category not in GLOBAL_ASSET_CATEGORIES:
        raise InvalidUploadError(
            f"Unknown category '{category}'. Allowed: {', '.join(GLOBAL_ASSET_CATEGORIES)}"
        )
    return category


def _default_category(display_name: str) -> str:
    name = display_name.casefold()
    return "markers" if "marker" in name or "arrow" in name else "tools"


def _refuse_taken_name(records: list[dict[str, Any]], display_name: str, own_id: str | None = None) -> None:
    wanted = display_name.casefold()
    for record in records:
        if record["assetId"] != own_id and record["displayName"].casefold() == wanted:
            raise AssetConflictError({"reason": "name", "existing": record})


def _refuse_reserved_name(records: list[dict[str, Any]], display_name: str) -> None:
    """
    S9-4: the web and the headset's picker insert an upload as prefab `<displayName>`. If the headset knows
    that prefab, it shows the compiled one instead of this upload. A plain refusal, not a `name` conflict:
    no record holds the name, so there's nothing the prompt could offer to replace.
    The headset also takes `<prefab>(Clone)` as that prefab (`GuideReplacementCoordinator.cs`), so that
    trailing form counts as the prefab here.
    """
    reserved = {name.casefold() for name in _FUNCTIONAL_PREFABS}
    reserved.update(
        record["prefabName"].casefold() for record in records if record.get("builtIn") and record.get("prefabName")
    )
    name = display_name.casefold().removesuffix("(clone)")
    if name in reserved:
        raise InvalidUploadError(
            f"The name '{display_name}' is reserved: the headset has a built-in model of that name and would "
            "show that one instead of this upload. Rename the file and upload it again."
        )


def _refuse_same_content(records: list[dict[str, Any]], content_hash: str, own_id: str | None = None) -> None:
    for record in records:
        if record["assetId"] != own_id and record["currentHash"] == content_hash:
            raise AssetConflictError({"reason": "content", "existing": record})


# ── Staging ──────────────────────────────────────────────────────────────────────────────────────


async def _stage_upload(file: UploadFile, staging_id: str, upload_name: str) -> str:
    """Stream the body to staging with the size cap (A9) and return its hash."""
    source_path = _staging_path(staging_id, "source", upload_name)
    await run_in_threadpool(source_path.parent.mkdir, parents=True, exist_ok=True)

    cap = media_cap_bytes("cad_models")
    digest = hashlib.sha256()
    size = 0
    buffer = await run_in_threadpool(source_path.open, "wb")
    try:
        while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > cap:
                raise PayloadTooLargeError(f"This file is over the {cap}-byte limit for models.")
            digest.update(chunk)
            await run_in_threadpool(buffer.write, chunk)
    finally:
        buffer.close()
    return digest.hexdigest()


def _canonicalize_converted(staging_id: str, source: Path, ext: str, temporary: Path) -> CanonicalGlb:
    work_dir = _staging_path(staging_id, "convert")
    work_dir.mkdir()
    try:
        canonical = canonicalize_glb(convert_upload_to_glb(source, ext, work_dir), temporary)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # The staging folder can become the asset's folder, and this must not go with it.
    if work_dir.exists():
        raise OSError(f"Asset store: could not remove the conversion folder in staging '{staging_id}'.")
    return canonical


def _stage_canonical(staging_id: str, upload_name: str) -> CanonicalGlb:
    temporary = _staging_path(staging_id, "model.tmp")
    source = _staging_path(staging_id, "source", upload_name)
    ext = Path(upload_name).suffix.lower()
    if ext == ".glb":
        canonical = canonicalize_glb(source, temporary)
    else:
        canonical = _canonicalize_converted(staging_id, source, ext, temporary)
    temporary.replace(_staging_path(staging_id, _model_file_name(canonical.sha256)))
    return canonical


def _remove_staging(staging_id: str) -> None:
    shutil.rmtree(_staging_path(staging_id), ignore_errors=True)


# ── Create ───────────────────────────────────────────────────────────────────────────────────────


def _new_asset_record(
    asset_id: str,
    display_name: str,
    category: str,
    canonical: CanonicalGlb,
    upload_name: str,
    source_hash: str,
) -> dict[str, Any]:
    now = now_iso()
    return {
        "assetId": asset_id,
        "displayName": display_name,
        "category": category,
        "currentHash": canonical.sha256,
        "sizeBytes": canonical.size_bytes,
        "bounds": canonical.bounds,
        "defaultScale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "builtIn": False,
        "prefabName": "",
        "sourceFileName": upload_name,
        "sourceHash": source_hash,
        "createdAt": now,
        "updatedAt": now,
    }


def _finish_staged_asset(record: dict[str, Any]) -> None:
    """The record and the source in their final places inside staging, leaving only the rename."""
    asset_id = record["assetId"]
    atomic_write_json(_staging_path(asset_id, _RECORD_FILE), record)

    # Moved inside staging, so a new asset still commits with one rename.
    source_path = _staging_path(asset_id, "source", record["sourceHash"], record["sourceFileName"])
    source_path.parent.mkdir()
    _staging_path(asset_id, "source", record["sourceFileName"]).replace(source_path)


def _commit_new_asset(
    asset_id: str,
    display_name: str,
    category: str,
    canonical: CanonicalGlb,
    upload_name: str,
    source_hash: str,
) -> dict[str, Any]:
    with _store_lock:
        records = _read_all_records()
        _refuse_taken_name(records, display_name)
        _refuse_reserved_name(records, display_name)
        _refuse_same_content(records, canonical.sha256)

        record = _new_asset_record(asset_id, display_name, category, canonical, upload_name, source_hash)
        _finish_staged_asset(record)

        _staging_path(asset_id).rename(_asset_path(asset_id))
    return record


async def create_store_asset(
    file: UploadFile,
    display_name: str | None,
    category: str | None,
) -> dict[str, Any]:
    """POST /sync/store. The reply is the record plus this upload's `warnings`, which aren't stored."""
    upload_name = _checked_upload_name(file)
    name = validate_upload_filename(display_name or Path(upload_name).stem)
    resolved_category = _checked_category(category) if category else _default_category(name)

    # Saves checking a file that can't go in. The name is checked again under the lock.
    records = await run_in_threadpool(_read_all_records)
    _refuse_taken_name(records, name)
    _refuse_reserved_name(records, name)

    # The staging folder becomes the asset's folder, so it already has the asset's id.
    asset_id = str(uuid4())
    try:
        source_hash = await _stage_upload(file, asset_id, upload_name)
        canonical = await run_in_threadpool(_stage_canonical, asset_id, upload_name)
        record = await run_in_threadpool(
            _commit_new_asset, asset_id, name, resolved_category, canonical, upload_name, source_hash
        )
    finally:
        await run_in_threadpool(_remove_staging, asset_id)

    return {**record, "warnings": canonical.warnings}


# ── Generator inputs, `/upload` (Slices 9 and 9a) ────────────────────────────────────────────────


@dataclass
class _StagedInput:
    asset_id: str
    upload_name: str
    display_name: str
    source_hash: str
    canonical: CanonicalGlb


@dataclass
class _RecordChange:
    """A replacement or an attached source. `previous` is what a failure writes back."""

    item: _StagedInput
    previous: dict[str, Any]
    updated: dict[str, Any]


_COLLISION_ACTIONS = ("useBuiltIn", "replace", "rename")


def parse_collision_decisions(text: str | None) -> dict[str, dict[str, Any]]:
    """Slice 9a: the `/upload` form field `collisions`, keyed by upload file name. Malformed is a 400."""
    if text is None:
        return {}
    try:
        decisions = json.loads(text)
    except ValueError as exc:
        raise InvalidUploadError("Das Feld 'collisions' ist kein gültiges JSON.") from exc
    if not isinstance(decisions, dict):
        raise InvalidUploadError("Das Feld 'collisions' muss ein Objekt mit Dateinamen als Schlüsseln sein.")

    for file_name, decision in decisions.items():
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in _COLLISION_ACTIONS:
            raise InvalidUploadError(f"Unbekannte Entscheidung für '{file_name}'.")
        if action == "rename" and not isinstance(decision.get("name"), str):
            raise InvalidUploadError(f"Für '{file_name}' fehlt der neue Name.")
    return decisions


def _rename_problem(
    requested: str,
    own_stem: str,
    records: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    batch_names: set[str],
) -> tuple[str, dict[str, Any], str] | None:
    """
    Why this name can't be used, for the 409 list. Never a 400: that would lose every other answer.
    The code and its params are for the web's translation (C3 Slice 8-B); the text is the German as before.
    """
    try:
        validate_upload_filename(requested)
    except InvalidUploadError as exc:
        return "invalidName", {"reason": str(exc)}, f"Der Name ist ungültig: {exc}"

    wanted = requested.casefold()
    if wanted != own_stem and wanted in batch_names:
        return "takenInUpload", {}, "Eine andere Datei in diesem Upload hat schon diesen Namen."
    if wanted in by_name:
        return "taken", {}, "Der Name ist schon vergeben."
    try:
        _refuse_reserved_name(records, requested)
    except InvalidUploadError:
        return "reserved", {}, (
            "Der Name ist reserviert: Das Headset hat ein eingebautes Modell mit diesem Namen "
            "und würde dieses statt des hochgeladenen Modells zeigen."
        )
    return None


def _move_change_in(change: _RecordChange) -> None:
    """Under its hash, where nothing names it until the record is written."""
    item = change.item
    asset_id = change.previous["assetId"]
    if not change.previous.get("builtIn"):
        _move_replacement_in(asset_id, item.asset_id, item.canonical, item.upload_name, item.source_hash)
        return

    # Left alone if it's there: a running generator may be copying it, which fails on Windows (S9a-7).
    if not _asset_path(asset_id, "source", item.source_hash, item.upload_name).exists():
        _move_source_in(asset_id, item.asset_id, item.upload_name, item.source_hash)


def _restore_records(changes: list[_RecordChange]) -> None:
    for change in changes:
        asset_id = change.previous["assetId"]
        try:
            atomic_write_json(_asset_path(asset_id, _RECORD_FILE), change.previous)
        except OSError as exc:
            # The others still get their turn, and the failure that got us here is the one reported.
            log.error(f"Asset store: could not restore the record of '{asset_id}': {exc}")


def _commit_generator_inputs(  # noqa: C901 - ported suite contract logic
    staged: list[_StagedInput],
    decisions: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], list[str]]]:
    """
    All or nothing; for each input its record and its warnings. Slice 9a's steps, in file order. The
    decisions are checked against what's here now, never against what the page saw. A file left without
    a usable decision goes in the 409 list, and then nothing moves.
    """
    with _store_lock:
        records = _read_all_records()
        by_hash = {record["currentHash"]: record for record in records}
        by_name = {record["displayName"].casefold(): record for record in records}
        latest = {record["assetId"]: record for record in records}
        batch_names = {item.display_name.casefold() for item in staged}

        results: list[tuple[str, list[str]]] = []
        new_records: list[dict[str, Any]] = []
        changes: list[_RecordChange] = []
        collisions: list[dict[str, Any]] = []
        for item in staged:
            # Step 1: any decision for this file is ignored.
            reused = by_hash.get(item.canonical.sha256)
            if reused is not None:
                results.append((reused["assetId"], []))
                continue

            decision = decisions.get(item.upload_name, {})
            action = decision.get("action")
            stem = item.display_name.casefold()
            taken = by_name.get(stem)
            name = item.display_name

            if action == "rename":
                name = decision["name"]
                refused = _rename_problem(name, stem, records, by_name, batch_names)
                if refused is not None:
                    problem_code, problem_params, problem = refused
                    collisions.append(
                        {
                            "file": item.upload_name,
                            "displayName": item.display_name,
                            "requestedName": name,
                            "problem": problem,
                            "problemCode": problem_code,
                            "problemParams": problem_params,
                        }
                    )
                    continue
                batch_names.add(name.casefold())
            elif taken is None:
                # Step 2: a `replace` or `useBuiltIn` for a name that's free is ignored.
                _refuse_reserved_name(records, name)
            elif taken.get("builtIn") and action == "useBuiltIn":
                results.append((taken["assetId"], []))
                # The same source again: not even the record is rewritten (S9a-7).
                if taken.get("sourceHash") != item.source_hash:
                    updated = {
                        **taken,
                        "sourceFileName": item.upload_name,
                        "sourceHash": item.source_hash,
                        "updatedAt": now_iso(),
                    }
                    changes.append(_RecordChange(item, taken, updated))
                    latest[taken["assetId"]] = updated
                continue
            elif not taken.get("builtIn") and action == "replace":
                updated = _replaced_record(taken, item.canonical, item.upload_name, item.source_hash)
                changes.append(_RecordChange(item, taken, updated))
                latest[taken["assetId"]] = updated
                # A later file with the old content no longer matches this record; one with the new does.
                del by_hash[taken["currentHash"]]
                by_hash[item.canonical.sha256] = updated
                results.append((taken["assetId"], item.canonical.warnings))
                continue
            else:
                collisions.append({"file": item.upload_name, "displayName": item.display_name, "existing": taken})
                continue

            record = _new_asset_record(
                item.asset_id,
                name,
                _default_category(name),
                item.canonical,
                item.upload_name,
                item.source_hash,
            )
            by_hash[item.canonical.sha256] = record
            by_name[name.casefold()] = record
            latest[item.asset_id] = record
            new_records.append(record)
            results.append((item.asset_id, item.canonical.warnings))

        if collisions:
            raise AssetConflictError({"reason": "collisions", "collisions": collisions})

        for record in new_records:
            _finish_staged_asset(record)

        # The order is load-bearing: every file in place with no record touched, then the records.
        moved_folders: list[str] = []
        written: list[_RecordChange] = []
        try:
            for record in new_records:
                _staging_path(record["assetId"]).rename(_asset_path(record["assetId"]))
                moved_folders.append(record["assetId"])
            for change in changes:
                _move_change_in(change)
            for change in changes:
                atomic_write_json(_asset_path(change.previous["assetId"], _RECORD_FILE), change.updated)
                written.append(change)
        except Exception:
            # Files left unnamed go at the next sweep. A replacement written here is seen by every guide
            # using the model, so it can't just stand. Only a process crash can leave one (accepted limit).
            _restore_records(written)
            for asset_id in moved_folders:
                shutil.rmtree(_asset_path(asset_id), ignore_errors=True)
            raise
    return [(latest[asset_id], warnings) for asset_id, warnings in results]


async def store_generator_inputs(
    files: list[UploadFile],
    decisions: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], list[str]]]:
    """
    §2l: every CAD file of one `/upload`, staged and checked first and then committed together. A bad
    file or a name collision the decisions don't settle writes nothing. Each file gets its stem as its
    name unless it's renamed.
    """
    names = [_checked_upload_name(file) for file in files]
    display_names = [validate_upload_filename(Path(name).stem) for name in names]

    # Slice 9a: first, because no decision can help, so nobody is asked about a file that can't be stored.
    stems: set[str] = set()
    for upload_name, display_name in zip(names, display_names, strict=False):
        if display_name.casefold() in stems:
            raise AssetConflictError({"reason": "sameNameInUpload", "file": upload_name})
        stems.add(display_name.casefold())

    asset_ids: list[str] = []
    staged: list[_StagedInput] = []
    try:
        for file, upload_name, display_name in zip(files, names, display_names, strict=False):
            # As in `create_store_asset`: the staging folder can become the asset's folder.
            asset_id = str(uuid4())
            asset_ids.append(asset_id)
            source_hash = await _stage_upload(file, asset_id, upload_name)
            canonical = await run_in_threadpool(_stage_canonical, asset_id, upload_name)
            staged.append(_StagedInput(asset_id, upload_name, display_name, source_hash, canonical))
        return await run_in_threadpool(_commit_generator_inputs, staged, decisions)
    finally:
        for asset_id in asset_ids:
            await run_in_threadpool(_remove_staging, asset_id)


# ── Replace ──────────────────────────────────────────────────────────────────────────────────────


def _require_user_model(asset_id: str) -> dict[str, Any]:
    """§2l: the headset keeps showing a compiled prefab, so a built-in stays as it is."""
    record = _require_record(asset_id)
    if record.get("builtIn"):
        raise AssetConflictError({"reason": "builtIn", "existing": record})
    return record


def _move_source_in(asset_id: str, staging_id: str, upload_name: str, source_hash: str) -> None:
    # In its own folder, so the source the record still names is never touched. An existing target
    # has the same name and bytes.
    source_path = _asset_path(asset_id, "source", source_hash, upload_name)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    _staging_path(staging_id, "source", upload_name).replace(source_path)


def _move_replacement_in(
    asset_id: str,
    staging_id: str,
    canonical: CanonicalGlb,
    upload_name: str,
    source_hash: str,
) -> None:
    # Never overwritten: going back to an earlier version finds its file still there.
    model_path = _asset_path(asset_id, _model_file_name(canonical.sha256))
    if not model_path.exists():
        _staging_path(staging_id, _model_file_name(canonical.sha256)).replace(model_path)

    _move_source_in(asset_id, staging_id, upload_name, source_hash)


def _replaced_record(
    current: dict[str, Any],
    canonical: CanonicalGlb,
    upload_name: str,
    source_hash: str,
) -> dict[str, Any]:
    return {
        **current,
        "currentHash": canonical.sha256,
        "sizeBytes": canonical.size_bytes,
        "bounds": canonical.bounds,
        "sourceFileName": upload_name,
        "sourceHash": source_hash,
        "updatedAt": now_iso(),
    }


def _commit_replacement(
    asset_id: str,
    staging_id: str,
    display_name: str | None,
    category: str | None,
    canonical: CanonicalGlb,
    upload_name: str,
    source_hash: str,
) -> tuple[dict[str, Any], bool]:
    """The record in force afterwards, and whether this call changed it."""
    with _store_lock:
        current = _require_user_model(asset_id)
        if canonical.sha256 == current["currentHash"]:
            return current, False

        records = _read_all_records()
        if display_name is not None:
            _refuse_taken_name(records, display_name, own_id=current["assetId"])
            _refuse_reserved_name(records, display_name)
        _refuse_same_content(records, canonical.sha256, own_id=current["assetId"])

        _move_replacement_in(asset_id, staging_id, canonical, upload_name, source_hash)

        updated = {
            **_replaced_record(current, canonical, upload_name, source_hash),
            "displayName": display_name if display_name is not None else current["displayName"],
            "category": category if category is not None else current.get("category"),
        }
        atomic_write_json(_asset_path(asset_id, _RECORD_FILE), updated)
    return updated, True


async def replace_store_asset(
    asset_id: str,
    file: UploadFile,
    display_name: str | None,
    category: str | None,
) -> dict[str, Any]:
    """PUT /sync/store/{assetId}/file. The name and category stay unless they're sent."""
    checked_id = _checked_asset_id(asset_id)
    await run_in_threadpool(_require_user_model, checked_id)

    upload_name = _checked_upload_name(file)
    name = validate_upload_filename(display_name) if display_name else None
    resolved_category = _checked_category(category) if category else None

    staging_id = str(uuid4())
    try:
        source_hash = await _stage_upload(file, staging_id, upload_name)
        canonical = await run_in_threadpool(_stage_canonical, staging_id, upload_name)
        record, changed = await run_in_threadpool(
            _commit_replacement,
            checked_id,
            staging_id,
            name,
            resolved_category,
            canonical,
            upload_name,
            source_hash,
        )
    finally:
        await run_in_threadpool(_remove_staging, staging_id)

    return {**record, "warnings": canonical.warnings if changed else []}


# ── Built-in models (Slice 8) ────────────────────────────────────────────────────────────────────


def _checked_default_scale(default_scale: Any) -> dict[str, float]:
    if not isinstance(default_scale, dict):
        raise InvalidUploadError("defaultScale must have x, y and z.")
    scale: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        value = default_scale.get(axis)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise InvalidUploadError(f"defaultScale.{axis} must be a finite number above 0.")
        scale[axis] = float(value)
    return scale


def _commit_built_in(
    staging_id: str,
    prefab_name: str,
    display_name: str,
    category: str,
    default_scale: dict[str, float],
    canonical: CanonicalGlb,
) -> tuple[dict[str, Any], str]:
    """The record in force afterwards, and whether it was `created`, `updated` or left `unchanged`."""
    with _store_lock:
        records = _read_all_records()
        current = next((record for record in records if record.get("prefabName") == prefab_name), None)
        own_id = current["assetId"] if current else None
        _refuse_taken_name(records, display_name, own_id=own_id)
        _refuse_same_content(records, canonical.sha256, own_id=own_id)

        fields = {
            "displayName": display_name,
            "category": category,
            "currentHash": canonical.sha256,
            "sizeBytes": canonical.size_bytes,
            "bounds": canonical.bounds,
            "defaultScale": default_scale,
            "builtIn": True,
            "prefabName": prefab_name,
        }
        now = now_iso()

        if current is None:
            record = {
                "assetId": staging_id,
                **fields,
                "sourceFileName": "",
                "sourceHash": "",
                "createdAt": now,
                "updatedAt": now,
            }
            atomic_write_json(_staging_path(staging_id, _RECORD_FILE), record)
            _staging_path(staging_id).rename(_asset_path(staging_id))
            return record, "created"

        asset_id = current["assetId"]
        model_path = _asset_path(asset_id, _model_file_name(canonical.sha256))
        if {**current, **fields} == current and model_path.exists():
            return current, "unchanged"

        # As a replacement: copied in, record swapped, nothing deleted until the next start.
        if not model_path.exists():
            _staging_path(staging_id, _model_file_name(canonical.sha256)).replace(model_path)

        updated = {**current, **fields, "updatedAt": now}
        atomic_write_json(_asset_path(asset_id, _RECORD_FILE), updated)
    return updated, "updated"


def seed_built_in_asset(
    glb_path: Path,
    prefab_name: str,
    display_name: str,
    category: str,
    default_scale: Any,
) -> tuple[dict[str, Any], str, list[str]]:
    """
    `backend/scripts/seed_asset_store.py`: creates or updates one built-in, keeping an `assetId` found by
    `prefabName`. Returns the record, `created`, `updated` or `unchanged`, and the check's warnings.
    """
    validate_upload_filename(prefab_name)
    name = validate_upload_filename(display_name)
    resolved_category = _checked_category(category)
    scale = _checked_default_scale(default_scale)
    if not glb_path.is_file():
        raise InvalidUploadError(f"{glb_path.name} is listed in the manifest but isn't in the folder.")

    # The staging folder becomes a new asset's folder, as in `create_store_asset`.
    staging_id = str(uuid4())
    try:
        _staging_path(staging_id).mkdir(parents=True)
        temporary = _staging_path(staging_id, "model.tmp")
        canonical = canonicalize_glb(glb_path, temporary)
        # On the canonical file: an export's raw size is mostly textures the check scales down.
        cap = media_cap_bytes("cad_models")
        if canonical.size_bytes > cap:
            raise PayloadTooLargeError(f"This model is over the {cap}-byte limit for models.")
        temporary.replace(_staging_path(staging_id, _model_file_name(canonical.sha256)))

        record, outcome = _commit_built_in(staging_id, prefab_name, name, resolved_category, scale, canonical)
    finally:
        _remove_staging(staging_id)
    return record, outcome, canonical.warnings


# ── Delete ───────────────────────────────────────────────────────────────────────────────────────


def _guide_references_asset(guide: dict[str, Any], asset_id: str) -> bool:
    step_lists = [guide.get("steps")]
    translations = guide.get("translations")
    if isinstance(translations, dict):
        step_lists.extend(
            snapshot.get("steps") for snapshot in translations.values() if isinstance(snapshot, dict)
        )

    for steps in step_lists:
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            for obj in step.get("objects") or []:
                if isinstance(obj, dict) and str(obj.get("assetId") or "").strip().lower() == asset_id:
                    return True
    return False


def _titles_of_guides_using(guides: list[dict[str, Any]], asset_id: str) -> list[str]:
    return [
        str(guide.get("title") or guide.get("guideId") or "")
        for guide in guides
        if _guide_references_asset(guide, asset_id)
    ]


def readable_guides() -> list[dict[str, Any]]:
    """
    Every guide, as `guide_repository.find_all` gives them, passing over a file that can't be read. For
    the replies that only inform: `find_all` raises on invalid JSON, and one such file shouldn't take the
    guide list's flags or a replace question with it.
    """
    guides: list[dict[str, Any]] = []
    for raw_doc in guide_repository.find_all():
        if isinstance(raw_doc, dict):
            guides.append(raw_doc)
    guides.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return guides


def guide_titles_using(asset_id: str) -> list[str]:
    """C2 §2q: for the replace questions. Reads every guide, so never under `_store_lock`."""
    return _titles_of_guides_using(readable_guides(), asset_id.strip().lower())


def delete_store_asset(asset_id: str) -> None:
    """
    DELETE /sync/store/{assetId}. Limit (§2l): a push can add a reference just after the scan; the
    guide survives and shows the model as missing.
    """
    checked_id = _checked_asset_id(asset_id)
    record = _require_user_model(checked_id)

    guide_titles = _titles_of_guides_using(guide_repository.find_all(), checked_id)
    if guide_titles:
        raise AssetConflictError({"reason": "referenced", "guides": guide_titles, "existing": record})

    # The folder leaves the store in one rename; what's left of it is only staging from then on.
    trash_id = str(uuid4())
    with _store_lock:
        current = _require_record(checked_id)
        ASSET_STORE_STAGING_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _asset_path(checked_id).rename(_staging_path(trash_id))
        except PermissionError as exc:
            # Windows refuses while a file inside is open, such as a running download. Nothing has moved.
            raise AssetConflictError({"reason": "inUse", "existing": current}) from exc
    _remove_staging(trash_id)


# ── Start-up sweep ───────────────────────────────────────────────────────────────────────────────


def sweep_asset_store() -> None:
    """
    Removes `.staging/*`, and any `model-*.glb` or `source/<folder>` its folder's record doesn't name.
    A folder with no readable record is left entirely alone: its files may be the only copy of something.
    """
    if ASSET_STORE_STAGING_DIR.exists():
        for staged in ASSET_STORE_STAGING_DIR.iterdir():
            _remove_staging(staged.name)
            if staged.exists():
                log.warning(f"Asset store: could not remove staging folder '{staged.name}'.")

    for name in _asset_folder_names():
        record = _read_record(name)
        if record is None:
            continue

        current = _model_file_name(record["currentHash"])
        for model_path in (ASSET_STORE_DIR / name).glob("model-*.glb"):
            if model_path.name == current:
                continue
            try:
                model_path.unlink()
            except OSError as exc:
                log.warning(f"Asset store: could not remove '{model_path.name}' from '{name}': {exc}")

        for source_folder in _asset_path(name, "source").glob("*"):
            if not source_folder.is_dir() or source_folder.name == record.get("sourceHash"):
                continue
            try:
                shutil.rmtree(source_folder)
            except OSError as exc:
                log.warning(f"Asset store: could not remove source '{source_folder.name}' from '{name}': {exc}")
