"""THE one guide mutation primitive — ported from the suite's
`guide_mutation_service` (C1 Slice 2, §7a/§7b) onto Postgres row locks.

Every productive guide writer goes through `mutate_guide` / `create_guide` /
`delete_guide` / `replace_guide_document`. There is exactly one such primitive;
do not add a second one and do not write `Guide` rows directly.

What the primitive guarantees, per call (identical to the suite, plus the
process-local RLock is reinforced by `SELECT … FOR UPDATE` on Postgres, so the
critical section also holds across processes — API and arq worker alike):

  1. Per-guide lock (process-local fast path).
  2. Inside it, the guide row is fetched FOR UPDATE — a stale read across an
     await/lock boundary cannot be written back.
  3. The compare-and-swap precondition is checked against that freshly read
     document.
  4. The caller's mutation callback runs (pure dict → dict; no I/O/LLM inside).
  5. `contentHash`/`version` are stamped from CONTENT only.
  6. Exactly one commit.

A rejected precondition raises before step 4 — nothing mutates (§7a.6).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sqlalchemy.exc import IntegrityError

from kiara.guides.media_index import media_repository
from kiara.guides.persistence import authoring_content_hash, now_iso
from kiara.guides.repository import guide_repository
from kiara.guides.schemas import GUIDE_FORMAT_VERSION

log = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────────────


class GuideMutationError(Exception):
    """Base class for every precondition/state failure of the mutation primitive."""


class GuideNotFoundError(GuideMutationError):
    """The guide does not exist. → HTTP 404."""


class GuideAlreadyExistsError(GuideMutationError):
    """Creation was requested but the guideId is already taken. → HTTP 409."""


class ExpectedVersionRequiredError(GuideMutationError):
    """
    §7a.5 — an existing guide was written without an expected base version.
    → HTTP 428 Precondition Required. NEVER a blind overwrite.
    """

    def __init__(self, guide_id: str, current: dict[str, Any]):
        super().__init__(
            f"Guide '{guide_id}' exists; an expectedBaseVersion is required for this write."
        )
        self.guide_id = guide_id
        self.current = current


class GuideVersionConflictError(GuideMutationError):
    """
    §7a.5/§7a.6 — the client's expected base is not the current version.
    → HTTP 409 Conflict, carrying the current authoritative document.
    """

    def __init__(self, guide_id: str, expected: int | None, current: dict[str, Any]):
        super().__init__(
            f"Guide '{guide_id}' has moved on: expected base version {expected}, "
            f"current version is {current.get('version')}."
        )
        self.guide_id = guide_id
        self.expected = expected
        self.current = current


# ── C1 Slice 4 (§9c.13) — conflict evidence for F-048 ──────────────────────────

CONFLICT_LOG_PREFIX = "C1-CONFLICT"


def log_version_conflict(exc: GuideVersionConflictError, *, path: str) -> None:
    """§9c.13 — exactly ONE structured log line per version-conflict 409."""
    log.warning(
        "%s guideId=%s side=backend path=%s baseVersion=%s serverVersion=%s resolution=unknown",
        CONFLICT_LOG_PREFIX,
        exc.guide_id,
        path,
        exc.expected,
        current_version(exc.current),
    )


# ── Per-guide lock registry (in-process fast path) ────────────────────────────

_registry_lock = threading.Lock()
_guide_locks: dict[str, threading.RLock] = {}


def _lock_for(guide_id: str) -> threading.RLock:
    key = str(guide_id or "").strip()
    with _registry_lock:
        existing = _guide_locks.get(key)
        if existing is None:
            existing = threading.RLock()
            _guide_locks[key] = existing
        return existing


# ── Metadata helpers ─────────────────────────────────────────────────────────


def current_version(doc: dict[str, Any] | None) -> int:
    try:
        return int((doc or {}).get("version") or 1)
    except (TypeError, ValueError):
        return 1


def guide_meta(doc: dict[str, Any] | None) -> dict[str, Any]:
    """The version/hash triple a client needs to build its next compare-and-swap."""
    doc = doc or {}
    return {
        "guideId": str(doc.get("guideId") or ""),
        "version": current_version(doc),
        "baseVersion": doc.get("baseVersion", current_version(doc)),
        "contentHash": doc.get("contentHash"),
        "updatedAt": str(doc.get("updatedAt") or ""),
    }


def _stamp_authoring_metadata(
    doc: dict[str, Any],
    previous: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """
    §7a.3/§7a.4 — decide the new revision from CONTENT, never from a timestamp
    and never from a number the client supplied.

    Returns (document, changed). `changed` is False when the canonical authoring
    hash is unchanged: same content twice is success/no-change and does not
    advance the version (§7f.1).
    """
    new_hash = authoring_content_hash(doc)
    previous_hash = (previous or {}).get("contentHash")
    previous_version = current_version(previous) if previous else 0

    if previous is not None and previous_hash == new_hash:
        # No-change: keep the stored revision AND the stored updatedAt, so a
        # repeated identical save is genuinely idempotent.
        doc["version"] = previous_version
        doc["baseVersion"] = previous.get("baseVersion", previous_version)
        doc["contentHash"] = new_hash
        doc["updatedAt"] = previous.get("updatedAt") or now_iso()
        return doc, False

    next_version = previous_version + 1 if previous is not None else 1
    doc["version"] = next_version
    doc["baseVersion"] = next_version
    doc["contentHash"] = new_hash
    doc["updatedAt"] = now_iso()
    return doc, True


# ── The primitive ────────────────────────────────────────────────────────────

MutationFn = Callable[[dict[str, Any]], dict[str, Any]]


def create_guide(doc: dict[str, Any]) -> dict[str, Any]:
    """
    §7a.5/§7b — creation asserts ABSENCE. It carries no expected version by
    definition, and it can never overwrite an existing guide.
    """
    guide_id = str((doc or {}).get("guideId") or "").strip()
    if not guide_id:
        raise ValueError("Guide document requires 'guideId'.")

    with _lock_for(guide_id), guide_repository.locked(guide_id) as (s, row):
        if row is not None:
            raise GuideAlreadyExistsError(
                f"Guide with guideId '{guide_id}' already exists."
            )

        fresh = deepcopy(doc)
        fresh["guideId"] = guide_id
        fresh.pop("version", None)
        fresh.pop("baseVersion", None)
        fresh.pop("contentHash", None)
        fresh.setdefault("createdAt", now_iso())

        stamped, _ = _stamp_authoring_metadata(fresh, previous=None)
        try:
            return guide_repository._commit_doc(s, stamped, row)
        except IntegrityError as exc:
            raise GuideAlreadyExistsError(
                f"Guide with guideId '{guide_id}' already exists."
            ) from exc


def mutate_guide(
    guide_id: str,
    mutate: MutationFn,
    *,
    expected_base_version: int | None = None,
    require_expected_version: bool = True,
    derived_only: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    The single authoring/derived write path.

    Args:
        guide_id: canonical guideId.
        mutate: pure-ish callback taking the FRESHLY RELOADED current document
            and returning the document to store. Runs inside the lock+tx, so it
            must be short: no I/O beyond the in-memory document, no LLM call.
        expected_base_version: the client's compare-and-swap base. Checked
            against the version read inside the lock.
        require_expected_version: True for public Unity/web edits and
            user-driven deletes (§7b). False only for trusted internal
            mutations that legitimately have no client base (pipeline writes),
            still under the lock and still re-reading the current document.
        derived_only: True for translation writers (§7b, F7). A derived write
            NEVER advances the authoring `version`/`contentHash`, so a client's
            outstanding base does not go stale behind its back.

    Returns: (stored_document, changed)
    """
    key = str(guide_id or "").strip()
    if not key:
        raise ValueError("Guide mutation requires a guideId.")

    with _lock_for(key), guide_repository.locked(key) as (s, row):
        if row is None:
            raise GuideNotFoundError(f"Guide with guideId '{key}' not found.")
        existing = guide_repository._row_to_doc(row, fallback_guide_id=key)

        # (3) Precondition — rejection here is side-effect free.
        if not derived_only:
            server_version = current_version(existing)
            if expected_base_version is None:
                if require_expected_version:
                    raise ExpectedVersionRequiredError(key, existing)
            elif int(expected_base_version) != server_version:
                raise GuideVersionConflictError(key, expected_base_version, existing)

        # (4) Apply.
        working = deepcopy(existing)
        mutated = mutate(working)
        if not isinstance(mutated, dict):
            raise ValueError("A guide mutation callback must return the document dict.")

        mutated["guideId"] = existing["guideId"]
        mutated["databaseId"] = existing["databaseId"]
        mutated["_id"] = existing["_id"]
        mutated["createdAt"] = existing["createdAt"]

        # (5) Metadata.
        if derived_only:
            mutated["version"] = current_version(existing)
            mutated["baseVersion"] = existing.get("baseVersion", current_version(existing))
            mutated["contentHash"] = existing.get("contentHash") or authoring_content_hash(
                existing
            )
            mutated["updatedAt"] = now_iso()
            changed = True
        else:
            mutated, changed = _stamp_authoring_metadata(mutated, previous=existing)
            if not changed:
                # §7f.1: same content twice does at most one canonical write.
                return existing, False

        # (6) Exactly one commit.
        stored = guide_repository._commit_doc(s, mutated, row)
        return stored, changed


def delete_guide(
    guide_id: str,
    *,
    expected_base_version: int | None = None,
    require_expected_version: bool = False,
) -> bool:
    """
    §7b — a user-driven delete is a guide mutation and takes the same critical
    section, so it can never race a concurrent CAS write. When an expected base
    is supplied it is enforced.
    """
    key = str(guide_id or "").strip()
    if not key:
        return False

    with _lock_for(key), guide_repository.locked(key) as (s, row):
        if row is None:
            return False
        existing = guide_repository._row_to_doc(row, fallback_guide_id=key)

        server_version = current_version(existing)
        if expected_base_version is None:
            if require_expected_version:
                raise ExpectedVersionRequiredError(key, existing)
        elif int(expected_base_version) != server_version:
            raise GuideVersionConflictError(key, expected_base_version, existing)

        s.delete(row)
        try:
            from kiara.guides.paths import guide_instruction_path

            guide_instruction_path(key).unlink(missing_ok=True)
        except Exception:
            pass
        return True


def replace_guide_document(
    doc: dict[str, Any],
    *,
    project_id: Any = None,
    source_version_id: Any = None,
) -> dict[str, Any]:
    """
    Trusted internal upsert used ONLY where a component legitimately owns the
    whole document and has no client base version (guide creation shells, the
    AR/CAD processing pipeline, our own publish path).

    It is NOT a bypass: same lock, same re-read, same merge+stamp rules — a
    concurrent CAS write cannot be lost. ``project_id``/``source_version_id``
    carry the publish linkage — denormalized row columns set atomically in
    the same commit (never part of the client-facing document).
    """
    guide_id = str((doc or {}).get("guideId") or "").strip()
    if not guide_id:
        raise ValueError("Guide document requires 'guideId'.")

    import uuid as _uuid

    def _as_uuid(value: Any) -> _uuid.UUID | None:
        try:
            return _uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None

    with _lock_for(guide_id), guide_repository.locked(guide_id) as (s, row):
        if row is None:
            # Server's own creation may default the format (C2 §2k, R2).
            stored = create_guide({"formatVersion": GUIDE_FORMAT_VERSION, **doc})
            if project_id is not None or source_version_id is not None:
                with guide_repository.locked(guide_id) as (s2, row2):
                    if row2 is not None:
                        if project_id is not None:
                            row2.project_id = _as_uuid(project_id)
                        if source_version_id is not None:
                            row2.source_version_id = _as_uuid(source_version_id)
                        s2.add(row2)
            return stored

        existing = guide_repository._row_to_doc(row, fallback_guide_id=guide_id)
        merged = deepcopy(existing)
        merged.update(deepcopy(doc))
        merged["guideId"] = existing["guideId"]
        merged["databaseId"] = existing["databaseId"]
        merged["_id"] = existing["_id"]
        merged["createdAt"] = existing["createdAt"]

        stamped, changed = _stamp_authoring_metadata(merged, previous=existing)
        if project_id is not None:
            row.project_id = _as_uuid(project_id)
        if source_version_id is not None:
            row.source_version_id = _as_uuid(source_version_id)
        if not changed:
            s.add(row)
            return existing
        return guide_repository._commit_doc(s, stamped, row)


def read_guide_for_copy(guide_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    C1 Slice 4 (§9c.11) — read a SOURCE guide and its media_index rows under
    that guide's lock only, briefly. Media copying happens afterwards with NO
    lock held; the target is created last via `create_guide`.
    """
    key = str(guide_id or "").strip()
    if not key:
        return None, []

    with _lock_for(key), guide_repository.locked(key) as (_s, row):
        if row is None:
            return None, []
        existing = guide_repository._row_to_doc(row, fallback_guide_id=key)
        rows = media_repository.find_project_media_by_guide_id(key)
        return deepcopy(existing), deepcopy(rows)
