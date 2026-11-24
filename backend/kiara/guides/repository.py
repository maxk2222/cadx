"""Postgres-backed guide document store — the suite's `guide_repository` port.

The suite persists the canonical doc as
`project_data/<guideId>/instruction/instruction.json` behind a process-local
lock. We keep the identical document shape but store it in the `guide` table's
`doc` JSON column, so `SELECT … FOR UPDATE` on the row replaces the RLock and
the primitive is safe across processes (API + arq worker alike). The
`instruction/instruction.json` file is additionally mirrored on every write so
the `/sync/assets/{guideId}/instruction/…` asset route and on-disk inspection
see the same content — the DB remains authoritative.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, select

from kiara.config import get_settings
from kiara.db.models import Guide
from kiara.guides.paths import guide_instruction_path
from kiara.guides.persistence import atomic_write_json, new_legacy_id, now_iso


def _sync_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("sqlite+aiosqlite:///"):
        return url.replace("sqlite+aiosqlite:///", "sqlite:///", 1)
    if url.startswith("sqlite:///"):
        return url
    return url


_sync_engine = None
_sync_sessionmaker = None
_sync_url_seen: str | None = None


def _engine():
    """Process-shared sync engine — bound to the CURRENT database_url.

    evalkit's local driver rebinds DATABASE_URL per scratch run; a blind
    singleton would keep writing every later run's guide document into the
    first run's eval.db. Re-create when the url changes."""
    global _sync_engine, _sync_sessionmaker, _sync_url_seen
    url = _sync_url(get_settings().database_url)
    if _sync_engine is None or _sync_url_seen != url:
        if _sync_engine is not None:
            _sync_engine.dispose()
        _sync_engine = create_engine(url, pool_pre_ping=True)
        _sync_sessionmaker = sessionmaker(_sync_engine, class_=Session, expire_on_commit=False)
        _sync_url_seen = url
    return _sync_engine


def _sessionmaker() -> sessionmaker:
    _engine()
    return _sync_sessionmaker


def _supports_for_update() -> bool:
    return _engine().dialect.name != "sqlite"


class GuideRepository:
    """Dict-shaped document store mirroring the suite's repository interface.

    Callers see normalized guide dicts exactly as the file-backed version
    produced; nothing outside `kiara/guides/mutation.py` may call the writers.
    """

    # ── reads ────────────────────────────────────────────────────────────────

    def _normalize_guide_doc(
        self, doc: dict[str, Any], fallback_guide_id: str | None = None
    ) -> dict[str, Any]:
        normalized = deepcopy(doc or {})

        guide_id = str(normalized.get("guideId") or fallback_guide_id or "").strip()
        if not guide_id:
            raise ValueError("Guide document requires 'guideId'.")

        database_id = str(
            normalized.get("databaseId") or normalized.get("_id") or new_legacy_id()
        ).strip()

        created_at = str(normalized.get("createdAt") or now_iso())
        updated_at = str(normalized.get("updatedAt") or created_at)

        normalized["guideId"] = guide_id
        normalized["_id"] = database_id
        normalized["databaseId"] = database_id
        normalized["createdAt"] = created_at
        normalized["updatedAt"] = updated_at

        if "title" not in normalized:
            normalized["title"] = "Unbenannte Anleitung"
        if "version" not in normalized:
            normalized["version"] = 1
        if "steps" not in normalized:
            normalized["steps"] = []

        return normalized

    def _row_to_doc(self, row: Guide, fallback_guide_id: str | None = None) -> dict[str, Any]:
        return self._normalize_guide_doc(row.doc or {}, fallback_guide_id=fallback_guide_id)

    def find_all(self) -> list[dict[str, Any]]:
        with _sessionmaker()() as s:
            rows = s.exec(select(Guide)).all()
        docs: list[dict[str, Any]] = []
        for row in rows:
            try:
                docs.append(self._row_to_doc(row, fallback_guide_id=row.guide_id))
            except ValueError:
                continue
        docs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
        return docs

    def find_by_guide_id(self, guide_id: str) -> dict[str, Any] | None:
        with _sessionmaker()() as s:
            row = s.exec(select(Guide).where(Guide.guide_id == str(guide_id))).first()
        if row is None:
            return None
        return self._row_to_doc(row, fallback_guide_id=guide_id)

    def find_by_database_id(self, database_id: str) -> dict[str, Any] | None:
        with _sessionmaker()() as s:
            row = s.exec(select(Guide).where(Guide.database_id == str(database_id))).first()
        if row is None:
            return None
        return self._row_to_doc(row, fallback_guide_id=row.guide_id)

    def find_by_source_file_id(self, source_file_id: str) -> dict[str, Any] | None:
        with _sessionmaker()() as s:
            row = s.exec(
                select(Guide).where(Guide.source_file_id == str(source_file_id))
            ).first()
        if row is None:
            return None
        return self._row_to_doc(row, fallback_guide_id=row.guide_id)

    def find_by_identifier(self, identifier: str) -> dict[str, Any] | None:
        by_guide_id = self.find_by_guide_id(identifier)
        if by_guide_id:
            return by_guide_id
        return self.find_by_database_id(identifier)

    def find_by_project_id(self, project_id: Any) -> dict[str, Any] | None:
        with _sessionmaker()() as s:
            row = s.exec(select(Guide).where(Guide.project_id == project_id)).first()
        if row is None:
            return None
        return self._row_to_doc(row, fallback_guide_id=row.guide_id)

    # ── row-level locking for the mutation primitive ─────────────────────────

    @contextmanager
    def locked(self, guide_id: str) -> Iterator[tuple[Session, Guide | None]]:
        """Yield `(session, row)` with the guide's row held FOR UPDATE on
        Postgres; on SQLite the surrounding transaction serializes writes.

        Only `kiara/guides/mutation.py` may use this. The yielded row is
        expired-free; mutate `row.doc` and `write_guide_document`-equivalent
        stamping happens via `_commit_doc` before the transaction closes.
        """
        gid = str(guide_id or "").strip()
        with _sessionmaker()() as s:
            if _supports_for_update():
                row = s.exec(
                    select(Guide).where(Guide.guide_id == gid).with_for_update()
                ).first()
            else:
                row = s.exec(select(Guide).where(Guide.guide_id == gid)).first()
            try:
                yield s, row
            except Exception:
                s.rollback()
                raise
            else:
                s.commit()

    def _commit_doc(self, s: Session, doc: dict[str, Any], row: Guide | None) -> dict[str, Any]:
        """Normalize, upsert the row and mirror `instruction.json`. IN-TX only."""
        normalized = self._normalize_guide_doc(doc)

        if row is None:
            row = Guide(
                guide_id=normalized["guideId"],
                database_id=normalized["databaseId"],
            )
            s.add(row)
        row.title = str(normalized.get("title") or "")
        row.doc = normalized
        row.version = int(normalized.get("version") or 1)
        row.base_version = normalized.get("baseVersion", row.version)
        row.content_hash = normalized.get("contentHash")
        row.format_version = normalized.get("formatVersion")
        row.source_file_id = normalized.get("source_file_id")
        for fld, col in (("createdAt", "created_at"), ("updatedAt", "updated_at")):
            try:
                from datetime import datetime

                setattr(row, col, datetime.fromisoformat(str(normalized[fld]).replace("Z", "+00:00")))
            except (KeyError, ValueError):
                pass
        s.add(row)
        s.flush()
        # mirror the canonical file for the instruction asset route (derived,
        # non-authoritative — write failure must not poison the tx)
        with suppress(Exception):
            atomic_write_json(guide_instruction_path(row.guide_id), normalized)
        return normalized

    # ── writers — mutation primitive only ────────────────────────────────────
    # No public write helpers here on purpose: every productive write must go
    # through kiara/guides/mutation.py (lock + CAS + stamp + single commit).
    # `locked()` + `_commit_doc()` are the primitive's private seam.


guide_repository = GuideRepository()
