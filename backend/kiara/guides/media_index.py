from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from kiara.guides.paths_cfg import settings
from kiara.guides.persistence import atomic_write_json, new_legacy_id, now_iso, read_json_file

DATA_DIR = Path(settings.DATA_DIR)
PROJECT_DATA_DIR = DATA_DIR / "project_data"
GLOBAL_ASSETS_DIR = DATA_DIR / "global_assets"


class MediaRepository:
    def _project_media_index_path(self, guide_id: str) -> Path:
        # Imported here: backend.storage imports this repository, a module-level import is a cycle.
        from kiara.guides.paths import safe_guide_id

        return PROJECT_DATA_DIR / safe_guide_id(guide_id) / "media_index.json"

    def _global_media_index_path(self) -> Path:
        return GLOBAL_ASSETS_DIR / "media_index.json"

    def _read_index(self, path: Path) -> list[dict[str, Any]]:
        data = read_json_file(path, default=[])
        if not isinstance(data, list):
            raise ValueError(f"Media index '{path}' must contain a JSON list.")
        return [item for item in data if isinstance(item, dict)]

    def _write_index(self, path: Path, items: list[dict[str, Any]]) -> Path:
        return atomic_write_json(path, items)

    def _normalize_media_doc(
        self,
        doc: dict[str, Any],
        *,
        scope: str,
        guide_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = deepcopy(doc or {})
        media_id = str(normalized.get("_id") or normalized.get("id") or new_legacy_id()).strip()

        normalized["_id"] = media_id
        normalized["id"] = media_id
        normalized["scope"] = scope
        normalized["uploadDate"] = str(normalized.get("uploadDate") or now_iso())

        if scope == "project":
            if not guide_id:
                raise ValueError("Project media requires 'guide_id'.")
            normalized["guideId"] = str(normalized.get("guideId") or guide_id)

        return normalized

    def find_project_media_by_guide_id(self, guide_id: str) -> list[dict[str, Any]]:
        items = self._read_index(self._project_media_index_path(guide_id))
        return [self._normalize_media_doc(item, scope="project", guide_id=guide_id) for item in items]

    def find_global_media(self) -> list[dict[str, Any]]:
        items = self._read_index(self._global_media_index_path())
        return [self._normalize_media_doc(item, scope="global") for item in items]

    def insert_project_media(self, guide_id: str, media_doc: dict[str, Any]) -> dict[str, Any]:
        items = self._read_index(self._project_media_index_path(guide_id))
        normalized = self._normalize_media_doc(media_doc, scope="project", guide_id=guide_id)
        items.append(normalized)
        self._write_index(self._project_media_index_path(guide_id), items)
        return normalized

    def find_project_media_by_object(
        self,
        guide_id: str,
        object_id: str,
        asset_type: str,
        unique_filename: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Find the media row for (object_id, asset_type). Exact objectId match
        wins; a legacy row (no stored objectId) is matched ONLY by the exact
        deterministic filename — never by prefix ('abc' must not steal
        'abc_def_...'). A renamed legacy row cannot be safely correlated.
        """
        items = self._read_index(self._project_media_index_path(guide_id))
        legacy_match: dict[str, Any] | None = None

        for item in items:
            if str(item.get("assetType") or "") != str(asset_type):
                continue
            if str(item.get("objectId") or "") == str(object_id):
                return self._normalize_media_doc(item, scope="project", guide_id=guide_id)
            if (
                legacy_match is None
                and unique_filename
                and not item.get("objectId")
                and str(item.get("uniqueFileName") or "") == str(unique_filename)
            ):
                legacy_match = item

        if legacy_match is not None:
            return self._normalize_media_doc(legacy_match, scope="project", guide_id=guide_id)
        return None

    def upsert_project_media_by_object(
        self,
        guide_id: str,
        object_id: str,
        asset_type: str,
        media_doc: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Replace-or-insert keyed on (object_id, asset_type). Reuses the existing
        mediaId (also for a legacy row matched by exact deterministic filename)
        so retries never accumulate duplicate rows. One media type never
        replaces another for the same object.
        """
        path = self._project_media_index_path(guide_id)
        items = self._read_index(path)
        incoming = self._normalize_media_doc(
            {**media_doc, "objectId": str(object_id), "assetType": str(asset_type)},
            scope="project",
            guide_id=guide_id,
        )
        target_name = str(incoming.get("uniqueFileName") or "")

        for idx, item in enumerate(items):
            if str(item.get("assetType") or "") != str(asset_type):
                continue
            exact = str(item.get("objectId") or "") == str(object_id)
            legacy_exact = (
                not item.get("objectId")
                and target_name
                and str(item.get("uniqueFileName") or "") == target_name
            )
            if exact or legacy_exact:
                incoming["_id"] = incoming["id"] = str(
                    item.get("_id") or item.get("id") or incoming["_id"]
                )
                items[idx] = incoming
                self._write_index(path, items)
                return incoming

        items.append(incoming)
        self._write_index(path, items)
        return incoming

    # ── C1 Slice 3 (§8b) — attempt commit / rollback of the project media index ──────────────────

    def snapshot_project_media_index(self, guide_id: str) -> list[dict[str, Any]]:
        """
        §8b — the index as it stands BEFORE an attempt commit touches it.

        The attempt service takes this inside the guide lock and restores it if the guide JSON
        commit then fails, so "on JSON/index failure, restore the previous index" is a real
        restore of the previous bytes rather than a compensating guess.
        """
        return deepcopy(self._read_index(self._project_media_index_path(guide_id)))

    def restore_project_media_index(self, guide_id: str, items: list[dict[str, Any]]) -> Path:
        """
        §8b — put back exactly what `snapshot_project_media_index` returned.

        A guide whose index file did not exist yet is restored as an empty list, which reads back
        identically to an absent file (`_read_index` defaults to `[]`).
        """
        return self._write_index(
            self._project_media_index_path(guide_id),
            [deepcopy(item) for item in (items or [])],
        )

    def upsert_project_media_for_attempt(
        self,
        guide_id: str,
        object_id: str,
        asset_type: str,
        media_doc: dict[str, Any],
    ) -> dict[str, Any]:
        """
        §8b — the index row for a committed attempt final.

        Identical to `upsert_project_media_by_object` except that it ENFORCES the two facts §8b
        requires an attempt row to record: the verified `contentHash` and the exact final
        `fileUrl`. Enforcing them here means no caller can quietly write a half-described attempt
        row, whatever it does upstream.
        """
        if not str((media_doc or {}).get("contentHash") or "").strip():
            raise ValueError("an attempt media row must record its verified contentHash")
        if not str((media_doc or {}).get("fileUrl") or "").strip():
            raise ValueError("an attempt media row must record its exact final fileUrl")

        return self.upsert_project_media_by_object(guide_id, object_id, asset_type, media_doc)

    def insert_global_media(self, media_doc: dict[str, Any]) -> dict[str, Any]:
        items = self._read_index(self._global_media_index_path())
        normalized = self._normalize_media_doc(media_doc, scope="global")
        items.append(normalized)
        self._write_index(self._global_media_index_path(), items)
        return normalized

    def update_project_media(self, guide_id: str, media_id: str, update_doc: dict[str, Any]) -> dict[str, Any]:
        items = self._read_index(self._project_media_index_path(guide_id))

        for idx, item in enumerate(items):
            current_id = str(item.get("_id") or item.get("id") or "")
            if current_id != str(media_id):
                continue

            merged = deepcopy(item)
            merged.update(deepcopy(update_doc))
            normalized = self._normalize_media_doc(merged, scope="project", guide_id=guide_id)
            items[idx] = normalized
            self._write_index(self._project_media_index_path(guide_id), items)
            return normalized

        raise FileNotFoundError(f"Project media '{media_id}' for guide '{guide_id}' not found.")

    def update_global_media(self, media_id: str, update_doc: dict[str, Any]) -> dict[str, Any]:
        items = self._read_index(self._global_media_index_path())

        for idx, item in enumerate(items):
            current_id = str(item.get("_id") or item.get("id") or "")
            if current_id != str(media_id):
                continue

            merged = deepcopy(item)
            merged.update(deepcopy(update_doc))
            normalized = self._normalize_media_doc(merged, scope="global")
            items[idx] = normalized
            self._write_index(self._global_media_index_path(), items)
            return normalized

        raise FileNotFoundError(f"Global media '{media_id}' not found.")

    def delete_project_media(self, guide_id: str, media_id: str) -> bool:
        items = self._read_index(self._project_media_index_path(guide_id))

        for idx, item in enumerate(items):
            current_id = str(item.get("_id") or item.get("id") or "")
            if current_id != str(media_id):
                continue

            del items[idx]
            self._write_index(self._project_media_index_path(guide_id), items)
            return True

        return False

    def delete_global_media(self, media_id: str) -> bool:
        items = self._read_index(self._global_media_index_path())

        for idx, item in enumerate(items):
            current_id = str(item.get("_id") or item.get("id") or "")
            if current_id != str(media_id):
                continue

            del items[idx]
            self._write_index(self._global_media_index_path(), items)
            return True

        return False

    def ensure_project_media_entry(
        self,
        guide_id: str,
        media_doc: dict[str, Any],
        *,
        match_keys: tuple[str, ...] = ("filePath", "uniqueFileName", "fileUrl"),
    ) -> dict[str, Any]:
        items = self._read_index(self._project_media_index_path(guide_id))
        normalized = self._normalize_media_doc(media_doc, scope="project", guide_id=guide_id)

        for item in items:
            existing = self._normalize_media_doc(item, scope="project", guide_id=guide_id)

            for key in match_keys:
                existing_value = str(existing.get(key) or "").strip()
                incoming_value = str(normalized.get(key) or "").strip()
                if existing_value and incoming_value and existing_value == incoming_value:
                    return existing

        items.append(normalized)
        self._write_index(self._project_media_index_path(guide_id), items)
        return normalized

    def ensure_global_media_entry(
        self,
        media_doc: dict[str, Any],
        *,
        match_keys: tuple[str, ...] = ("filePath", "uniqueFileName", "fileUrl"),
    ) -> dict[str, Any]:
        items = self._read_index(self._global_media_index_path())
        normalized = self._normalize_media_doc(media_doc, scope="global")

        for item in items:
            existing = self._normalize_media_doc(item, scope="global")

            for key in match_keys:
                existing_value = str(existing.get(key) or "").strip()
                incoming_value = str(normalized.get(key) or "").strip()
                if existing_value and incoming_value and existing_value == incoming_value:
                    return existing

        items.append(normalized)
        self._write_index(self._global_media_index_path(), items)
        return normalized


media_repository = MediaRepository()
