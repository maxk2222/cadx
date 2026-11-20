from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool

from kiara.api.errors import CodedHTTPException
from kiara.guides import mutation
from kiara.guides.persistence import now_iso
from kiara.guides.repository import guide_repository
from kiara.guides.schemas import GUIDE_FORMAT_VERSION, GuideSchema, GuideSchemaForDB
from kiara.guides.semantics import normalize_step_semantic
from kiara.guides.sync_ops import retranslate_changed_texts_task
from kiara.guides.sync_schemas import conflict_detail, precondition_detail
from kiara.guides.translate import GuideTranslator as OllamaClient

log = logging.getLogger(__name__)

router = APIRouter()
ollama = OllamaClient()


# ── C1 Slice 2 (§7b, F8) — async routes must not take the lock on the event loop ─────────────────
#
# Every writer in this module is `async def`, so it runs ON the event loop. The mutation primitive
# is a short SYNCHRONOUS critical section guarded by a `threading.RLock`. Taking that lock directly
# from a coroutine would block the whole loop while another thread holds it, so all of them hand the
# call to FastAPI's threadpool through the two helpers below. Never call
# `guide_mutation_service.mutate_guide` directly from an `async def` in this file.


async def _mutate(guide_id: str, mutate_fn, **kwargs):
    return await run_in_threadpool(
        lambda: mutation.mutate_guide(guide_id, mutate_fn, **kwargs)
    )


def _raise_for_mutation_error(exc: Exception, path: str) -> None:
    """
    Translate the primitive's precondition failures into the §7c status codes.

    `path` names the writer for the §9c.13 `C1-CONFLICT` line, which is emitted for a
    `GuideVersionConflictError` response only.
    """
    if isinstance(exc, mutation.ExpectedVersionRequiredError):
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=precondition_detail(exc.guide_id, exc.current),
        ) from exc
    if isinstance(exc, mutation.GuideVersionConflictError):
        mutation.log_version_conflict(exc, path=path)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=conflict_detail(exc.guide_id, exc.expected, exc.current, _serialize_guide(exc.current)),
        ) from exc
    if isinstance(exc, mutation.GuideNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, mutation.GuideAlreadyExistsError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _expected_base_from(body: dict | None, fallback: Any = None) -> int | None:
    """
    The client's compare-and-swap base, read from `expectedBaseVersion` (preferred) or the
    `baseVersion`/`version` it last had acknowledged. Absent → the primitive answers 428.
    """
    for source in (body or {}, fallback or {}):
        if not isinstance(source, dict):
            continue
        for key in ("expectedBaseVersion", "baseVersion", "version"):
            value = source.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _serialize_guide(doc: dict | None) -> dict | None:
    if not doc:
        return doc

    serialized = dict(doc)

    if "_id" in serialized and not isinstance(serialized["_id"], str):
        try:
            serialized["_id"] = str(serialized["_id"])
        except Exception:
            serialized["_id"] = f"{serialized['_id']}"

    if not serialized.get("databaseId") and serialized.get("_id"):
        serialized["databaseId"] = serialized["_id"]

    return serialized


@router.get("/", response_model=list[GuideSchemaForDB])
@router.get("", response_model=list[GuideSchemaForDB], include_in_schema=False)
async def get_all_guides():
    """Ruft alle Anleitungen ab, sortiert nach dem neuesten Erstellungsdatum."""
    log.info("--> GET /instructions/ - Anfrage für alle Anleitungen erhalten.")
    try:
        guides_list = [_serialize_guide(doc) for doc in guide_repository.find_all()]
        log.info(f"<-- GET /instructions/ - {len(guides_list)} Anleitungen gefunden.")
        return guides_list
    except Exception as exc:
        log.error(f"!!! FEHLER in GET /instructions/: {exc}", exc_info=True)
        raise CodedHTTPException(
            status_code=500, code="internalError", detail="Ein interner Serverfehler ist aufgetreten."
        ) from exc


@router.get("/{guide_id}", response_model=GuideSchemaForDB)
async def get_guide_by_id(guide_id: str):
    """
    Ruft eine spezifische Anleitung ab.
    Funktioniert sowohl mit der `guideId` als auch mit der `databaseId`.
    """
    log.info(f"--> GET /instructions/{guide_id} - Anfrage für eine Anleitung erhalten.")

    guide = guide_repository.find_by_identifier(guide_id)

    if guide:
        log.info(f"<-- Anleitung mit ID '{guide_id}' gefunden.")
        return _serialize_guide(guide)

    log.warning(f"--- Anleitung mit ID '{guide_id}' nicht gefunden.")
    raise CodedHTTPException(status_code=404, code="guideNotFound", detail=f"Guide mit ID '{guide_id}' nicht gefunden.")


@router.post("/", response_model=GuideSchemaForDB, status_code=status.HTTP_201_CREATED)
@router.post("", response_model=GuideSchemaForDB, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_guide(guide: GuideSchema):
    """Erstellt eine komplett neue Anleitung und schreibt databaseId in das Dokument."""
    log.info(f"--> POST /instructions/ - Erstelle neue Anleitung mit Titel: {guide.title}")

    guide_doc = guide.model_dump(exclude={"id", "databaseId"})
    guide_doc["createdAt"] = now_iso()
    # C2 §2k: the dump above turns an absent formatVersion into None.
    if guide_doc.get("formatVersion") is None:
        guide_doc["formatVersion"] = GUIDE_FORMAT_VERSION

    # §7b: creation asserts ABSENCE through the same primitive; it never overwrites.
    try:
        created_guide = await run_in_threadpool(lambda: mutation.create_guide(guide_doc))
    except mutation.GuideAlreadyExistsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info(f"<-- Anleitung erfolgreich erstellt mit DB-ID: {created_guide.get('databaseId')}")
    return _serialize_guide(created_guide)


@router.put("/{guide_id}", response_model=GuideSchemaForDB)
async def update_guide(guide_id: str, guide_update: GuideSchema, background_tasks: BackgroundTasks):
    """
    Aktualisiert eine bestehende Anleitung komplett.

    §7b — a public web edit, so it carries an expected base version (`expectedBaseVersion`, or the
    `baseVersion`/`version` the client last had acknowledged) and goes through the one mutation
    primitive. Was: a direct `guide_repository.update_guide` with a self-incremented version, which
    could silently overwrite a concurrent write.
    """
    log.info(f"--> PUT /instructions/{guide_id} - Aktualisiere Anleitung.")

    existing_guide = guide_repository.find_by_identifier(guide_id)
    if not existing_guide:
        log.warning(f"--- Anleitung mit ID '{guide_id}' für Update nicht gefunden.")
        raise HTTPException(status_code=404, detail=f"Guide mit ID '{guide_id}' nicht gefunden.")

    old_guide_snapshot = dict(existing_guide)

    update_data = guide_update.model_dump(exclude_unset=True)
    expected_base_version = _expected_base_from(update_data)

    # Server-owned metadata is never taken from the client (§1c).
    for key in ["id", "_id", "databaseId", "guideId", "createdAt", "updatedAt",
                "version", "baseVersion", "contentHash", "expectedBaseVersion"]:
        update_data.pop(key, None)

    def _apply(current: dict) -> dict:
        merged = dict(current)
        merged.update(update_data)
        return merged

    try:
        updated_guide, changed = await _mutate(
            existing_guide["guideId"], _apply, expected_base_version=expected_base_version
        )
    except mutation.GuideMutationError as exc:
        _raise_for_mutation_error(exc, path="update")
        raise

    if changed:
        background_tasks.add_task(
            retranslate_changed_texts_task,
            existing_guide["guideId"],
            old_guide_snapshot,
            updated_guide,
        )

    log.info(f"<-- Anleitung '{guide_id}' erfolgreich aktualisiert.")
    return _serialize_guide(updated_guide)


@router.patch("/{guide_id}/title", response_model=GuideSchemaForDB, status_code=status.HTTP_200_OK)
async def update_guide_title(guide_id: str, title_update: dict = Body(...), background_tasks: BackgroundTasks = None):
    """
    Aktualisiert nur den Titel einer Anleitung und gibt den vollständigen Guide zurück.

    §7b — API-compatible (same path, same `{"title": ...}` body, same full-guide response), but it
    now carries an expected base (`expectedBaseVersion`, else the `baseVersion`/`version` the client
    holds) and calls the same canonical mutation primitive instead of incrementing the version
    itself. Absent expected base on an existing guide → 428; a stale one → 409.
    """
    new_title = title_update.get("title")
    log.info(f"--> PATCH /instructions/{guide_id}/title - Neuer Titel: '{new_title}'")

    if not new_title or not isinstance(new_title, str):
        raise HTTPException(status_code=400, detail="Ein gültiger Titel ist erforderlich.")

    existing_guide = guide_repository.find_by_identifier(guide_id)
    if not existing_guide:
        log.warning(f"--- Anleitung mit ID '{guide_id}' für Titel-Update nicht gefunden.")
        raise HTTPException(status_code=404, detail=f"Guide mit ID '{guide_id}' nicht gefunden.")

    old_guide_snapshot = dict(existing_guide)
    expected_base_version = _expected_base_from(title_update)

    def _apply(current: dict) -> dict:
        merged = dict(current)
        merged["title"] = new_title
        return merged

    try:
        updated_guide, changed = await _mutate(
            existing_guide["guideId"], _apply, expected_base_version=expected_base_version
        )
    except mutation.GuideMutationError as exc:
        _raise_for_mutation_error(exc, path="title")
        raise

    if changed and background_tasks is not None:
        background_tasks.add_task(
            retranslate_changed_texts_task,
            existing_guide["guideId"],
            old_guide_snapshot,
            updated_guide,
        )

    log.info(f"<-- Titel für Anleitung '{guide_id}' erfolgreich aktualisiert.")
    return _serialize_guide(updated_guide)


@router.delete("/{guide_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_guide(guide_id: str, expectedBaseVersion: int | None = None):
    """
    Löscht eine Anleitung aus der Persistenz.

    §7b — a user-driven delete takes the same per-guide critical section, and enforces the caller's
    expected base version when one is supplied, so it cannot discard a concurrent edit unnoticed.
    """
    log.info(f"--> DELETE /instructions/{guide_id} - Lösche Anleitung.")

    existing_guide = guide_repository.find_by_identifier(guide_id)
    if not existing_guide:
        log.warning(f"--- Anleitung mit ID '{guide_id}' zum Löschen nicht gefunden.")
        raise HTTPException(status_code=404, detail=f"Guide mit ID '{guide_id}' nicht gefunden.")

    try:
        deleted = await run_in_threadpool(
            lambda: mutation.delete_guide(
                existing_guide["guideId"], expected_base_version=expectedBaseVersion
            )
        )
    except mutation.GuideMutationError as exc:
        _raise_for_mutation_error(exc, path="delete")
        raise

    if not deleted:
        raise HTTPException(status_code=500, detail="Anleitung konnte nicht gelöscht werden.")

    log.info(f"<-- Anleitung '{guide_id}' erfolgreich gelöscht.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{guide_id}/semantics/rebuild", response_model=GuideSchemaForDB)
async def rebuild_guide_semantics(guide_id: str):
    """
    Baut Semantik für alle Schritte einer bestehenden Anleitung neu auf,
    ohne die äußere Guide-Struktur zu verändern.

    §7b — step semantic data IS meaningful authoring content (§7a.1), so this DOES advance the
    authoring revision. The expensive LLM inference runs OUTSIDE the lock against a known base
    version N; only the short commit takes the lock, with expected version N. If the base moved
    while the inference ran, the recompute is retried once against the new base rather than
    overwriting the concurrent change; a second miss is reported as 409.
    """
    log.info(f"--> POST /instructions/{guide_id}/semantics/rebuild - Rebuild angefordert.")

    existing_guide = guide_repository.find_by_identifier(guide_id)
    if not existing_guide:
        raise HTTPException(status_code=404, detail=f"Guide mit ID '{guide_id}' nicht gefunden.")

    canonical_id = existing_guide["guideId"]

    def _infer_steps(source_guide: dict[str, Any]) -> list[dict[str, Any]]:
        """Expensive: runs OUTSIDE the per-guide lock. Never call this from a mutation callback."""
        updated_steps: list[dict[str, Any]] = []

        for idx, step in enumerate(source_guide.get("steps") or [], start=1):
            step_copy = deepcopy(step or {})
            text_candidates: list[str] = []

            for obj in step_copy.get("objects") or []:
                if obj.get("isTextField") and str(obj.get("text") or "").strip():
                    text_candidates.append(str(obj.get("text") or "").strip())

            step_text = " ".join(text_candidates).strip()

            try:
                if step_text:
                    raw_semantic = ollama.infer_semantic_for_step(step_text)
                    step_copy["semantic"] = normalize_step_semantic(raw_semantic).model_dump()
                else:
                    step_copy["semantic"] = normalize_step_semantic(None).model_dump()
            except Exception as exc:
                log.warning(f"Semantik-Rebuild für Schritt {idx} fehlgeschlagen: {exc}")
                step_copy["semantic"] = normalize_step_semantic(None).model_dump()

            updated_steps.append(step_copy)

        return updated_steps

    source = existing_guide
    saved: dict[str, Any] | None = None

    for attempt in (1, 2):
        base_version = mutation.current_version(source)
        computed_steps = await run_in_threadpool(_infer_steps, source)

        def _apply(current: dict, steps=computed_steps) -> dict:
            merged = dict(current)
            merged["steps"] = steps
            return merged

        try:
            saved, _ = await _mutate(canonical_id, _apply, expected_base_version=base_version)
            break
        except mutation.GuideVersionConflictError as exc:
            if attempt == 2:
                _raise_for_mutation_error(exc, path="semantics")
                raise
            log.info(
                "Semantik-Rebuild für '%s': Basis hat sich während der Inferenz von %s auf %s "
                "bewegt — Neuberechnung statt Überschreiben.",
                guide_id, base_version, exc.current.get("version"),
            )
            source = exc.current
        except mutation.GuideMutationError as exc:
            _raise_for_mutation_error(exc, path="semantics")
            raise

    log.info(f"<-- Semantik-Rebuild für Anleitung '{guide_id}' abgeschlossen.")
    return _serialize_guide(saved)


@router.post("/{guide_id}/translate", response_model=GuideSchemaForDB)
async def translate_guide(guide_id: str, body: dict = Body(...)):
    """
    Übersetzt eine bestehende Anleitung in eine Zielsprache via LLM und speichert
    die Übersetzung als Sprachvariante. Erzeugt zusätzlich eine übersetzte Unity-Guide-Datei.
    """
    target_language = body.get("language")
    if not target_language or not isinstance(target_language, str):
        raise CodedHTTPException(
            status_code=400, code="targetLanguageRequired", detail="Eine gültige Zielsprache ist erforderlich."
        )
    target_language = target_language.strip().lower()

    force = bool(body.get("force", False))

    existing_guide = guide_repository.find_by_identifier(guide_id)
    if not existing_guide:
        log.warning(f"--- Anleitung mit ID '{guide_id}' für Übersetzung nicht gefunden.")
        raise CodedHTTPException(
            status_code=404, code="guideNotFound", detail=f"Guide mit ID '{guide_id}' nicht gefunden."
        )

    translations = existing_guide.get("translations") or {}
    if target_language in translations and not force:
        raise CodedHTTPException(
            status_code=409,
            code="translationExists",
            detail=f"Übersetzung für '{target_language}' existiert bereits. "
                   f"Setzen Sie force=true, um sie zu überschreiben.",
            params={"language": target_language},
        )

    log.info(f"--> POST /instructions/{guide_id}/translate - Zielsprache: {target_language}")

    # §7b — the expensive LLM call and the derived Unity-guide file write both run OUTSIDE the lock.
    try:
        translated = await run_in_threadpool(ollama.translate_guide, existing_guide, target_language)
    except Exception as exc:
        log.error(f"Übersetzung für Anleitung '{guide_id}' fehlgeschlagen: {exc}", exc_info=True)
        raise CodedHTTPException(
            status_code=500, code="translationFailed", detail=f"Übersetzung fehlgeschlagen: {exc}"
        ) from exc

    # Optional: Übersetzte Unity-Guide-Variante erzeugen — produced BEFORE the commit so the stored
    # translation and its unityGuidePath land in the SAME single write.
    unity_path: str | None = None
    try:
        from kiara.guides.translate import translate_unity_guide
        unity_path = await run_in_threadpool(
            translate_unity_guide, existing_guide["guideId"], target_language, translated
        )
    except Exception as exc:
        log.warning(f"Unity-Guide-Übersetzung für '{guide_id}' nicht möglich: {exc}")

    if unity_path:
        translated["unityGuidePath"] = unity_path

    # §7b/F7 — manual translation changes ONLY the derived `translations` subtree. `derived_only`
    # therefore preserves the authoring version/contentHash: a client editing this guide keeps a
    # valid base and its next save does not conflict because someone requested a translation.
    # The callback re-reads inside the lock and merges only this one language entry, so a
    # concurrent authoring change to steps/title made during the LLM call is not reverted.
    def _merge_language(current: dict) -> dict:
        merged = dict(current)
        translations_now = dict(merged.get("translations") or {})
        translations_now[target_language] = translated
        merged["translations"] = translations_now
        return merged

    try:
        updated_guide, _ = await _mutate(existing_guide["guideId"], _merge_language, derived_only=True)
    except mutation.GuideMutationError as exc:
        _raise_for_mutation_error(exc, path="translate")
        raise

    serialized = _serialize_guide(updated_guide)
    serialized["unityGuidePath"] = unity_path

    log.info(f"<-- Anleitung '{guide_id}' erfolgreich nach '{target_language}' übersetzt.")
    return serialized
