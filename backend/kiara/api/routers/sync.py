from __future__ import annotations

import contextlib
import logging
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from kiara.guides import attempts, mutation
from kiara.guides.asset_ops import upload_project_media
from kiara.guides.schemas import GuideSchema, GuideSchemaForDB
from kiara.guides.sync_ops import (
    ConflictCopySourceMissingError,
    check_sync_conflict,
    create_conflict_copy,
    delete_guide,
    get_guide_meta,
    get_guides_to_pull,
    get_health_payload,
    get_single_guide,
    incoming_authoring_document,
    retranslate_changed_texts_task,
    serialize_guide,
    sync_guide_cas,
)
from kiara.guides.sync_schemas import (
    AttemptMediaState,
    ClientSyncRequest,
    GuideCasEnvelope,
    GuideConflictCopyRequest,
    GuideMetaResponse,
    GuideUploadAttemptCommitRequest,
    GuideUploadAttemptRequest,
    GuideUploadAttemptResponse,
    SyncConflictResponse,
    attempt_response,
    conflict_detail,
    precondition_detail,
)
from kiara.guides.upload_errors import (
    GuideNotFoundError,
    InvalidUploadError,
    PayloadTooLargeError,
    StorageRecoveryError,
    UnsupportedMediaTypeError,
)

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    log.info("--> GET /health - Health check for sync router called.")
    try:
        return get_health_payload()
    except Exception as exc:
        log.error(f"❌ Health check failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {exc}",
        ) from exc


@router.post("/guide/check_conflict", response_model=SyncConflictResponse)
def check_sync_conflict_route(payload: GuideSchema = Body(...)):
    log.info(f"--> POST /guide/check_conflict for guideId: {payload.guideId}")
    return check_sync_conflict(payload)


def _parse_guide_upload(body: dict) -> tuple[GuideSchema, int | None]:
    """
    §7c.1/§7c.2 — accept the compare-and-swap envelope `{guide, expectedBaseVersion}`, and still
    accept a bare legacy guide document.

    A bare document carries no expected base, so it can only CREATE. For an existing guideId the
    mutation primitive answers 428 rather than blind-overwriting it (§7c.2).
    """
    if isinstance(body, dict) and isinstance(body.get("guide"), dict):
        envelope = GuideCasEnvelope.model_validate(body)
        return envelope.guide, envelope.expectedBaseVersion

    return GuideSchema.model_validate(body), None


@router.post("/guide", response_model=GuideSchemaForDB)
def sync_guide_route(body: dict = Body(...), force: bool = False, background_tasks: BackgroundTasks = None):
    """
    §7c.1 — the compare-and-swap guide upload. This route is SYNCHRONOUS: FastAPI runs it in a
    worker thread, so it may take the mutation primitive's per-guide lock directly (§7b, F8).

    `force` is retained for wire compatibility but is DELIBERATELY INERT. A forced blind overwrite
    is a rejected shortcut (§1c) — a stale write must never succeed, whatever the caller asks for.
    """
    try:
        payload, expected_base_version = _parse_guide_upload(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid guide upload payload: {exc}") from exc

    if force:
        log.warning(
            "POST /guide?force=true for %s — 'force' is inert under C1 §7: the compare-and-swap "
            "precondition still decides.", payload.guideId,
        )

    log.info(
        "--> POST /guide - Anleitung %s (expectedBaseVersion=%s).",
        payload.guideId, expected_base_version,
    )

    try:
        result, old_guide, changed = sync_guide_cas(payload, expected_base_version)
    except mutation.ExpectedVersionRequiredError as exc:
        # §7a.5 — 428, not a silent overwrite.
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=precondition_detail(exc.guide_id, exc.current),
        ) from exc
    except mutation.GuideVersionConflictError as exc:
        # §7a.6/§1c — a stale write NEVER returns HTTP success and mutated nothing.
        mutation.log_version_conflict(exc, path="push")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=conflict_detail(exc.guide_id, exc.expected, exc.current, serialize_guide(exc.current)),
        ) from exc
    except mutation.GuideAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except mutation.GuideNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # Retranslation only makes sense when authoring content actually moved. A no-change save
    # (§7f.1) schedules nothing.
    if old_guide is not None and changed and background_tasks is not None:
        background_tasks.add_task(
            retranslate_changed_texts_task,
            payload.guideId,
            old_guide,
            result.model_dump(by_alias=True),
        )
    return result


@router.post(
    "/guide/{guide_id}/conflict-copy",
    response_model=GuideSchemaForDB,
    status_code=status.HTTP_201_CREATED,
)
def create_conflict_copy_route(guide_id: str, payload: GuideConflictCopyRequest = Body(...)):
    """
    C1 Slice 4 (§9c.11) — the web conflict panel's "Save mine as a copy".

    Creates a NEW guide (fresh guideId and databaseId, conflict-copy title, no expected version)
    from the editor's visible document, with its own copies of every media file that document
    references in the source guide's folder. SYNCHRONOUS like the CAS route (§7b, F8): it takes the
    per-guide locks from a worker thread, never on the event loop.

    Any failure leaves no half-created guide and no orphaned files; the editor's edits stay dirty.
    """
    log.info("--> POST /guide/%s/conflict-copy", guide_id)
    try:
        created = create_conflict_copy(
            guide_id, payload.guide.model_dump(by_alias=True, exclude_unset=True)
        )
    except ConflictCopySourceMissingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return GuideSchemaForDB.model_validate(serialize_guide(created))


@router.post("/upload_media", status_code=status.HTTP_201_CREATED)
async def upload_media_route(
    file: UploadFile = File(...),
    guide_id: str = Form(...),
    object_id: str | None = Form(None),
):
    """
    Project media upload — the endpoint BOTH clients already target
    (Unity SyncHttpClient and the web editor). object_id is optional at the
    HTTP boundary only (web library uploads without an AR object): the route
    mints a UUID4; every layer below works with a concrete, validated id.
    """
    oid = (object_id or "").strip() or str(uuid4())
    log.info(f"--> POST /upload_media - guide {guide_id}, object {oid}.")
    try:
        return await upload_project_media(file, guide_id, oid)
    except UnsupportedMediaTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except PayloadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except GuideNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StorageRecoveryError as exc:
        log.error(f"❌ Media storage recovery required: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="media storage recovery required") from exc


# ── C1 Slice 3 (§8a) — the transactional upload-attempt protocol ────────────────────────────────
#
# Four endpoints, one visibility point. Only `/commit` publishes anything; create, media PUT and
# DELETE touch nothing but `DATA_DIR/upload_attempts/<attemptId>/`.
#
# The legacy `/upload_media` route above is NOT part of this protocol and is unchanged: it keeps
# serving independently uploaded web library media with its own replace-in-place model (§8b).


def _attempt_upload_http_error(exc: Exception) -> HTTPException:
    """The upload-error → status mapping the legacy media route already uses, reused verbatim."""
    if isinstance(exc, UnsupportedMediaTypeError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidUploadError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="upload attempt failed")


@router.post(
    "/guide-upload-attempts",
    response_model=GuideUploadAttemptResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_guide_upload_attempt_route(payload: GuideUploadAttemptRequest = Body(...)):
    """
    §8a.1 — open an attempt. Returns an opaque id; publishes nothing.

    Declaring media here is what makes the later commit able to say "every declared medium landed"
    instead of hoping the client uploaded what it meant to.
    """
    log.info(
        "--> POST /guide-upload-attempts - guide %s, %d media, expectedBaseVersion=%s.",
        payload.guideId, len(payload.media), payload.expectedBaseVersion,
    )
    try:
        manifest = attempts.create_attempt(
            payload.guideId,
            payload.expectedBaseVersion,
            [item.model_dump(by_alias=True) for item in payload.media],
        )
    except (UnsupportedMediaTypeError, PayloadTooLargeError, InvalidUploadError) as exc:
        raise _attempt_upload_http_error(exc) from exc

    return attempt_response(manifest)


@router.put(
    "/guide-upload-attempts/{attempt_id}/media/{media_key}",
    response_model=AttemptMediaState,
)
async def put_guide_upload_attempt_media_route(attempt_id: str, media_key: str, request: Request):
    """
    §8a.2/§8a.3 — the RAW body of this request IS the file.

    ASYNC on purpose: the body is consumed as an async stream and written chunk by chunk, so peak
    memory is one chunk no matter how large the video (§1c "no whole-video ReadAllBytes upload").
    No multipart parsing, which would buffer the part in memory first.
    """
    log.info("--> PUT /guide-upload-attempts/%s/media/%s", attempt_id, media_key)
    try:
        entry = await attempts.store_attempt_media(attempt_id, media_key, request.stream())
    except attempts.AttemptNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except attempts.AttemptMediaNotDeclaredError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnsupportedMediaTypeError, PayloadTooLargeError, InvalidUploadError) as exc:
        raise _attempt_upload_http_error(exc) from exc

    return AttemptMediaState.model_validate(entry)


@router.post(
    "/guide-upload-attempts/{attempt_id}/commit",
    response_model=GuideSchemaForDB,
)
def commit_guide_upload_attempt_route(
    attempt_id: str,
    payload: GuideUploadAttemptCommitRequest = Body(...),
    background_tasks: BackgroundTasks = None,
):
    """
    §8a.4 — the ONE commit point.

    SYNCHRONOUS for exactly the reason `sync_guide_route` is (§7b, F8): FastAPI runs it in a worker
    thread, so it may take the mutation primitive's per-guide lock directly.

    Every refusal below leaves the server byte-identical to what it was: a 409/428 never promotes
    media, and a failure after promotion is rolled back inside the service (§8b).
    """
    log.info("--> POST /guide-upload-attempts/%s/commit", attempt_id)
    try:
        # The SAME normalisation the CAS upload applies: server-owned metadata stripped (a client
        # never chooses its own revision, §1c) and the derived `translations` subtree left to the
        # server. Shared with `sync_guide_cas` rather than rebuilt here.
        stored, old_guide, changed = attempts.commit_attempt(
            attempt_id, incoming_authoring_document(payload.guide)
        )
    except attempts.AttemptNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except attempts.AttemptMediaNotDeclaredError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except attempts.AttemptIncompleteError as exc:
        # §1c — an incomplete attempt is NOT a success. Nothing was published.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except attempts.AttemptGuideMismatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except mutation.ExpectedVersionRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=precondition_detail(exc.guide_id, exc.current),
        ) from exc
    except mutation.GuideVersionConflictError as exc:
        mutation.log_version_conflict(exc, path="push")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=conflict_detail(exc.guide_id, exc.expected, exc.current, serialize_guide(exc.current)),
        ) from exc
    except mutation.GuideAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except mutation.GuideNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (UnsupportedMediaTypeError, PayloadTooLargeError, InvalidUploadError) as exc:
        raise _attempt_upload_http_error(exc) from exc

    result = GuideSchemaForDB.model_validate(serialize_guide(stored))

    # Same rule as the CAS route: retranslate only when authoring content actually moved.
    if old_guide is not None and changed and background_tasks is not None:
        background_tasks.add_task(
            retranslate_changed_texts_task,
            result.guideId,
            old_guide,
            result.model_dump(by_alias=True),
        )
    return result


@router.delete(
    "/guide-upload-attempts/{attempt_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_guide_upload_attempt_route(attempt_id: str):
    """
    §8a.5 — abort. Deletes only paths in this attempt's own directory; a committed guide and its
    media are unreachable from here by construction.

    Deleting an attempt that is already gone is success, not 404: an abort must be safely
    repeatable by a client that lost the response to its first try.
    """
    log.info("--> DELETE /guide-upload-attempts/%s", attempt_id)
    with contextlib.suppress(attempts.AttemptNotFoundError):
        attempts.abort_attempt(attempt_id)
    return


@router.get("/guide/{guide_id}/meta", response_model=GuideMetaResponse)
def get_guide_meta_route(guide_id: str):
    log.info(f"--> GET /guide/{guide_id}/meta - load guide metadata.")
    try:
        return get_guide_meta(guide_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/guide/{guide_id}", response_model=GuideSchemaForDB)
def get_single_guide_route(guide_id: str):
    log.info(f"--> GET /guide/{guide_id} - load single guide.")
    try:
        return get_single_guide(guide_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/guides_from_db", response_model=list[GuideSchemaForDB])
def get_guides_to_pull_route(request: ClientSyncRequest):
    log.info(f"--> POST /guides_from_db - client has {len(request.localGuides)} guides.")
    try:
        return get_guides_to_pull(request)
    except Exception as exc:
        log.error(f"Error while loading guides for pull sync: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Datenbankabfrage fehlgeschlagen.") from exc


@router.delete("/guide/{guide_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_guide_route(guide_id: str, expectedBaseVersion: int | None = None):
    """
    §7b — a user-driven delete runs inside the same per-guide critical section as every other
    writer. When the caller supplies the version it was looking at, a concurrent edit makes the
    delete fail with 409 instead of silently discarding that edit.
    """
    log.info(f"--> DELETE /guide/{guide_id} - full deletion started (expectedBaseVersion={expectedBaseVersion}).")
    try:
        delete_guide(guide_id, expected_base_version=expectedBaseVersion)
    except mutation.GuideVersionConflictError as exc:
        mutation.log_version_conflict(exc, path="delete")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=conflict_detail(exc.guide_id, exc.expected, exc.current, serialize_guide(exc.current)),
        ) from exc
    return
