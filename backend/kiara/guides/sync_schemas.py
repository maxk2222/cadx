from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from kiara.guides.schemas import GuideSchema, GuideSchemaForDB


class ClientGuideInfo(BaseModel):
    guideId: str
    updatedAt: str
    # C1 Slice 2: pull SELECTION may prefer the server-authoritative version when the client knows
    # it. This is not a conflict decision (§7f.8) — the conflict is decided by CAS on push and by
    # the clean/dirty dialog on pull. `updatedAt` remains only as a legacy-client fallback for
    # deciding which guides to offer for download.
    version: int | None = None


class ClientSyncRequest(BaseModel):
    localGuides: list[ClientGuideInfo]


class MediaUploadResponse(BaseModel):
    mediaId: str
    fileUrl: str
    status: str


class SyncConflictResponse(BaseModel):
    hasConflict: bool
    conflictType: str
    serverVersion: GuideSchemaForDB | None = None
    recommendedAction: str


class GuideMetaResponse(BaseModel):
    guideId: str
    updatedAt: str
    version: int | None = None
    # C1 Slice 2 (§7a.7): what a client needs to build its next compare-and-swap.
    baseVersion: int | None = None
    contentHash: str | None = None


# ── C1 Slice 2 (§7c) — the compare-and-swap envelope ─────────────────────────────────────────────


class GuideCasEnvelope(BaseModel):
    """
    `POST /api/sync/guide` body.

    `expectedBaseVersion` is the version the client last had ACKNOWLEDGED by this server (§7a.7).
    It is not a revision the client invented — the server never trusts a client-computed number as
    the new revision (§1c); it only compares this one against the stored version.

    Omitting it is legal ONLY for creation of a guideId that does not exist yet. For an existing
    guide it produces `428 Precondition Required`, never a blind overwrite (§7a.5, §7c.2).
    """

    guide: GuideSchema
    expectedBaseVersion: int | None = None
    model_config = ConfigDict(extra="allow")


class GuideConflictResponse(BaseModel):
    """
    `409 Conflict` body (§7c.4). Carries the current authoritative metadata plus the full server
    guide, which is what the decided clean/dirty conflict workflow needs (§7e) to offer
    "Save mine first / Discard mine / Keep working" without a second round trip.
    """

    error: str = "version_conflict"
    guideId: str
    expectedBaseVersion: int | None = None
    currentVersion: int | None = None
    currentContentHash: str | None = None
    serverGuide: GuideSchemaForDB | None = None


class GuidePreconditionRequiredResponse(BaseModel):
    """`428 Precondition Required` body (§7a.5)."""

    error: str = "expected_base_version_required"
    guideId: str
    currentVersion: int | None = None
    currentContentHash: str | None = None


# ── C1 Slice 3 (§8a) — the upload-attempt protocol ───────────────────────────────────────────────


class AttemptMediaDescriptor(BaseModel):
    """
    ONE medium a client declares up front (§8a.1).

    `mediaKey` is the client's own label for this medium and becomes the last path segment of the
    upload URL. `length`/`sha256` are optional but strongly recommended: when present the server
    verifies the streamed bytes against them and refuses the medium otherwise (§8a.3), which is
    what lets the client trust that "uploaded" means "the server has exactly my file".

    `assetType` is cross-checked against the file extension; a mismatch is refused rather than
    quietly corrected, because the extension decides where the file lands and which cap applies.
    """

    mediaKey: str
    objectId: str
    fileName: str
    assetType: str | None = None
    length: int | None = None
    sha256: str | None = None
    required: bool = True
    model_config = ConfigDict(extra="allow")


class GuideUploadAttemptRequest(BaseModel):
    """
    `POST /api/sync/guide-upload-attempts` body (§8a.1).

    `expectedBaseVersion` has exactly the meaning it has on the CAS upload route: the version this
    client last had ACKNOWLEDGED by this server. It is recorded now and RECHECKED under the guide
    lock at commit — declaring an attempt is not itself a precondition check and publishes nothing.
    Omitting it is legal only for a guideId that does not exist yet.
    """

    guideId: str
    expectedBaseVersion: int | None = None
    media: list[AttemptMediaDescriptor] = []
    model_config = ConfigDict(extra="allow")


class AttemptMediaState(BaseModel):
    """One medium's server-side state, as the attempt manifest holds it."""

    mediaKey: str
    objectId: str
    assetType: str
    fileName: str
    required: bool = True
    state: str
    receivedLength: int | None = None
    sha256: str | None = None
    contentType: str | None = None
    fileUrl: str | None = None
    model_config = ConfigDict(extra="allow")


class GuideUploadAttemptResponse(BaseModel):
    """`POST /api/sync/guide-upload-attempts` response — the opaque attempt id and its manifest."""

    attemptId: str
    guideId: str
    expectedBaseVersion: int | None = None
    createdAt: str
    expiresAt: str
    media: list[AttemptMediaState] = []


class GuideUploadAttemptCommitRequest(BaseModel):
    """
    `POST /api/sync/guide-upload-attempts/{attemptId}/commit` body (§8a.4).

    It carries the extra-preserving guide document only: the expected base version already belongs
    to the attempt, so a client cannot change the precondition between declaring and committing.
    """

    guide: GuideSchema
    model_config = ConfigDict(extra="allow")


class GuideConflictCopyRequest(BaseModel):
    """
    `POST /api/sync/guide/{guideId}/conflict-copy` body (C1 Slice 4, §9c.11).

    `guide` is the editor's VISIBLE document. Its identity/lineage fields are stripped server-side
    and the copy gets a fresh guideId and databaseId; no expected version applies, because the copy
    is a creation.
    """

    guide: GuideSchema
    model_config = ConfigDict(extra="allow")


def attempt_response(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project an attempt manifest onto the wire shape, media dict flattened to a list."""
    media = list((manifest.get("media") or {}).values())
    return GuideUploadAttemptResponse(
        attemptId=str(manifest.get("attemptId") or ""),
        guideId=str(manifest.get("guideId") or ""),
        expectedBaseVersion=manifest.get("expectedBaseVersion"),
        createdAt=str(manifest.get("createdAt") or ""),
        expiresAt=str(manifest.get("expiresAt") or ""),
        media=[AttemptMediaState.model_validate(item) for item in media],
    ).model_dump(by_alias=True)


def conflict_detail(
    guide_id: str,
    expected: int | None,
    current: dict[str, Any],
    serialized: dict[str, Any] | None,
) -> dict[str, Any]:
    return GuideConflictResponse(
        guideId=guide_id,
        expectedBaseVersion=expected,
        currentVersion=current.get("version"),
        currentContentHash=current.get("contentHash"),
        serverGuide=GuideSchemaForDB.model_validate(serialized) if serialized else None,
    ).model_dump(by_alias=True)


def precondition_detail(guide_id: str, current: dict[str, Any]) -> dict[str, Any]:
    return GuidePreconditionRequiredResponse(
        guideId=guide_id,
        currentVersion=current.get("version"),
        currentContentHash=current.get("contentHash"),
    ).model_dump(by_alias=True)
