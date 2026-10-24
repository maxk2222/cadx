"""Part service — the shared resolution/lookup helpers for project parts.

One chokepoint for "which Part does this operation belong to" so the REST
routers, the chat agent and the MCP server never diverge:

- `resolve_part` — validate an explicit part_id (project-scoped), match a
  name, or auto-create a part derived from the prompt. Spawn callers that
  pass nothing get a deterministic new part — full backward compatibility
  for evalkit/MCP/wizard paths that predate parts.
- `part_meta` — version count + best thumbnail for part cards.
- `delete_part` — subtree cascade (deepest-first) reusing the same cleanup
  order as version deletion; refused while any subtree job is non-terminal.

No FastAPI imports — routers translate the SpawnError codes.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func
from sqlalchemy import update as sa_update
from sqlmodel import col, select

from cadx import storage
from cadx.db.models import (
    AgentTrace,
    Artifact,
    Job,
    JobStatus,
    Part,
    Version,
    VersionStatus,
    utcnow,
)
from cadx.spawn import SpawnError

_NON_TERMINAL = (JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input)


async def next_part_number(db: Any, project_id: uuid.UUID) -> int:
    """Per-project sequential part number — mirrors next_version_number."""
    c = await db.execute(
        select(func.coalesce(func.max(Part.number), 0)).where(Part.project_id == project_id)
    )
    return int(c.scalar_one()) + 1


def derive_part_name(prompt: str, number: int) -> str:
    """Short display name from a generation prompt — char-limited so CJK
    prompts (no word boundaries) still produce sane names."""
    flat = re.sub(r"\s+", " ", prompt or "").strip()
    if not flat:
        return f"Part {number}"
    if len(flat) > 48:
        cut = flat[:48]
        # prefer ending on a space so latin prompts don't clip mid-word;
        # CJK has no spaces — the char cut stands then
        flat = cut.rsplit(" ", 1)[0].strip() if " " in cut[:-1] else cut
    return flat or f"Part {number}"


async def create_part(db: Any, project_id: uuid.UUID, name: str = "") -> Part:
    """Explicit part creation (REST `POST /parts`, agent resolve)."""
    p = Part(
        project_id=project_id,
        number=await next_part_number(db, project_id),
        name=(name or "").strip()[:200],
    )
    db.add(p)
    await db.flush()
    return p


async def resolve_part(
    db: Any,
    project_id: uuid.UUID,
    *,
    part_id: uuid.UUID | str | None = None,
    part_name: str | None = None,
    prompt_for_name: str = "",
) -> Part:
    """Resolve the part a NEW ROOT version should belong to.

    - `part_id` given → must exist in this project (else 404)
    - `part_name` given → case-insensitive match wins; no match → create
      the part under that name
    - neither → auto-create a part named from `prompt_for_name`
    """
    if part_id is not None:
        try:
            pid = uuid.UUID(str(part_id))
        except ValueError:
            raise SpawnError("invalid part id", status=422) from None
        p = await db.get(Part, pid)
        if p is None or p.project_id != project_id:
            raise SpawnError("part not found in this project", status=404)
        return p
    if part_name is not None and part_name.strip():
        want = part_name.strip()
        rows = await db.execute(
            select(Part).where(Part.project_id == project_id).order_by(col(Part.number))
        )
        for p in rows.scalars().all():
            if p.name.casefold() == want.casefold():
                return p
        return await create_part(db, project_id, want[:200])
    number = await next_part_number(db, project_id)
    return await create_part(db, project_id, derive_part_name(prompt_for_name, number))


async def _subtree_versions(db: Any, part: Part) -> list[Version]:
    """All versions of a part, parents before children."""
    rows = await db.execute(
        select(Version).where(Version.part_id == part.id).order_by(col(Version.number))
    )
    return list(rows.scalars().all())


async def part_meta(db: Any, part: Part) -> dict[str, Any]:
    """`version_count` + `thumbnail_url` for part cards — the best version's
    image (highest score, newest tiebreak; else latest), same ranking as the
    project-card thumbnail."""
    versions = await _subtree_versions(db, part)
    if not versions:
        return {"version_count": 0, "thumbnail_url": None, "accepted_version_id": None}
    done = [
        v for v in versions if v.status in (VersionStatus.succeeded, VersionStatus.needs_review)
    ]
    accepted = next((v for v in versions if v.accepted_at is not None), None)
    # the user-verified master is the part's representative once it has
    # geometry; a still-running accepted version falls back to the best pick
    pick = (
        accepted
        if accepted is not None
        and accepted.status in (VersionStatus.succeeded, VersionStatus.needs_review)
        else max(done or versions, key=lambda v: (v.score if v.score is not None else -1, v.number))
    )
    arts = await db.execute(
        select(Artifact).where(
            Artifact.version_id == pick.id,
            col(Artifact.kind).in_(["thumbnail", "render", "contact_sheet"]),
        )
    )
    rank = {"thumbnail": 0, "render": 1, "contact_sheet": 2}
    picks = sorted(
        arts.scalars().all(),
        key=lambda a: (rank.get(str(a.kind), 3), 0 if a.view == "iso" else 1),
    )
    thumb = f"/api/artifacts/{picks[0].id}/file" if picks else f"/api/versions/{pick.id}/preview"
    return {
        "version_count": len(versions),
        "thumbnail_url": thumb,
        "accepted_version_id": accepted.id if accepted else None,
    }


async def set_version_accepted(db: Any, v: Version, accepted: bool) -> None:
    """Mark/unmark a version as the user-verified master of its part.

    At most one version per part is accepted — accepting clears every other
    version in the same part in the same transaction (last writer wins).
    The caller commits. Idempotent: re-accepting refreshes the timestamp."""
    if accepted:
        await db.execute(
            sa_update(Version)
            .where(Version.part_id == v.part_id, Version.id != v.id)
            .values(accepted_at=None)
        )
        v.accepted_at = utcnow()
    else:
        v.accepted_at = None


async def delete_versions_leaf(db: Any, versions: list[Version]) -> None:
    """Shared leaf cleanup for single-version delete and the part cascade:
    signal cancel once for the whole set → commit → traces → jobs →
    artifacts → storage dirs → rows (highest number first so children go
    before their parents). Callers guard upfront: the single delete checks
    for children; the part cascade checks for non-terminal jobs."""
    vids = [v.id for v in versions]
    if not vids:
        return
    jids = select(Job.id).where(col(Job.version_id).in_(vids))
    await db.execute(
        sa_update(Job).where(col(Job.version_id).in_(vids)).values(cancel_requested=True)
    )
    await db.commit()
    await db.execute(sa_delete(AgentTrace).where(col(AgentTrace.job_id).in_(jids)))
    await db.execute(sa_delete(Job).where(col(Job.version_id).in_(vids)))
    await db.execute(sa_delete(Artifact).where(col(Artifact.version_id).in_(vids)))
    for v in versions:
        storage.delete_version_dir(v.project_id, v.id)
    for v in sorted(versions, key=lambda x: x.number, reverse=True):
        await db.delete(v)


async def delete_part(db: Any, part: Part) -> int:
    """Cascade-delete the part's whole version subtree.

    Refuses while any job in the subtree is non-terminal (409) — same guard
    the REST delete uses. `delete_versions_leaf` signals cancel on every
    subtree job first so in-flight workers stop before their rows vanish.
    Returns the number of deleted versions.
    """
    versions = await _subtree_versions(db, part)
    vids = [v.id for v in versions]
    if vids:
        active = await db.execute(
            select(Job.id).where(col(Job.version_id).in_(vids), col(Job.status).in_(_NON_TERMINAL))
        )
        if active.scalars().first() is not None:
            raise SpawnError("part has running jobs — cancel them first", status=409)
        await delete_versions_leaf(db, versions)
    await db.delete(part)
    await db.commit()
    return len(versions)
