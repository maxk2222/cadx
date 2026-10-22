"""Project CRUD."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadx import storage
from cadx.api.deps import get_session
from cadx.api.schemas import ProjectCreate, ProjectOut, ProjectUpdate
from cadx.db.models import (
    AgentTrace,
    Artifact,
    Asset,
    Chat,
    ChatMessage,
    Job,
    Project,
    Version,
    VersionStatus,
)

router = APIRouter(prefix="/projects", tags=["projects"])


async def _latest_meta(
    project_id: uuid.UUID, db: AsyncSession
) -> tuple[uuid.UUID | None, VersionStatus | None, str | None]:
    """(id, status, image url) for project cards.

    id/status describe the highest-numbered version (drives the live overlay
    via running_version_id); the image comes from the *best* version — the
    highest-scoring succeeded/needs_review version, newest as tiebreak, else
    the latest version — preferring its thumbnail, then a render, then the
    scan-based preview route which also finds unregistered render files."""
    rows = list(
        (await db.execute(select(Version).where(Version.project_id == project_id))).scalars().all()
    )
    if not rows:
        return None, None, None
    latest = max(rows, key=lambda v: v.number)
    done = [v for v in rows if v.status in (VersionStatus.succeeded, VersionStatus.needs_review)]
    best = (
        max(done, key=lambda v: (v.score if v.score is not None else -1, v.number))
        if done
        else latest
    )
    arts = await db.execute(
        select(Artifact).where(
            Artifact.version_id == best.id,
            col(Artifact.kind).in_(["thumbnail", "render", "contact_sheet"]),
        )
    )
    rank = {"thumbnail": 0, "render": 1, "contact_sheet": 2}
    picks = sorted(
        arts.scalars().all(),
        key=lambda a: (rank.get(str(a.kind), 3), 0 if a.view == "iso" else 1),
    )
    if picks:
        return latest.id, latest.status, f"/api/artifacts/{picks[0].id}/file"
    # no registered image artifacts — the preview endpoint still resolves
    # render files on disk (unregistered iteration dirs etc.)
    return latest.id, latest.status, f"/api/versions/{best.id}/preview"


def _out(
    p: Project,
    count: int = 0,
    latest_version_id: uuid.UUID | None = None,
    latest_status: VersionStatus | None = None,
    latest_thumbnail_url: str | None = None,
) -> ProjectOut:
    return ProjectOut(
        id=p.id,
        name=p.name,
        description=p.description,
        settings=p.settings,
        created_at=p.created_at,
        updated_at=p.updated_at,
        version_count=count,
        latest_version_status=latest_status,
        latest_thumbnail_url=latest_thumbnail_url,
        latest_version_id=latest_version_id,
        running_version_id=(
            latest_version_id
            if latest_status
            in (VersionStatus.queued, VersionStatus.running, VersionStatus.waiting_for_input)
            else None
        ),
    )


@router.post("", response_model=ProjectOut)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_session)):
    p = Project(
        name=body.name,
        description=body.description,
        settings=body.settings.model_dump() if body.settings else {},
    )
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return _out(p)


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: AsyncSession = Depends(get_session)):
    rows = await db.execute(select(Project).order_by(col(Project.updated_at).desc()))
    projects = list(rows.scalars().all())
    out: list[ProjectOut] = []
    for p in projects:
        c = await db.execute(
            select(func.count()).select_from(Version).where(Version.project_id == p.id)
        )
        vid, status, thumb = await _latest_meta(p.id, db)
        out.append(_out(p, c.scalar_one(), vid, status, thumb))
    return out


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "project not found")
    c = await db.execute(
        select(func.count()).select_from(Version).where(Version.project_id == p.id)
    )
    vid, status, thumb = await _latest_meta(p.id, db)
    return _out(p, c.scalar_one(), vid, status, thumb)


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: uuid.UUID, body: ProjectUpdate, db: AsyncSession = Depends(get_session)
):
    p = await db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "project not found")
    if body.name is not None:
        p.name = body.name
    if body.description is not None:
        p.description = body.description
    if body.settings is not None:
        p.settings = body.settings.model_dump()
    p.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(p)
    return _out(p)


@router.delete("/{project_id}")
async def delete_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "project not found")
    vids = select(Version.id).where(col(Version.project_id) == project_id)
    jids = select(Job.id).where(col(Job.version_id).in_(vids))
    await db.execute(sa_delete(AgentTrace).where(col(AgentTrace.job_id).in_(jids)))
    await db.execute(sa_delete(Job).where(col(Job.version_id).in_(vids)))
    await db.execute(sa_delete(Artifact).where(col(Artifact.version_id).in_(vids)))
    await db.execute(sa_delete(Version).where(col(Version.project_id) == project_id))
    await db.execute(sa_delete(Asset).where(col(Asset.project_id) == project_id))
    # chats bound to the project go too — the FK would otherwise fire
    cids = select(Chat.id).where(col(Chat.project_id) == project_id)
    await db.execute(sa_delete(ChatMessage).where(col(ChatMessage.chat_id).in_(cids)))
    await db.execute(sa_delete(Chat).where(col(Chat.project_id) == project_id))
    await db.delete(p)
    await db.commit()
    storage.delete_project_dir(project_id)
    return {"ok": True}
