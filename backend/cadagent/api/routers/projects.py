"""Project CRUD."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadagent import storage
from cadagent.api.deps import get_session
from cadagent.api.schemas import ProjectCreate, ProjectOut, ProjectUpdate
from cadagent.db.models import AgentTrace, Artifact, Asset, Job, Project, Version

router = APIRouter(prefix="/projects", tags=["projects"])


def _out(p: Project, count: int = 0) -> ProjectOut:
    return ProjectOut(
        id=p.id,
        name=p.name,
        description=p.description,
        settings=p.settings,
        created_at=p.created_at,
        updated_at=p.updated_at,
        version_count=count,
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
        out.append(_out(p, c.scalar_one()))
    return out


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "project not found")
    c = await db.execute(
        select(func.count()).select_from(Version).where(Version.project_id == p.id)
    )
    return _out(p, c.scalar_one())


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
    await db.delete(p)
    await db.commit()
    storage.delete_project_dir(project_id)
    return {"ok": True}
