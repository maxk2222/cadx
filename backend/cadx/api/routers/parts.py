"""Part CRUD — named subtrees inside a project.

List/create/rename/delete parts; deletion cascades the whole version
subtree (deepest-first, same cleanup as version deletion) and refuses
while any job inside is non-terminal. Business logic lives in
`cadx.parts` — these handlers only translate errors to HTTP.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx import parts as parts_svc
from cadx.api.deps import get_session
from cadx.api.schemas import PartCreate, PartOut, PartUpdate
from cadx.db.models import Part, Project
from cadx.spawn import SpawnError

router = APIRouter(tags=["parts"])
log = structlog.get_logger()


async def _out(db: AsyncSession, part: Part) -> PartOut:
    meta = await parts_svc.part_meta(db, part)
    return PartOut(
        id=part.id,
        project_id=part.project_id,
        number=part.number,
        name=part.name,
        version_count=meta["version_count"],
        thumbnail_url=meta["thumbnail_url"],
        created_at=part.created_at,
    )


@router.get("/projects/{project_id}/parts", response_model=list[PartOut])
async def list_parts(project_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    rows = await db.execute(
        select(Part).where(Part.project_id == project_id).order_by(col(Part.number))
    )
    return [await _out(db, p) for p in rows.scalars().all()]


@router.post("/projects/{project_id}/parts", response_model=PartOut)
async def create_part(
    project_id: uuid.UUID, body: PartCreate, db: AsyncSession = Depends(get_session)
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    p = await parts_svc.create_part(db, project_id, body.name)
    await db.commit()
    return await _out(db, p)


@router.patch("/parts/{part_id}", response_model=PartOut)
async def update_part(
    part_id: uuid.UUID, body: PartUpdate, db: AsyncSession = Depends(get_session)
):
    p = await db.get(Part, part_id)
    if not p:
        raise HTTPException(404, "part not found")
    if body.name is not None:
        p.name = body.name.strip()[:200]
        p.updated_at = datetime.now(UTC)
    await db.commit()
    return await _out(db, p)


@router.delete("/parts/{part_id}")
async def delete_part(part_id: uuid.UUID, db: AsyncSession = Depends(get_session)):
    p = await db.get(Part, part_id)
    if not p:
        raise HTTPException(404, "part not found")
    try:
        n = await parts_svc.delete_part(db, p)
    except SpawnError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"ok": True, "versions_deleted": n}
