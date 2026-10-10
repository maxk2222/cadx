"""Mesh generator sidecar proxy: health + model list for the settings UI."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadagent.agent.mesh_client import make_mesh_client
from cadagent.api.deps import get_session
from cadagent.api.schemas import MeshHealthOut, MeshModelOut
from cadagent.db.models import AppSetting

router = APIRouter(prefix="/mesh", tags=["mesh"])


async def _client(db: AsyncSession):
    row = await db.get(AppSetting, "mesh")
    base_url = (row.value or {}).get("mesh_base_url") if row else None
    return make_mesh_client(base_url=base_url)


@router.get("/health", response_model=MeshHealthOut)
async def mesh_health(db: AsyncSession = Depends(get_session)):
    client = await _client(db)
    if client is None:
        return MeshHealthOut(ok=False, error="mesh server not configured (MESH_BASE_URL)")
    h = await client.health()
    return MeshHealthOut(
        ok=h.ok,
        device=h.device,
        backend=h.backend,
        error=h.error,
        models=[MeshModelOut(**m.model_dump()) for m in h.models],
    )


@router.get("/models", response_model=list[MeshModelOut])
async def mesh_models(db: AsyncSession = Depends(get_session)):
    client = await _client(db)
    if client is None:
        return []
    return [MeshModelOut(**m.model_dump()) for m in await client.models()]
