"""Mesh generator sidecar proxy: health, model list + model-management API."""

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.agent.mesh_client import make_mesh_client
from cadx.api.deps import get_session
from cadx.api.schemas import (
    DownloadStatusOut,
    MeshHealthOut,
    MeshModelOut,
    MeshRegisterIn,
    QueueInfoOut,
)
from cadx.runtime_config import load_runtime_settings

router = APIRouter(prefix="/mesh", tags=["mesh"])


async def _client(db: AsyncSession):
    return make_mesh_client(runtime_settings=await load_runtime_settings(db))


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


# --------------------------------------------------------------------------- #
# model-management proxies (sidecar /v1/* — independent of mesh_backend)      #
# --------------------------------------------------------------------------- #


def _sidecar_detail(r: httpx.Response) -> Any:
    try:
        data = r.json()
        if isinstance(data, dict) and "detail" in data:
            return data["detail"]
        return data
    except Exception:
        return r.text[:500]


async def _proxy(
    db: AsyncSession,
    method: str,
    path: str,
    *,
    timeout_s: float = 15.0,
    body: dict[str, Any] | None = None,
) -> Any:
    """Forward one request to the sidecar; map its error body onto the API.
    Carries the configured Hugging Face token so model downloads/loads can
    reach gated repos."""
    settings = await load_runtime_settings(db)
    base = settings.mesh_base_url.strip().rstrip("/")
    if not base:
        raise HTTPException(503, "mesh server not configured")
    headers = {}
    if tok := settings.hf_token.get_secret_value():
        headers["X-HF-Token"] = tok
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.request(method, f"{base}{path}", json=body, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"mesh server unreachable: {e}") from e
    if r.status_code >= 400:
        raise HTTPException(r.status_code, detail=_sidecar_detail(r))
    try:
        return r.json()
    except Exception:
        return {}


@router.get("/queue", response_model=QueueInfoOut)
async def mesh_queue(db: AsyncSession = Depends(get_session)):
    return QueueInfoOut(**await _proxy(db, "GET", "/v1/queue"))


@router.post("/models/{model_id:path}/download", response_model=DownloadStatusOut)
async def mesh_model_download(model_id: str, db: AsyncSession = Depends(get_session)):
    # model weights can be GBs — the POST only starts the download (202-ish),
    # but older sidecars may block until done, so allow a wider window
    return DownloadStatusOut(
        **await _proxy(db, "POST", f"/v1/models/{model_id}/download", timeout_s=30.0)
    )


@router.get("/models/{model_id:path}/download", response_model=DownloadStatusOut)
async def mesh_model_download_status(model_id: str, db: AsyncSession = Depends(get_session)):
    return DownloadStatusOut(**await _proxy(db, "GET", f"/v1/models/{model_id}/download"))


@router.delete("/models/{model_id:path}/download")
async def mesh_model_download_cancel(model_id: str, db: AsyncSession = Depends(get_session)):
    return await _proxy(db, "DELETE", f"/v1/models/{model_id}/download")


@router.delete("/models/{model_id:path}/weights")
async def mesh_model_weights_delete(model_id: str, db: AsyncSession = Depends(get_session)):
    return await _proxy(db, "DELETE", f"/v1/models/{model_id}/weights")


@router.post("/models/register", response_model=MeshModelOut)
async def mesh_model_register(body: MeshRegisterIn, db: AsyncSession = Depends(get_session)):
    return MeshModelOut(
        **await _proxy(db, "POST", "/v1/models/register", body=body.model_dump(exclude_none=True))
    )


@router.delete("/models/{model_id:path}/custom")
async def mesh_model_custom_delete(model_id: str, db: AsyncSession = Depends(get_session)):
    return await _proxy(db, "DELETE", f"/v1/models/{model_id}/custom")
