"""Read and update the database-backed application runtime configuration."""

from typing import Any

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.api.deps import get_session
from cadx.api.schemas import SettingsOut, SettingsUpdate
from cadx.config import get_settings
from cadx.llm.client import make_llm
from cadx.runtime_config import (
    RuntimeSettings,
    load_runtime_settings,
    save_runtime_settings,
)

router = APIRouter(prefix="/settings", tags=["settings"])

# write-only secret fields — excluded from the generic field loop and
# handled explicitly below (set / clear via dedicated flags)
_SECRET_KEYS = (
    "llm_api_key",
    "tripo_api_key",
    "image_api_key",
)


def _out(settings: RuntimeSettings) -> SettingsOut:
    return SettingsOut(**settings.public_values(), data_dir=get_settings().data_dir)


@router.get("", response_model=SettingsOut)
async def read_settings(db: AsyncSession = Depends(get_session)) -> SettingsOut:
    return _out(await load_runtime_settings(db))


@router.put("", response_model=SettingsOut)
async def update_settings(
    body: SettingsUpdate, db: AsyncSession = Depends(get_session)
) -> SettingsOut:
    current = await load_runtime_settings(db)
    values = current.to_storage()
    # explicitly provided null/"" resets the field to its code default
    # (sparse storage: an absent key means "use default")
    handled = {k for name in _SECRET_KEYS for k in (name, f"clear_{name}")}
    for key in body.model_fields_set - handled:
        v = getattr(body, key)
        if v is None or v == "":
            values.pop(key, None)
        else:
            values[key] = v
    for name in _SECRET_KEYS:
        if getattr(body, f"clear_{name}"):
            values[name] = ""
        elif name in body.model_fields_set and getattr(body, name):
            values[name] = getattr(body, name)

    updated = RuntimeSettings.model_validate(values)
    await save_runtime_settings(db, updated)
    await db.commit()
    return _out(updated)


@router.post("/test")
async def test_provider(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Test the saved provider config without returning or logging credentials."""
    settings = await load_runtime_settings(db)
    try:
        client = make_llm(settings)
    except ValueError as e:
        return {"ok": False, "vision_ok": False, "error": str(e), "latency_ms": 0}
    return await client.test()


@router.get("/health")
async def llm_health(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Polled LLM reachability probe for the status cards / sidebar dot —
    same probe as POST /test, GET so it can sit in a cached query."""
    settings = await load_runtime_settings(db)
    try:
        client = make_llm(settings)
    except ValueError as e:
        return {"ok": False, "vision_ok": False, "error": str(e), "latency_ms": 0}
    return await client.test()


@router.get("/llm-models")
async def llm_models(db: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """List the models the ACTIVE llm backend can serve — for model pickers.

    external: GET {llm_base_url}/models (OpenAI-compatible `data[]` — vision
    capability is not reported there, so `vision` stays False). sidecar:
    GET {mesh_base_url}/v1/models filtered to kind=="llm"."""
    settings = await load_runtime_settings(db)
    backend = settings.llm_backend
    try:
        if backend == "sidecar":
            base = settings.mesh_base_url.strip().rstrip("/")
            if not base:
                return {
                    "backend": backend,
                    "models": [],
                    "error": "local sidecar not configured",
                }
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{base}/v1/models")
            r.raise_for_status()
            models = [
                {
                    "id": str(m.get("id")),
                    "vision": bool(m.get("supports_vision")),
                    "downloaded": bool(m.get("downloaded")),
                }
                for m in r.json().get("models", [])
                if m.get("kind") == "llm" and m.get("id")
            ]
        else:
            headers = {}
            key = settings.llm_api_key.get_secret_value()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"{settings.llm_base_url.rstrip('/')}/models", headers=headers
                )
            r.raise_for_status()
            models = [
                {"id": str(m.get("id")), "vision": False}
                for m in r.json().get("data", [])
                if m.get("id")
            ]
        return {"backend": backend, "models": models, "error": None}
    except Exception as e:
        return {"backend": backend, "models": [], "error": str(e)}
