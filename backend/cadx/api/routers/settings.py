"""Read and update the database-backed application runtime configuration."""

from typing import Any

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.api.deps import get_session
from cadx.api.schemas import SettingsOut, SettingsUpdate
from cadx.config import get_settings
from cadx.db.models import AppSetting
from cadx.llm.capabilities import (
    caps_from_card,
    load_caps_session,
    load_llm_caps,
    record_caps_session,
)
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
    "hf_token",
)


async def _out(db: AsyncSession, settings: RuntimeSettings) -> SettingsOut:
    caps = await load_llm_caps(db)
    detected = {k: c.context_tokens for k, c in caps.items() if c.context_tokens}
    # executor-reported resources (written by cad_startup into an AppSetting
    # row); falls back to local detection for dev mode where api+executor
    # share the host anyway
    row = await db.get(AppSetting, "resources_detected")
    res: dict[str, int] = dict(row.value) if isinstance(row and row.value, dict) else {}
    if not res:
        from cadx.cad.resources import detected as _local

        res = _local()
    return SettingsOut(
        **settings.public_values(),
        data_dir=get_settings().data_dir,
        llm_detected=detected,
        resources_detected=res,
    )


@router.get("", response_model=SettingsOut)
async def read_settings(db: AsyncSession = Depends(get_session)) -> SettingsOut:
    return await _out(db, await load_runtime_settings(db))


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
    return await _out(db, updated)


async def _test_client(settings: RuntimeSettings) -> Any:
    return make_llm(settings, caps=await load_caps_session(), caps_cb=record_caps_session)


@router.post("/test")
async def test_provider(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Test the saved provider config without returning or logging credentials."""
    settings = await load_runtime_settings(db)
    try:
        client = await _test_client(settings)
    except ValueError as e:
        return {"ok": False, "vision_ok": False, "error": str(e), "latency_ms": 0}
    return await client.test()


@router.get("/health")
async def llm_health(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Polled LLM reachability probe for the status cards / sidebar dot —
    same probe as POST /test, GET so it can sit in a cached query."""
    settings = await load_runtime_settings(db)
    try:
        client = await _test_client(settings)
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
                    "context_tokens": m.get("context_len"),
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
                r = await client.get(f"{settings.llm_base_url.rstrip('/')}/models", headers=headers)
            r.raise_for_status()
            models = [
                {
                    "id": str(m.get("id")),
                    "vision": False,
                    "context_tokens": (
                        caps.context_tokens if (caps := caps_from_card(m)) else None
                    ),
                }
                for m in r.json().get("data", [])
                if m.get("id")
            ]
        return {"backend": backend, "models": models, "error": None}
    except Exception as e:
        return {"backend": backend, "models": [], "error": str(e)}
