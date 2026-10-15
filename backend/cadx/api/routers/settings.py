"""Read and update the database-backed application runtime configuration."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.api.deps import get_session
from cadx.api.schemas import SettingsOut, SettingsUpdate
from cadx.config import get_settings
from cadx.llm.client import LLMClient
from cadx.runtime_config import (
    RuntimeSettings,
    load_runtime_settings,
    save_runtime_settings,
)

router = APIRouter(prefix="/settings", tags=["settings"])


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
    for key in body.model_fields_set - {"llm_api_key", "clear_llm_api_key"}:
        v = getattr(body, key)
        if v is None or v == "":
            values.pop(key, None)
        else:
            values[key] = v
    if body.clear_llm_api_key:
        values["llm_api_key"] = ""
    elif "llm_api_key" in body.model_fields_set and body.llm_api_key:
        values["llm_api_key"] = body.llm_api_key

    updated = RuntimeSettings.model_validate(values)
    await save_runtime_settings(db, updated)
    await db.commit()
    return _out(updated)


@router.post("/test")
async def test_provider(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Test the saved provider config without returning or logging credentials."""
    settings = await load_runtime_settings(db)
    return await LLMClient(runtime_settings=settings).test()


@router.get("/health")
async def llm_health(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    """Polled LLM reachability probe for the status cards / sidebar dot —
    same probe as POST /test, GET so it can sit in a cached query."""
    settings = await load_runtime_settings(db)
    return await LLMClient(runtime_settings=settings).test()
