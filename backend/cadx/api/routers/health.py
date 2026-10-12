"""Health check."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.api.deps import get_session
from cadx.runtime_config import load_runtime_settings

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"


@router.get("/health")
async def health(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    settings = await load_runtime_settings(db)
    return {
        "ok": True,
        "version": APP_VERSION,
        "llm_configured": bool(settings.llm_api_key.get_secret_value()),
    }
