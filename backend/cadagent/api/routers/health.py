"""Health check."""

from fastapi import APIRouter

from cadagent.config import get_settings

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "version": APP_VERSION,
        "llm_configured": bool(s.llm_api_key),
    }
