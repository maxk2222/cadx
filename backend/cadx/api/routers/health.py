"""Health check.

`ok`/200 is pinned to the database session — the compose healthcheck must not
bounce the api because a worker or redis is flapping. Everything else is
reported as `degraded` detail: redis reachability and the age of each
worker's heartbeat key (`cadx:hb:<role>`, written by the worker loop — when
the loop wedges the key expires, which is the stall signal)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.api.deps import get_pool, get_session
from cadx.queue import HB_KEY_PREFIX
from cadx.runtime_config import load_runtime_settings

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"

# heartbeat TTL is 45s — beyond this the key is gone and the worker loop is
# either dead or wedged
HB_STALE_S = 60.0


async def _worker_ages() -> dict[str, dict[str, float | bool | None]]:
    """Seconds since each worker's last heartbeat (None = no key / redis down)."""
    out: dict[str, dict[str, float | bool | None]] = {
        "agent": {"age_s": None, "stale": True},
        "cad": {"age_s": None, "stale": True},
    }
    try:
        pool = await get_pool()
        for role in out:
            raw = await pool.get(f"{HB_KEY_PREFIX}{role}")
            if raw:
                try:
                    ts = datetime.fromisoformat(raw.decode())
                    age = round((datetime.now(UTC) - ts).total_seconds(), 1)
                    out[role] = {"age_s": age, "stale": age > HB_STALE_S}
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return out


@router.get("/health")
async def health(db: AsyncSession = Depends(get_session)) -> dict[str, object]:
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    redis_ok = False
    workers = await _worker_ages()
    try:
        await (await get_pool()).ping()
        redis_ok = True
    except Exception:
        pass
    degraded = (not redis_ok) or any(bool(w["stale"]) for w in workers.values())
    settings = await load_runtime_settings(db) if db_ok else None
    body = {
        "ok": db_ok,
        "version": APP_VERSION,
        "llm_configured": bool(settings.llm_api_key.get_secret_value()) if settings else False,
        "db": db_ok,
        "redis": redis_ok,
        "workers": workers,
        "degraded": degraded,
    }
    # the compose healthcheck (curl -sf) only sees the status code — a dead DB
    # must surface as unhealthy so the watchdog can react
    if not db_ok:
        from fastapi.responses import JSONResponse

        return JSONResponse(body, status_code=503)
    return body
