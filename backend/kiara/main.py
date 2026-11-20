"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from kiara.api.routers import (
    app_updates,
    assets,
    catalog,
    chat,
    editor,
    files,
    guide,
    guides,
    health,
    images,
    jobs,
    library,
    mesh,
    parts,
    projects,
    settings,
    sse,
    stats,
    sync,
    sync_assets,
    system,
    versions,
)
from kiara.config import get_settings
from kiara.logging import configure_logging, get_logger
from kiara.mcpserver.http import MCPDispatchMiddleware, mcp_asgi_app

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


async def _bootstrap_runtime_config() -> None:
    """Seed the DB runtime row once (MESH_BASE_URL/MESH_MODEL env seed)."""
    from kiara.db.session import session_scope
    from kiara.runtime_config import ensure_runtime_seeded

    try:
        async with session_scope() as db:
            if await ensure_runtime_seeded(db):
                log.info("runtime_config.seeded")
            await db.commit()
    except Exception as e:  # never block startup on config bootstrap
        log.warning("runtime_config.bootstrap_failed", error=str(e)[:300])


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    s.data_path.mkdir(parents=True, exist_ok=True)
    log.info("kiara.start", data_dir=s.data_dir)
    await _bootstrap_runtime_config()
    # Guide-sync recovery sweeps (web-suite parity): drop stale upload attempts
    # left by a crash mid-upload and clear orphaned store staging dirs. Both are
    # bounded, idempotent and run off the event loop.
    from fastapi.concurrency import run_in_threadpool

    from kiara.guides import attempts
    from kiara.guides.asset_store import sweep_asset_store

    await run_in_threadpool(attempts.expire_attempts)
    await run_in_threadpool(sweep_asset_store)
    # the MCP session manager's task group lives for the app's lifetime —
    # /mcp requests fail until it is running
    from kiara.mcpserver.http import mcp_lifespan

    async with mcp_lifespan(app):
        yield
    from kiara.api.deps import close_pool

    await close_pool()


app = FastAPI(title="KIARA", version="0.1.0", lifespan=lifespan)


# ---------- global error envelope ----------
# Unhandled exceptions must never leak internals (LLM keys, tracebacks with
# file paths/env) to clients — log the full context server-side with a
# correlation id, return a stable JSON body.
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    import secrets

    from fastapi.responses import JSONResponse

    err_id = secrets.token_hex(6)
    # NOTE: never log exc.args blobs blindly — LLM payloads can carry keys.
    # type + bounded str(exc) only; secrets never leave this process.
    log.error(
        "api.unhandled",
        err_id=err_id,
        method=request.method,
        path=request.url.path,
        error=str(exc)[:500],
        exc_type=type(exc).__name__,
        exc_info=exc,
    )
    return JSONResponse(
        {"detail": f"internal error — reference {err_id}", "error_id": err_id},
        status_code=500,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (
    health.router,
    projects.router,
    assets.router,
    versions.router,
    jobs.router,
    parts.router,
    files.router,
    catalog.router,
    editor.router,
    library.router,
    settings.router,
    mesh.router,
    images.router,
    sse.router,
    stats.router,
    system.router,
    chat.router,
    guide.router,
):
    app.include_router(r, prefix="/api")


# Headset-sync surface (KIARA Web Suite parity): bare route paths mounted at
# both /sync and /api/sync; sync_assets/app_updates carry their own full
# paths; instructions CRUD is dual-mounted like the suite.
app.include_router(sync.router, prefix="/sync")
app.include_router(sync.router, prefix="/api/sync")
app.include_router(sync_assets.router)
app.include_router(app_updates.router)
app.include_router(guides.router, prefix="/instructions")
app.include_router(guides.router, prefix="/api/instructions")
# suite parity: the settings surface is dual-mounted bare + /api
# (the router carries its own /settings prefix — mount it unprefixed here)
app.include_router(settings.router)


# MCP server (Streamable HTTP) — ASGI dispatch instead of Mount (a Mount
# 307-redirects bare /mcp; MCP clients need POST /mcp to answer directly).
# Added after CORS so it wraps outermost and never falls through to the
# SPA StaticFiles mount.
app.add_middleware(MCPDispatchMiddleware, prefix="/mcp", subapp=mcp_asgi_app())


# Serve the built frontend (SPA fallback) when frontend/dist exists.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
