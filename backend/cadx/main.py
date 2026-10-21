"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cadx.api.routers import (
    assets,
    catalog,
    chat,
    editor,
    files,
    health,
    images,
    jobs,
    library,
    mesh,
    projects,
    settings,
    sse,
    stats,
    system,
    versions,
)
from cadx.config import get_settings
from cadx.logging import configure_logging, get_logger
from cadx.mcpserver.http import MCPDispatchMiddleware, mcp_asgi_app

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


async def _bootstrap_runtime_config() -> None:
    """Seed the DB runtime row once (MESH_BASE_URL/MESH_MODEL env seed)."""
    from cadx.db.session import session_scope
    from cadx.runtime_config import ensure_runtime_seeded

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
    log.info("cadx.start", data_dir=s.data_dir)
    await _bootstrap_runtime_config()
    # the MCP session manager's task group lives for the app's lifetime —
    # /mcp requests fail until it is running
    from cadx.mcpserver.http import mcp_lifespan

    async with mcp_lifespan(app):
        yield
    from cadx.api.deps import close_pool

    await close_pool()


app = FastAPI(title="CADX", version="0.1.0", lifespan=lifespan)

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
):
    app.include_router(r, prefix="/api")


# MCP server (Streamable HTTP) — ASGI dispatch instead of Mount (a Mount
# 307-redirects bare /mcp; MCP clients need POST /mcp to answer directly).
# Added after CORS so it wraps outermost and never falls through to the
# SPA StaticFiles mount.
app.add_middleware(MCPDispatchMiddleware, prefix="/mcp", subapp=mcp_asgi_app())


# Serve the built frontend (SPA fallback) when frontend/dist exists.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
