"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cadagent.api.routers import (
    assets,
    catalog,
    files,
    health,
    jobs,
    mesh,
    projects,
    settings,
    sse,
    versions,
)
from cadagent.config import get_settings
from cadagent.logging import configure_logging, get_logger

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    s.data_path.mkdir(parents=True, exist_ok=True)
    log.info("cadagent.start", data_dir=s.data_dir)
    yield
    from cadagent.api.deps import close_pool

    await close_pool()


app = FastAPI(title="cadagent", version="0.1.0", lifespan=lifespan)

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
    settings.router,
    mesh.router,
    sse.router,
):
    app.include_router(r, prefix="/api")


# Serve the built frontend (SPA fallback) when frontend/dist exists.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
