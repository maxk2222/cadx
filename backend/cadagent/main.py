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
    library,
    mesh,
    projects,
    settings,
    sse,
    versions,
)
from cadagent.config import REPO_ROOT, get_settings
from cadagent.logging import configure_logging, get_logger

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


async def _bootstrap_runtime_config() -> None:
    """Seed the DB runtime row once, then import a legacy .env if present.

    Runs in the API process (native or docker). A host-side .env only exists
    for native/dev installs — the docker image never contains it. The file is
    removed only after the import committed successfully.
    """
    from cadagent.db.session import session_scope
    from cadagent.runtime_config import ensure_runtime_seeded, import_legacy_env

    try:
        async with session_scope() as db:
            if await ensure_runtime_seeded(db):
                log.info("runtime_config.seeded")
            legacy = REPO_ROOT / ".env"
            if legacy.exists():
                result = await import_legacy_env(db, legacy)
                await db.commit()
                if not result.get("already_imported"):
                    log.info(
                        "runtime_config.legacy_imported",
                        keys=len(result.get("imported") or []),
                    )
                legacy.unlink()
                log.info("runtime_config.legacy_env_removed")
            else:
                await db.commit()
    except Exception as e:  # never block startup on config bootstrap
        log.warning("runtime_config.bootstrap_failed", error=str(e)[:300])


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    s.data_path.mkdir(parents=True, exist_ok=True)
    log.info("cadagent.start", data_dir=s.data_dir)
    await _bootstrap_runtime_config()
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
    library.router,
    settings.router,
    mesh.router,
    sse.router,
):
    app.include_router(r, prefix="/api")


# Serve the built frontend (SPA fallback) when frontend/dist exists.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
