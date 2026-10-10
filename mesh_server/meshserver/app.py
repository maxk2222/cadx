"""FastAPI app: /v1/health, /v1/models, /v1/jobs CRUD + mesh download."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from meshserver.config import ServerConfig, get_config
from meshserver.jobs import JobManager
from meshserver.logging import configure_logging, get_logger
from meshserver.registry import Registry
from meshserver.schemas import GenParams, Health, JobCreated, JobStatus, ModelInfo

log = get_logger(__name__)


def create_app(cfg: ServerConfig | None = None) -> FastAPI:
    cfg = cfg or get_config()
    configure_logging()
    registry = Registry(cfg)
    manager = JobManager(cfg, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.job_dir.mkdir(parents=True, exist_ok=True)
        manager.start()
        log.info("meshserver.start", device=registry.device, port=cfg.port)
        yield
        await manager.stop()

    app = FastAPI(title="mesh-server", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/health", response_model=Health)
    async def health() -> Health:
        return Health(
            ok=True, device=registry.device, backend="local", models=registry.models()
        )

    @app.get("/v1/models", response_model=dict)
    async def models() -> dict:
        return {"models": [m.model_dump() for m in registry.models()]}

    @app.post("/v1/jobs", response_model=JobCreated)
    async def create_job(
        params_file: UploadFile = File(alias="json"),
        images: list[UploadFile] = File(...),
    ) -> JobCreated:
        params = GenParams(**json_loads(await params_file.read()))
        if not images:
            raise HTTPException(422, "at least one image is required")
        job_dir = cfg.job_dir / "incoming"
        job_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, img in enumerate(images):
            name = img.filename or f"image{i}.png"
            p = job_dir / f"{name}.{i}"
            p.write_bytes(await img.read())
            paths.append(p)
        job = await manager.submit(params, paths)
        if job is None:
            raise HTTPException(429, "job queue full")
        return JobCreated(job_id=job.id)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    async def job_status(job_id: str) -> JobStatus:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.to_status()

    @app.get("/v1/jobs/{job_id}/mesh")
    async def job_mesh(job_id: str) -> Response:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != "succeeded" or job.glb is None:
            raise HTTPException(409, f"job is {job.status}, mesh unavailable")
        return Response(
            content=job.glb,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": "attachment; filename=mesh.glb"},
        )

    @app.delete("/v1/jobs/{job_id}")
    async def job_delete(job_id: str) -> dict:
        await manager.cancel(job_id)
        deleted = await manager.delete(job_id)
        return {"ok": True, "deleted": deleted}

    app.state.registry = registry
    app.state.manager = manager
    app.state.cfg = cfg
    return app


def json_loads(data: bytes) -> dict:
    try:
        return json.loads(data)
    except Exception as e:
        raise HTTPException(422, f"invalid params json: {e}") from e


app = create_app()
