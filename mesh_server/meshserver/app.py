"""FastAPI app: /v1/health, /v1/models, /v1/jobs CRUD + mesh download,
/v1/image-jobs CRUD + PNG download (text-to-image adapter family),
/v1/chat/completions (OpenAI-compatible proxy to local llm adapters),
/v1/queue + model download/registration management routes."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from meshserver.config import ServerConfig, get_config
from meshserver.downloads import DownloadManager, downloadable
from meshserver.jobs import ImageJob, JobManager, MeshJob
from meshserver.llm_models import register_custom_model, unregister_custom_model
from meshserver.llm_proxy import chat_completions
from meshserver.logging import configure_logging, get_logger
from meshserver.registry import Registry
from meshserver.schemas import (
    DownloadStatus,
    GenParams,
    Health,
    ImageGenParams,
    JobCreated,
    JobStatus,
    ModelInfo,
    QueueInfo,
    RegisterModelRequest,
)

log = get_logger(__name__)


def create_app(cfg: ServerConfig | None = None) -> FastAPI:
    cfg = cfg or get_config()
    configure_logging()
    registry = Registry(cfg)
    manager = JobManager(cfg, registry)

    def _is_loaded(model_id: str) -> bool:
        inst = registry._instances.get(model_id)
        return bool(inst and inst.loaded)

    downloads = DownloadManager(cfg, is_loaded=_is_loaded)
    registry.downloads = downloads

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.job_dir.mkdir(parents=True, exist_ok=True)
        manager.start()
        log.info(
            "meshserver.start",
            device=registry.device,
            port=cfg.port,
            llm_models=len(getattr(registry, "_llm_classes", [])),
        )
        yield
        await manager.stop()
        registry.unload_all()

    app = FastAPI(title="mesh-server", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/health", response_model=Health)
    async def health() -> Health:
        import psutil

        return Health(
            ok=True,
            device=registry.device,
            backend="local",
            models=registry.models(),
            mem_free_mb=int(psutil.virtual_memory().available / 1e6),
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
    async def job_mesh(job_id: str, i: int = 0) -> Response:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != "succeeded" or job.glb is None:
            raise HTTPException(409, f"job is {job.status}, mesh unavailable")
        glbs = getattr(job, "glbs", None) or {}
        if i in glbs:
            return Response(
                content=glbs[i],
                media_type="model/gltf-binary",
                headers={"Content-Disposition": f"attachment; filename=mesh_{i}.glb"},
            )
        if i == 0:
            return Response(
                content=job.glb,
                media_type="model/gltf-binary",
                headers={"Content-Disposition": "attachment; filename=mesh.glb"},
            )
        raise HTTPException(404, f"no variant {i} (outputs: {sorted(glbs)})")

    @app.delete("/v1/jobs/{job_id}")
    async def job_delete(job_id: str) -> dict:
        await manager.cancel(job_id)
        deleted = await manager.delete(job_id)
        return {"ok": True, "deleted": deleted}

    # -- text-to-image jobs ------------------------------------------------ #

    @app.post("/v1/image-jobs", response_model=JobCreated)
    async def create_image_job(params: ImageGenParams) -> JobCreated:
        job = await manager.submit_image(params)
        if job is None:
            raise HTTPException(429, "job queue full")
        return JobCreated(job_id=job.id)

    @app.post("/v1/image-edit-jobs", response_model=JobCreated)
    async def create_image_edit_job(
        params_file: UploadFile = File(alias="json"),
        image: UploadFile = File(...),
    ) -> JobCreated:
        """Multipart variant of /v1/image-jobs carrying an init image for
        img2img / instruction-edit adapters (supports_edit=True)."""
        params = ImageGenParams(**json_loads(await params_file.read()))
        job_dir = cfg.job_dir / "incoming"
        job_dir.mkdir(parents=True, exist_ok=True)
        # filename is client-controlled — strip any directory components
        safe_name = Path(image.filename or "init.png").name or "init.png"
        init_path = job_dir / f"edit_{uuid.uuid4().hex[:8]}_{safe_name}"
        init_path.write_bytes(await image.read())
        job = await manager.submit_image(params, init_image_path=init_path)
        if job is None:
            raise HTTPException(429, "job queue full")
        return JobCreated(job_id=job.id)

    @app.get("/v1/image-jobs/{job_id}", response_model=JobStatus)
    async def image_job_status(job_id: str) -> JobStatus:
        job = manager.get(job_id)
        if job is None or not isinstance(job, ImageJob):
            raise HTTPException(404, "job not found")
        return job.to_status()

    @app.get("/v1/image-jobs/{job_id}/image")
    async def image_job_image(job_id: str, i: int = 0) -> Response:
        job = manager.get(job_id)
        if job is None or not isinstance(job, ImageJob):
            raise HTTPException(404, "job not found")
        if job.status != "succeeded":
            raise HTTPException(409, f"job is {job.status}, image unavailable")
        if i < 0 or i >= len(job.images):
            raise HTTPException(400, f"image index {i} out of range ({len(job.images)})")
        return Response(
            content=job.images[i],
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=image{i}.png"},
        )

    @app.delete("/v1/image-jobs/{job_id}")
    async def image_job_delete(job_id: str) -> dict:
        await manager.cancel(job_id)
        deleted = await manager.delete(job_id)
        return {"ok": True, "deleted": deleted}

    # -- llm chat completions (OpenAI-compatible proxy) -------------------- #

    @app.post("/v1/chat/completions")
    async def llm_chat(request: Request) -> Response:
        return await chat_completions(registry, cfg, json_loads(await request.body()))

    # -- scheduling / model management ------------------------------------- #

    @app.get("/v1/queue", response_model=QueueInfo)
    async def queue_info() -> QueueInfo:
        return manager.queue_snapshot()

    @app.post("/v1/models/{model_id}/download", response_model=DownloadStatus)
    async def model_download(model_id: str) -> DownloadStatus:
        cls = registry.find_class(model_id)
        if cls is None:
            raise HTTPException(404, f"unknown model '{model_id}'")
        if not downloadable(cls):
            raise HTTPException(400, f"model '{model_id}' has no managed download")
        return downloads.start(model_id, cls)

    @app.get("/v1/models/{model_id}/download", response_model=DownloadStatus)
    async def model_download_status(model_id: str) -> DownloadStatus:
        return downloads.status(model_id)

    @app.delete("/v1/models/{model_id}/download")
    async def model_download_cancel(model_id: str) -> dict:
        return {"ok": downloads.cancel(model_id)}

    @app.delete("/v1/models/{model_id}/weights")
    async def model_weights_delete(model_id: str) -> dict:
        cls = registry.find_class(model_id)
        if cls is None:
            raise HTTPException(404, f"unknown model '{model_id}'")
        deleted = downloads.delete_weights(model_id, cls)
        if not deleted:
            return {"deleted": False, "error": "model is loaded or downloading"}
        return {"deleted": True, "error": None}

    @app.post("/v1/models/register", response_model=ModelInfo)
    async def model_register(req: RegisterModelRequest) -> ModelInfo:
        try:
            info = register_custom_model(cfg, req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        registry.refresh_llm_classes()
        return info

    @app.delete("/v1/models/{model_id}/custom")
    async def model_unregister(model_id: str) -> dict:
        if _is_loaded(model_id):
            raise HTTPException(409, "model is loaded — unload it first")
        ok = unregister_custom_model(cfg, model_id)
        if ok:
            registry.refresh_llm_classes()
        return {"ok": ok}

    app.state.registry = registry
    app.state.manager = manager
    app.state.downloads = downloads
    app.state.cfg = cfg
    return app


def json_loads(data: bytes) -> dict:
    try:
        return json.loads(data)
    except Exception as e:
        raise HTTPException(422, f"invalid params json: {e}") from e


app = create_app()
