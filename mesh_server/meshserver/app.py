"""FastAPI app: /v1/health, /v1/models, /v1/jobs CRUD + mesh download,
/v1/image-jobs CRUD + PNG download (text-to-image adapter family),
/v1/chat/completions + /v1/embeddings + /v1/images/* (OpenAI-compatible),
/v1/queue + /v1/config + model download/registration management routes.

LAN exposure: a non-loopback bind requires MESH_API_KEY(S) — the
BearerAuthMiddleware enforces it per-request and create_app refuses to
start without keys, so an open GPU endpoint can never be published
accidentally."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from meshserver import runtime_config as rcfg
from meshserver.auth import BearerAuthMiddleware
from meshserver.config import ServerConfig, _is_loopback, get_config
from meshserver.downloads import DownloadManager, downloadable
from meshserver.jobs import ImageJob, JobManager, MeshJob
from meshserver.llm_models import register_custom_model, unregister_custom_model
from meshserver.llm_proxy import chat_completions, embeddings as llm_embeddings
from meshserver.logging import configure_logging, get_logger
from meshserver.openai_compat import (
    images_edits,
    images_generations,
    openai_models,
)
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
    if not _is_loopback(cfg.host) and not cfg.api_keys:
        raise RuntimeError(
            f"refusing to bind '{cfg.host}' without API keys — set MESH_API_KEY "
            "(or MESH_API_KEYS) or bind a loopback host"
        )
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

    # middleware order (add_middleware wraps outermost-last): the hf-token
    # forwarder (added below via @app.middleware) ends up outermost, then
    # CORS (handles preflights + decorates error responses), then bearer
    # auth innermost before routing — pure ASGI on purpose, since
    # BaseHTTPMiddleware would buffer and break the SSE relay
    app.add_middleware(BearerAuthMiddleware, api_keys=cfg.api_keys)
    origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    if origins or cfg.lan:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins or ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def hf_token_forward(request: Request, call_next):
        """The CADX backend forwards the configured Hugging Face token as a
        per-request header. Apply it process-wide so every huggingface_hub
        call — managed weight downloads, adapter loads, gen_worker children
        (they inherit os.environ at spawn) — can reach gated repos."""
        token = request.headers.get("x-hf-token", "").strip()
        if token:
            os.environ["HF_TOKEN"] = token
        return await call_next(request)

    @app.get("/v1/health", response_model=Health)
    async def health() -> Health:
        mem_free_mb: int | None = None
        with contextlib.suppress(Exception):
            import psutil

            mem_free_mb = int(psutil.virtual_memory().available / 1e6)
        return Health(
            ok=True,
            device=registry.device,
            backend="local",
            models=registry.models(),
            mem_free_mb=mem_free_mb,
        )

    @app.get("/v1/models", response_model=dict)
    async def models() -> dict:
        # dual-shape: internal `models[]` (CADX management view) plus the
        # OpenAI `data[]` listing so stock SDKs/clients can discover the
        # served llm + image model ids
        listing = {"models": [m.model_dump() for m in registry.models()]}
        listing.update(await openai_models(registry))
        return listing

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

    # -- OpenAI-compatible surface (llm proxy + image shims) --------------- #

    @app.post("/v1/chat/completions")
    async def llm_chat(request: Request) -> Response:
        return await chat_completions(
            registry, cfg, json_loads(await request.body()), manager=manager
        )

    @app.post("/v1/embeddings")
    async def openai_embeddings(request: Request) -> Response:
        return await llm_embeddings(
            registry, cfg, json_loads(await request.body()), manager=manager
        )

    @app.post("/v1/images/generations")
    async def openai_image_generations(request: Request) -> Response:
        return await images_generations(
            manager,
            cfg,
            json_loads(await request.body()),
            base_url=str(request.base_url).rstrip("/"),
        )

    @app.post("/v1/images/edits")
    async def openai_image_edits(
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
    ) -> Response:
        body = {
            k: v
            for k, v in (await request.form()).items()
            if k not in ("image", "mask") and isinstance(v, str)
        }
        return await images_edits(
            manager,
            cfg,
            body,
            image_name=image.filename or "init.png",
            image_bytes=await image.read(),
            has_mask=mask is not None,
            base_url=str(request.base_url).rstrip("/"),
        )

    # -- runtime config (scheduling knobs, auth-protected) ----------------- #

    @app.get("/v1/config")
    async def get_runtime_config() -> dict:
        return rcfg.config_view(cfg)

    @app.put("/v1/config")
    async def put_runtime_config(request: Request) -> dict:
        return rcfg.apply_patch(cfg, json_loads(await request.body()))

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
