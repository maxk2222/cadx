"""In-process job queue: one GPU job at a time, coarse cancellation, TTL cleanup."""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from pathlib import Path

from meshserver.adapters.base import GenOpts
from meshserver.config import ServerConfig
from meshserver.postprocess import clean_mesh, export_glb
from meshserver.preprocess import prep_images
from meshserver.registry import Registry
from meshserver.schemas import GenParams, JobStatus
from meshserver.logging import get_logger

log = get_logger(__name__)


class MeshJob:
    def __init__(self, job_id: str, params: GenParams, image_paths: list[Path]) -> None:
        self.id = job_id
        self.params = params
        self.image_paths = image_paths
        self.status = "queued"
        self.progress = 0.0
        self.message = "queued"
        self.error: str | None = None
        self.cancelled = False
        self.created = time.monotonic()
        self.started: float | None = None
        self.finished: float | None = None
        self.glb: bytes | None = None
        self.triangles: int | None = None
        self.model = params.model
        self.backend = ""
        self.seed = params.seed
        self.steps = params.steps
        self.guidance = params.guidance
        self.resolution = params.resolution

    def to_status(self) -> JobStatus:
        dur = (self.finished or time.monotonic()) - (self.started or self.created)
        return JobStatus(
            status=self.status,
            progress=round(self.progress, 3),
            eta_s=None,
            message=self.message,
            model=self.model,
            backend=self.backend,
            seed=self.seed,
            steps=self.steps,
            guidance=self.guidance,
            resolution=self.resolution,
            duration_s=round(max(dur, 0.0), 2),
            triangles=self.triangles,
            error=self.error,
        )


class JobManager:
    def __init__(self, cfg: ServerConfig, registry: Registry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.jobs: dict[str, MeshJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    # -- public API ------------------------------------------------------- #

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, params: GenParams, image_paths: list[Path]) -> MeshJob | None:
        self._gc()
        queued = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
        if queued >= self.cfg.max_queue:
            return None
        job = MeshJob(str(uuid.uuid4()), params, image_paths)
        self.jobs[job.id] = job
        await self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> MeshJob | None:
        self._gc()
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.cancelled = True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.monotonic()
        # running jobs finish their model call but discard the result
        return True

    async def delete(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        d = self.cfg.job_dir / job_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        return True

    def _gc(self) -> None:
        now = time.monotonic()
        for jid, job in list(self.jobs.items()):
            if job.status in ("succeeded", "failed", "cancelled") and (
                now - (job.finished or job.created) > self.cfg.job_ttl_s
            ):
                self.jobs.pop(jid, None)
                shutil.rmtree(self.cfg.job_dir / jid, ignore_errors=True)

    # -- worker ------------------------------------------------------------ #

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                await asyncio.to_thread(self._run_job, job)
            except Exception as e:
                log.exception("job.crashed", job=job_id)
                job.status = "failed"
                job.error = str(e)
                job.finished = time.monotonic()

    def _run_job(self, job: MeshJob) -> None:  # runs in a thread
        job.status = "running"
        job.started = time.monotonic()
        job.message = "preprocessing images"

        images = prep_images(job.image_paths, self.cfg.bg_removal)
        n = len(images)

        cls = self.registry.select(job.params.model, n)
        if cls is None:
            job.status = "failed"
            job.error = (
                f"no adapter available for model '{job.params.model}' "
                f"on device '{self.registry.device}' — see /v1/models"
            )
            job.finished = time.monotonic()
            return
        job.model = cls.id
        job.backend = cls.backend

        if len(images) > cls.max_images:
            images = images[: cls.max_images]

        def progress(frac: float, msg: str) -> None:
            job.progress = 0.05 + min(frac, 1.0) * 0.9
            job.message = msg

        def cancelled() -> bool:
            return job.cancelled

        adapter = self.registry.get(cls)
        job.message = f"generating with {cls.name}"
        result = adapter.generate(
            images,
            GenOpts(
                seed=job.params.seed,
                steps=job.params.steps,
                guidance=job.params.guidance,
                resolution=job.params.resolution,
                subject=job.params.subject,
                image_roles=job.params.image_roles,
                work_dir=self.cfg.job_dir / job.id,
            ),
            progress=progress,
            cancelled=cancelled,
        )

        if job.cancelled:
            job.status = "cancelled"
            job.finished = time.monotonic()
            return

        job.message = "cleaning + exporting"
        job.progress = 0.95
        mesh = clean_mesh(result.mesh)
        job.glb = export_glb(mesh)
        job.triangles = len(mesh.faces)
        job.seed = result.seed
        job.steps = result.steps
        job.guidance = result.guidance
        job.resolution = result.resolution
        job.status = "succeeded"
        job.progress = 1.0
        job.message = "done"
        job.finished = time.monotonic()
        self.registry.maybe_unload_idle()
