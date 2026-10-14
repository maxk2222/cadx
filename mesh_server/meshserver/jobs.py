"""In-process job queue: one GPU job at a time, coarse cancellation, TTL cleanup.

Two job kinds share the single worker thread — image-to-3D (MeshJob) and
text-to-image (ImageJob) — so a mesh model and an image model can never be
resident at the same time and single-flight GPU usage holds automatically.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from meshserver.adapters.base import GenOpts
from meshserver.adapters.image_base import ImageGenOpts
from meshserver.config import ServerConfig
from meshserver.postprocess import clean_mesh, export_glb
from meshserver.preprocess import prep_images
from meshserver.registry import Registry
from meshserver.schemas import GenParams, ImageGenParams, JobStatus
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
        # monotonic timestamp of the last *observable* progress change —
        # the stall watchdog reads this via JobManager.get()
        self.last_progress_at = time.monotonic()

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


class ImageJob:
    def __init__(
        self, job_id: str, params: ImageGenParams, init_image_path: Path | None = None
    ) -> None:
        self.id = job_id
        self.params = params
        self.status = "queued"
        self.progress = 0.0
        self.message = "queued"
        self.error: str | None = None
        self.cancelled = False
        self.created = time.monotonic()
        self.started: float | None = None
        self.finished: float | None = None
        self.images: list[bytes] = []  # PNG payloads
        self.init_image_path = init_image_path
        self.model = params.model
        self.backend = ""
        self.seed = params.seed
        self.steps = params.steps
        self.guidance = params.guidance
        self.last_progress_at = time.monotonic()

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
            duration_s=round(max(dur, 0.0), 2),
            images=len(self.images),
            error=self.error,
        )


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(
        m in msg
        for m in ("out of memory", "cublas", "cudnn", "metal", "allocation failed", "mps")
    )


class JobManager:
    def __init__(self, cfg: ServerConfig, registry: Registry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.jobs: dict[str, MeshJob | ImageJob] = {}
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
        if self._queued() >= self.cfg.max_queue:
            return None
        job = MeshJob(str(uuid.uuid4()), params, image_paths)
        self.jobs[job.id] = job
        await self._queue.put(job.id)
        return job

    async def submit_image(
        self, params: ImageGenParams, init_image_path: Path | None = None
    ) -> ImageJob | None:
        self._gc()
        if self._queued() >= self.cfg.max_queue:
            return None
        job = ImageJob(str(uuid.uuid4()), params, init_image_path)
        self.jobs[job.id] = job
        await self._queue.put(job.id)
        return job

    def _queued(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))

    def get(self, job_id: str) -> MeshJob | ImageJob | None:
        self._gc()
        job = self.jobs.get(job_id)
        # stall watchdog: a 'running' job whose progress hasn't moved is
        # dead — the adapter thread may be frozen on MPS/MLX memory waits.
        # Mark it failed so clients stop polling; the thread still gets to
        # unwind and its result is discarded via the cancelled flag.
        if (
            job is not None
            and job.status == "running"
            and time.monotonic() - job.last_progress_at > self.cfg.job_stall_s
        ):
            job.status = "failed"
            job.error = (
                f"stalled — no progress for {self.cfg.job_stall_s}s; "
                "the worker thread may still be unwinding"
            )
            job.finished = time.monotonic()
            job.cancelled = True  # adapter checks this and aborts early
            log.warning("job.stalled", job=job_id, model=job.model)
        return job

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

    @staticmethod
    def _cleanup_job_files(job: MeshJob | ImageJob, job_dir: Path) -> None:
        shutil.rmtree(job_dir / job.id, ignore_errors=True)
        # staged edit-job init image lives in job_dir/incoming/, outside the
        # per-job dir — remove it explicitly
        init = getattr(job, "init_image_path", None)
        if init is not None:
            init.unlink(missing_ok=True)

    async def delete(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        self._cleanup_job_files(job, self.cfg.job_dir)
        return True

    def _gc(self) -> None:
        now = time.monotonic()
        for jid, job in list(self.jobs.items()):
            if job.status in ("succeeded", "failed", "cancelled") and (
                now - (job.finished or job.created) > self.cfg.job_ttl_s
            ):
                self.jobs.pop(jid, None)
                self._cleanup_job_files(job, self.cfg.job_dir)

    # -- worker ------------------------------------------------------------ #

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                if isinstance(job, ImageJob):
                    await asyncio.to_thread(self._run_image_job, job)
                else:
                    await asyncio.to_thread(self._run_job, job)
            except Exception as e:
                log.exception("job.crashed", job=job_id)
                job.status = "failed"
                job.finished = time.monotonic()
                if _is_oom(e):
                    # memory exhausted — unload everything, purge caches and
                    # badlist the adapter so 'auto' picks a fallback next time
                    self.registry.unload_all()
                    failed_id = getattr(job, "model", "")
                    if failed_id and failed_id != "auto":
                        self.registry.mark_failed(failed_id, str(e))
                    job.error = (
                        f"VRAM/memory exhausted — try a smaller model, fewer steps "
                        f"or a smaller resolution ({type(e).__name__}: {str(e)[:200]})"
                    )
                else:
                    job.error = str(e)[:500]
            finally:
                self.registry.maybe_unload_idle()

    def _load_adapter(
        self,
        job: Any,
        *,
        explicit: bool,
        select: Callable[[], type | None],
        kind_label: str,
    ) -> tuple[Any | None, type | None]:
        """Select + load an adapter; on 'auto' fall through to the next adapter
        when loading fails (missing entry point, gated weights, OOM at load).

        Load failures badlist the adapter so later jobs skip it too; an
        explicitly requested model still fails immediately (client asked for
        that model, a silent swap would be wrong)."""
        attempted: set[str] = set()
        while True:
            cls = select()
            if cls is None or cls.id in attempted:
                job.status = "failed"
                job.error = (
                    f"no {'image ' if kind_label == 'image' else ''}adapter available "
                    f"for model '{job.params.model}' "
                    f"on device '{self.registry.device}' — see /v1/models"
                )
                job.finished = time.monotonic()
                return None, None
            attempted.add(cls.id)
            try:
                return self.registry.get(cls), cls
            except Exception as e:
                self.registry.mark_failed(cls.id, str(e))
                log.warning("job.adapter_load_failed", model=cls.id, error=str(e)[:300])
                if explicit:
                    raise
                job.message = f"{cls.name} failed to load — trying next adapter"

    def _run_job(self, job: MeshJob) -> None:  # runs in a thread
        job.status = "running"
        job.started = time.monotonic()
        job.message = "preprocessing images"

        images = prep_images(job.image_paths, self.cfg.bg_removal)
        n = len(images)

        adapter, cls = self._load_adapter(
            job,
            explicit=job.params.model not in ("", "auto"),
            select=lambda: self.registry.select(job.params.model, n),
            kind_label="mesh",
        )
        if cls is None:
            return
        job.model = cls.id
        job.backend = cls.backend

        if len(images) > cls.max_images:
            images = images[: cls.max_images]

        def progress(frac: float, msg: str) -> None:
            new_p = 0.05 + min(frac, 1.0) * 0.9
            if new_p != job.progress or msg != job.message:
                job.last_progress_at = time.monotonic()
            job.progress = new_p
            job.message = msg

        def cancelled() -> bool:
            return job.cancelled

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
        job.last_progress_at = time.monotonic()
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

    def _run_image_job(self, job: ImageJob) -> None:  # runs in a thread
        job.status = "running"
        job.started = time.monotonic()
        job.message = "selecting image adapter"

        adapter, cls = self._load_adapter(
            job,
            explicit=job.params.model not in ("", "auto"),
            select=lambda: self.registry.select_image(
                job.params.model, need_edit=job.init_image_path is not None
            ),
            kind_label="image",
        )
        if cls is None:
            if job.init_image_path is not None:
                job.error = (job.error or "") + " (edit job needs a supports_edit adapter)"
            return
        job.model = cls.id
        job.backend = cls.backend

        # clamp request params to the adapter's safe envelope (OOM prevention)
        count = min(job.params.count, getattr(cls, "max_count", 4))
        width = min(job.params.width, getattr(cls, "max_side", 1024))
        height = min(job.params.height, getattr(cls, "max_side", 1024))

        def progress(frac: float, msg: str) -> None:
            new_p = 0.05 + min(frac, 1.0) * 0.9
            if new_p != job.progress or msg != job.message:
                job.last_progress_at = time.monotonic()
            job.progress = new_p
            job.message = msg

        def cancelled() -> bool:
            return job.cancelled

        adapter = self.registry.get(cls)
        job.message = f"generating with {cls.name}"
        result = adapter.generate(
            ImageGenOpts(
                prompt=job.params.prompt,
                negative=job.params.negative,
                seed=job.params.seed,
                count=count,
                width=width,
                height=height,
                steps=job.params.steps,
                guidance=job.params.guidance,
                work_dir=self.cfg.job_dir / job.id,
                image_path=job.init_image_path,
                image_strength=job.params.strength,
            ),
            progress=progress,
            cancelled=cancelled,
        )

        if job.cancelled:
            job.status = "cancelled"
            job.finished = time.monotonic()
            return

        job.images = [_png_bytes(im) for im in result.images]
        job.seed = result.seed
        job.steps = result.steps
        job.guidance = result.guidance
        job.status = "succeeded"
        job.progress = 1.0
        job.message = "done"
        job.finished = time.monotonic()
