"""In-process job queue: one GPU job at a time, coarse cancellation, TTL cleanup.

Two job kinds share the single worker thread — image-to-3D (MeshJob) and
text-to-image (ImageJob) — so a mesh model and an image model can never be
resident at the same time and single-flight GPU usage holds automatically.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import io
import json
import os
import shutil
import signal
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from meshserver.adapters.base import GenOpts
from meshserver.adapters.image_base import ImageGenOpts
from meshserver.config import ServerConfig
from meshserver.logging import get_logger
from meshserver.postprocess import clean_mesh, export_glb
from meshserver.preprocess import prep_images
from meshserver.registry import Registry
from meshserver.schemas import (
    GenParams,
    ImageGenParams,
    JobStatus,
    QueueEntry,
    QueueInfo,
)

log = get_logger(__name__)

# config knobs the gen-worker child can't see via env (test fakes,
# programmatic overrides) — sent in request.json["cfg"] for cold jobs and
# via MESH_SERVE_CFG for warm workers; applied over a fresh ServerConfig()
_CHILD_CFG_KEYS = (
    "fake",
    "allow_cpu",
    "device",
    "bg_removal",
    "default_model",
    "image_default_model",
    "cuda_offload",
    "cuda_offload_vram_gb",
    "flux_mlx_quant",
    "mlx_mem_limit_gb",
    # --serve mode only: the warm worker self-exits after this much stdin
    # silence (harmless on the single-shot path)
    "warm_idle_s",
)


def _open_log(path: Path) -> io.BufferedWriter:
    """stderr sink for a generation subprocess — the handle must outlive the
    call, so no context manager here by design."""
    return path.open("wb")


def _peak_rss_mb() -> float:
    """Process peak RSS in MB — ru_maxrss is bytes on macOS, KiB on Linux.
    Process-wide highwater (adapters share this process), logged per job so a
    growing baseline across jobs shows a leak, a spike shows model weight."""
    import resource
    import sys

    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(v / (1e6 if sys.platform == "darwin" else 1e3), 1)


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
        # batch jobs (params.batch): per-variant GLBs keyed by variant index —
        # /v1/jobs/{id}/mesh?i=N serves them; job.glb mirrors index 0
        self.glbs: dict[int, bytes] = {}
        self.outputs: list[dict[str, Any]] = []
        self.model = params.model
        self.backend = ""
        # adapter picked for this job — explicit models resolve at submit,
        # 'auto' at pick time; the child may refine it (ladder fallback)
        self.resolved_model = ""
        self.seed = params.seed
        self.steps = params.steps
        self.guidance = params.guidance
        self.resolution = params.resolution
        # monotonic timestamp of the last *observable* progress change —
        # the stall watchdog reads this via JobManager.get()
        self.last_progress_at = time.monotonic()
        # subprocess-mode liveness: set by the watch loop when the child is
        # computing — surfaced as `busy` so clients don't mistake a silent
        # stage for a stall
        self.last_active_at = self.last_progress_at

    def to_status(self) -> JobStatus:
        dur = (self.finished or time.monotonic()) - (self.started or self.created)
        return JobStatus(
            status=self.status,
            progress=round(self.progress, 3),
            eta_s=None,
            message=self.message,
            busy=self.status == "running"
            and time.monotonic() - self.last_active_at < 30,
            model=self.model,
            backend=self.backend,
            seed=self.seed,
            steps=self.steps,
            guidance=self.guidance,
            resolution=self.resolution,
            duration_s=round(max(dur, 0.0), 2),
            triangles=self.triangles,
            outputs=[o["index"] for o in self.outputs if o.get("ok")],
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
        self.resolved_model = ""
        self.seed = params.seed
        self.steps = params.steps
        self.guidance = params.guidance
        self.last_progress_at = time.monotonic()
        self.last_active_at = self.last_progress_at

    def to_status(self) -> JobStatus:
        dur = (self.finished or time.monotonic()) - (self.started or self.created)
        return JobStatus(
            status=self.status,
            progress=round(self.progress, 3),
            eta_s=None,
            message=self.message,
            busy=self.status == "running"
            and time.monotonic() - self.last_active_at < 30,
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
        for m in (
            "out of memory",
            "outofmemory",  # hipErrorOutOfMemory — ROCm reports without spaces
            "cublas",
            "cudnn",
            "metal",
            "allocation failed",
            "mps",
        )
    )


class _WarmWorker:
    """Bookkeeping for one `gen_worker --serve` child.

    The child resolves and loads a model on its first request and keeps it
    resident; subsequent stdin lines are request-dir paths it serves against
    that pinned model (a mismatching request makes it exit 4 — "wrong-model").
    `lines` is fed by a reader task on the child's stdout: "ready" at
    startup, then "done|fail|wrong-model <dir>" per request, None on EOF.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.proc: asyncio.subprocess.Process | None = None
        self.lines: asyncio.Queue[str | None] = asyncio.Queue()
        self.state = "starting"  # starting | ready | busy | dead
        self.last_used = time.monotonic()
        self.stderr_file: io.BufferedWriter | None = None
        self.reader_task: asyncio.Task | None = None

    @property
    def idle(self) -> bool:
        return self.state == "ready"


class JobManager:
    def __init__(self, cfg: ServerConfig, registry: Registry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.jobs: dict[str, MeshJob | ImageJob] = {}
        # model-affine scheduling: a deque + event instead of asyncio.Queue
        # so _pick_next can scan all queued jobs and prefer ones matching
        # the resident/warm model
        self._queue: collections.deque[str] = collections.deque()
        self._queue_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        # warm `gen_worker --serve` children keyed by their pinned adapter id
        self._warm: dict[str, _WarmWorker] = {}
        # adapter the previous job ran on — _pick_next scores same-model
        # jobs higher so a run of same-model submissions batches on the
        # resident weights instead of thrashing loads
        self._last_model = ""

    # -- public API ------------------------------------------------------- #

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        for warm in list(self._warm.values()):
            await self._drop_warm(warm)

    def _free_mem_ok(self) -> bool:
        """Refuse new jobs under host memory pressure — a job started with no
        headroom stalls at 0% CPU and can swap the host into a panic."""
        limit = self.cfg.job_min_free_mb
        if limit <= 0:
            return True
        free = self.registry.ledger.free_mb()
        return free is None or free >= limit

    async def submit(self, params: GenParams, image_paths: list[Path]) -> MeshJob | None:
        self._gc()
        if self._queued() >= self.cfg.max_queue or not self._free_mem_ok():
            return None
        job = MeshJob(str(uuid.uuid4()), params, image_paths)
        # explicit models resolve at submit — 'auto' waits for pick time
        # (the preference ladder may shift while the job sits in the queue)
        if params.model not in ("", "auto"):
            job.resolved_model = self._resolve_model(job) or ""
        self.jobs[job.id] = job
        self._queue.append(job.id)
        self._queue_event.set()
        return job

    async def submit_image(
        self, params: ImageGenParams, init_image_path: Path | None = None
    ) -> ImageJob | None:
        self._gc()
        if self._queued() >= self.cfg.max_queue or not self._free_mem_ok():
            return None
        job = ImageJob(str(uuid.uuid4()), params, init_image_path)
        if params.model not in ("", "auto"):
            job.resolved_model = self._resolve_model(job) or ""
        self.jobs[job.id] = job
        self._queue.append(job.id)
        self._queue_event.set()
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
            # a stalled adapter thread cannot be killed — badlist the model so
            # 'auto' and candidate screening stop picking it, and release
            # whatever model memory the registry still owns. Subprocess jobs
            # are killed by the watcher instead and share no adapter state —
            # a transient stall must not poison the model for later jobs.
            if job.model and job.model != "auto" and not self.cfg.jobs_subprocess:
                self.registry.mark_failed(job.model, job.error)
            self.registry.unload_all()
            log.warning("job.stalled", job=job_id, model=job.model, peak_rss_mb=_peak_rss_mb())
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

    # -- scheduling -------------------------------------------------------- #

    def _resolve_model(self, job: MeshJob | ImageJob) -> str | None:
        """Adapter id the registry would pick for this job right now."""
        try:
            if isinstance(job, ImageJob):
                cls = self.registry.select_image(
                    job.params.model, need_edit=job.init_image_path is not None
                )
            else:
                cls = self.registry.select(job.params.model, max(1, len(job.image_paths)))
        except Exception:
            return None
        return cls.id if cls else None

    def _declared_mem(self, adapter_id: str) -> int | None:
        classes = getattr(self.registry, "_classes", []) + getattr(
            self.registry, "_llm_classes", []
        )
        for cls in classes:
            if cls.id == adapter_id:
                return getattr(cls, "est_mem_mb", None)
        return None

    def _admit(self, job: MeshJob | ImageJob) -> bool:
        """Memory admission for a queued job.

        No *new* memory is needed when an idle warm worker or the in-process
        registry already holds the job's model. Otherwise the ledger estimate
        must fit while keeping mem_headroom_mb free — evicting idle warm
        workers (LRU) then resident registry instances to make room."""
        ledger = self.registry.ledger
        resolved = job.resolved_model
        if resolved:
            warm = self._warm.get(resolved)
            if warm is not None and warm.idle:
                return True  # the worker already holds the model
            if not self.cfg.jobs_subprocess and ledger.has(resolved):
                return True  # in-process instance already resident
        need = ledger.est_mb(resolved, self._declared_mem(resolved)) if resolved else 0.0
        while not ledger.fits(need):
            victim = self._lru_idle_warm(exclude={resolved} if resolved else set())
            if victim is not None:
                log.info("warm.evicted_for_memory", model=victim.model)
                self._evict_warm(victim)
                continue
            if self.registry.evict_lru(exclude={resolved} if resolved else set()) is None:
                break
        return ledger.fits(need)

    def _pick_next(self) -> MeshJob | ImageJob | None:
        """Pick the next queued job.

        Score = waited_s + queue_fair_s when the job has model affinity
        (its resolved model is the last one used, or an idle warm worker
        holds it) — same-model bursts batch together while a job older than
        queue_fair_s always wins, so affinity can never starve. Jobs whose
        model can't fit wait ("waiting for memory") and fail after
        queue_mem_wait_s instead of head-of-line blocking the queue."""
        self._warm_gc()
        now = time.monotonic()
        queued: list[MeshJob | ImageJob] = []
        for jid in list(self._queue):
            job = self.jobs.get(jid)
            if job is None or job.status != "queued":
                with contextlib.suppress(ValueError):
                    self._queue.remove(jid)
                continue
            queued.append(job)
        if not queued:
            return None
        # 'auto' resolves at pick time — the preference ladder may have
        # shifted since submit (badlist changes, downloads finishing)
        for job in queued:
            if not job.resolved_model:
                job.resolved_model = self._resolve_model(job) or ""
        fittable: list[MeshJob | ImageJob] = []
        for job in queued:
            if self._admit(job):
                if job.message == "waiting for memory":
                    job.message = "queued"
                fittable.append(job)
                continue
            if self.cfg.queue_mem_wait_s > 0 and now - job.created > self.cfg.queue_mem_wait_s:
                job.status = "failed"
                job.error = "insufficient host memory"
                job.finished = now
                with contextlib.suppress(ValueError):
                    self._queue.remove(job.id)
                log.warning("job.mem_wait_timeout", job=job.id, model=job.resolved_model)
            else:
                job.message = "waiting for memory"
        if not fittable:
            return None

        def _score(job: MeshJob | ImageJob) -> tuple[float, float]:
            waited = now - job.created
            warm = self._warm.get(job.resolved_model)
            affinity = bool(job.resolved_model) and (
                job.resolved_model == self._last_model
                or (warm is not None and warm.idle)
            )
            # -created keeps the order FIFO when scores tie
            return (waited + (self.cfg.queue_fair_s if affinity else 0), -job.created)

        pick = max(fittable, key=_score)
        with contextlib.suppress(ValueError):
            self._queue.remove(pick.id)
        return pick

    def queue_snapshot(self) -> QueueInfo:
        """Queued + running jobs for /v1/queue — scheduling view, plus free
        host memory and the resident set (loaded models + warm workers)."""
        self._warm_gc()
        now = time.monotonic()
        ledger = self.registry.ledger
        entries: list[QueueEntry] = []

        def _entry(job: MeshJob | ImageJob) -> QueueEntry:
            est = (
                ledger.est_mb(job.resolved_model, self._declared_mem(job.resolved_model))
                if job.resolved_model
                else 0.0
            )
            return QueueEntry(
                id=job.id,
                kind="image" if isinstance(job, ImageJob) else "mesh",
                model=job.params.model,
                resolved_model=job.resolved_model,
                status=job.status,
                waited_s=round(now - job.created, 1),
                est_mem_mb=round(est) if est else None,
                message=job.message,
            )

        for jid in list(self._queue):
            job = self.jobs.get(jid)
            if job is not None and job.status == "queued":
                entries.append(_entry(job))
        for job in self.jobs.values():
            if job.status == "running":
                entries.insert(0, _entry(job))
        return QueueInfo(
            jobs=entries,
            free_mb=ledger.free_mb(),
            residents=ledger.residents(),
        )

    # -- worker ------------------------------------------------------------ #

    async def _worker(self) -> None:
        while True:
            job = self._pick_next()
            if job is None:
                # nothing runnable: either the queue is empty (wait forever)
                # or every queued job is blocked on memory — poll on a slow
                # tick so queue_mem_wait_s fires without needing a new submit
                self._queue_event.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._queue_event.wait(),
                        timeout=1.0 if self._queue else None,
                    )
                continue
            job_id = job.id
            try:
                if self.cfg.jobs_subprocess:
                    await self._run_job_proc(job)
                elif isinstance(job, ImageJob):
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
                self._last_model = job.resolved_model or job.model or self._last_model
                self.registry.maybe_unload_idle()
                log.info(
                    "job.finished",
                    job=job_id,
                    status=job.status,
                    model=getattr(job, "model", ""),
                    peak_rss_mb=_peak_rss_mb(),
                )

    async def _run_job_proc(self, job: MeshJob | ImageJob) -> None:
        """Run one generation in a child process (`meshserver.gen_worker`).

        The child owns the model weights and device buffers; on stall, cancel
        or the RSS cap it gets SIGKILLed and the OS reclaims all its memory —
        the main crash vector this single-process queue cannot fix otherwise.
        Progress is reported via progress.json in the job dir; result.json +
        binary payloads land there on success."""
        work = self.cfg.job_dir / job.id
        work.mkdir(parents=True, exist_ok=True)
        # 'auto' jobs retry on the next preference-ladder adapter when the child
        # reports a deterministic error naming a specific adapter —
        # _finish_proc_failed badlists it, so the next dispatch can't land on
        # the same broken model (observed: corrupt qwen-image-edit snapshot →
        # every synth-view job failed instead of falling back to flux/sdxl).
        # Explicit requests keep their single attempt. 4 attempts cover the
        # full edit ladder (qwen → kontext → schnell-mlx → sdxl) — observed:
        # three broken/gated adapters in a row before sdxl could serve.
        max_attempts = 4 if job.params.model in ("", "auto") else 1
        for attempt in range(1, max_attempts + 1):
            # prefer a warm worker holding this job's model — it skips the
            # model load each cold subprocess pays (~30-60s on torch/MPS)
            warm = await self._warm_target(job)
            if warm is not None:
                rc, kill_reason, peak_rss_mb = await self._dispatch_warm(warm, job, work)
            else:
                rc, kill_reason, peak_rss_mb = await self._dispatch_cold(job, work)

            if rc == 0 and kill_reason is None:
                self._finish_proc_ok(job, work, peak_rss_mb)
                self._record_peak(job, peak_rss_mb)
                return
            if (job.cancelled or rc == 2) and kill_reason is None:
                job.status = "cancelled"
                job.finished = time.monotonic()
                self._record_peak(job, peak_rss_mb)
                return
            self._finish_proc_failed(job, work, rc, kill_reason, peak_rss_mb)
            self._record_peak(job, peak_rss_mb)
            # rc 4 = a warm worker rejected the resolved model — fall back to
            # a cold subprocess on the next attempt
            retryable = self._proc_retryable(work) or rc == 4
            if attempt < max_attempts and not job.cancelled and retryable:
                log.info(
                    "job.auto_retry",
                    job=job.id,
                    attempt=attempt,
                    error=(job.error or "")[:200],
                )
                job.status = "running"
                job.error = None
                job.finished = None
                job.message = (
                    f"adapter failed — trying next model (attempt {attempt + 1}/{max_attempts})"
                )
                job.last_progress_at = time.monotonic()
                # the badlist grew — re-resolve so affinity/warm dispatch
                # aim at the next ladder entry instead of the dead adapter
                job.resolved_model = self._resolve_model(job) or job.resolved_model
                # stale attempt artifacts would corrupt the next finish check
                for f in ("error.json", "result.json", "progress.json"):
                    (work / f).unlink(missing_ok=True)
                continue
            return

    async def _dispatch_cold(
        self, job: MeshJob | ImageJob, work: Path
    ) -> tuple[int | None, str | None, float]:
        """Spawn a fresh gen_worker for this request — returns
        (returncode, kill_reason, peak_rss_mb)."""
        (work / "request.json").write_text(json.dumps(self._gen_request(job)))
        # stderr to a file, not a pipe — an orphaned PIPE transport raises
        # "Event loop is closed" in __del__ when the loop ends first
        stderr_file = _open_log(work / "stderr.log")
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "meshserver.gen_worker",
                str(work / "request.json"),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_file,
                # own process group — killpg also reaps torch/MLX helper children
                start_new_session=True,
            )
            job.status = "running"
            job.started = time.monotonic()
            job.message = "generating (subprocess)"
            job.last_progress_at = time.monotonic()
            job.last_active_at = job.last_progress_at
            kill_reason, peak_rss_mb = await self._watch_gen_proc(proc, job, work)
            return proc.returncode, kill_reason, peak_rss_mb
        finally:
            # whatever happens above — exception in the watcher, task cancel —
            # a live child must never leak past this point
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
            stderr_file.close()

    # -- warm generation workers (gen_worker --serve) ----------------------- #

    async def _warm_target(self, job: MeshJob | ImageJob) -> _WarmWorker | None:
        """Idle warm worker for the job's resolved model, or a freshly
        spawned one when allowed. 'auto' jobs keep the cold retry ladder —
        a pinned worker can't fall through to the next adapter."""
        if not (self.cfg.jobs_subprocess and self.cfg.warm_workers):
            return None
        resolved = job.resolved_model
        if not resolved or resolved == "auto":
            return None
        warm = self._warm.get(resolved)
        if warm is not None:
            return warm if warm.idle else None
        if job.params.model in ("", "auto") or len(self._warm) >= self.cfg.warm_max:
            return None
        return await self._spawn_warm(job)

    async def _spawn_warm(self, job: MeshJob | ImageJob) -> _WarmWorker | None:
        """Start a `gen_worker --serve` child and wait for its 'ready' line.

        The child pins whatever model its first request resolves to; the
        parent only ever sends dirs matching ww.model."""
        model = job.resolved_model
        env = dict(os.environ)
        env["MESH_SERVE_CFG"] = json.dumps(
            {k: getattr(self.cfg, k) for k in _CHILD_CFG_KEYS}
            | {"model_dir": str(self.cfg.model_dir)}
        )
        stderr_file = _open_log(self.cfg.job_dir / f"warm-{model}.stderr.log")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "meshserver.gen_worker",
                "--serve",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_file,
                # own process group — killpg reaps torch/MLX helper children
                start_new_session=True,
                env=env,
            )
        except Exception:
            stderr_file.close()
            log.exception("warm.spawn_failed", model=model)
            return None
        warm = _WarmWorker(model)
        warm.proc = proc
        warm.stderr_file = stderr_file
        self._warm[model] = warm
        ledger = self.registry.ledger
        ledger.reserve(
            f"warm:{model}",
            ledger.est_mb(model, self._declared_mem(model)),
            pinned=False,
            kind="warm",
        )
        warm.reader_task = asyncio.create_task(self._warm_reader(warm))
        try:
            line = await asyncio.wait_for(warm.lines.get(), timeout=120)
        except TimeoutError:
            line = None
        if line != "ready" or proc.returncode is not None:
            log.warning("warm.spawn_no_ready", model=model, line=line)
            await self._drop_warm(warm)
            return None
        warm.state = "ready"
        log.info("warm.spawned", model=model, pid=proc.pid)
        return warm

    async def _warm_reader(self, warm: _WarmWorker) -> None:
        """Forward child stdout lines into warm.lines; EOF -> None sentinel."""
        proc = warm.proc
        try:
            while proc is not None and proc.stdout is not None:
                line = await proc.stdout.readline()
                if not line:
                    break
                warm.lines.put_nowait(line.decode(errors="replace").strip())
        except Exception:
            pass
        warm.lines.put_nowait(None)
        if warm.state != "dead":
            warm.state = "dead"

    async def _dispatch_warm(
        self, warm: _WarmWorker, job: MeshJob | ImageJob, work: Path
    ) -> tuple[int | None, str | None, float]:
        """Send one request dir to a warm worker; returns
        (rc, kill_reason, peak_rss_mb) with the cold path's semantics —
        'done' -> 0, 'fail' -> 1, 'wrong-model' -> 4, dead child -> its rc."""
        # pin the request to the worker's model — for 'auto' jobs that means
        # the resolved pick, not the literal request
        (work / "request.json").write_text(
            json.dumps(self._gen_request(job, model=warm.model))
        )
        proc = warm.proc
        assert proc is not None
        job.status = "running"
        job.started = time.monotonic()
        job.message = f"generating (warm {warm.model})"
        job.last_progress_at = time.monotonic()
        job.last_active_at = job.last_progress_at
        done: dict[str, str | None] = {"tok": None}

        async def _wait_done() -> None:
            while True:
                line = await warm.lines.get()
                if line is None:
                    done["tok"] = "eof"
                    return
                tok, _, path = line.partition(" ")
                if tok in ("done", "fail", "wrong-model") and Path(path.strip()) == work:
                    done["tok"] = tok
                    return
                # stray output (adapter print, protocol noise) — ignore it

        done_task = asyncio.create_task(_wait_done())
        warm.state = "busy"
        try:
            assert proc.stdin is not None
            proc.stdin.write((str(work) + "\n").encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            done["tok"] = "eof"
        kill_reason, peak_rss_mb = await self._watch_gen_proc(
            proc, job, work, is_done=lambda: done["tok"] is not None
        )
        warm.last_used = time.monotonic()
        if not done_task.done():
            # the watcher returned because the child died — the reader's EOF
            # sentinel lands on its own; don't leak the consumer task
            done_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await done_task
        if (
            kill_reason is not None
            or proc.returncode is not None
            or done["tok"] in ("eof", "wrong-model", None)
        ):
            # worker died mid-job, got watchdog-killed, or rejected the model —
            # drop it; the job result path treats this like a cold rc != 0
            await self._drop_warm(warm)
            if done["tok"] == "wrong-model":
                return 4, kill_reason, peak_rss_mb
            rc = proc.returncode if proc.returncode is not None else -1
            return rc, kill_reason, peak_rss_mb
        warm.state = "ready"
        self.registry.ledger.touch(f"warm:{warm.model}")
        return (0 if done["tok"] == "done" else 1), kill_reason, peak_rss_mb

    def _lru_idle_warm(self, exclude: set[str]) -> _WarmWorker | None:
        cands = [w for m, w in self._warm.items() if w.idle and m not in exclude]
        return min(cands, key=lambda w: w.last_used) if cands else None

    def _warm_gc(self) -> None:
        """Drop bookkeeping for dead warm workers (idle self-exit, crash).
        Busy workers are reaped by their dispatch path instead."""
        for model, warm in list(self._warm.items()):
            proc = warm.proc
            dead = warm.state == "dead" or (
                proc is not None and proc.returncode is not None
            )
            if not dead or warm.state == "busy":
                continue
            self._warm.pop(model, None)
            self.registry.ledger.release(f"warm:{model}")
            if warm.stderr_file is not None:
                with contextlib.suppress(Exception):
                    warm.stderr_file.close()

    def _evict_warm(self, warm: _WarmWorker) -> None:
        """Synchronous eviction for the admission path — kill the process
        group and drop the bookkeeping without awaiting (memory pressure
        wants the slot freed now; the reader task reaps the dead child)."""
        self._warm.pop(warm.model, None)
        self.registry.ledger.release(f"warm:{warm.model}")
        warm.state = "dead"
        proc = warm.proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            # reap the child so the transport finalizer doesn't run into a
            # closed loop — _pick_next runs on the event loop, but stay
            # safe for sync-only contexts (tests, teardown)
            try:
                asyncio.get_running_loop().create_task(self._reap_proc(proc))
            except RuntimeError:
                pass
        if warm.reader_task is not None:
            warm.reader_task.cancel()
        if warm.stderr_file is not None:
            with contextlib.suppress(Exception):
                warm.stderr_file.close()

    @staticmethod
    async def _reap_proc(proc: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)

    async def _drop_warm(self, warm: _WarmWorker) -> None:
        """Stop a warm worker: close stdin (the child exits on EOF), killpg
        as a fallback, release the ledger reservation."""
        self._warm.pop(warm.model, None)
        self.registry.ledger.release(f"warm:{warm.model}")
        warm.state = "dead"
        proc = warm.proc
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
                    await proc.stdin.wait_closed()
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
        if warm.reader_task is not None:
            warm.reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warm.reader_task
        if warm.stderr_file is not None:
            with contextlib.suppress(Exception):
                warm.stderr_file.close()

    def _record_peak(self, job: MeshJob | ImageJob, peak_rss_mb: float) -> None:
        """Feed the ledger's learned per-model footprint from the child's
        measured peak — resolved_model reflects what actually ran ('auto'
        placeholders excluded)."""
        adapter_id = job.resolved_model or job.model
        if adapter_id and adapter_id != "auto":
            self.registry.ledger.record_peak(adapter_id, peak_rss_mb)

    def _gen_request(self, job: MeshJob | ImageJob, model: str | None = None) -> dict[str, Any]:
        return {
            "kind": "image" if isinstance(job, ImageJob) else "mesh",
            "model": model or job.params.model,
            "params": job.params.model_dump(mode="json"),
            "image_paths": [str(p) for p in getattr(job, "image_paths", [])],
            "init_image_path": str(getattr(job, "init_image_path", None) or "") or None,
            # resolved config knobs the child can't see via env (test fakes,
            # programmatic overrides)
            "cfg": {k: getattr(self.cfg, k) for k in _CHILD_CFG_KEYS}
            | {"model_dir": str(self.cfg.model_dir)},
            # parent's badlist carries over — adapters that stalled/OOM'd in a
            # previous child must not be retried by this one
            "badlist": dict(getattr(self.registry, "_failed", {})),
        }

    async def _watch_gen_proc(
        self,
        proc: asyncio.subprocess.Process,
        job: MeshJob | ImageJob,
        work: Path,
        is_done: Callable[[], bool] | None = None,
    ) -> tuple[str | None, float]:
        """Poll progress.json + child RSS; returns (kill_reason, peak_rss_mb).

        Kills the process group on stall (no progress.json update within
        job_stall_s), on job.cancelled (after a short cancel.flag grace) and
        when the child's RSS crosses job_max_rss_mb. `is_done` lets warm
        workers end the watch on their stdout protocol (the child outlives
        the request) while cold children end it by exiting."""
        progress_file = work / "progress.json"
        last_raw = ""
        peak_rss_mb = 0.0
        kill_reason: str | None = None
        # a generation may legitimately stay silent between stage callbacks
        # (hunyuan only reports 5% then 90%) — the real stall signal is an
        # idle CPU: track the last time the child either progressed or
        # actually computed something
        last_active_at = time.monotonic()
        last_rss: float | None = None
        cancel_seen = False
        child_proc = None
        with contextlib.suppress(Exception):
            import psutil

            child_proc = psutil.Process(proc.pid)
            child_proc.cpu_percent()  # prime the sampler

        def _kill() -> None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

        while proc.returncode is None and not (is_done is not None and is_done()):
            await asyncio.sleep(0.5)
            try:
                if progress_file.exists():
                    raw = progress_file.read_text()
                    if raw != last_raw:
                        last_raw = raw
                        p = json.loads(raw)
                        job.progress = 0.05 + min(p.get("progress", 0.0), 1.0) * 0.9
                        job.message = p.get("message", job.message)
                        job.last_progress_at = time.monotonic()
            except (json.JSONDecodeError, OSError):
                pass
            rss = self._child_rss_mb(proc.pid)
            peak_rss_mb = max(peak_rss_mb, rss or 0.0)
            if (
                self.cfg.job_max_rss_mb > 0
                and rss is not None
                and rss > self.cfg.job_max_rss_mb
            ):
                kill_reason = f"job exceeded {self.cfg.job_max_rss_mb}MB RSS — killed"
                _kill()
            if child_proc is not None:
                # liveness = CPU activity OR allocation churn. A deadlocked
                # child sits at exactly 0% CPU with a frozen RSS; GPU-bound
                # MLX stages only burn ~1.5-3.5% CPU — 0.5% separates them.
                active = False
                with contextlib.suppress(Exception):
                    active = child_proc.cpu_percent() > 0.5
                if rss is not None and last_rss is not None and abs(rss - last_rss) > 32:
                    active = True
                if rss is not None:
                    last_rss = rss
                if active:
                    last_active_at = time.monotonic()
                    # a computing child is alive even between progress
                    # callbacks — feed both watchdogs
                    job.last_progress_at = last_active_at
                    job.last_active_at = last_active_at
            if job.cancelled and not cancel_seen:
                cancel_seen = True
                # the job dir may already be gone — a client DELETE cleans
                # files concurrently with this loop
                with contextlib.suppress(OSError):
                    (work / "cancel.flag").touch()
                # a client abort *on an idle child* is the stall signature —
                # record it so the adapter gets badlisted in _finish_proc_*;
                # threshold stays well below job_stall_s because the client's
                # own watchdog aborts first. Must run only on the *first*
                # cancelled tick — after the grace kill the child is dead and
                # "idle" would always exceed the threshold.
                if kill_reason is None and (
                    time.monotonic() - last_active_at > self.cfg.job_cancel_idle_s
                ):
                    kill_reason = (
                        f"stalled — idle >{self.cfg.job_cancel_idle_s}s when "
                        "client aborted (subprocess killed)"
                    )
                # brief grace for a clean abort, then the hard kill
                await asyncio.sleep(2)
                _kill()
            elif job.cancelled:
                _kill()
            if time.monotonic() - last_active_at > self.cfg.job_stall_s:
                kill_reason = (
                    f"stalled — idle for {self.cfg.job_stall_s}s (subprocess killed)"
                )
                _kill()
        return kill_reason, peak_rss_mb

    def _finish_proc_ok(self, job: MeshJob | ImageJob, work: Path, peak_rss_mb: float) -> None:
        try:
            meta = json.loads((work / "result.json").read_text())
        except (json.JSONDecodeError, OSError) as e:
            job.status = "failed"
            job.error = f"missing result.json ({e})"
            job.finished = time.monotonic()
            return
        if isinstance(job, MeshJob):
            glb_path = work / "out.glb"
            if not glb_path.exists():
                job.status = "failed"
                job.error = "subprocess exited 0 without out.glb"
                job.finished = time.monotonic()
                return
            job.glb = glb_path.read_bytes()
            job.triangles = meta.get("triangles")
            job.resolution = meta.get("resolution") or job.resolution
            outs = meta.get("outputs") or []
            if len(outs) > 1 or job.params.batch:
                job.outputs = outs
                for o in outs:
                    if not o.get("ok"):
                        continue
                    p = work / f"out_{o['index']}.glb"
                    if p.exists():
                        job.glbs[o["index"]] = p.read_bytes()
        else:
            job.images = [p.read_bytes() for p in sorted(work.glob("img_*.png"))]
        job.model = meta.get("model") or job.model
        # what the child actually ran — 'auto' jobs resolve late
        job.resolved_model = meta.get("model") or job.resolved_model
        job.backend = meta.get("backend") or ""
        job.seed = meta.get("seed") or 0
        job.steps = meta.get("steps")
        job.guidance = meta.get("guidance")
        job.status = "succeeded"
        job.progress = 1.0
        job.message = "done"
        job.finished = time.monotonic()
        log.info(
            "job.proc_done",
            job=job.id,
            model=job.model,
            child_peak_rss_mb=round(peak_rss_mb, 1),
        )

    @staticmethod
    def _proc_retryable(work: Path) -> bool:
        """A child error.json naming a specific adapter is deterministic —
        that adapter is now badlisted, so a retry lands on the next ladder
        entry. Generic crashes (no model, e.g. preprocessing) would re-fail;
        stall/RSS kills (kill_reason, no child error) stay non-retryable."""
        try:
            err = json.loads((work / "error.json").read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return bool(err.get("error") and err.get("model"))

    def _finish_proc_failed(
        self,
        job: MeshJob | ImageJob,
        work: Path,
        rc: int | None,
        kill_reason: str | None,
        peak_rss_mb: float,
    ) -> None:
        stderr_tail = ""
        with contextlib.suppress(OSError):
            stderr_tail = (work / "stderr.log").read_text(errors="replace")[-600:]
        child_error = ""
        child_model = ""
        with contextlib.suppress(json.JSONDecodeError, OSError):
            err = json.loads((work / "error.json").read_text())
            child_error = err.get("error", "")
            child_model = err.get("model", "")
        if child_model:
            # the child's real pick (auto ladder) — feeds affinity + the
            # learned footprint even on failure
            job.resolved_model = child_model
        job.status = "failed"
        job.error = (
            kill_reason or child_error or f"gen worker exited rc={rc}: {stderr_tail.strip()[:400]}"
        )
        job.finished = time.monotonic()
        # a child-reported adapter error (import crash, missing weights) is
        # deterministic — every retry fails the same way, so badlist it.
        # A kill_reason (stall/RSS/cancel) in subprocess mode leaves NO shared
        # state: the dead child owned the model, so a transient MPS hang must
        # not poison the pinned model for the rest of this server process
        # (observed: one stalled candidate badlisted mv-turbo → the winner
        # regen refused instantly → hybrid fell back to CAD). In in-process
        # mode a stalled adapter *thread* stays frozen — badlist still applies.
        bad_model = child_model or (job.model if job.model != "auto" else "")
        if bad_model and (child_error or (kill_reason and not self.cfg.jobs_subprocess)):
            self.registry.mark_failed(bad_model, job.error)
        log.warning(
            "job.proc_failed",
            job=job.id,
            rc=rc,
            model=job.model,
            reason=kill_reason,
            child_peak_rss_mb=round(peak_rss_mb, 1),
        )

    @staticmethod
    def _child_rss_mb(pid: int) -> float | None:
        try:
            import psutil

            return psutil.Process(pid).memory_info().rss / 1e6
        except Exception:
            return None

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
        # batch variants share the loaded adapter — same model, different
        # seed/params, one generate() call per variant (subprocess mode does
        # the same inside gen_worker so the model loads once per job)
        variants = job.params.batch or []
        todo = variants if variants else [job.params]
        n_v = len(todo)
        for i, vp in enumerate(todo):
            if job.cancelled:
                break

            def vprog(frac: float, msg: str, _i: int = i) -> None:
                progress((_i + min(frac, 1.0)) / n_v, f"[{_i + 1}/{n_v}] {msg}")

            def getv(k: str, _vp: Any = vp) -> Any:
                return _vp.get(k) if isinstance(_vp, dict) else getattr(_vp, k, None)

            try:
                result = adapter.generate(
                    images,
                    GenOpts(
                        seed=getv("seed") or 0,
                        steps=getv("steps") or job.params.steps,
                        guidance=(
                            getv("guidance")
                            if getv("guidance") is not None
                            else job.params.guidance
                        ),
                        resolution=getv("resolution") or job.params.resolution,
                        subject=getv("subject") or job.params.subject,
                        image_roles=job.params.image_roles,
                        # job-level conditioning (bbox/voxel/point/pose) —
                        # GenVariant has no per-seed override
                        control_type=job.params.control_type,
                        control_input=job.params.control_input,
                        work_dir=self.cfg.job_dir / job.id,
                    ),
                    progress=vprog,
                    cancelled=cancelled,
                )
            except Exception as e:
                if not variants:
                    raise
                job.outputs.append({"index": i, "ok": False, "error": str(e)[:300]})
                continue
            mesh = clean_mesh(result.mesh)
            job.glbs[i] = export_glb(mesh)
            job.outputs.append(
                {
                    "index": i,
                    "ok": True,
                    "seed": result.seed,
                    "steps": result.steps,
                    "resolution": result.resolution,
                    "triangles": len(mesh.faces),
                }
            )
            if i == 0 or job.glb is None:
                job.glb = job.glbs[i]
                job.triangles = len(mesh.faces)
                job.seed = result.seed
                job.steps = result.steps
                job.guidance = result.guidance
                job.resolution = result.resolution

        if job.cancelled or not job.glbs:
            job.status = "cancelled" if job.cancelled else job.status
            if not job.glbs and not job.cancelled:
                job.status = "failed"
                first_err = next(
                    (o["error"] for o in job.outputs if o.get("error")), "no output"
                )
                job.error = f"all {n_v} batch variants failed ({first_err})"
            job.finished = time.monotonic()
            return

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
