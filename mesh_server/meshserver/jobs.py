"""In-process job queue: one GPU job at a time, coarse cancellation, TTL cleanup.

Two job kinds share the single worker thread — image-to-3D (MeshJob) and
text-to-image (ImageJob) — so a mesh model and an image model can never be
resident at the same time and single-flight GPU usage holds automatically.
"""

from __future__ import annotations

import asyncio
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
from meshserver.schemas import GenParams, ImageGenParams, JobStatus

log = get_logger(__name__)


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
        self.model = params.model
        self.backend = ""
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
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

    def _host_free_mb(self) -> float | None:
        try:
            import psutil

            return psutil.virtual_memory().available / 1e6
        except Exception:
            return None

    def _free_mem_ok(self) -> bool:
        """Refuse new jobs under host memory pressure — a job started with no
        headroom stalls at 0% CPU and can swap the host into a panic."""
        limit = self.cfg.job_min_free_mb
        if limit <= 0:
            return True
        free = self._host_free_mb()
        return free is None or free >= limit

    async def submit(self, params: GenParams, image_paths: list[Path]) -> MeshJob | None:
        self._gc()
        if self._queued() >= self.cfg.max_queue or not self._free_mem_ok():
            return None
        job = MeshJob(str(uuid.uuid4()), params, image_paths)
        self.jobs[job.id] = job
        await self._queue.put(job.id)
        return job

    async def submit_image(
        self, params: ImageGenParams, init_image_path: Path | None = None
    ) -> ImageJob | None:
        self._gc()
        if self._queued() >= self.cfg.max_queue or not self._free_mem_ok():
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
            # a stalled adapter thread cannot be killed — badlist the model so
            # 'auto' and candidate screening stop picking it, and release
            # whatever model memory the registry still owns
            if job.model and job.model != "auto":
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

    # -- worker ------------------------------------------------------------ #

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
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
        finally:
            # whatever happens above — exception in the watcher, task cancel —
            # a live child must never leak past this point
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
            stderr_file.close()

        if proc.returncode == 0:
            self._finish_proc_ok(job, work, peak_rss_mb)
            return
        if (job.cancelled or proc.returncode == 2) and kill_reason is None:
            job.status = "cancelled"
            job.finished = time.monotonic()
            return
        self._finish_proc_failed(job, work, proc.returncode, kill_reason, peak_rss_mb)

    def _gen_request(self, job: MeshJob | ImageJob) -> dict[str, Any]:
        return {
            "kind": "image" if isinstance(job, ImageJob) else "mesh",
            "model": job.params.model,
            "params": job.params.model_dump(mode="json"),
            "image_paths": [str(p) for p in getattr(job, "image_paths", [])],
            "init_image_path": str(getattr(job, "init_image_path", None) or "") or None,
            # resolved config knobs the child can't see via env (test fakes,
            # programmatic overrides)
            "cfg": {
                k: getattr(self.cfg, k)
                for k in (
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
                )
            }
            | {"model_dir": str(self.cfg.model_dir)},
            # parent's badlist carries over — adapters that stalled/OOM'd in a
            # previous child must not be retried by this one
            "badlist": dict(getattr(self.registry, "_failed", {})),
        }

    async def _watch_gen_proc(
        self, proc: asyncio.subprocess.Process, job: MeshJob | ImageJob, work: Path
    ) -> tuple[str | None, float]:
        """Poll progress.json + child RSS; returns (kill_reason, peak_rss_mb).

        Kills the process group on stall (no progress.json update within
        job_stall_s), on job.cancelled (after a short cancel.flag grace) and
        when the child's RSS crosses job_max_rss_mb."""
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

        while proc.returncode is None:
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
        else:
            job.images = [p.read_bytes() for p in sorted(work.glob("img_*.png"))]
        job.model = meta.get("model") or job.model
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
        job.status = "failed"
        job.error = (
            kill_reason or child_error or f"gen worker exited rc={rc}: {stderr_tail.strip()[:400]}"
        )
        job.finished = time.monotonic()
        # stall/RSS kill or a child-reported adapter error both mean this
        # model is unsafe to retry in this process — badlist it
        bad_model = child_model or (job.model if job.model != "auto" else "")
        if bad_model and (kill_reason or child_error):
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
