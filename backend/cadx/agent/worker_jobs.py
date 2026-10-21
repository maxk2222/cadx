"""Worker-side job runners: wire AgentCtx -> orchestrator -> DB.

Runs in the `worker` service (queue "agent"). CAD work is delegated to the
`cad` queue via QueueExecutor.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func
from sqlmodel import col, select

from cadx import events, storage
from cadx.agent.executor_client import QueueExecutor, exec_error
from cadx.agent.image_client import ImagePort, make_image_client
from cadx.agent.mesh_client import MeshGenRequest, MeshPort, make_mesh_client
from cadx.agent.orchestrator import RunOutcome, run_generation
from cadx.agent.trace import MemoryTrace, sanitize_messages
from cadx.agent.types import AgentCtx, InputAsset, ParentContext
from cadx.cad.allowlist import check_code
from cadx.cad.mesh_ops import (
    MESH_OPS_VERSION,
    MeshOp,
    MeshPostParams,
    merge_post_params,
    mesh_params_schema,
)
from cadx.config import get_settings
from cadx.db.models import (
    IMAGE_ASSET_KINDS,
    AgentTrace,
    Artifact,
    ArtifactKind,
    Asset,
    AssetKind,
    Job,
    JobStatus,
    JobType,
    LibraryPart,
    Project,
    ProjectSettings,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import session_scope
from cadx.llm.capabilities import load_caps_session, record_caps_session
from cadx.llm.client import LLMClient, make_llm
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings, load_runtime_settings

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


def _json_safe(obj: Any) -> Any:
    """Replace NaN/±Inf recursively (Postgres JSON rejects non-finite floats)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


async def _publish(job_id: str, event: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        await events.publish(job_id, event)


async def _append_job_log(db: AsyncSession, job: Job, text: str) -> None:
    """Append to job.log without clobbering emit() lines written by other
    sessions — the caller's `job` object may be stale since load time."""
    cur = (await db.execute(select(Job.log).where(Job.id == job.id))).scalar_one_or_none()
    job.log = (cur or "") + text


def make_emit(
    job_id: uuid.UUID, tokens_fn: Callable[[], dict[str, Any]] | None = None
) -> Callable[..., Awaitable[None]]:
    """Job emit helper: writes step/iteration/progress/log + publishes the event.

    Deduplicates consecutive identical (step, iteration, message) emissions —
    mesh-server polls otherwise flood the log with e.g. 'preprocessing images
    (0%)' dozens of times.

    Optional structured fields on the call (`score=`, `max_iterations=`,
    `sub_progress=`, `sub_label=`, `metrics=`, `run=`) are merged into the
    published SSE event and rolled into the persisted Job.live snapshot
    (iteration budget, score history, per-step timings, token usage, named
    scalar metrics, completed-run rows) so reconnects restore the full UI.
    """
    last: tuple[str, int, str] | None = None
    live: dict[str, Any] = {"scores": [], "step_history": []}
    cur_step: str | None = None
    step_t0 = time.monotonic()

    async def emit(step: str, iteration: int, message: str, **fields: Any) -> None:
        nonlocal last, cur_step, step_t0
        sig = (step, iteration, message)
        if sig == last:
            return
        last = sig
        now = time.monotonic()
        if step != cur_step:
            if cur_step is not None:
                hist = live.setdefault("step_history", [])
                hist.append({"step": cur_step, "it": iteration, "dt_s": round(now - step_t0, 1)})
                del hist[:-40]
            cur_step = step
            step_t0 = now
            live["step"] = step
            live["step_started_ts"] = datetime.now(UTC).isoformat()
            live.pop("sub_progress", None)
            live.pop("sub_label", None)
        for k in ("max_iterations", "sub_progress", "sub_label"):
            if fields.get(k) is not None:
                live[k] = fields[k]
        if (score := fields.get("score")) is not None:
            scores = live.setdefault("scores", [])
            if not any(s["it"] == iteration for s in scores):
                scores.append({"it": iteration, "score": score})
        if (metrics := fields.get("metrics")) is not None:
            # flat named scalars (worst_sf, probe_sf, …) — latest value wins;
            # None entries are dropped so a failed probe never erases a value
            merged = live.setdefault("metrics", {})
            merged.update({k: v for k, v in metrics.items() if v is not None})
        if (run := fields.get("run")) is not None:
            # one row per completed topopt/refine run — same cap as step_history
            runs = live.setdefault("runs", [])
            runs.append(run)
            del runs[:-40]
        if tokens_fn is not None:
            with contextlib.suppress(Exception):
                live["tokens"] = tokens_fn()
        max_it = live.get("max_iterations")
        if isinstance(max_it, int) and max_it > 0 and iteration > 0:
            sub = float(live.get("sub_progress") or 0.0)
            progress = min(0.95, max(0.0, (iteration - 1 + sub) / max_it))
        else:
            progress = None  # no budget known yet — keep the +0.05 bump
        job_progress: float | None = None
        async with session_scope() as s2:
            j = await s2.get(Job, job_id)
            if j is not None:
                j.step = step
                j.iteration = iteration
                # monotonic: iteration-derived progress must never pull the
                # bar backwards after setup-step bumps (it=0 events)
                j.progress = (
                    max(j.progress, progress)
                    if progress is not None
                    else min(0.95, j.progress + 0.05)
                )
                j.live = dict(live)  # reassign — JSON columns don't track in-place edits
                j.log = (j.log or "") + f"[{step}:{iteration}] {message}\n"
                job_progress = j.progress
                await s2.commit()
        await _publish(
            str(job_id),
            {
                "type": "step",
                "step": step,
                "iteration": iteration,
                "message": message,
                "ts": datetime.now(UTC).isoformat(),
                "progress": job_progress,
                "live": dict(live),
                **fields,
            },
        )

    return emit


async def _fail_job_and_version(payload: dict[str, Any], error: str) -> None:
    """Crash guard: mark job+version failed so a crashed task never leaves
    them stuck in 'running'."""
    try:
        async with session_scope() as db:
            job = await db.get(Job, _u(payload["job_id"]))
            version = await db.get(Version, _u(payload["version_id"]))
            # a user-requested cancel is not a failure — keep the distinction
            # even when the task died via timeout/shutdown mid-cancel
            user_cancelled = bool(job.cancel_requested) if job is not None else False
            if job is not None and job.status in (JobStatus.queued, JobStatus.running):
                job.status = JobStatus.cancelled if user_cancelled else JobStatus.failed
                job.finished_at = datetime.now(UTC)
                await _append_job_log(
                    db, job, f"{'cancelled' if user_cancelled else 'failed'}: {error}\n"
                )
            if version is not None and version.status in (
                VersionStatus.queued,
                VersionStatus.running,
            ):
                version.status = VersionStatus.cancelled if user_cancelled else VersionStatus.failed
                version.error = error[:500]
            await db.commit()
        await _publish(payload["job_id"], {"type": "done", "status": "failed"})
    except Exception:
        log.exception("job.fail_guard_failed", job=payload.get("job_id"))


_MIN_FREE_GB = 2.0  # refuse to start a job below this — full disks corrupt mid-run


def _disk_free_gb() -> float:
    try:
        return shutil.disk_usage(storage.data_root()).free / 1e9
    except Exception:
        return float("inf")


def _guarded(
    fn: Callable[..., Awaitable[dict[str, Any]]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Top-level crash guard for arq job runners."""

    @functools.wraps(fn)
    async def wrapper(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        free = _disk_free_gb()
        if free < _MIN_FREE_GB:
            err = f"refusing to start: only {free:.1f} GB free on the data disk"
            log.warning("job.disk_guard", job=payload.get("job_id"), free_gb=free)
            await _fail_job_and_version(payload, err)
            return {"ok": False, "error": err}
        try:
            return await fn(ctx, payload)
        except asyncio.CancelledError:
            # arq job_timeout / shutdown cancels the task; CancelledError is
            # BaseException so `except Exception` misses it and the job row
            # would stay 'running' forever
            log.warning("job.task_cancelled", job=payload.get("job_id"))
            await _fail_job_and_version(payload, "task cancelled (worker timeout or shutdown)")
            raise
        except Exception as e:
            log.exception("job.unhandled_exception", job=payload.get("job_id"))
            await _fail_job_and_version(payload, f"{type(e).__name__}: {e}")
            return {"ok": False, "error": str(e)[:300]}

    return wrapper


class DbTrace(MemoryTrace):
    """Memory trace that also persists each record as an AgentTrace row.

    Uses its own session per write: a failed trace insert (e.g. FK violation
    after the job row was deleted) must never poison the job's main session —
    that previously cascaded into bogus codegen/critique step failures.
    """

    def __init__(self, job_id: uuid.UUID) -> None:
        super().__init__()
        self._job_id = job_id
        self._dead = False

    async def record(self, **kwargs: Any) -> None:
        await super().record(**kwargs)
        if self._dead:
            # job row was deleted mid-run — the task is orphaned and will stop
            # at the next cancellation checkpoint; keep recording to memory but
            # stop hammering the DB with FK violations
            return
        try:
            async with session_scope() as s:
                s.add(
                    AgentTrace(
                        job_id=self._job_id,
                        step=kwargs["step"],
                        iteration=kwargs["iteration"],
                        model=kwargs["model"],
                        messages=sanitize_messages(kwargs["messages"]),
                        response=kwargs["response_text"][:20_000],
                        parsed=_json_safe(kwargs["parsed"]),
                        prompt_tokens=kwargs["prompt_tokens"],
                        completion_tokens=kwargs["completion_tokens"],
                        latency_ms=kwargs["latency_ms"],
                    )
                )
                await s.commit()
        except Exception as e:
            if "agenttrace" in str(e) and (
                "ForeignKeyViolation" in str(e) or "violates foreign key" in str(e)
            ):
                self._dead = True
                log.info("agent.trace_orphaned", job=str(self._job_id))
            else:
                log.warning("agent.trace_persist_failed", job=str(self._job_id), error=str(e)[:200])


def register_artifacts(
    session: AsyncSession,
    version_id: uuid.UUID,
    artifacts: dict[str, str],
    *,
    tessellated_step: bool = False,
) -> None:
    """artifacts: {"render:front": relpath, "export:step": relpath, "contact_sheet": rel, ...}"""
    for key, rel in artifacts.items():
        p = storage.resolve(rel)
        size = p.stat().st_size if p.exists() else 0
        if key.startswith("render:"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.render,
                    view=key.split(":", 1)[1],
                    path=rel,
                    size=size,
                )
            )
        elif key.startswith("export:"):
            fmt = key.split(":", 1)[1]
            meta = {"tessellated": True} if fmt == "step" and tessellated_step else {}
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind(fmt),
                    path=rel,
                    size=size,
                    meta=meta,
                )
            )
        elif key == "mesh_raw":
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.mesh_raw,
                    path=rel,
                    size=size,
                )
            )
        elif key.startswith("fem:"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind.fem,
                    view=key.split(":", 1)[1],
                    path=rel,
                    size=size,
                )
            )
        elif key in ("contact_sheet", "thumbnail"):
            session.add(
                Artifact(
                    version_id=version_id,
                    kind=ArtifactKind(key),
                    path=rel,
                    size=size,
                )
            )


def _input_assets(assets: list[Asset]) -> list[InputAsset]:
    return [
        InputAsset(
            asset_id=str(a.id),
            kind=a.kind.value,
            path=storage.resolve(a.path),
            caption=a.caption,
            filename=a.filename,
            role_hint=str((a.gen_params or {}).get("role_hint") or ""),
        )
        for a in assets
    ]


async def _load_library(
    db: AsyncSession, version: Version
) -> tuple[list[InputAsset], list[dict[str, Any]]]:
    """Resolve version.library_part_ids into extra reference assets + templates."""
    extra_assets: list[InputAsset] = []
    templates: list[dict[str, Any]] = []
    if not version.library_part_ids:
        return extra_assets, templates
    rows = await db.execute(
        select(LibraryPart).where(
            col(LibraryPart.id).in_([_u(p) for p in version.library_part_ids])
        )
    )
    for part in rows.scalars().all():
        if part.kind == "template":
            templates.append(
                {
                    "name": part.name,
                    "caption": part.caption,
                    "code": part.code or "",
                    "params": part.params or {},
                }
            )
        elif part.path:
            extra_assets.append(
                InputAsset(
                    asset_id=str(part.id),
                    kind="reference_step" if part.kind == "step" else "reference_stl",
                    path=storage.resolve(part.path),
                    caption=f"{part.name} — {part.caption}".rstrip(" —"),
                    filename=part.filename,
                )
            )
    return extra_assets, templates


def _parent_ctx(version: Version, parent: Version | None) -> ParentContext:
    if parent is None:
        return ParentContext()
    v_dir = storage.version_dir(parent.project_id, parent.id)
    sheet = v_dir / "contact_sheet.png"
    solid = v_dir / "mesh" / "solid.brep"
    return ParentContext(
        code=parent.code or "",
        spec=PartSpec(**parent.spec) if parent.spec else None,
        contact_sheet=sheet if sheet.exists() else None,
        number=parent.number,
        resolved_mode=parent.resolved_mode,
        mesh=parent.mesh,
        mesh_solid=solid if solid.exists() else None,
        analysis=parent.analysis,
    )


# --------------------------------------------------------------------------- #
# job runners                                                                  #
# --------------------------------------------------------------------------- #


@_guarded
async def run_agent_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq function on queue 'agent' for types generate|edit|resume."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def check_cancel() -> bool:
            async with session_scope() as s3:
                j = await s3.get(Job, job_id)
                # a deleted job row means the version was deleted mid-run; a
                # terminal status means the reaper or another guard already
                # finalised it — either way the task is orphaned and must
                # stop, not run to completion
                return (
                    j is None
                    or bool(j.cancel_requested)
                    or j.status in (JobStatus.failed, JobStatus.cancelled, JobStatus.succeeded)
                )

        async def check_finish() -> bool:
            async with session_scope() as s3:
                j = await s3.get(Job, job_id)
                return (
                    j is None
                    or bool(j.finish_requested)
                    or j.status in (JobStatus.failed, JobStatus.cancelled, JobStatus.succeeded)
                )

        runtime_settings = await load_runtime_settings(db)
        project = await db.get(Project, version.project_id)
        merged = _merge_settings((project.settings or {}) if project else {}, version.options or {})
        # per-job backend/model overrides (RunOptions < project settings)
        job_runtime = _job_runtime(runtime_settings, merged)
        llm = _make_llm(
            job_runtime,
            backend=merged.llm_backend,
            caps=await load_caps_session(),
            caps_cb=record_caps_session,
        )
        emit = make_emit(job_id, tokens_fn=lambda: dict(getattr(llm, "total_usage", {}) or {}))
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=job_runtime)
        trace = DbTrace(job_id)
        mesh = _make_mesh(job_runtime, backend=merged.mesh_backend)
        image = _make_image(job_runtime, backend=merged.image_backend)

        explicit_max_iterations = merged.max_iterations is not None
        settings = _fill_settings(merged, job_runtime)

        # assets
        asset_ids = version.asset_ids or []
        assets: list[Asset] = []
        if asset_ids:
            rows = await db.execute(
                select(Asset).where(col(Asset.id).in_([_u(a) for a in asset_ids]))
            )
            assets = list(rows.scalars().all())
        else:
            rows = await db.execute(select(Asset).where(Asset.project_id == version.project_id))
            assets = list(rows.scalars().all())

        # reusable library parts selected for this version: file parts become
        # reference assets (loadable via load_reference(part_id)), code
        # templates are surfaced in the codegen context only
        input_assets = _input_assets(assets)
        extra_assets, library_templates = await _load_library(db, version)
        input_assets += extra_assets

        parent = await db.get(Version, version.parent_id) if version.parent_id else None
        parent_ctx = _parent_ctx(version, parent)

        mode = "edit" if job.type == JobType.edit else "generate"
        gen_mode = version.gen_mode or settings.gen_mode or runtime_settings.agent_default_gen_mode
        # hybrid edit keeps the parent's solid unless the user asked for regeneration
        regenerate_mesh = bool(job.resume_state and job.resume_state.get("regenerate_mesh"))
        if mode == "edit" and gen_mode in ("auto", "mesh", "hybrid"):
            effective = gen_mode if gen_mode != "auto" else (parent_ctx.resolved_mode or "cad")
            if effective == "hybrid" and parent_ctx.resolved_mode == "hybrid":
                gen_mode = "hybrid"
                if regenerate_mesh:
                    parent_ctx.mesh_solid = None
            elif effective == "mesh" and parent_ctx.resolved_mode == "mesh":
                gen_mode = "mesh"

        mesh_client: MeshPort | None = mesh
        mesh_available = await _mesh_available(mesh_client)
        image_client: ImagePort | None = image
        image_available = await _image_available(image_client)

        actx = AgentCtx(
            llm=llm,
            executor=executor,
            trace=trace,
            emit=emit,
            check_cancel=check_cancel,
            check_finish=check_finish,
            job_id=str(job_id),
            project_id=version.project_id,
            version_id=version.id,
            mode=mode,
            prompt=version.prompt,
            settings=settings,
            runtime_settings=job_runtime,
            assets=input_assets,
            library_templates=library_templates,
            parent=parent_ctx,
            gen_mode=gen_mode,
            mesh=mesh_client,
            mesh_available=mesh_available,
            image=image_client,
            image_available=image_available,
            budget_tokens=runtime_settings.agent_max_tokens_per_job,
            explicit_max_iterations=explicit_max_iterations,
        )
        if job.type == JobType.resume and job.resume_state and job.resume_state.get("spec"):
            actx.spec = PartSpec(**job.resume_state["spec"])

        try:
            outcome = await run_generation(actx)
        except Exception as e:
            log.exception("agent.job_crashed", job=str(job_id))
            outcome = RunOutcome(
                status="failed", error=f"agent crashed: {type(e).__name__}: {e!r}"[:500]
            )

        if actx.library_attached:
            # record library parts the agent selected itself via tools
            version.library_part_ids = sorted(
                set(version.library_part_ids) | set(actx.library_attached)
            )
        if actx.generated_asset_ids:
            # hero/concept images the agent generated itself become permanent
            # inputs of this version (provenance + mesh source on edits)
            version.asset_ids = sorted(set(version.asset_ids or []) | set(actx.generated_asset_ids))

        await _persist_outcome(db, job, version, outcome, emit)
        await _spawn_refine_after_topopt(
            db, job, version, outcome, settings, runtime_settings, emit
        )
        return {"ok": outcome.status in ("succeeded", "needs_review"), "status": outcome.status}


@_guarded
async def run_params_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-execute parent with new params (no LLM). Child version already exists.

    CAD/hybrid parents: code re-run with PARAMS overrides.
    Mesh parents: mesh pipeline re-run with merged post-parameters.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        parent_mode = (parent.resolved_mode if parent else None) or "cad"
        # the version's own code wins over the inherited parent code — MCP
        # `code` children and feature edits carry their own source
        src_code = ((version.code if version else None) or (parent.code if parent else None)) or ""
        if (
            job is None
            or version is None
            or parent is None
            or (parent_mode != "mesh" and not src_code)
        ):
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "params job: missing parent code/mesh\n")
                await db.commit()
            return {"ok": False, "error": "missing parent code/mesh"}

        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings(
                (project.settings or {}) if project else {},
                version.options or (parent.options if parent else {}) or {},
            ),
            runtime_settings,
        )

        # design-robustness pre-flight (M7): perturb each numeric PARAMS value
        # of the executed code ±ε and re-run — a quality signal only, it never
        # blocks the sweep. Mesh parents carry no PARAMS dict to audit.
        audit_report: dict[str, Any] | None = None
        parent_v_dir = storage.version_dir(parent.project_id, parent.id)
        solid = parent_v_dir / "mesh" / "solid.brep"
        if parent_mode != "mesh" and src_code:
            # baseline = declared PARAMS overridden by the parent's stored
            # params and this job's sweep values — the audit probes around
            # the configuration actually being produced
            declared = check_code(src_code).params
            base_params = {
                **declared,
                **(parent.params or {}),
                **(payload.get("params") or {}),
            }
            audit_report = await _design_audit(
                db,
                job,
                version,
                executor,
                emit,
                code=src_code,
                base_params=base_params,
                settings=settings,
                runtime_settings=runtime_settings,
                mesh_solid=solid if solid.exists() and parent_mode == "hybrid" else None,
                mesh_bbox=(parent.mesh or {}).get("bbox_size"),
            )

        if parent_mode == "mesh":
            exec_result = await _run_mesh_params(
                executor,
                job_id,
                version,
                parent,
                settings,
                runtime_settings,
                payload.get("params") or {},
                emit,
            )
        else:
            await emit("execute", 1, "re-executing with new parameters")
            exec_result = await executor.execute(
                {
                    "job_id": str(job_id),
                    "project_id": str(version.project_id),
                    "version_id": str(version.id),
                    "iteration": 1,
                    "code": src_code,
                    "params_override": payload.get("params") or {},
                    "references": [],
                    "settings": settings.model_dump(),
                    "runtime_settings": runtime_settings.executor_values(),
                    "full_exports": True,
                    "name": "model",
                    "mesh_solid": str(solid)
                    if solid.exists() and parent_mode == "hybrid"
                    else None,
                    "mesh_bbox": (parent.mesh or {}).get("bbox_size"),
                }
            )

        if not exec_result.get("ok"):
            was_cancelled = (exec_result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = exec_error(exec_result, "execution failed")
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error, "design_audit": audit_report}

        version.code = src_code or None
        await _finalize_exec_version(db, job, version, exec_result, mode=parent_mode, parent=parent)
        return {"ok": True, "design_audit": audit_report}


async def _design_audit(
    db: AsyncSession,
    job: Job,
    version: Version,
    executor: QueueExecutor,
    emit: Any,
    *,
    code: str,
    base_params: dict[str, Any],
    settings: ProjectSettings,
    runtime_settings: RuntimeSettings,
    references: list[dict[str, Any]] | None = None,
    mesh_solid: Path | None = None,
    mesh_bbox: Any = None,
) -> dict[str, Any] | None:
    """Design-robustness pre-flight (M7): perturb each numeric PARAMS value
    ±ε and re-run — a quality signal only, never blocks the job. Returns the
    audit report, or None when no numeric params exist."""
    if not any(
        isinstance(v, int | float) and not isinstance(v, bool) for v in base_params.values()
    ):
        return None
    await emit("audit", 1, "design audit: perturbing parameters")
    report = await executor.audit_params(
        {
            "job_id": str(job.id),
            "project_id": str(version.project_id),
            "version_id": str(version.id),
            "iteration": 1,
            "code": code,
            "params": base_params,
            "references": references or [],
            "settings": settings.model_dump(),
            "runtime_settings": runtime_settings.executor_values(),
            "mesh_solid": str(mesh_solid) if mesh_solid else None,
            "mesh_bbox": mesh_bbox,
        }
    )
    a_status = str(report.get("status") or "fails")
    flagged = [
        n
        for n, e in (report.get("params") or {}).items()
        if isinstance(e, dict) and e.get("verdict") in ("brittle", "fails")
    ]
    what = (
        f"param audit {a_status} — brittle/failing: {', '.join(flagged)}"
        if flagged
        else f"param audit {a_status} ({report.get('audited', 0)} audited)"
    )
    version.issues = (version.issues or []) + [
        {
            "source": "design_audit",
            "severity": "minor" if a_status == "robust" else "major",
            "what": what,
            "report": _json_safe(report),
        }
    ]
    await _append_job_log(db, job, f"design audit: {what}\n")
    # flush the issue+log now — emit writes the job row on its own session and
    # an open transaction here can lock it out (sqlite tests), plus the audit
    # finding should survive a later crash
    await db.commit()
    if a_status != "robust":
        await emit("audit", 1, f"design audit: {what}")
    return report


async def _finalize_exec_version(
    db: AsyncSession,
    job: Job,
    version: Version,
    exec_result: dict[str, Any],
    *,
    mode: str,
    parent: Version | None = None,
    mesh_extra: dict[str, Any] | None = None,
) -> None:
    """Shared success tail for LLM-free execute/process_mesh jobs (params,
    code, meshgen): artifacts -> version fields -> model.py/params.json ->
    job done. Caller must set ``version.code`` beforehand when the executed
    source should be persisted (params children already did)."""
    import json

    artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, exec_result)
    for fmt, rel in (exec_result.get("exports") or {}).items():
        if rel:
            artifacts[f"export:{fmt}"] = rel
    tessellated = mode == "mesh"

    version.gen_mode = mode
    version.resolved_mode = mode
    if mode == "mesh":
        # merged post params live under mesh.post in the exec result
        version.params = (exec_result.get("mesh") or {}).get("post") or version.params
    else:
        version.params = exec_result.get("params_used") or version.params
    if parent is not None:
        version.params_schema = parent.params_schema
    version.analysis = exec_result.get("analysis")
    version.dfm = exec_result.get("dfm")
    version.iterations = 1
    version.status = VersionStatus.succeeded

    if mode == "mesh":
        # mesh child: inherit provenance + copy intermediates (raw GLB,
        # cleaned STL, solid BREP) into versions/<vid>/mesh/
        version.mesh = dict(parent.mesh or {}) if parent else {}
        version.mesh.update(mesh_extra or {})
        version.mesh.update(exec_result.get("mesh") or {})
        version.mesh["post"] = version.params
        if parent is not None:
            version.mesh["from_params_of"] = str(parent.id)
        artifacts.update(_copy_mesh_artifacts_ctxless(version.project_id, version.id, 1, parent))
        m_dir = storage.ensure(storage.version_dir(version.project_id, version.id) / "mesh")
        (m_dir / "mesh.json").write_text(json.dumps(version.mesh, indent=2), encoding="utf-8")
    register_artifacts(db, version.id, artifacts, tessellated_step=tessellated)

    v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
    if version.code:
        (v_dir / "model.py").write_text(version.code, encoding="utf-8")
    (v_dir / "params.json").write_text(json.dumps(version.params or {}, indent=2), encoding="utf-8")
    job.status = JobStatus.succeeded
    job.finished_at = datetime.now(UTC)
    job.progress = 1.0
    await _append_job_log(db, job, "done\n")
    await db.commit()
    await _publish(str(job.id), {"type": "done", "status": "succeeded"})


@_guarded
async def run_code_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Execute externally-supplied build123d code (kind=code, no LLM, MCP).

    The version already carries the contract-checked source — this job runs
    the same sandboxed execute pipeline + design audit the params job does,
    without requiring a parent version.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        src_code = version.code or ""
        if not src_code.strip():
            version.status = VersionStatus.failed
            version.error = "code version carries no code"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, "code job: version has no code\n")
            await db.commit()
            return {"ok": False, "error": version.error}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings((project.settings or {}) if project else {}, version.options or {}),
            runtime_settings,
        )

        # reference solids loadable via load_reference() inside the sandbox —
        # same payload shape as orchestrator._exec_payload
        ref_rows = await db.execute(
            select(Asset).where(
                Asset.project_id == version.project_id,
                col(Asset.kind).in_([AssetKind.reference_step, AssetKind.reference_stl]),
            )
        )
        references = [
            {"asset_id": str(a.id), "path": str(storage.resolve(a.path))}
            for a in ref_rows.scalars().all()
        ]

        # design-robustness pre-flight when the code declares numeric PARAMS
        declared = check_code(src_code).params
        base_params = {**declared, **(version.params or {})}
        audit_report = await _design_audit(
            db,
            job,
            version,
            executor,
            emit,
            code=src_code,
            base_params=base_params,
            settings=settings,
            runtime_settings=runtime_settings,
            references=references,
        )

        await emit("execute", 1, "executing submitted code")
        exec_result = await executor.execute(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "code": src_code,
                "params_override": {},
                "references": references,
                "settings": settings.model_dump(),
                "runtime_settings": runtime_settings.executor_values(),
                "full_exports": True,
                "name": "model",
            }
        )

        if not exec_result.get("ok"):
            was_cancelled = (exec_result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = exec_error(exec_result, "execution failed")
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error, "design_audit": audit_report}

        await _finalize_exec_version(db, job, version, exec_result, mode="cad")
        return {"ok": True, "design_audit": audit_report}


# generator view roles accepted from asset gen_params.role_hint — anything
# else falls back to front-for-first/other-for-rest (mirrors the intake-free
# default in orchestrator._mesh_source_images)
_GEN_ROLES = {"front", "back", "left", "right", "iso", "other", "scribble"}


@_guarded
async def run_meshgen_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Direct image->mesh generation (kind=meshgen, no LLM, MCP).

    The version's image assets feed the mesh sidecar verbatim; the produced
    GLB goes through the standard executor mesh pipeline (baseline
    orient/scale + exports) and lands in versions/<vid>/mesh/ exactly like
    an agent-mode mesh version.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        async def _fail(msg: str, *, cancelled: bool = False) -> dict[str, Any]:
            version.status = VersionStatus.cancelled if cancelled else VersionStatus.failed
            version.error = msg[:500]
            job.status = JobStatus.cancelled if cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"{'cancelled' if cancelled else 'failed'}: {msg}\n")
            await db.commit()
            return {"ok": False, "error": msg}

        # image inputs, kept in the version's declared asset_ids order
        wanted = [_u(a) for a in (version.asset_ids or [])]
        rows = await db.execute(select(Asset).where(col(Asset.id).in_(wanted)))
        by_id = {a.id: a for a in rows.scalars().all()}
        images = [by_id[a] for a in wanted if a in by_id and by_id[a].kind in IMAGE_ASSET_KINDS]
        if not images:
            return await _fail("no usable image assets for mesh generation")
        roles = [
            str((a.gen_params or {}).get("role_hint") or "")
            if str((a.gen_params or {}).get("role_hint") or "") in _GEN_ROLES
            else ("front" if i == 0 else "other")
            for i, a in enumerate(images)
        ]

        runtime_settings = await load_runtime_settings(db)
        project = await db.get(Project, version.project_id)
        merged = _merge_settings((project.settings or {}) if project else {}, version.options or {})
        job_runtime = _job_runtime(runtime_settings, merged)
        mesh_client = _make_mesh(job_runtime, backend=merged.mesh_backend)
        if not await _mesh_available(mesh_client):
            return await _fail("mesh server not configured or unreachable")
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=job_runtime)
        settings = _fill_settings(merged, job_runtime)

        opts = version.options or {}
        subject = str(opts.get("subject") or version.prompt or "part")[:200]
        model = str(opts.get("model") or settings.mesh_model or "auto")
        seed = int(opts.get("seed") or 0)
        try:
            post = MeshPostParams(**(opts.get("post") or {}))
        except Exception as e:
            return await _fail(f"invalid mesh post params: {e}")
        if post.ops:
            post.ops_version = MESH_OPS_VERSION

        async def check_cancel() -> bool:
            async with session_scope() as s3:
                j = await s3.get(Job, job_id)
                return (
                    j is None
                    or bool(j.cancel_requested)
                    or j.status in (JobStatus.failed, JobStatus.cancelled, JobStatus.succeeded)
                )

        async def on_progress(msg: str, p: float) -> None:
            await emit("meshgen", 1, f"{msg} ({int(p * 100)}%)", sub_progress=p, sub_label=msg)

        it_dir = storage.ensure(storage.iteration_dir(version.project_id, version.id, 1))
        out_path = it_dir / "mesh_raw.glb"
        await emit("meshgen", 1, f"generating mesh ({model}, seed {seed})")
        req = MeshGenRequest(
            images=[storage.resolve(a.path) for a in images],
            image_roles=roles,
            subject=subject,
            model=model,
            seed=seed,
            steps=settings.mesh_steps,
            guidance=settings.mesh_guidance,
            resolution=settings.mesh_resolution,
            out_path=out_path,
            job_id=str(job_id),
        )
        try:
            gen_result = await mesh_client.generate(req, on_progress, should_abort=check_cancel)
        except Exception as e:
            msg = str(e) or "mesh generation failed"
            return await _fail(msg, cancelled="cancel" in msg.lower())

        gen = gen_result.model_dump(mode="json")
        gen.pop("mesh_path", None)
        mesh_rel = storage.relpath(out_path)
        gen["mesh_rel"] = mesh_rel
        gen["images"] = [str(a.id) for a in images]

        await emit("meshprocess", 1, "processing mesh")
        exec_result = await executor.process_mesh(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "mode": "final",
                "mesh_rel": mesh_rel,
                "post": post.model_dump(),
                "anchors": None,
                "spec": None,
                "settings": settings.model_dump(),
                "runtime_settings": runtime_settings.executor_values(),
                "full_exports": True,
                "name": "model",
            }
        )

        if not exec_result.get("ok"):
            was_cancelled = (exec_result.get("error") or "") == "cancelled"
            return await _fail(
                exec_error(exec_result, "mesh processing failed"), cancelled=was_cancelled
            )

        # params carries the effective post stack so a params child can
        # re-run the processing deterministically — same contract as
        # agent-mode mesh versions
        version.params = post.model_dump()
        version.params_schema = [p.model_dump() for p in mesh_params_schema()]
        await _finalize_exec_version(
            db,
            job,
            version,
            exec_result,
            mode="mesh",
            mesh_extra={"generation": gen},
        )
        return {"ok": True}


async def _run_mesh_params(
    executor: QueueExecutor,
    job_id: uuid.UUID,
    version: Version,
    parent: Version,
    settings: ProjectSettings,
    runtime_settings: RuntimeSettings,
    overrides: dict[str, Any],
    emit: Any,
) -> dict[str, Any]:
    """Mesh-mode params job: re-run the post-pipeline on the parent's raw mesh."""
    parent_mesh_dir = storage.version_dir(parent.project_id, parent.id) / "mesh"
    raw = parent_mesh_dir / "mesh_raw.glb"
    if not raw.exists():
        return {"ok": False, "error": "parent mesh_raw.glb not found"}

    base = MeshPostParams(**(parent.params or {}))
    post = merge_post_params(base, overrides)
    await emit("meshprocess", 1, f"re-processing mesh ({overrides})")

    return await executor.process_mesh(
        {
            "job_id": str(job_id),
            "project_id": str(version.project_id),
            "version_id": str(version.id),
            "iteration": 1,
            "mode": "final",
            "mesh_rel": storage.relpath(raw),
            "post": post.model_dump(),
            "anchors": (parent.analysis or {}).get("anchors"),
            "spec": parent.spec,
            "settings": settings.model_dump(),
            "runtime_settings": runtime_settings.executor_values(),
            "full_exports": True,
            "name": "model",
        }
    )


def _minimal_spec(parent: Version) -> PartSpec:
    """Fallback PartSpec for optimize sessions whose parent has no stored spec —
    dims come from the parent's measured analysis bbox."""
    from cadx.llm.schemas import Dim

    bb = ((parent.analysis or {}).get("bbox_size") or [0, 0, 0])[:3]
    dims = [Dim(value=float(v or 0), unit="mm", source="inferred") for v in bb]
    while len(dims) < 3:
        dims.append(Dim(value=0.0, unit="mm", source="inferred"))
    return PartSpec(
        name="part",
        summary="(no stored spec — parent analysis only)",
        units="mm",
        length_x=dims[0],
        width_y=dims[1],
        height_z=dims[2],
        features=[],
        image_roles=[],
    )


async def _try_optimize_session(
    db: AsyncSession,
    ctx: dict[str, Any],
    job_id: uuid.UUID,
    version: Version,
    parent: Version,
    spec: PartSpec | None,
    executor: QueueExecutor,
    emit: Any,
    settings: ProjectSettings,
    runtime_settings: RuntimeSettings,
    mesh_rel: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Agent-driven optimize session for the standalone endpoint. Returns
    {"result", "result_it", "agent_runs"} on a chosen run, {"early_return": res}
    when the agent kept the base part, or {} when the session is unavailable —
    the caller then takes the deterministic path."""
    from cadx.agent.steps import optimize as optimize_step

    async def check_cancel() -> bool:
        async with session_scope() as s3:
            j = await s3.get(Job, job_id)
            # deleted row or externally-finalised status → orphaned task
            return (
                j is None
                or bool(j.cancel_requested)
                or j.status in (JobStatus.failed, JobStatus.cancelled, JobStatus.succeeded)
            )

    sheet = storage.version_dir(parent.project_id, parent.id) / "contact_sheet.png"
    job_runtime = _job_runtime(runtime_settings, settings)
    actx = AgentCtx(
        llm=_make_llm(
            job_runtime,
            backend=settings.llm_backend,
            caps=await load_caps_session(),
            caps_cb=record_caps_session,
        ),
        executor=executor,
        trace=DbTrace(job_id),
        emit=emit,
        check_cancel=check_cancel,
        job_id=str(job_id),
        project_id=version.project_id,
        version_id=version.id,
        mode="generate",
        prompt=parent.prompt or "",
        settings=settings,
        runtime_settings=job_runtime,
        parent=_parent_ctx(version, parent),
        anchors=(parent.analysis or {}).get("anchors"),
    )
    try:
        agent = await optimize_step.run(
            actx,
            spec or _minimal_spec(parent),
            base_stl=storage.resolve(mesh_rel),
            base_it=0,
            analysis=parent.analysis,
            contact_sheet=sheet if sheet.exists() else None,
            request={
                "load_cases": payload.get("load_cases"),
                "optimize": payload.get("optimize"),
                "material": payload.get("material"),
            },
            # bound by the arq job timeout minus a finalize margin — a chain
            # of full-length SIMP runs must not outlive the job
            deadline=time.monotonic() + get_settings().agent_job_timeout_s - 300,
        )
    except Exception as e:
        log.warning("optimize.agent_session_failed", error=str(e)[:300])
        return {}
    if agent is None:
        return {}
    if agent.chosen_run == -1:
        version.status = VersionStatus.needs_review
        version.error = f"agent kept the base part: {agent.reasoning or 'no note'}"
        job = await db.get(Job, job_id)
        if job is not None:
            job.status = JobStatus.succeeded
            job.finished_at = datetime.now(UTC)
            job.progress = 1.0
            await _append_job_log(db, job, f"done: kept base — {version.error}\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "needs_review"})
        return {"early_return": {"ok": True}}
    if agent.res is None or agent.iteration is None:
        return {}
    await emit(
        "optimize",
        agent.iteration,
        f"agent selected run {agent.chosen_run} — {agent.reasoning or 'no note'}",
    )
    return {"result": agent.res, "result_it": agent.iteration, "agent_runs": agent.runs}


def _parent_stl(parent: Version) -> str | None:
    """Best available triangle-mesh artifact of a version (relpath)."""
    return storage.best_mesh_relpath(parent.project_id, parent.id, parent.iterations or 1)


def _injob_refine_committed(version: Version) -> bool:
    """True when the version's chosen topopt agent run is a REFINE output —
    the in-job optimize session already finished features (`refine_features`
    runs land in `mesh.topopt.agent_runs` with `params.refine_of` set; the
    entry marked `chosen` is the committed result). Spawning a standalone
    refine child on top would redo the same re-cuts."""
    top = (version.mesh or {}).get("topopt") or {}
    runs = top.get("agent_runs") or []
    return any(
        isinstance(r, dict)
        and r.get("chosen")
        and (r.get("params") or {}).get("refine_of") is not None
        for r in runs
    )


async def _maybe_spawn_refine(
    db: AsyncSession,
    parent: Version,
    *,
    spec: PartSpec | None,
    optimize: dict[str, Any],
    load_cases: list[dict[str, Any]],
    material: str,
    runtime: RuntimeSettings | None = None,
) -> bool:
    """Create + enqueue a 'refine' child version on a finished optimize run.

    Triggers: OptimizeSpec.auto_refine_features, the request-level flag, or
    the runtime-wide `topopt_auto_refine` default. Never spawns a child with
    nothing to re-cut — requires declared spec features or measured topopt
    anchors. Returns True when enqueued."""
    runtime = runtime or await load_runtime_settings(db)
    auto_refine = bool(
        optimize.get("auto_refine_features")
        or (spec.optimize.auto_refine_features if spec and spec.optimize else False)
        or runtime.topopt_auto_refine
    )
    if not auto_refine or parent.status not in (
        VersionStatus.succeeded,
        VersionStatus.needs_review,
    ):
        return False
    if _injob_refine_committed(parent):
        log.info(
            "refine.spawn_skipped",
            version=str(parent.id),
            reason="refine already applied in generation job",
        )
        return False
    has_features = bool(spec and spec.features)
    topopt_anchors = ((parent.mesh or {}).get("topopt") or {}).get("anchors") or []
    if not (has_features or topopt_anchors):
        log.info(
            "refine.spawn_skipped",
            version=str(parent.id),
            reason="no spec features or topopt anchors to re-cut",
        )
        return False
    from cadx.api.deps import enqueue as enqueue_agent

    max_num = (
        await db.execute(
            select(func.max(Version.number)).where(Version.project_id == parent.project_id)
        )
    ).scalar() or 0
    child = Version(
        project_id=parent.project_id,
        parent_id=parent.id,
        number=max_num + 1,
        kind=VersionKind.refine,
        prompt=f"refine {parent.prompt}"[:500],
        gen_mode=parent.gen_mode,
        asset_ids=parent.asset_ids or [],
        options={**(parent.options or {}), "auto_refined": True},
    )
    db.add(child)
    await db.flush()
    rjob = Job(version_id=child.id, type=JobType.refine)
    db.add(rjob)
    await db.commit()
    keep_regions = optimize.get("keep_regions") or []
    await enqueue_agent(
        "run_refine_job",
        {
            "job_id": str(rjob.id),
            "version_id": str(child.id),
            "load_cases": load_cases,
            "keep_regions": keep_regions,
            # no explicit keep regions but measured anchors exist — the
            # executor may auto-derive anchor keep-boxes from them
            "auto_keep_anchors": bool(not keep_regions and topopt_anchors),
            "material": material,
            "helical_threads": bool(optimize.get("helical_threads")),
            "auto_plane_restore": bool(optimize.get("auto_plane_restore")),
            "restore_planes": optimize.get("restore_planes", True),
        },
    )
    return True


async def _spawn_refine_after_topopt(
    db: AsyncSession,
    job: Job,
    version: Version,
    outcome: RunOutcome,
    settings: ProjectSettings,
    runtime: RuntimeSettings,
    emit: Any,
) -> None:
    """In-job topopt leaves bores voxel-rough and threads destroyed —
    enqueue the same feature-finishing refine child the standalone optimize
    job spawns via `_maybe_spawn_refine` (which holds the trigger +
    nothing-to-refine guards). Best-effort: a failure here must never fail
    the already-persisted job."""
    if outcome.resolved_mode != "topopt" or outcome.status not in (
        "succeeded",
        "needs_review",
    ):
        return
    try:
        if _injob_refine_committed(version):
            # the agent session already committed a refine run — a child
            # job would re-cut the same features a second time
            await _append_job_log(db, job, "refine already applied in generation job — skipped\n")
            await emit("refine", 1, "refine already applied in generation job — skipped")
            return
        spec = PartSpec(**version.spec) if version.spec else None
        optimize = spec.optimize.model_dump() if spec and spec.optimize else {}
        load_cases = [c.model_dump() for c in (spec.load_cases if spec else [])]
        material = settings.fem_material or (spec.material if spec else "")
        refine_enqueued = await _maybe_spawn_refine(
            db,
            version,
            spec=spec,
            optimize=optimize,
            load_cases=load_cases,
            material=material,
            runtime=runtime,
        )
        if refine_enqueued:
            await _append_job_log(db, job, "auto-refine enqueued\n")
            await emit("refine", 1, "feature refine enqueued")
    except Exception:
        log.warning("agent.auto_refine_failed", job=str(job.id), exc_info=True)


@_guarded
async def run_optimize_job(  # noqa: C901
    ctx: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Manual topology optimization (LLM-free, like params): child version gets
    the optimized mesh + FEM report + artifacts."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "optimize job: missing parent\n")
                await db.commit()
            return {"ok": False, "error": "missing parent version"}

        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        mesh_rel = _parent_stl(parent)
        if mesh_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no mesh artifact to optimize"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        # load cases: request payload -> parent spec -> none
        spec = PartSpec(**parent.spec) if parent.spec else None
        load_cases = payload.get("load_cases") or [
            c.model_dump() for c in (spec.load_cases if spec else [])
        ]
        if not load_cases:
            version.status = VersionStatus.failed
            version.error = "no load cases — declare loads in the prompt or the optimize dialog"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        runtime_settings = await load_runtime_settings(db)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            ProjectSettings(**(project.settings or {})) if project else ProjectSettings(),
            runtime_settings,
        )
        optimize = {
            **(spec.optimize.model_dump() if spec and spec.optimize else {}),
            **(
                {"target_volume_fraction": settings.topopt_volfrac}
                if settings.topopt_volfrac
                else {}
            ),
            "max_iterations": settings.topopt_iterations,
            **(payload.get("optimize") or {}),
        }
        # multires is default-on executor-side — a serialized OptimizeSpec
        # carries schema-False which would read as an opt-out
        if not optimize.get("multires"):
            optimize.pop("multires", None)
        # stated spec envelope → executor clips the base mesh to the design
        # domain before voxelizing (raw hero meshes can overshoot it)
        from cadx.cad.verifyspec import _stated_envelope

        spec_env = _stated_envelope(parent.spec if parent else None)
        spec_dims = [spec_env.get(a) for a in ("x", "y", "z")]
        if any(d is not None for d in spec_dims):
            optimize["spec_dims_mm"] = spec_dims
        material = (
            payload.get("material") or settings.fem_material or (spec.material if spec else "")
        )

        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        result_it = 1
        agent_runs: list[dict[str, Any]] = []

        # agent-driven session first (preview/FEM/SIMP probes, picks best run);
        # falls back to the deterministic single shot below
        result: dict[str, Any] | None = None
        if runtime_settings.agent_struct_tools:
            agent_out = await _try_optimize_session(
                db,
                ctx,
                job_id,
                version,
                parent,
                spec,
                executor,
                emit,
                settings,
                runtime_settings,
                mesh_rel,
                payload,
            )
            if agent_out.get("early_return") is not None:
                return dict(agent_out["early_return"])
            result = agent_out.get("result")
            result_it = agent_out.get("result_it") or result_it
            agent_runs = agent_out.get("agent_runs") or []

        if result is None:
            await emit("optimize", 1, f"SIMP on {mesh_rel} ({len(load_cases)} load case(s))")
            result = await executor.optimize(
                {
                    "job_id": str(job_id),
                    "project_id": str(version.project_id),
                    "version_id": str(version.id),
                    "iteration": 1,
                    "mesh_rel": mesh_rel,
                    "load_cases": load_cases,
                    "optimize": optimize,
                    "material": material,
                    "voxel_mm": settings.fem_voxel_mm,
                    "runtime_settings": runtime_settings.executor_values(),
                    "name": "model",
                    "anchors": (parent.analysis or {}).get("anchors", {}).get("holes"),
                    "spec": spec.model_dump(mode="json") if spec else None,
                }
            )

        if not result.get("ok"):
            was_cancelled = (result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = exec_error(result, "optimization failed")
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error}
        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, result_it, result)
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        it_dir = storage.iteration_dir(version.project_id, version.id, result_it)
        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        fem_dir = storage.ensure(v_dir / "fem")
        fem_art = 0
        import shutil

        for png in (it_dir / "fem").glob("*.png") if (it_dir / "fem").exists() else []:
            shutil.copy2(png, fem_dir / png.name)
            artifacts[f"fem:{png.stem}"] = storage.relpath(fem_dir / png.name)
            fem_art += 1
        for extra in ("topopt/density.npz", "topopt/history.json", "fem/report.json"):
            src = it_dir / extra
            if src.exists():
                dst = (
                    fem_dir / src.name
                    if src.parent.name == "fem"
                    else storage.ensure(v_dir / "topopt") / src.name
                )
                shutil.copy2(src, dst)
                artifacts[f"fem:{src.stem}"] = storage.relpath(dst)
                fem_art += 1

        topopt = result.get("topopt") or {}
        fem = result.get("fem") or {}
        version.gen_mode = "topopt"
        version.resolved_mode = "topopt"
        version.spec = parent.spec
        version.fem = fem or None
        version.mesh = {
            **(result.get("mesh") or {}),
            "topopt": {**topopt, **({"agent_runs": agent_runs} if agent_runs else {})},
            "from_optimize_of": str(parent.id),
        }
        version.analysis = result.get("analysis")
        version.params = {"optimize": optimize, "load_cases": load_cases, "material": material}
        version.iterations = result_it
        # structural review: flag unsafe results instead of silent success;
        # a FEM-less result (no solvable case) is unverified, not clean;
        # disconnected keep/load islands make a multi-body part = defect
        sfs = [c.get("safety_factor") for c in fem.get("cases", []) if c.get("safety_factor")]
        worst_sf = min(sfs) if sfs else None
        n_bodies = max(
            int(topopt.get("n_components") or 1), int(topopt.get("n_bodies_export") or 1)
        )
        version.status = (
            VersionStatus.needs_review
            if worst_sf is None or worst_sf < 1.0 or n_bodies > 1
            else VersionStatus.succeeded
        )
        register_artifacts(db, version.id, artifacts, tessellated_step=False)
        await emit(
            "optimize",
            1,
            f"done: volfrac {topopt.get('volfrac_achieved')}, SF {worst_sf}, "
            f"{topopt.get('iterations')} iterations",
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")

        # auto feature-finishing: OptimizeSpec.auto_refine_features (or the
        # request-level flag) spawns a refine child version right after the
        # optimize succeeded — bores/threads/keep surfaces are re-cut exact
        refine_enqueued = await _maybe_spawn_refine(
            db,
            version,
            spec=spec,
            optimize=optimize,
            load_cases=load_cases,
            material=material,
            runtime=runtime_settings,
        )
        if refine_enqueued:
            await _append_job_log(db, job, "auto-refine enqueued\n")

        await db.commit()
        await _publish(
            str(job_id),
            {
                "type": "done",
                "status": "succeeded",
                **({"refine_enqueued": True} if refine_enqueued else {}),
            },
        )
        return {"ok": True}


@_guarded
async def run_topopt_resample_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-extract a stored TopOpt density field at a new density_threshold /
    min_member without re-running SIMP. Produces a child version of kind
    'optimize' carrying the re-verified mesh + FEM reports."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "resample job: missing parent\n")
                await db.commit()
            return {"ok": False, "error": "missing parent version"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        organic_parent = parent.kind == VersionKind.organic
        density_rel = ((parent.mesh or {}).get("topopt") or {}).get("density_relpath") or (
            (parent.mesh or {}).get("organic") or {}
        ).get("density_relpath")
        if not density_rel or not storage.resolve(density_rel).exists():
            version.status = VersionStatus.failed
            version.error = "parent has no stored density field (re-run the optimization)"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        # load cases: request payload -> parent optimize params -> parent spec
        spec = PartSpec(**parent.spec) if parent.spec else None
        parent_params = parent.params or {}
        load_cases = (
            payload.get("load_cases")
            or parent_params.get("load_cases")
            or [c.model_dump() for c in (spec.load_cases if spec else [])]
        )
        optimize = {
            **(parent_params.get("optimize") or {}),
            **(payload.get("optimize") or {}),
        }
        material = (
            payload.get("material")
            or parent_params.get("material")
            or (spec.material if spec else "")
        )
        keep_regions = payload.get("keep_regions") or (
            spec.optimize.keep_regions if spec and spec.optimize else []
        )
        keep_regions = [r.model_dump() if hasattr(r, "model_dump") else r for r in keep_regions]
        optimize.setdefault("keep_regions", keep_regions)

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        await emit(
            "optimize",
            1,
            f"re-extracting density at threshold "
            f"{optimize.get('density_threshold', 0.5):g} (no new SIMP run)",
        )
        result = await executor.topopt_resample(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "density_relpath": density_rel,
                "load_cases": load_cases,
                "optimize": optimize,
                "material": material,
                "runtime_settings": runtime_settings.executor_values(),
                "name": "model",
                # reuse the anchors the parent optimize run actually resolved
                # (mesh-scanned + merged); fall back to declared anchors
                "anchors": ((parent.mesh or {}).get("topopt") or {}).get("anchors")
                or (parent.analysis or {}).get("anchors", {}).get("holes"),
                "spec": spec.model_dump(mode="json") if spec else None,
            }
        )

        if not result.get("ok"):
            version.status = VersionStatus.failed
            version.error = exec_error(result, "resample failed")
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"failed: {version.error}\n")
            await db.commit()
            return {"ok": False, "error": version.error}

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, result)
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        it_dir = storage.iteration_dir(version.project_id, version.id, 1)
        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        fem_dir = storage.ensure(v_dir / "fem")
        import shutil

        for png in (it_dir / "fem").glob("*.png") if (it_dir / "fem").exists() else []:
            shutil.copy2(png, fem_dir / png.name)
            artifacts[f"fem:{png.stem}"] = storage.relpath(fem_dir / png.name)
        for extra in ("fem/report.json", "fem/report_export.json"):
            src = it_dir / extra
            if src.exists():
                shutil.copy2(src, fem_dir / src.name)
                artifacts[f"fem:{src.stem}"] = storage.relpath(fem_dir / src.name)

        topopt = result.get("topopt") or {}
        fem = result.get("fem") or {}
        if organic_parent:
            # organic resample — aesthetic restyle, no FEM verdict
            version.gen_mode = "mesh"
            version.resolved_mode = "mesh"
            topopt = {**topopt, "organic": True}
        else:
            version.gen_mode = "topopt"
            version.resolved_mode = "topopt"
        version.spec = parent.spec
        version.fem = fem or None
        version.mesh = {
            **(result.get("mesh") or {}),
            "topopt": topopt,
            "resampled_from": str(parent.id),
            # the density field is shared with the parent run
            "density_relpath": density_rel,
        }
        version.mesh["topopt"]["density_relpath"] = density_rel
        version.analysis = result.get("analysis")
        version.params = {
            "optimize": optimize,
            "load_cases": load_cases,
            "material": material,
            "resample": True,
        }
        version.iterations = 1
        sfs = [c.get("safety_factor") for c in fem.get("cases", []) if c.get("safety_factor")]
        worst_sf = min(sfs) if sfs else None
        n_bodies = max(
            int(topopt.get("n_components") or 1), int(topopt.get("n_bodies_export") or 1)
        )
        version.status = (
            VersionStatus.needs_review
            if n_bodies > 1 or (not organic_parent and (worst_sf is None or worst_sf < 1.0))
            else VersionStatus.succeeded
        )
        register_artifacts(db, version.id, artifacts, tessellated_step=organic_parent)
        await emit(
            "optimize",
            1,
            f"resample done: threshold {topopt.get('threshold')}, SF {worst_sf}",
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


@_guarded
async def run_refine_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Post-TopOpt feature finishing (LLM-free): re-cut exact bores/threads/
    inserts at measured anchors and restore keep-region surfaces from the
    pre-optimization part. Produces a 'refine' child version."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "refine job: missing parent\n")
                await db.commit()
            return {"ok": False, "error": "missing parent version"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        mesh_rel = _parent_stl(parent)
        if mesh_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no mesh artifact to refine"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        # keep-restore source: the part BEFORE optimization — for a topopt
        # child that is its own parent (the base); otherwise the parent itself
        base_rel = _parent_stl(parent)
        if parent.resolved_mode == "topopt" and parent.parent_id:
            grand = await db.get(Version, parent.parent_id)
            if grand is not None:
                base_rel = _parent_stl(grand) or base_rel

        spec = PartSpec(**parent.spec) if parent.spec else None
        parent_params = parent.params or {}
        load_cases = (
            payload.get("load_cases")
            or parent_params.get("load_cases")
            or [c.model_dump() for c in (spec.load_cases if spec else [])]
        )
        optimize_params = parent_params.get("optimize") or {}
        keep_regions = (
            payload.get("keep_regions")
            or optimize_params.get("keep_regions")
            or (spec.optimize.keep_regions if spec and spec.optimize else [])
        )
        keep_regions = [r.model_dump() if hasattr(r, "model_dump") else r for r in keep_regions]
        material = (
            payload.get("material")
            or parent_params.get("material")
            or (spec.material if spec else "")
        )
        anchors = (
            payload.get("anchors")
            or ((parent.mesh or {}).get("topopt") or {}).get("anchors")
            or (parent.analysis or {}).get("anchors", {}).get("holes")
        )
        spec_features = payload.get("spec_features") or [
            f.model_dump() for f in (spec.features if spec else [])
        ]

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        n_feat = len(payload.get("features") or []) or len(spec_features)
        await emit(
            "refine", 1, f"feature finishing — {n_feat} feature(s), {len(anchors or [])} anchor(s)"
        )
        result = await executor.refine(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "mesh_rel": mesh_rel,
                "base_mesh_rel": base_rel if base_rel != mesh_rel else None,
                "features": payload.get("features"),
                "spec_features": spec_features,
                "anchors": anchors,
                "keep_regions": keep_regions,
                "load_cases": load_cases,
                "material": material,
                "voxel_mm": payload.get("voxel_mm"),
                "restore_planes": bool(
                    payload.get("restore_planes")
                    if payload.get("restore_planes") is not None
                    else (spec.optimize.restore_planes if spec and spec.optimize else True)
                ),
                "helical_threads": bool(
                    payload.get("helical_threads")
                    or (spec.optimize.helical_threads if spec and spec.optimize else False)
                ),
                "auto_plane_restore": bool(
                    payload.get("auto_plane_restore")
                    or (spec.optimize.auto_plane_restore if spec and spec.optimize else False)
                ),
                # the executor may auto-derive anchor keep-boxes when no
                # explicit keep_regions were resolved (topopt auto-keep gap)
                "auto_keep_anchors": bool(payload.get("auto_keep_anchors")),
                "runtime_settings": runtime_settings.executor_values(),
                "name": "model",
                "spec": spec.model_dump(mode="json") if spec else None,
            }
        )

        if not result.get("ok"):
            version.status = VersionStatus.failed
            version.error = exec_error(result, "refine failed")
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"failed: {version.error}\n")
            await db.commit()
            return {"ok": False, "error": version.error}

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, result)
        # promote the refined exports to versions/<vid>/exports/ — API/UI
        # consumers (and best_mesh_relpath) resolve the version-level dir
        # first; iteration-only paths serve the parent's UN-refined files
        artifacts.update(
            _copy_export_artifacts_ctxless(version.project_id, version.id, 1, result.get("exports"))
        )
        it_dir = storage.iteration_dir(version.project_id, version.id, 1)
        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        fem_dir = storage.ensure(v_dir / "fem")
        import shutil

        for png in (it_dir / "fem").glob("*.png") if (it_dir / "fem").exists() else []:
            shutil.copy2(png, fem_dir / png.name)
            artifacts[f"fem:{png.stem}"] = storage.relpath(fem_dir / png.name)
        for extra in ("fem/report.json", "fem/report_export.json"):
            src = it_dir / extra
            if src.exists():
                shutil.copy2(src, fem_dir / src.name)
                artifacts[f"fem:{src.stem}"] = storage.relpath(fem_dir / src.name)

        refine_report = (result.get("features") or {}).get("refine") or {}
        fem = result.get("fem") or {}
        version.gen_mode = parent.gen_mode
        version.resolved_mode = parent.resolved_mode
        version.spec = parent.spec
        version.fem = fem or None
        version.mesh = {
            **(result.get("mesh") or {}),
            "refine": refine_report,
            "refined_from": str(parent.id),
        }
        version.analysis = result.get("analysis")
        version.params = {
            "refine": {
                "features": payload.get("features"),
                "load_cases": load_cases,
                "material": material,
            }
        }
        version.iterations = 1
        sfs = [c.get("safety_factor") for c in fem.get("cases", []) if c.get("safety_factor")]
        worst_sf = min(sfs) if sfs else None
        n_skipped = len(refine_report.get("skipped") or [])

        # deterministic post-refine verification: the report's `applied` list
        # is self-reported — rescan the refined mesh and require every applied
        # cuttable +z feature to have left a real tunnel (a cut landing in a
        # SIMP void reports huge dia_deviation_mm yet still "applied")
        from cadx.cad.fem.refine import refine_verify

        misses = refine_verify(
            it_dir / "_run" / "model.stl",
            refine_report,
            anchors=anchors,
            request_features=payload.get("features"),
        )
        issues: list[dict[str, Any]] = [
            {"source": "refine", "severity": "minor", "what": str(w)}
            for w in (refine_report.get("warnings") or [])
        ]
        issues += [
            {
                "source": "refine_verify",
                "severity": "major",
                "what": (
                    f"re-cut feature '{m['entry'].get('name')}' left no tunnel at "
                    f"({m['x']:.1f}, {m['y']:.1f}) — expected Ø{m['dia']:g} mm"
                ),
                "report": _json_safe(m["entry"]),
            }
            for m in misses
        ]
        if issues:
            version.issues = (version.issues or []) + issues

        version.status = (
            VersionStatus.needs_review
            if (worst_sf is not None and worst_sf < 1.0)
            or (n_skipped and not refine_report.get("applied"))
            or bool(misses)
            else VersionStatus.succeeded
        )
        register_artifacts(db, version.id, artifacts, tessellated_step=False)
        await emit(
            "refine",
            1,
            f"done: {len(refine_report.get('applied') or [])} feature(s) re-cut, "
            f"{refine_report.get('restored_regions') or 0} keep surface(s) restored"
            + (f", SF {worst_sf}" if worst_sf else "")
            + (f", {len(misses)} feature(s) failed verification" if misses else ""),
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


@_guarded
async def run_organic_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Procedural organic restyle (LLM-free): child version gets the restyled
    mesh + exports + stored density field (resample-compatible npz layout).

    Pseudo-topology-optimization — no FEM, no SIMP: the result is aesthetic
    styling and explicitly NOT structurally verified.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "organic job: missing parent\n")
                await db.commit()
            return {"ok": False, "error": "missing parent version"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            # e.g. an arq retry after the guard already finalised the job
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)

        mesh_rel = _parent_stl(parent)
        if mesh_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no mesh artifact to restyle"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        organic = payload.get("organic") or {}
        runtime_settings = await load_runtime_settings(db)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            ProjectSettings(**(project.settings or {})) if project else ProjectSettings(),
            runtime_settings,
        )
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)

        await emit(
            "organic",
            1,
            f"organic restyle ({organic.get('style', 'gyroid')}, seed "
            f"{organic.get('seed', 0)}) on {mesh_rel}",
        )
        result = await executor.organic(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "mesh_rel": mesh_rel,
                "organic": organic,
                "voxel_mm": settings.fem_voxel_mm,
                "runtime_settings": runtime_settings.executor_values(),
                "name": "model",
                "anchors": (parent.analysis or {}).get("anchors", {}).get("holes"),
            }
        )

        if not result.get("ok"):
            was_cancelled = (result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = exec_error(result, "organic restyle failed")
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error}

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, result)
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        it_dir = storage.iteration_dir(version.project_id, version.id, 1)
        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        import shutil

        for extra in ("organic/density.npz",):
            src = it_dir / extra
            if src.exists():
                dst_dir = storage.ensure(v_dir / "organic")
                shutil.copy2(src, dst_dir / src.name)
                artifacts[f"organic:{src.stem}"] = storage.relpath(dst_dir / src.name)

        org = (result.get("mesh") or {}).get("organic") or {}
        version.gen_mode = "mesh"  # tessellated output — ops-only edits apply
        version.resolved_mode = "mesh"
        version.spec = parent.spec
        version.mesh = {
            **(result.get("mesh") or {}),
            "organic": org,
            "from_organic_of": str(parent.id),
        }
        version.analysis = result.get("analysis")
        version.params = {"organic": organic}
        version.iterations = 1
        # unprintable multi-body results are flagged; the result is
        # intentionally NOT structural — "needs_review" is only about
        # manufacturability here, never about strength
        n_bodies = max(
            int(org.get("n_components") or 1), int(org.get("n_bodies_export") or 1)
        )
        version.status = VersionStatus.needs_review if n_bodies > 1 else VersionStatus.succeeded
        register_artifacts(db, version.id, artifacts, tessellated_step=True)
        await emit(
            "organic",
            1,
            f"done: {org.get('style')}, volfrac {org.get('volfrac_achieved')}, "
            f"{org.get('triangles')} tris",
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


@_guarded
async def run_mesh_edit_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """GUI visual edit: a recorded MeshOp stack applied to the parent's mesh —
    produces a tessellated child version (kind=mesh_edit, no LLM).

    mesh parents: baseline + ops replayed on mesh/mesh_raw.glb (same path as a
    params rerun). Other parents: ops-only pipeline on the finished STL —
    already in the mm frame, so no orient/scale baseline runs.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = await db.get(Version, version.parent_id) if version and version.parent_id else None
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                await _append_job_log(db, job, "mesh_edit job: missing parent\n")
                await db.commit()
            return {"ok": False, "error": "missing parent version"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)
        parent_mode = parent.resolved_mode or "cad"
        raw_ops = payload.get("ops") or []

        try:
            ops = [MeshOp.model_validate(o) for o in raw_ops]
        except Exception as e:
            version.status = VersionStatus.failed
            version.error = f"invalid mesh ops: {e}"[:500]
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        mesh_rel: str | None = None
        ops_only = parent_mode != "mesh"
        spec: dict[str, Any] | None = None
        if parent_mode == "mesh":
            raw = storage.version_dir(parent.project_id, parent.id) / "mesh" / "mesh_raw.glb"
            if raw.exists():
                mesh_rel = storage.relpath(raw)
                spec = parent.spec
        else:
            mesh_rel = storage.best_mesh_relpath(
                parent.project_id, parent.id, parent.iterations or 1
            )
        if mesh_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no mesh to edit"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await db.commit()
            return {"ok": False, "error": version.error}

        if ops_only:
            post = MeshPostParams(ops=ops)
        else:
            post = merge_post_params(
                MeshPostParams(**(parent.params or {})), payload.get("post") or {}
            )
            post = post.model_copy(update={"ops": ops})
        post.ops_version = MESH_OPS_VERSION

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings(
                (project.settings or {}) if project else {},
                version.options or (parent.options if parent else {}) or {},
            ),
            runtime_settings,
        )
        await emit(
            "meshedit",
            1,
            f"applying {len(ops)} mesh op(s) ({'ops-only' if ops_only else 'baseline+ops'})",
        )
        exec_result = await executor.process_mesh(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "mode": "final",
                "mesh_rel": mesh_rel,
                "post": post.model_dump(),
                "anchors": (parent.analysis or {}).get("anchors"),
                "spec": spec,
                "settings": settings.model_dump(),
                "runtime_settings": runtime_settings.executor_values(),
                "full_exports": True,
                "name": "model",
                "ops_only": ops_only,
            }
        )

        if not exec_result.get("ok"):
            was_cancelled = (exec_result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = exec_error(exec_result, "mesh edit failed")
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error}

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, exec_result)
        for fmt, rel in (exec_result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        artifacts.update(_copy_mesh_artifacts_ctxless(version.project_id, version.id, 1, parent))

        import json

        version.gen_mode = "mesh" if ops_only else parent_mode
        version.resolved_mode = "mesh"
        version.spec = parent.spec
        version.params = post.model_dump()
        version.params_schema = parent.params_schema if parent_mode == "mesh" else None
        version.analysis = exec_result.get("analysis")
        version.dfm = exec_result.get("dfm")
        version.iterations = 1
        version.status = VersionStatus.succeeded
        version.mesh = dict(parent.mesh or {})
        version.mesh.update(exec_result.get("mesh") or {})
        version.mesh["post"] = version.params
        version.mesh["from_edit_of"] = str(parent.id)
        version.mesh["edited"] = True
        register_artifacts(db, version.id, artifacts, tessellated_step=True)

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        m_dir = storage.ensure(v_dir / "mesh")
        (m_dir / "mesh.json").write_text(json.dumps(version.mesh, indent=2), encoding="utf-8")
        (v_dir / "params.json").write_text(
            json.dumps(version.params or {}, indent=2), encoding="utf-8"
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


@_guarded
async def run_fem_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Standalone FEM check on the SAME version (no child) — GUI-declared load
    cases incl. painted 'points' regions. Writes version.fem + fem artifacts;
    flags the version needs_review when the worst safety factor drops < 1."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        prev_status = version.status
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)
        mesh_rel = storage.best_mesh_relpath(
            version.project_id, version.id, version.iterations or 1
        )
        load_cases = payload.get("load_cases") or []
        if mesh_rel is None or not load_cases:
            version.status = prev_status
            version.error = (
                "no load cases declared" if load_cases else "version has no mesh to analyse"
            )
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"failed: {version.error}\n")
            await db.commit()
            return {"ok": False, "error": version.error}

        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        await emit("fem", 1, f"FEM check on {mesh_rel} ({len(load_cases)} load case(s))")
        result = await executor.fem_analyze(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": version.iterations or 1,
                "mesh_rel": mesh_rel,
                "load_cases": load_cases,
                "keep_regions": payload.get("keep_regions") or [],
                "material": payload.get("material") or "",
                "voxel_mm": payload.get("voxel_mm"),
                "runtime_settings": runtime_settings.executor_values(),
                "anchors": (version.analysis or {}).get("anchors", {}).get("holes"),
            }
        )

        if not result.get("ok"):
            version.status = prev_status
            version.error = exec_error(result, "FEM failed")
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"failed: {version.error}\n")
            await db.commit()
            return {"ok": False, "error": version.error}

        # persist report + stress/deformation renders on the version itself
        import shutil

        it_dir = storage.iteration_dir(version.project_id, version.id, version.iterations or 1)
        fem_dir = storage.ensure(storage.version_dir(version.project_id, version.id) / "fem")
        artifacts: dict[str, str] = {}
        if (it_dir / "fem").exists():
            for png in (it_dir / "fem").glob("*.png"):
                shutil.copy2(png, fem_dir / png.name)
                artifacts[f"fem:{png.stem}"] = storage.relpath(fem_dir / png.name)
            report = it_dir / "fem" / "report.json"
            if report.exists():
                shutil.copy2(report, fem_dir / "report.json")
                artifacts["fem:report"] = storage.relpath(fem_dir / "report.json")

        fem = result.get("fem") or {}
        version.fem = fem or None
        sfs = [c.get("safety_factor") for c in fem.get("cases", []) if c.get("safety_factor")]
        worst_sf = min(sfs) if sfs else None
        version.status = (
            VersionStatus.needs_review
            if (worst_sf is not None and worst_sf < 1.0)
            or prev_status == VersionStatus.needs_review
            else VersionStatus.succeeded
        )
        version.error = None
        register_artifacts(db, version.id, artifacts)
        await emit(
            "fem",
            1,
            f"done: worst SF {worst_sf}",
            metrics={"worst_sf": float(worst_sf)} if worst_sf is not None else None,
        )
        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(db, job, "done\n")
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


@_guarded
async def run_reexport_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-export formats from the best iteration BREP."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False}
        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        await db.commit()
        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        result = await executor.export_all(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": payload.get("iteration") or 1,
                "name": "model",
                "runtime_settings": runtime_settings.executor_values(),
            }
        )
        artifacts = {}
        for fmt, rel in (result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        register_artifacts(db, version.id, artifacts)
        job.status = JobStatus.succeeded if result.get("ok") else JobStatus.failed
        job.finished_at = datetime.now(UTC)
        await db.commit()
        return result


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #


def _merge_settings(
    project_settings: dict[str, Any], version_options: dict[str, Any]
) -> ProjectSettings:
    """project defaults < per-run options (GUI options panel).

    `ui_*` keys (e.g. ui_editor) are GUI state persisted on the version —
    they must never reach run settings."""
    s = ProjectSettings(
        **{
            **project_settings,
            **{k: v for k, v in version_options.items() if not k.startswith("ui_")},
        }
    )
    if s.fem and not s.fem_material and s.material:
        s.fem_material = s.material
    return s


def _fill_settings(settings: ProjectSettings, runtime: RuntimeSettings) -> ProjectSettings:
    for key, default in (
        ("clarification_mode", runtime.agent_clarification_mode),
        ("max_iterations", runtime.agent_max_iterations),
        ("max_repairs", runtime.agent_max_repairs),
        ("pass_score", runtime.agent_pass_score),
        ("dim_tolerance_pct", runtime.agent_dim_tolerance_pct),
        ("dim_tolerance_mm", runtime.agent_dim_tolerance_mm),
        ("fem_voxel_mm", runtime.fem_voxel_mm),
        ("topopt_volfrac", runtime.topopt_default_volfrac),
        ("topopt_iterations", runtime.topopt_max_iterations),
    ):
        if getattr(settings, key) is None:
            setattr(settings, key, default)
    if not settings.mesh_model:
        settings.mesh_model = runtime.mesh_model
    return settings


def _job_runtime(runtime: RuntimeSettings, merged: ProjectSettings) -> RuntimeSettings:
    """Effective runtime snapshot for one job: per-job provider/model
    overrides (RunOptions keys merged into ProjectSettings) win over the
    global defaults wherever the job's code path reads
    ``ctx.runtime_settings.*``."""
    updates: dict[str, Any] = {}
    for key in (
        "llm_backend",
        "mesh_backend",
        "image_backend",
        "mesh_model",
        "image_model",
        "image_edit_model",
    ):
        v = getattr(merged, key, None)
        if v:
            updates[key] = v
    return runtime.model_copy(update=updates) if updates else runtime


def _make_mesh(runtime: RuntimeSettings, backend: str | None = None) -> MeshPort | None:
    return make_mesh_client(
        runtime_settings=runtime,
        backend=backend,
    )


def _make_image(runtime: RuntimeSettings, backend: str | None = None) -> ImagePort | None:
    return make_image_client(
        runtime_settings=runtime,
        backend=backend,
    )


async def _image_available(client: ImagePort | None) -> bool:
    """health() is ok even with zero image adapters — check models() instead."""
    if client is None:
        return False
    try:
        return any(m.available for m in await client.models())
    except Exception:
        return False


async def _mesh_available(client: MeshPort | None) -> bool:
    if client is None:
        return False
    try:
        return (await client.health()).ok
    except Exception:
        return False


def _make_llm(
    runtime: RuntimeSettings,
    backend: str | None = None,
    *,
    caps: dict[str, Any] | None = None,
    caps_cb: Any = None,
) -> LLMClient:
    return make_llm(runtime, backend=backend, caps=caps, caps_cb=caps_cb)


def _copy_best_artifacts_ctxless(
    project_id: uuid.UUID, version_id: uuid.UUID, iteration: int, exec_result: dict[str, Any]
) -> dict[str, str]:
    """Same as orchestrator._copy_best_artifacts but without an AgentCtx."""
    import shutil
    from pathlib import Path

    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    v_dir = storage.version_dir(project_id, version_id)
    out: dict[str, str] = {}
    r_dir = storage.ensure(v_dir / "renders")
    for view, rel in (exec_result.get("renders") or {}).items():
        src = it_dir / "renders" / Path(rel).name
        if src.exists():
            dst = r_dir / f"{view}.png"
            shutil.copy2(src, dst)
            out[f"render:{view}"] = storage.relpath(dst)
    for name, kind in (("contact_sheet.png", "contact_sheet"), ("thumbnail.png", "thumbnail")):
        src = it_dir / name
        if src.exists():
            dst = v_dir / name
            shutil.copy2(src, dst)
            out[kind] = storage.relpath(dst)
    if exec_result.get("dfm"):
        import json

        (v_dir / "dfm.json").write_text(json.dumps(exec_result["dfm"], indent=2), encoding="utf-8")
    return out


def _copy_export_artifacts_ctxless(
    project_id: uuid.UUID,
    version_id: uuid.UUID,
    iteration: int,
    exports: dict[str, Any] | None,
) -> dict[str, str]:
    """Promote iteration-level exports to the version-level ``exports/`` dir.

    LLM-free child jobs (refine) write exports under ``iterations/<n>/exports``
    only — but API consumers and ``storage.best_mesh_relpath`` resolve
    ``versions/<vid>/exports/model.<fmt>`` first, so without this copy the
    child version keeps serving the PARENT's files. Keys stay ``export:<fmt>``
    and point at the promoted path (iteration path as fallback)."""
    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    out: dict[str, str] = {}
    dst_dir: Path | None = None
    for fmt, rel in (exports or {}).items():
        if not rel:
            continue
        src = it_dir / "exports" / Path(str(rel)).name
        if src.exists():
            if dst_dir is None:
                dst_dir = storage.ensure(storage.export_dir(project_id, version_id))
            dst = dst_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            out[f"export:{fmt}"] = storage.relpath(dst)
        else:
            out[f"export:{fmt}"] = str(rel)
    return out


# applied refine kinds that must leave a real through-feature in the mesh
def _copy_mesh_artifacts_ctxless(
    project_id: uuid.UUID, version_id: uuid.UUID, iteration: int, parent: Version | None = None
) -> dict[str, str]:
    """Version-level mesh intermediates for params children (no AgentCtx)."""
    import shutil

    it_dir = storage.iteration_dir(project_id, version_id, iteration)
    mesh_dir = storage.ensure(storage.version_dir(project_id, version_id) / "mesh")
    out: dict[str, str] = {}
    parent_mesh = storage.version_dir(parent.project_id, parent.id) / "mesh" if parent else None
    for src, dst_name, key in (
        (it_dir / "mesh_raw.glb", "mesh_raw.glb", "mesh_raw"),
        (it_dir / "_run" / "model.stl", "mesh_clean.stl", "mesh_clean"),
        (it_dir / "_run" / "model.brep", "solid.brep", "mesh_solid"),
    ):
        src_path = src
        if not src_path.exists() and parent_mesh is not None:
            # params children don't regenerate the raw mesh — inherit from parent
            candidate = parent_mesh / dst_name
            if candidate.exists() and key == "mesh_raw":
                src_path = candidate
        if src_path.exists():
            dst = mesh_dir / dst_name
            shutil.copy2(src_path, dst)
            out[key] = storage.relpath(dst)
    return out


async def _persist_outcome(
    db: AsyncSession, job: Job, version: Version, outcome: RunOutcome, emit: Any
) -> None:
    job.finished_at = datetime.now(UTC)
    if outcome.status == "waiting_for_input":
        job.status = JobStatus.waiting_for_input
        job.questions = outcome.questions
        job.resume_state = outcome.resume_state
        version.status = VersionStatus.waiting_for_input
        version.spec = _json_safe(outcome.spec)
        await db.commit()
        await _publish(str(job.id), {"type": "question", "questions": outcome.questions})
        return

    version.spec = _json_safe(outcome.spec)
    version.plan = _json_safe(outcome.plan)
    version.code = outcome.code
    version.params = _json_safe(outcome.params)
    version.params_schema = _json_safe(outcome.params_schema)
    version.analysis = _json_safe(outcome.analysis)
    version.critique = _json_safe(outcome.critique)
    version.dfm = _json_safe(outcome.dfm)
    version.bom = _json_safe(outcome.bom)
    version.issues = _json_safe(outcome.issues) or None
    version.mesh = _json_safe(outcome.mesh)
    version.fem = _json_safe(outcome.fem) or None  # {} would be truthy-but-shapeless in the UI
    version.resolved_mode = outcome.resolved_mode
    version.score = outcome.score
    version.iterations = outcome.iterations
    version.usage = outcome.usage
    version.error = outcome.error
    tessellated = outcome.resolved_mode == "mesh" or bool(
        outcome.mesh and outcome.mesh.get("step_tessellated")
    )
    m_dir = storage.version_dir(version.project_id, version.id) / "measure"
    if m_dir.exists():
        for png in m_dir.glob("*_overlay.png"):
            outcome.artifacts.setdefault(
                f"render:measure_{png.name.removesuffix('_overlay.png')}",
                storage.relpath(png),
            )
    register_artifacts(db, version.id, outcome.artifacts, tessellated_step=tessellated)
    if outcome.mode_note:
        # the emit helper appends step lines on separate sessions — this
        # session's `job.log` is stale since load time and would clobber them
        current_log = (
            await db.execute(select(Job.log).where(Job.id == job.id))
        ).scalar_one_or_none()
        job.log = (current_log or "") + f"mode: {outcome.mode_note}\n"

    if outcome.code or outcome.params is not None:
        import json

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        if outcome.code:
            (v_dir / "model.py").write_text(outcome.code, encoding="utf-8")
        (v_dir / "params.json").write_text(
            json.dumps(_json_safe(outcome.params) or {}, indent=2), encoding="utf-8"
        )

    mapping = {
        "succeeded": (VersionStatus.succeeded, JobStatus.succeeded),
        "needs_review": (VersionStatus.needs_review, JobStatus.succeeded),
        "failed": (VersionStatus.failed, JobStatus.failed),
        "cancelled": (VersionStatus.cancelled, JobStatus.cancelled),
    }
    v_status, j_status = mapping.get(outcome.status, (VersionStatus.failed, JobStatus.failed))
    version.status = v_status
    job.status = j_status
    job.progress = 1.0
    await _append_job_log(db, job, f"final: {outcome.status}\n")
    await db.commit()
    await _publish(str(job.id), {"type": "done", "status": outcome.status, "score": outcome.score})


def _parent_source_model(artifacts: list[Artifact], parent: Version) -> str | None:
    """Best analytic B-Rep of ``parent`` for a verified feature edit, as a
    storage relpath. Preference order:

    1. ``mesh/solid.brep`` — canonical sewn solid for mesh/hybrid versions
    2. the winning run's ``_run/model.brep``, found via the registered STEP
       artifact (its path embeds the attempt dir, incl. ``aN`` subdirs)
    3. the newest ``iterations/<n>[/<aN>]/_run/model.brep`` on disk
    4. ``exports/model.step`` — always written by ``export_all``; STEP is a
       faithful analytic exchange for the plain cylindrical faces we edit
    """
    pid, vid = parent.project_id, parent.id
    v_dir = storage.version_dir(pid, vid)

    mesh_solid = v_dir / "mesh" / "solid.brep"
    if mesh_solid.exists():
        return storage.relpath(mesh_solid)

    for art in artifacts:
        if art.kind != ArtifactKind.step:
            continue
        step_path = storage.resolve(art.path)
        brep = step_path.parent.parent / "_run" / "model.brep"  # <run>/exports/model.step
        if brep.exists():
            return storage.relpath(brep)

    it_root = v_dir / "iterations"
    if it_root.is_dir():
        run_dirs: list[Path] = []
        for it in it_root.iterdir():
            if not it.is_dir():
                continue
            if (it / "_run" / "model.brep").exists():
                run_dirs.append(it)
            run_dirs.extend(sub for sub in it.iterdir() if (sub / "_run" / "model.brep").exists())
        if run_dirs:
            newest = max(run_dirs, key=lambda d: (d / "_run" / "model.brep").stat().st_mtime)
            return storage.relpath(newest / "_run" / "model.brep")

    step = v_dir / "exports" / "model.step"
    if step.exists():
        return storage.relpath(step)
    return None


@_guarded
async def run_edit_feature_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Verified parametric edit of a recognised feature on the parent's B-Rep
    (M9: plain through-holes only). The child version already exists.

    Transactional: the executor gate must pass or the job/version fails with
    the refusal — a mutated shape is never persisted as a result.
    """
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        parent = (
            await db.get(Version, version.parent_id)
            if version is not None and version.parent_id
            else None
        )
        if job is None or version is None or parent is None:
            if job:
                job.status = JobStatus.failed
                job.finished_at = datetime.now(UTC)
                await _append_job_log(db, job, "edit_feature job: missing version/parent\n")
                await db.commit()
            return {"ok": False, "error": "missing version or parent"}

        if job.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return {"ok": False, "error": f"job already {job.status}"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        emit = make_emit(job_id)
        runtime_settings = await load_runtime_settings(db)
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            _merge_settings(
                (project.settings or {}) if project else {},
                version.options or (parent.options or {}),
            ),
            runtime_settings,
        )

        art_rows = (
            (await db.execute(select(Artifact).where(Artifact.version_id == parent.id)))
            .scalars()
            .all()
        )
        source_rel = _parent_source_model(list(art_rows), parent)
        if source_rel is None:
            version.status = VersionStatus.failed
            version.error = "parent has no B-Rep solid to edit"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(db, job, f"failed: {version.error}\n")
            await db.commit()
            return {"ok": False, "error": version.error}

        parameter = str(payload.get("parameter") or "diameter")
        await emit(
            "edit",
            1,
            f"feature edit: {parameter} -> {payload.get('value')} on feature "
            f"{payload.get('feature_id')}",
        )
        exec_result = await executor.edit_feature(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": 1,
                "source_brep": source_rel,
                "feature_id": payload.get("feature_id"),
                "parameter": parameter,
                "value": payload.get("value"),
                "spec": parent.spec or {},
                "settings": settings.model_dump(),
                "runtime_settings": runtime_settings.executor_values(),
            }
        )

        if not exec_result.get("ok"):
            refusal = exec_error(exec_result, "edit refused")
            was_cancelled = refusal == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = refusal
            version.issues = (version.issues or []) + [
                {
                    "source": "edit_feature",
                    "severity": "major",
                    "what": refusal,
                    "report": _json_safe(exec_result.get("report") or {}),
                }
            ]
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'refused'}: {refusal}\n"
            )
            await db.commit()
            await _publish(
                str(job_id),
                {
                    "type": "done",
                    "status": "cancelled" if was_cancelled else "failed",
                },
            )
            return {
                "ok": False,
                "error": refusal,
                "report": exec_result.get("report"),
                "features": exec_result.get("features"),
            }

        report = exec_result.get("report") or {}
        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, exec_result)
        for fmt, rel in (exec_result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        artifacts.update(_copy_mesh_artifacts_ctxless(version.project_id, version.id, 1, parent))
        register_artifacts(db, version.id, artifacts)

        version.gen_mode = parent.gen_mode
        version.resolved_mode = parent.resolved_mode or parent.gen_mode
        version.spec = parent.spec
        version.code = parent.code
        version.params = parent.params
        version.params_schema = parent.params_schema
        version.analysis = exec_result.get("analysis")
        version.dfm = exec_result.get("dfm")
        version.iterations = 1
        version.issues = (version.issues or []) + [
            {
                "source": "edit_feature",
                "severity": "info",
                "what": (
                    f"resized feature {payload.get('feature_id')} "
                    f"{parameter}: {report.get('old_diameter')} -> "
                    f"{report.get('new_diameter')} mm (gate passed)"
                ),
                "report": _json_safe(report),
            }
        ]
        version.status = VersionStatus.succeeded

        import json

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        if version.code:
            (v_dir / "model.py").write_text(version.code, encoding="utf-8")
        (v_dir / "params.json").write_text(
            json.dumps(version.params or {}, indent=2), encoding="utf-8"
        )
        (v_dir / "edit_feature_report.json").write_text(
            json.dumps(_json_safe(report), indent=2), encoding="utf-8"
        )

        job.status = JobStatus.succeeded
        job.finished_at = datetime.now(UTC)
        job.progress = 1.0
        await _append_job_log(
            db,
            job,
            f"done: {parameter} {report.get('old_diameter')} -> {report.get('new_diameter')} mm\n",
        )
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True, "report": _json_safe(report)}
