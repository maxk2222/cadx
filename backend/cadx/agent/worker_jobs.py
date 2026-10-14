"""Worker-side job runners: wire AgentCtx -> orchestrator -> DB.

Runs in the `worker` service (queue "agent"). CAD work is delegated to the
`cad` queue via QueueExecutor.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadx import events, storage
from cadx.agent.executor_client import QueueExecutor
from cadx.agent.image_client import ImagePort, make_image_client
from cadx.agent.mesh_client import MeshPort, make_mesh_client
from cadx.agent.orchestrator import RunOutcome, run_generation
from cadx.agent.trace import MemoryTrace, sanitize_messages
from cadx.agent.types import AgentCtx, InputAsset, ParentContext
from cadx.cad.mesh_ops import MESH_OPS_VERSION, MeshOp, MeshPostParams, merge_post_params
from cadx.db.models import (
    AgentTrace,
    Artifact,
    ArtifactKind,
    Asset,
    Job,
    JobStatus,
    JobType,
    LibraryPart,
    Project,
    ProjectSettings,
    Version,
    VersionStatus,
)
from cadx.db.session import session_scope
from cadx.llm.client import LLMClient
from cadx.llm.schemas import PartSpec
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings, load_runtime_settings

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


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
    `sub_progress=`, `sub_label=`) are merged into the published SSE event and
    rolled into the persisted Job.live snapshot (iteration budget, score
    history, per-step timings, token usage) so reconnects restore the full UI.
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
            if job is not None and job.status in (JobStatus.queued, JobStatus.running):
                job.status = JobStatus.failed
                job.finished_at = datetime.now(UTC)
                await _append_job_log(db, job, f"failed: {error}\n")
            if version is not None and version.status in (
                VersionStatus.queued,
                VersionStatus.running,
            ):
                version.status = VersionStatus.failed
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
                        parsed=kwargs["parsed"],
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
        llm = _make_llm(runtime_settings)
        emit = make_emit(job_id, tokens_fn=lambda: dict(getattr(llm, "total_usage", {}) or {}))
        executor = QueueExecutor(ctx.get("arq_redis"), runtime_settings=runtime_settings)
        trace = DbTrace(job_id)
        mesh = _make_mesh(runtime_settings)
        image = _make_image(runtime_settings)

        project = await db.get(Project, version.project_id)
        merged = _merge_settings((project.settings or {}) if project else {}, version.options or {})
        explicit_max_iterations = merged.max_iterations is not None
        settings = _fill_settings(merged, runtime_settings)

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
            runtime_settings=runtime_settings,
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
            outcome = RunOutcome(status="failed", error=f"agent crashed: {e}")

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
        if (
            job is None
            or version is None
            or parent is None
            or (parent_mode != "mesh" and not parent.code)
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
            parent_v_dir = storage.version_dir(parent.project_id, parent.id)
            solid = parent_v_dir / "mesh" / "solid.brep"
            exec_result = await executor.execute(
                {
                    "job_id": str(job_id),
                    "project_id": str(version.project_id),
                    "version_id": str(version.id),
                    "iteration": 1,
                    "code": parent.code,
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
            version.error = exec_result.get("error") or "execution failed"
            job.status = JobStatus.cancelled if was_cancelled else JobStatus.failed
            job.finished_at = datetime.now(UTC)
            await _append_job_log(
                db, job, f"{'cancelled' if was_cancelled else 'failed'}: {version.error}\n"
            )
            await db.commit()
            return {"ok": False, "error": version.error}

        # copy artifacts to version root

        artifacts = _copy_best_artifacts_ctxless(version.project_id, version.id, 1, exec_result)
        for fmt, rel in (exec_result.get("exports") or {}).items():
            if rel:
                artifacts[f"export:{fmt}"] = rel
        tessellated = parent_mode == "mesh"

        version.gen_mode = parent_mode
        version.resolved_mode = parent_mode
        if parent_mode == "mesh":
            # merged post params live under mesh.post in the exec result
            version.params = (exec_result.get("mesh") or {}).get("post") or version.params
        else:
            version.params = exec_result.get("params_used") or version.params
        version.params_schema = parent.params_schema
        version.code = parent.code
        version.analysis = exec_result.get("analysis")
        version.dfm = exec_result.get("dfm")
        version.iterations = 1
        version.status = VersionStatus.succeeded

        import json

        if parent_mode == "mesh":
            # mesh params child: inherit mesh provenance + copy intermediates;
            # overlay the fresh run's mesh info (post, triangles, ops_report)
            version.mesh = dict(parent.mesh or {})
            version.mesh.update(exec_result.get("mesh") or {})
            version.mesh["post"] = version.params
            version.mesh["from_params_of"] = str(parent.id)
            artifacts.update(
                _copy_mesh_artifacts_ctxless(version.project_id, version.id, 1, parent)
            )
            m_dir = storage.ensure(storage.version_dir(version.project_id, version.id) / "mesh")
            (m_dir / "mesh.json").write_text(json.dumps(version.mesh, indent=2), encoding="utf-8")
        register_artifacts(db, version.id, artifacts, tessellated_step=tessellated)

        v_dir = storage.ensure(storage.version_dir(version.project_id, version.id))
        if version.code:
            (v_dir / "model.py").write_text(version.code, encoding="utf-8")
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
    actx = AgentCtx(
        llm=_make_llm(runtime_settings),
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
        runtime_settings=runtime_settings,
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


@_guarded
async def run_optimize_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
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
                }
            )

        if not result.get("ok"):
            was_cancelled = (result.get("error") or "") == "cancelled"
            version.status = VersionStatus.cancelled if was_cancelled else VersionStatus.failed
            version.error = result.get("error") or "optimization failed"
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
        n_bodies = topopt.get("n_components") or 1
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
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
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

        density_rel = ((parent.mesh or {}).get("topopt") or {}).get("density_relpath")
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
            }
        )

        if not result.get("ok"):
            version.status = VersionStatus.failed
            version.error = result.get("error") or "resample failed"
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
        n_bodies = topopt.get("n_components") or 1
        version.status = (
            VersionStatus.needs_review
            if worst_sf is None or worst_sf < 1.0 or n_bodies > 1
            else VersionStatus.succeeded
        )
        register_artifacts(db, version.id, artifacts, tessellated_step=False)
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
            version.error = exec_result.get("error") or "mesh edit failed"
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
            version.error = result.get("error") or "FEM failed"
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
        await emit("fem", 1, f"done: worst SF {worst_sf}")
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


def _make_mesh(runtime: RuntimeSettings) -> MeshPort | None:
    return make_mesh_client(
        base_url=runtime.mesh_base_url,
        timeout_s=runtime.mesh_timeout_s,
    )


def _make_image(runtime: RuntimeSettings) -> ImagePort | None:
    return make_image_client(
        base_url=runtime.mesh_base_url,  # same sidecar hosts the image adapters
        timeout_s=runtime.image_timeout_s,
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


def _make_llm(runtime: RuntimeSettings) -> LLMClient:
    return LLMClient(runtime_settings=runtime)


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
        version.spec = outcome.spec
        await db.commit()
        await _publish(str(job.id), {"type": "question", "questions": outcome.questions})
        return

    version.spec = outcome.spec
    version.plan = outcome.plan
    version.code = outcome.code
    version.params = outcome.params
    version.params_schema = outcome.params_schema
    version.analysis = outcome.analysis
    version.critique = outcome.critique
    version.dfm = outcome.dfm
    version.bom = outcome.bom
    version.issues = outcome.issues or None
    version.mesh = outcome.mesh
    version.fem = outcome.fem or None  # {} would be truthy-but-shapeless in the UI
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
            json.dumps(outcome.params or {}, indent=2), encoding="utf-8"
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
