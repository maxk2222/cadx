"""Worker-side job runners: wire AgentCtx -> orchestrator -> DB.

Runs in the `worker` service (queue "agent"). CAD work is delegated to the
`cad` queue via QueueExecutor.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from cadagent import events, storage
from cadagent.agent.executor_client import QueueExecutor
from cadagent.agent.mesh_client import MeshPort, make_mesh_client
from cadagent.agent.orchestrator import RunOutcome, run_generation
from cadagent.agent.trace import MemoryTrace, sanitize_messages
from cadagent.agent.types import AgentCtx, InputAsset, ParentContext
from cadagent.cad.mesh_ops import MeshPostParams, merge_post_params
from cadagent.config import get_settings
from cadagent.db.models import (
    AgentTrace,
    Artifact,
    ArtifactKind,
    Asset,
    Job,
    JobStatus,
    JobType,
    Project,
    ProjectSettings,
    Version,
    VersionStatus,
)
from cadagent.db.session import session_scope
from cadagent.llm.client import LLMClient
from cadagent.llm.schemas import PartSpec
from cadagent.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _u(s: str | uuid.UUID) -> uuid.UUID:
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))


async def _publish(job_id: str, event: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        await events.publish(job_id, event)


class DbTrace(MemoryTrace):
    """Memory trace that also persists each record as an AgentTrace row."""

    def __init__(self, session: AsyncSession, job_id: uuid.UUID) -> None:
        super().__init__()
        self._db = session
        self._job_id = job_id

    async def record(self, **kwargs: Any) -> None:
        await super().record(**kwargs)
        self._db.add(
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
        await self._db.commit()


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
        )
        for a in assets
    ]


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
    )


# --------------------------------------------------------------------------- #
# job runners                                                                  #
# --------------------------------------------------------------------------- #


async def run_agent_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """arq function on queue 'agent' for types generate|edit|resume."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False, "error": "job or version not found"}
        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def emit(step: str, iteration: int, message: str) -> None:
            async with session_scope() as s2:
                j = await s2.get(Job, job_id)
                if j is not None:
                    j.step = step
                    j.iteration = iteration
                    j.progress = min(0.95, j.progress + 0.05)
                    j.log = (j.log or "") + f"[{step}:{iteration}] {message}\n"
                    await s2.commit()
            with contextlib.suppress(Exception):
                await events.publish(
                    str(job_id),
                    {"type": "step", "step": step, "iteration": iteration, "message": message},
                )

        async def check_cancel() -> bool:
            async with session_scope() as s3:
                j = await s3.get(Job, job_id)
                return bool(j and j.cancel_requested)

        # llm config (AppSetting overrides env)
        llm = await _make_llm(db)
        executor = QueueExecutor(ctx.get("arq_redis"))
        trace = DbTrace(db, job_id)
        mesh = await _make_mesh(db)

        project = await db.get(Project, version.project_id)
        settings = ProjectSettings(**(project.settings or {})) if project else ProjectSettings()
        settings = _fill_settings(settings)

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

        parent = await db.get(Version, version.parent_id) if version.parent_id else None
        parent_ctx = _parent_ctx(version, parent)

        mode = "edit" if job.type == JobType.edit else "generate"
        gen_mode = version.gen_mode or settings.gen_mode or s_default_gen_mode()
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
        mesh_available = False
        if mesh_client is not None:
            try:
                mesh_available = (await mesh_client.health()).ok
            except Exception:
                mesh_available = False

        actx = AgentCtx(
            llm=llm,
            executor=executor,
            trace=trace,
            emit=emit,
            check_cancel=check_cancel,
            job_id=str(job_id),
            project_id=version.project_id,
            version_id=version.id,
            mode=mode,
            prompt=version.prompt,
            settings=settings,
            assets=_input_assets(assets),
            parent=parent_ctx,
            gen_mode=gen_mode,
            mesh=mesh_client,
            mesh_available=mesh_available,
        )
        if job.type == JobType.resume and job.resume_state and job.resume_state.get("spec"):
            actx.spec = PartSpec(**job.resume_state["spec"])

        try:
            outcome = await run_generation(actx)
        except Exception as e:
            log.exception("agent.job_crashed", job=str(job_id))
            outcome = RunOutcome(status="failed", error=f"agent crashed: {e}")

        await _persist_outcome(db, job, version, outcome, emit)
        return {"ok": outcome.status in ("succeeded", "needs_review"), "status": outcome.status}


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
                job.log = (job.log or "") + "params job: missing parent code/mesh\n"
                await db.commit()
            return {"ok": False, "error": "missing parent code/mesh"}

        job.status = JobStatus.running
        job.started_at = datetime.now(UTC)
        version.status = VersionStatus.running
        await db.commit()

        async def emit(step: str, iteration: int, message: str) -> None:
            async with session_scope() as s2:
                j = await s2.get(Job, job_id)
                if j is not None:
                    j.step = step
                    j.log = (j.log or "") + f"[{step}] {message}\n"
                    await s2.commit()
            with contextlib.suppress(Exception):
                await events.publish(
                    str(job_id),
                    {"type": "step", "step": step, "iteration": iteration, "message": message},
                )

        executor = QueueExecutor(ctx.get("arq_redis"))
        project = await db.get(Project, version.project_id)
        settings = _fill_settings(
            ProjectSettings(**(project.settings or {})) if project else ProjectSettings()
        )

        if parent_mode == "mesh":
            exec_result = await _run_mesh_params(
                executor, job_id, version, parent, settings, payload.get("params") or {}, emit
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
                    "full_exports": True,
                    "name": "model",
                    "mesh_solid": str(solid)
                    if solid.exists() and parent_mode == "hybrid"
                    else None,
                    "mesh_bbox": (parent.mesh or {}).get("bbox_size"),
                }
            )

        if not exec_result.get("ok"):
            version.status = VersionStatus.failed
            version.error = exec_result.get("error") or "execution failed"
            job.status = JobStatus.failed
            job.finished_at = datetime.now(UTC)
            job.log = (job.log or "") + f"failed: {version.error}\n"
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
            # mesh params child: inherit mesh provenance + copy intermediates
            version.mesh = dict(parent.mesh or {})
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
        job.log = (job.log or "") + "done\n"
        await db.commit()
        await _publish(str(job_id), {"type": "done", "status": "succeeded"})
        return {"ok": True}


async def _run_mesh_params(
    executor: QueueExecutor,
    job_id: uuid.UUID,
    version: Version,
    parent: Version,
    settings: ProjectSettings,
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
            "spec": parent.spec,
            "settings": settings.model_dump(),
            "full_exports": True,
            "name": "model",
        }
    )


async def run_reexport_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Re-export formats from the best iteration BREP."""
    job_id = _u(payload["job_id"])
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        version = await db.get(Version, _u(payload["version_id"]))
        if job is None or version is None:
            return {"ok": False}
        job.status = JobStatus.running
        await db.commit()
        executor = QueueExecutor(ctx.get("arq_redis"))
        result = await executor.export_all(
            {
                "job_id": str(job_id),
                "project_id": str(version.project_id),
                "version_id": str(version.id),
                "iteration": payload.get("iteration") or 1,
                "name": "model",
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


def _fill_settings(settings: ProjectSettings) -> ProjectSettings:
    s = get_settings()
    for k, default in (
        ("clarification_mode", s.agent_clarification_mode),
        ("max_iterations", s.agent_max_iterations),
        ("max_repairs", s.agent_max_repairs),
        ("pass_score", s.agent_pass_score),
        ("dim_tolerance_pct", s.agent_dim_tolerance_pct),
        ("dim_tolerance_mm", s.agent_dim_tolerance_mm),
    ):
        if getattr(settings, k) is None:
            setattr(settings, k, default)
    return settings


def s_default_gen_mode() -> str:
    return get_settings().agent_default_gen_mode


async def _make_mesh(db: AsyncSession) -> MeshPort | None:
    """Mesh generator client honoring AppSetting 'mesh' overrides + env."""
    from cadagent.db.models import AppSetting

    base_url: str | None = None
    try:
        row = await db.get(AppSetting, "mesh")
        if row and row.value:
            base_url = row.value.get("mesh_base_url") or None
    except Exception:
        pass
    return make_mesh_client(base_url=base_url)


async def _make_llm(db: AsyncSession) -> Any:
    """LLMClient with optional AppSetting 'llm' overrides (key stays in env).

    LLM_FAKE_SCRIPT (path to JSON script or inline JSON) swaps in FakeLLM —
    used for deterministic end-to-end tests without a real provider.
    """
    import json as _json
    from pathlib import Path

    from cadagent.db.models import AppSetting
    from cadagent.llm.fake import FakeLLM

    fake = get_settings().llm_fake_script
    if fake:
        p = Path(fake)
        try:
            text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except OSError:
            text = fake
        return FakeLLM(_json.loads(text))

    row = await db.get(AppSetting, "llm")
    kwargs: dict[str, Any] = {}
    if row and row.value:
        for src, dst in (
            ("llm_base_url", "base_url"),
            ("llm_model_vision", "model_vision"),
            ("llm_model_code", "model_code"),
            ("llm_json_mode", "json_mode"),
        ):
            if row.value.get(src):
                kwargs[dst] = row.value[src]
    return LLMClient(**kwargs)


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
    version.mesh = outcome.mesh
    version.resolved_mode = outcome.resolved_mode
    version.score = outcome.score
    version.iterations = outcome.iterations
    version.usage = outcome.usage
    version.error = outcome.error
    tessellated = outcome.resolved_mode == "mesh" or bool(
        outcome.mesh and outcome.mesh.get("step_tessellated")
    )
    register_artifacts(db, version.id, outcome.artifacts, tessellated_step=tessellated)
    if outcome.mode_note:
        job.log = (job.log or "") + f"mode: {outcome.mode_note}\n"

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
    job.log = (job.log or "") + f"final: {outcome.status}\n"
    await db.commit()
    await _publish(str(job.id), {"type": "done", "status": outcome.status, "score": outcome.score})
