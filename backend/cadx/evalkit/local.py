"""Local driver — quick dev runs without API server, Redis or real DB writes.

Bootstrap redirects this process to a scratch sqlite DB + scratch data dir
(DATABASE_URL/DATA_DIR env + engine rebind), seeds it with a runtime
settings snapshot (read-only from the real DB when reachable) and an
ephemeral eval project. Stage dispatch then runs the *production* path:
``cadx.spawn.spawn_*`` creates real Version/Job rows (with a no-op enqueue
function) and ``worker_jobs.run_*_job`` executes them — with
``QueueExecutor`` patched to ``InProcessExecutor`` so no Redis/queue is
needed. All persistence lands in the scratch tree under ``evals/scratch/``
and is never baselined.

Keep cadx imports function-local: bootstrap must run before the first
``cadx.config.get_settings()`` / ``cadx.db.session`` evaluation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from cadx.evalkit.models import StageOutcome
from cadx.evalkit.suite import ASSET_STAGES, EvalStage
from cadx.logging import get_logger

log = get_logger(__name__)

_TERMINAL = {"succeeded", "failed", "cancelled"}


class LocalCtx:
    """Per-run state for the local driver."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.project_id: uuid.UUID = uuid.uuid4()
        self.project_name: str = ""
        self.parent_version_id: uuid.UUID | None = None
        self.stage_assets: dict[str, list[str]] = {}  # stage id -> scratch Asset uuids
        self.stage_files: dict[str, list[str]] = {}  # stage id -> generated file paths
        self.case_assets: list[str] = []
        self.live_log: Path = out_dir / "live.jsonl"


# ---------------------------------------------------------------------------
# bootstrap — must run before heavy cadx imports bind engine/settings
# ---------------------------------------------------------------------------


def _rebind_session(url: str) -> None:
    """Rebind cadx.db.session's module engine — safe even if it was already
    imported (unit tests, same-process reuse)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    import cadx.db.session as dbs

    eng = dbs.make_engine(url)
    dbs.engine = eng
    dbs.async_session = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)


async def bootstrap(out_dir: Path) -> str | None:
    """Redirect this process at the scratch DB + data dir.

    Returns the previously configured DATABASE_URL (for the settings
    snapshot). Caller must invoke this before touching cadx modules that
    bind ``get_settings()`` — evalkit.local keeps those imports
    function-local for that reason.
    """
    prev_url = os.environ.get("DATABASE_URL")
    # multi-case runs re-enter bootstrap after the first case already pointed
    # DATABASE_URL at its scratch sqlite — pin the real URL once so every
    # case snapshots the live runtime settings instead of falling back to
    # code defaults (which then dial e.g. OpenAI without a key)
    if prev_url and not prev_url.startswith("sqlite"):
        os.environ.setdefault("CADX_REAL_DATABASE_URL", prev_url)
    else:
        prev_url = os.environ.get("CADX_REAL_DATABASE_URL")
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{out_dir / 'eval.db'}"
    os.environ["DATA_DIR"] = str(out_dir / "data")
    from cadx.config import get_settings

    get_settings.cache_clear()
    url = os.environ["DATABASE_URL"]
    _rebind_session(url)
    from cadx.db.session import create_all

    await create_all()
    return prev_url


async def snapshot_runtime(real_url: str | None) -> Any:
    """Read-only RuntimeSettings snapshot from the real DB when reachable;
    code defaults otherwise. Uses a standalone engine — never the module one."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from cadx.db.session import make_engine
    from cadx.runtime_config import RuntimeSettings, load_runtime_settings

    url = (
        real_url
        or os.environ.get("CADX_REAL_DATABASE_URL")
        or ("postgresql://cadx:cadx@localhost:5432/cadx")
    )
    if url.startswith("sqlite"):
        return RuntimeSettings()
    eng = make_engine(url)
    try:
        sm = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
        async with sm() as db:
            return await load_runtime_settings(db)
    except Exception as e:
        log.info("eval.local.runtime_snapshot_failed", error=str(e)[:120])
        return RuntimeSettings()
    finally:
        await eng.dispose()


async def seed_project(ctx: LocalCtx, name: str, settings: dict[str, Any], runtime: Any) -> None:
    """Insert the ephemeral eval project + seeded runtime row in scratch DB."""
    from cadx import storage
    from cadx.db.models import AppSetting, Project
    from cadx.db.session import session_scope

    async with session_scope() as db:
        db.add(AppSetting(key="runtime", value=runtime.to_storage()))
        merged = {"_eval": True, "clarification_mode": "assume", **(settings or {})}
        db.add(
            Project(
                id=ctx.project_id,
                name=name,
                description="evalkit local scratch project",
                settings=merged,
            )
        )
        await db.commit()
    ctx.project_name = name
    storage.project_dir(ctx.project_id).mkdir(parents=True, exist_ok=True)


async def stage_assets(ctx: LocalCtx, paths: list[Path]) -> list[str]:
    """Create scratch Asset rows + files for input images; returns ids."""
    from cadx import storage
    from cadx.db.models import Asset, AssetKind
    from cadx.db.session import session_scope

    ids: list[str] = []
    async with session_scope() as db:
        for p in paths:
            aid = uuid.uuid4()
            dest = storage.asset_path(ctx.project_id, aid, p.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            db.add(
                Asset(
                    id=aid,
                    project_id=ctx.project_id,
                    kind=AssetKind.image,
                    filename=p.name,
                    path=storage.relpath(dest),
                    mime="image/png",
                )
            )
            ids.append(str(aid))
        await db.commit()
    return ids


async def import_parent(ctx: LocalCtx, src: Path) -> uuid.UUID:
    """`--from DIR`: synthesize a succeeded parent version from a directory
    containing an STL/BREP (e.g. a previous scratch version dir)."""
    from cadx import storage
    from cadx.db.models import Version, VersionKind, VersionStatus
    from cadx.db.session import session_scope

    vid = uuid.uuid4()
    vdir = storage.version_dir(ctx.project_id, vid)
    (vdir / "exports").mkdir(parents=True, exist_ok=True)
    mesh_rel = None
    stls = await asyncio.to_thread(lambda: sorted(src.rglob("*.stl")))
    breps = await asyncio.to_thread(lambda: sorted(src.rglob("*.brep")))
    for cand in (
        src / "exports" / "model.stl",
        src / "model.stl",
        (stls[0] if stls else None),
        src / "solid.brep",
        (breps[0] if breps else None),
    ):
        if cand and await asyncio.to_thread(Path(cand).exists):
            dest = vdir / "exports" / Path(cand).name
            await asyncio.to_thread(shutil.copy2, cand, dest)
            if dest.suffix == ".stl" and mesh_rel is None:
                mesh_rel = storage.relpath(dest)
    if mesh_rel is None:
        raise ValueError(f"--from {src}: no .stl/.brep found")
    # the persisted spec (written next to model.py/params.json on finalize)
    # carries declared features — downstream refine/spec_features paths then
    # re-cut to nominal instead of degrading to scanned-anchor reams
    spec: dict[str, Any] | None = None
    for cand in (src / "spec.json", src.parent / "spec.json"):
        if cand.exists():
            try:
                spec = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                spec = None
            break
    if spec is None:
        specs = await asyncio.to_thread(lambda: sorted(src.rglob("spec.json")))
        if specs:
            try:
                spec = json.loads(specs[0].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                spec = None
    async with session_scope() as db:
        from cadx.parts import create_part

        part = await create_part(db, ctx.project_id, "imported")
        v = Version(
            id=vid,
            project_id=ctx.project_id,
            part_id=part.id,
            parent_id=None,
            number=1,
            kind=VersionKind.generate,
            prompt=f"imported parent ({src.name})",
            gen_mode="cad",
            resolved_mode="cad",
            status=VersionStatus.succeeded,
            iterations=1,
            spec=spec,
        )
        db.add(v)
        await db.commit()
    ctx.parent_version_id = vid
    return vid


# ---------------------------------------------------------------------------
# dispatch — spawn (with no-op enqueue) + run_*_job against scratch DB
# ---------------------------------------------------------------------------


async def _noop_enqueue(func: str, payload: dict[str, Any]) -> None:
    return None


_JOB_FNS = {
    "generate": "run_agent_job",
    "edit": "run_agent_job",
    "params": "run_params_job",
    "edit_feature": "run_edit_feature_job",
    "optimize": "run_optimize_job",
    "resample": "run_topopt_resample_job",
    "refine": "run_refine_job",
    "organic": "run_organic_job",
    "fem": "run_fem_job",
    "reexport": "run_reexport_job",
    "mesh_edit": "run_mesh_edit_job",
    "meshgen": "run_meshgen_job",
}


async def _spawn(stage: EvalStage, ctx: LocalCtx) -> tuple[uuid.UUID, uuid.UUID | None]:
    """Create the Version+Job rows in scratch DB — production spawn path
    with the enqueue replaced by a no-op. Returns (version_id, job_id)."""
    from cadx.db.models import Job, JobType, Version
    from cadx.db.session import session_scope
    from cadx.spawn import (
        spawn_edit,
        spawn_edit_feature,
        spawn_fem,
        spawn_generate,
        spawn_mesh_edit,
        spawn_meshgen,
        spawn_optimize,
        spawn_organic,
        spawn_params,
        spawn_refine,
        spawn_resample,
    )

    t = stage.type
    async with session_scope() as db:
        parent = await db.get(Version, ctx.parent_version_id) if ctx.parent_version_id else None
        if t not in ("generate", "meshgen") and parent is None:
            raise ValueError(f"stage {stage.id}: no parent version resolved")
        asset_ids = [uuid.UUID(a) for a in ctx.case_assets]
        if stage.assets_from:
            asset_ids += [uuid.UUID(a) for a in ctx.stage_assets.get(stage.assets_from, [])]

        if t == "generate":
            from cadx.db.models import Project

            project = await db.get(Project, ctx.project_id)
            v, j = await spawn_generate(
                db,
                project,
                stage.prompt,
                gen_mode=stage.mode or "auto",
                asset_ids=asset_ids,
                options=stage.options or None,
                enqueue_fn=_noop_enqueue,
            )
        elif t == "edit":
            v, j = await spawn_edit(
                db,
                parent,
                stage.prompt,
                gen_mode=stage.mode,
                asset_ids=asset_ids,
                options=stage.options or None,
                enqueue_fn=_noop_enqueue,
            )
        elif t == "params":
            v, j = await spawn_params(db, parent, stage.params or {}, enqueue_fn=_noop_enqueue)
        elif t == "edit_feature":
            f = stage.feature or {}
            v, j = await spawn_edit_feature(
                db,
                parent,
                feature_id=f.get("feature_id", f.get("id", 0)),
                parameter=str(f.get("parameter", "diameter")),
                value=float(f.get("value", f.get("dia", 0)) or 0),
                enqueue_fn=_noop_enqueue,
            )
        elif t == "optimize":
            v, j = await spawn_optimize(
                db,
                parent,
                load_cases=stage.load_cases or None,
                optimize=stage.optimize or None,
                material=stage.material or None,
                allow_reopt=bool(stage.options.get("allow_reopt")),
                enqueue_fn=_noop_enqueue,
            )
        elif t == "resample":
            r = stage.resample or {}
            v, j = await spawn_resample(
                db,
                parent,
                density_threshold=r.get("density_threshold"),
                min_member_mm=r.get("min_member_mm"),
                material=stage.material or None,
                load_cases=stage.load_cases or None,
                keep_regions=stage.keep_regions or None,
                enqueue_fn=_noop_enqueue,
            )
        elif t == "refine":
            v, j = await spawn_refine(
                db,
                parent,
                features=stage.features or None,
                load_cases=stage.load_cases or None,
                keep_regions=stage.keep_regions or None,
                material=stage.material or None,
                voxel_mm=stage.voxel_mm,
                restore_planes=stage.restore_planes,
                enqueue_fn=_noop_enqueue,
            )
        elif t == "organic":
            v, j = await spawn_organic(db, parent, stage.organic or {}, enqueue_fn=_noop_enqueue)
        elif t == "fem":
            j = await spawn_fem(
                db,
                parent,
                load_cases=stage.load_cases or None,
                keep_regions=stage.keep_regions or None,
                material=stage.material or None,
                enqueue_fn=_noop_enqueue,
            )
            return parent.id, j.id
        elif t == "reexport":
            j = Job(version_id=parent.id, type=JobType.reexport)
            db.add(j)
            await db.commit()
            return parent.id, j.id
        elif t == "mesh_edit":
            v, j = await spawn_mesh_edit(
                db,
                parent,
                ops=stage.ops,
                post=stage.post,
                note=stage.prompt or f"eval {stage.id}",
                enqueue_fn=_noop_enqueue,
            )
        elif t == "meshgen":
            from cadx.db.models import Project

            project = await db.get(Project, ctx.project_id)
            ids = asset_ids or [
                uuid.UUID(a)
                for a in (
                    ctx.stage_assets.get(stage.assets_from or "", [])
                    or ([stage.reference] if stage.reference else [])
                )
                if a
            ]
            v, j = await spawn_meshgen(
                db,
                project,
                asset_ids=ids,
                subject=stage.subject or stage.prompt,
                model=stage.model,
                seed=stage.seed,
                post=stage.post,
                prompt=stage.prompt,
                enqueue_fn=_noop_enqueue,
            )
        else:
            raise ValueError(f"local driver: unsupported stage type {t}")
        return v.id, j.id


def _mesh_artifact_files(v: Any, arts: list[Any]) -> list[str]:
    """Mesh files authoritative-first — the committed stl_relpath, then the
    version-level export, then the latest iteration dir, then raw artifact
    rows (registered during intermediate runs). Mesh gates must measure the
    FINAL committed mesh, not the first STL that happened to be registered."""
    from cadx import storage

    cands: list[Path] = []
    if v is not None:
        v_dir = storage.version_dir(v.project_id, v.id)
        vmesh = getattr(v, "mesh", None)
        stl_rel = (vmesh or {}).get("stl_relpath") if isinstance(vmesh, dict) else None
        if stl_rel:
            with contextlib.suppress(ValueError):
                cands.append(storage.resolve(stl_rel))
        cands.append(v_dir / "exports" / "model.stl")
        it_root = v_dir / "iterations"
        if it_root.exists():
            for it in sorted(
                (d for d in it_root.iterdir() if d.is_dir() and d.name.isdigit()),
                key=lambda d: int(d.name),
                reverse=True,
            ):
                cands.append(it / "exports" / "model.stl")
                cands.append(it / "_run" / "model.stl")
        cands.append(v_dir / "mesh" / "mesh_clean.stl")
    for a in arts:
        if str(a.kind) in ("stl", "glb", "3mf", "obj"):
            try:
                cands.append(storage.resolve(a.path))
            except ValueError:
                continue
    for p in cands:
        try:
            if p.exists():
                return [str(p)]
        except OSError:
            continue
    return []


async def _read_outcome(
    vid: uuid.UUID, jid: uuid.UUID | None, wall: float, *, same_version: bool
) -> StageOutcome:
    """Read back the version/job/trace rows the job function wrote."""
    from sqlmodel import select

    from cadx import storage
    from cadx.db.models import AgentTrace, Artifact, Job, Version
    from cadx.db.session import session_scope

    async with session_scope() as db:
        v = await db.get(Version, vid)
        j = await db.get(Job, jid) if jid else None
        arts = (
            (await db.execute(select(Artifact).where(Artifact.version_id == vid))).scalars().all()
        )
        records = []
        if jid:
            rows = (
                (
                    await db.execute(
                        select(AgentTrace)
                        .where(AgentTrace.job_id == jid)
                        .order_by(AgentTrace.created_at)
                    )
                )
                .scalars()
                .all()
            )
            records = [
                {
                    "step": r.step,
                    "iteration": r.iteration,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency_ms": r.latency_ms,
                    "parsed": r.parsed,
                }
                for r in rows
            ]
    artifact_files: list[str] = _mesh_artifact_files(v, arts)
    artifacts: dict[str, str] = {}
    for a in arts:
        key = f"{a.kind}|{a.view or ''}"
        artifacts[key] = a.path
    # fall back to the working STL when no mesh artifact row exists
    if not artifact_files:
        rel = storage.best_mesh_relpath(
            v.project_id if v else "", vid, (v.iterations if v else 1) or 1
        )
        if rel:
            try:
                p = storage.resolve(rel)
                if p.exists():
                    artifact_files.append(str(p))
            except ValueError:
                pass
    status = str(j.status) if j and j.status else (str(v.status) if v else "failed")
    if status == "queued" and j is None:
        status = str(v.status) if v else "failed"
    return StageOutcome(
        status=status,
        version_id=str(vid),
        job_id=str(jid) if jid else None,
        score=v.score if v else None,
        iterations=int(v.iterations or 0) if v else 0,
        resolved_mode=v.resolved_mode if v else None,
        mode_note="",
        analysis=v.analysis if v else None,
        features=(v.mesh or {}).get("features") if v and isinstance(v.mesh, dict) else None,
        spec=v.spec if v else None,
        plan=v.plan if v else None,
        code=v.code if v else None,
        fem=v.fem if v else None,
        mesh=v.mesh if v else None,
        issues=v.issues or [] if v else [],
        dfm=v.dfm if v else None,
        critique=v.critique if v else None,
        usage=v.usage or {} if v else {},
        artifacts=artifacts,
        artifact_files=artifact_files,
        trace_records=records,
        live=j.live if j else None,
        asset_ids=[str(x) for x in (v.asset_ids or [])] if v else [],
        wall_s=wall,
        error=(v.error if v else None) or (j.live or {}).get("error") if j else None,
    )


class LocalDriver:
    """Runs stages against the scratch DB — production spawn + job path."""

    name = "local"

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose
        self._patched: Any = None
        self.exec_stats: dict[str, float] = {"n": 0, "s": 0.0}

    def patch_executor(self) -> None:
        """worker_jobs constructs QueueExecutor(arq_redis) — replace the name
        it resolves so the in-process executor runs instead, wrapped in a
        call counter for the execs/exec_s metrics."""
        import cadx.agent.worker_jobs as wj
        from cadx.agent.executor_client import InProcessExecutor

        if self._patched is None:
            self._patched = wj.QueueExecutor
            stats = self.exec_stats

            class _CountingExecutor(InProcessExecutor):
                async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
                    stats["n"] += 1
                    t = time.monotonic()
                    try:
                        return await super().execute(payload)
                    finally:
                        stats["s"] += time.monotonic() - t

            def _inprocess(*_a: Any, **_k: Any) -> InProcessExecutor:
                return _CountingExecutor()

            wj.QueueExecutor = _inprocess  # type: ignore[assignment]

    def unpatch_executor(self) -> None:
        if self._patched is not None:
            import cadx.agent.worker_jobs as wj

            wj.QueueExecutor = self._patched  # type: ignore[assignment]
            self._patched = None
            self.exec_stats = {"n": 0, "s": 0.0}

    async def dispatch(self, stage: EvalStage, ctx: LocalCtx) -> StageOutcome:
        t0 = time.monotonic()
        if stage.type in ASSET_STAGES:
            return await self._dispatch_image(stage, ctx, t0)
        vid, jid = await _spawn(stage, ctx)
        func = _JOB_FNS[stage.type]
        import cadx.agent.worker_jobs as wj

        run_fn = getattr(wj, func)
        # spawn_* stores the verbatim {func,args} envelope on job.payload —
        # replay it exactly like arq would, so job functions that read request
        # fields (load_cases/optimize/material/...) see the full payload, not
        # just job_id+version_id
        run_args: dict[str, Any] = {"job_id": str(jid), "version_id": str(vid)}
        if jid is not None:
            from cadx.db.models import Job
            from cadx.db.session import session_scope

            async with session_scope() as db:
                env = await db.get(Job, jid)
                args = (env.payload or {}).get("args") if env else None
                if isinstance(args, dict) and args.get("job_id"):
                    run_args = args
        if self.verbose:
            log.info("eval.local.job", stage=stage.id, func=func, version=str(vid)[:8])
        try:
            await asyncio.wait_for(
                run_fn({"arq_redis": None}, run_args),
                timeout=stage.timeout_s,
            )
        except TimeoutError:
            # mark job+version failed in scratch so read-back is consistent
            from cadx.db.models import Job, JobStatus, Version, VersionStatus
            from cadx.db.session import session_scope

            async with session_scope() as db:
                j = await db.get(Job, jid)
                v = await db.get(Version, vid)
                if j:
                    j.status = JobStatus.failed
                if v:
                    v.status = VersionStatus.failed
                    v.error = f"eval local timeout after {stage.timeout_s}s"
                await db.commit()
        wall = time.monotonic() - t0
        out = await _read_outcome(vid, jid, wall, same_version=stage.type in ("fem", "reexport"))
        out.extras["execs"] = self.exec_stats["n"]
        out.extras["exec_s"] = self.exec_stats["s"]
        if stage.type not in ("fem", "reexport"):
            ctx.parent_version_id = vid
        # dump trace records for the archive
        if out.trace_records and ctx.out_dir:
            tpath = ctx.out_dir / f"trace.{str(vid)[:8]}.jsonl"
            with tpath.open("a", encoding="utf-8") as fh:
                for r in out.trace_records:
                    fh.write(json.dumps(r, default=str) + "\n")
        return out

    async def _dispatch_image(self, stage: EvalStage, ctx: LocalCtx, t0: float) -> StageOutcome:
        """imagegen/imageedit via the image sidecar client — no job row."""
        from cadx.agent.image_client import ImageGenRequest, make_image_client
        from cadx.db.session import session_scope
        from cadx.runtime_config import load_runtime_settings

        async with session_scope() as db:
            runtime = await load_runtime_settings(db)
        image = make_image_client(runtime_settings=runtime)
        if image is None:
            return StageOutcome(
                status="failed", wall_s=time.monotonic() - t0, error="no image client configured"
            )
        health = await image.health()
        if not health.ok:
            return StageOutcome(
                status="failed",
                wall_s=time.monotonic() - t0,
                error=f"image sidecar unreachable: {getattr(health, 'error', '')}",
            )
        out_dir = ctx.out_dir / "data" / "gen_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        # imageedit: init_image from an explicit path reference or the last
        # image stage's output file
        init = None
        ref = stage.reference
        if ref and ref != "prev-asset" and await asyncio.to_thread(Path(str(ref)).exists):
            init = Path(str(ref))
        elif stage.type == "imageedit" or ref == "prev-asset":
            for paths in reversed(list(ctx.stage_files.values())):
                if paths:
                    init = Path(str(paths[0]))
                    break
        req = ImageGenRequest(
            prompt=stage.image_prompt,
            negative=stage.negative,
            model=stage.model or "auto",
            seed=stage.seed,
            count=min(stage.count, 4),
            width=stage.width,
            height=stage.height,
            steps=stage.steps,
            guidance=stage.guidance,
            init_image=init,
            strength=stage.strength if init is not None else None,
            out_dir=out_dir,
        )
        try:
            result = await image.generate(req, None, None)
        except Exception as e:
            return StageOutcome(status="failed", wall_s=time.monotonic() - t0, error=str(e)[:200])
        paths = [str(p) for p in (result.image_paths or [])]
        ctx.stage_files[stage.id] = paths
        # register as scratch assets so a meshgen stage can consume real ids
        ids = await stage_assets(ctx, [Path(p) for p in paths])
        ctx.stage_assets[stage.id] = ids
        return StageOutcome(
            status="succeeded" if paths else "failed",
            asset_ids=ids or paths,
            artifact_files=paths,
            wall_s=time.monotonic() - t0,
            error=None if paths else "image stage produced no files",
        )
