"""HTTP driver — durable eval runs through the public REST API.

Every dispatched stage spawns real Version/Job rows under the case's fixed
``eval:*`` project (production spawn path — identical to GUI runs), then
harvests the result via ``GET /versions/{id}``, ``GET /jobs/{id}`` and the
``/jobs/{id}/trace`` endpoint. Nothing bypasses the API.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from cadx.evalkit.models import StageOutcome
from cadx.evalkit.suite import ASSET_STAGES, EvalStage
from cadx.logging import get_logger

log = get_logger(__name__)

_TERMINAL = {"succeeded", "failed", "cancelled"}


class EvalApiError(RuntimeError):
    pass


class DriverCtx:
    """Per-run state shared across stage dispatches."""

    def __init__(self) -> None:
        self.project_id: str | None = None
        self.project_name: str = ""
        self.parent_version_id: str | None = None  # resolved parent for next stage
        self.stage_assets: dict[str, list[str]] = {}  # stage id -> produced asset ids
        self.case_assets: list[str] = []  # case-level uploaded asset ids
        self.out_dir: Path | None = None


class HttpDriver:
    def __init__(self, base_url: str = "http://localhost:8080", *, verbose: bool = False) -> None:
        self.base = base_url.rstrip("/")
        self.api = f"{self.base}/api"
        self.verbose = verbose
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self._client.aclose()

    # -- low-level -----------------------------------------------------------

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = await self._client.request(method, f"{self.api}{path}", **kw)
        if r.status_code >= 400:
            raise EvalApiError(f"{method} {path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.content else None

    async def _get(self, path: str, **kw: Any) -> Any:
        return await self._req("GET", path, **kw)

    async def _post(self, path: str, **kw: Any) -> Any:
        return await self._req("POST", path, **kw)

    # -- high-level ----------------------------------------------------------

    async def find_or_create_project(self, name: str, settings: dict[str, Any]) -> dict[str, Any]:
        for p in await self._get("/projects"):
            if p.get("name") == name:
                # re-runs must honour the case's declared settings — a stored
                # project keeps its first-run settings otherwise (api_v8
                # helical kept min_wall_mm=0.8 and failed the gate despite
                # the case pinning 0.1)
                merged = {**(p.get("settings") or {}), **settings}
                if merged != (p.get("settings") or {}):
                    with contextlib.suppress(Exception):
                        p = await self._req(
                            "PATCH", f"/projects/{p['id']}", json={"settings": merged}
                        )
                return p
        body = {"name": name, "description": "evalkit benchmark project", "settings": settings}
        return await self._post("/projects", json=body)

    async def upload_assets(self, project_id: str, paths: list[Path]) -> list[dict[str, Any]]:
        if not paths:
            return []
        files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in paths]
        return await self._post(f"/projects/{project_id}/assets", files=files)

    async def version(self, vid: str) -> dict[str, Any]:
        return await self._get(f"/versions/{vid}")

    async def job(self, job_id: str) -> dict[str, Any]:
        return await self._get(f"/jobs/{job_id}")

    async def trace(self, job_id: str, limit: int = 5000) -> list[dict[str, Any]]:
        try:
            return await self._get(f"/jobs/{job_id}/trace", params={"limit": limit})
        except EvalApiError as e:
            log.warning("eval.trace_failed", job_id=job_id, error=str(e)[:120])
            return []

    async def wait_job(self, job_id: str, timeout_s: float) -> dict[str, Any]:
        """Poll until terminal; on timeout cancel and return the last state.

        Transient transport errors (dropped keep-alive, one-off connect
        failure) retry — a single socket hiccup must not fail a stage that
        polls for tens of minutes."""
        t0 = time.monotonic()
        delay = 1.0
        transient = 0
        while time.monotonic() - t0 < timeout_s:
            try:
                j = await self.job(job_id)
                transient = 0
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
                transient += 1
                if transient >= 10:
                    raise
                await asyncio.sleep(2.0)
                continue
            status = str(j.get("status"))
            if status in _TERMINAL:
                return j
            if status == "waiting_for_input":
                # clarification should be off for eval runs; don't hang on it
                await self._post(f"/jobs/{job_id}/cancel")
                j["status"] = "cancelled"
                j["error"] = "cancelled: job waited for user input during eval"
                return j
            await asyncio.sleep(delay)
            delay = min(delay * 1.4, 10.0)
        import contextlib

        with contextlib.suppress(EvalApiError):
            await self._post(f"/jobs/{job_id}/cancel")
        j = await self.job(job_id)
        j["status"] = "timeout"
        j["error"] = f"eval timeout after {timeout_s}s"
        return j

    async def download_artifact(
        self, url: str, dest_dir: Path, name: str | None = None
    ) -> Path | None:
        try:
            r = await self._client.get(url if url.startswith("http") else f"{self.base}{url}")
            if r.status_code >= 400:
                return None
            await asyncio.to_thread(dest_dir.mkdir, parents=True, exist_ok=True)
            name = (
                name
                or url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
                or uuid.uuid4().hex[:8]
            )
            dest = dest_dir / name
            await asyncio.to_thread(dest.write_bytes, r.content)
            return dest
        except Exception:
            return None

    async def settings_get(self) -> dict[str, Any]:
        return await self._get("/settings")

    async def settings_put(self, updates: dict[str, Any]) -> dict[str, Any]:
        return await self._req("PUT", "/settings", json=updates)

    async def health(self, *, retries: int = 4) -> bool:
        """Single dropped request must not abort a case — the api may be busy
        draining orphan jobs when a case starts."""
        for attempt in range(retries):
            try:
                r = await self._client.get(f"{self.base}/api/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            if attempt < retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
        return False

    # -- dispatch ------------------------------------------------------------

    async def _spawn_body(  # noqa: C901
        self, stage: EvalStage, ctx: DriverCtx
    ) -> tuple[str, dict[str, Any]]:
        """(path, json body) for a stage type."""
        t = stage.type
        pid = ctx.project_id
        parent = ctx.parent_version_id
        asset_ids = list(ctx.case_assets)
        if stage.assets_from:
            asset_ids += ctx.stage_assets.get(stage.assets_from, [])

        if t == "generate":
            body: dict[str, Any] = {"prompt": stage.prompt, "asset_ids": asset_ids}
            if stage.mode:
                body["mode"] = stage.mode
            if stage.options:
                body["options"] = stage.options
            return f"/projects/{pid}/generate", body
        if t == "edit":
            body = {"prompt": stage.prompt, "asset_ids": asset_ids}
            if stage.mode:
                body["mode"] = stage.mode
            if stage.options:
                body["options"] = stage.options
            return f"/versions/{parent}/edit", body
        if t == "params":
            return f"/versions/{parent}/params", {"params": stage.params or {}}
        if t == "edit_feature":
            return f"/versions/{parent}/edit-feature", dict(stage.feature or {})
        if t == "optimize":
            return f"/versions/{parent}/optimize", {
                "load_cases": stage.load_cases,
                "optimize": stage.optimize,
                "material": stage.material,
            }
        if t == "resample":
            body = {"load_cases": stage.load_cases, "material": stage.material}
            if stage.keep_regions:
                body["keep_regions"] = stage.keep_regions
            body.update(stage.resample or {})
            return f"/versions/{parent}/topopt-resample", body
        if t == "refine":
            body = {
                "features": stage.features,
                "load_cases": stage.load_cases,
                "keep_regions": stage.keep_regions,
                "material": stage.material,
            }
            if stage.voxel_mm is not None:
                body["voxel_mm"] = stage.voxel_mm
            if stage.restore_planes is not None:
                body["restore_planes"] = stage.restore_planes
            return f"/versions/{parent}/refine", body
        if t == "organic":
            return f"/versions/{parent}/organic", dict(stage.organic or {})
        if t == "reexport":
            return f"/versions/{parent}/reexport", dict(stage.options or {})
        if t == "fem":
            return f"/versions/{parent}/fem", {
                "load_cases": stage.load_cases,
                "keep_regions": stage.keep_regions,
                "material": stage.material,
            }
        if t == "mesh_edit":
            body = {"ops": stage.ops, "note": stage.prompt or f"eval {stage.id}"}
            if stage.post:
                body["post"] = stage.post
            return f"/versions/{parent}/mesh-edit", body
        if t == "meshgen":
            ids = asset_ids or self._ref_assets(stage, ctx)
            return f"/projects/{pid}/meshgen", {
                "asset_ids": ids,
                "subject": stage.subject or stage.prompt,
                "model": stage.model,
                "seed": stage.seed,
                "post": stage.post or {},
                "prompt": stage.prompt,
            }
        raise EvalApiError(f"no spawn route for stage type {t}")

    @staticmethod
    def _ref_assets(stage: EvalStage, ctx: DriverCtx) -> list[str]:
        if stage.reference and stage.reference != "prev-asset":
            return [stage.reference]
        for ids in reversed(list(ctx.stage_assets.values())):
            if ids:
                return ids
        return []

    async def dispatch(self, stage: EvalStage, ctx: DriverCtx) -> StageOutcome:
        t0 = time.monotonic()
        if stage.type in ASSET_STAGES:
            return await self._dispatch_image(stage, ctx, t0)

        path, body = await self._spawn_body(stage, ctx)
        if ctx.parent_version_id is None and "/versions/" in path:
            # upstream stage produced no version — don't POST to /versions/None/…
            return StageOutcome(
                status="failed",
                wall_s=time.monotonic() - t0,
                error=f"stage {stage.type} needs a parent version — none available",
            )
        if self.verbose:
            log.info("eval.http.spawn", stage=stage.id, path=path)
        enq = await self._post(path, json=body)
        vid, jid = str(enq.get("version_id")), str(enq.get("job_id"))

        job = await self.wait_job(jid, stage.timeout_s)
        wall = time.monotonic() - t0
        detail = await self.version(vid)
        records = await self.trace(jid)

        artifact_files: list[str] = []
        if ctx.out_dir:
            adir = ctx.out_dir / "artifacts" / stage.id
            rows = [
                a
                for a in (detail.get("artifacts") or [])
                if a.get("kind") in ("stl", "glb", "3mf", "obj", "step")
            ]
            # the version's committed mesh pointer first — artifact rows are
            # registered during intermediate runs, so row order is not
            # authoritative (mirrors the local driver's _mesh_artifact_files)
            stl_rel = (detail.get("mesh") or {}).get("stl_relpath")
            rows.sort(key=lambda a: 0 if stl_rel and a.get("path") == stl_rel else 1)
            for a in rows:
                # every artifact url ends in /file — name by kind + basename
                # of the storage path or downloads overwrite each other
                fname = f"{a.get('kind')}_{Path(str(a.get('path') or 'f')).name}"
                p = await self.download_artifact(str(a.get("url") or ""), adir, fname)
                if p:
                    artifact_files.append(str(p))

        status = str(job.get("status") or detail.get("status") or "failed")
        if status == "queued":
            status = "running"
        out = StageOutcome(
            status=status,
            version_id=vid,
            job_id=jid,
            score=detail.get("score"),
            iterations=int(detail.get("iterations") or 0),
            resolved_mode=detail.get("resolved_mode"),
            analysis=detail.get("analysis"),
            features=(detail.get("mesh") or {}).get("features")
            if isinstance(detail.get("mesh"), dict)
            else None,
            spec=detail.get("spec"),
            plan=detail.get("plan"),
            code=detail.get("code"),
            fem=detail.get("fem"),
            mesh=detail.get("mesh"),
            issues=detail.get("issues") or [],
            dfm=detail.get("dfm"),
            critique=detail.get("critique"),
            usage=detail.get("usage") or {},
            artifacts={
                f"{a.get('kind')}|{a.get('view') or ''}": a.get("path")
                for a in detail.get("artifacts") or []
            },
            artifact_files=artifact_files,
            trace_records=records if isinstance(records, list) else [],
            live=job.get("live"),
            asset_ids=[str(x) for x in detail.get("asset_ids") or []],
            wall_s=wall,
            error=job.get("error") or detail.get("error"),
        )
        # advance the chain: child-version stages produce a new parent,
        # same-version stages (fem/reexport) keep it
        if stage.type in ("fem", "reexport"):
            pass
        else:
            ctx.parent_version_id = vid
        return out

    async def _dispatch_image(self, stage: EvalStage, ctx: DriverCtx, t0: float) -> StageOutcome:
        body: dict[str, Any] = {
            "prompt": stage.image_prompt,
            "negative": stage.negative,
            "seed": stage.seed,
            "count": stage.count,
            "width": stage.width,
            "height": stage.height,
            "caption": f"eval {stage.id}",
        }
        if stage.model:
            body["model"] = stage.model
        if stage.steps is not None:
            body["steps"] = stage.steps
        if stage.guidance is not None:
            body["guidance"] = stage.guidance
        ref = self._ref_assets(stage, ctx)
        if stage.type == "imageedit" or stage.reference:
            if ref:
                body["reference_asset_id"] = ref[0]
            if stage.strength is not None:
                body["strength"] = stage.strength

        assets = await self._post(f"/projects/{ctx.project_id}/images", json=body)
        ids = [str(a.get("id")) for a in assets or []]
        ctx.stage_assets[stage.id] = ids
        return StageOutcome(
            status="succeeded" if ids else "failed",
            asset_ids=ids,
            wall_s=time.monotonic() - t0,
            error=None if ids else "image stage returned no assets",
        )
