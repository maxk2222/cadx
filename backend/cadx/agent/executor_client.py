"""Executor ports: in-process (CLI/tests) and queue-based (worker -> cad-executor)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import arq
from arq.connections import ArqRedis, create_pool
from arq.jobs import Job, JobStatus

from cadx.cad.tasks import (
    run_execute_payload,
    run_fem_payload,
    run_mesh_payload,
    run_optimize_payload,
)
from cadx.config import get_settings
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

CAD_QUEUE = "cad"


class InProcessExecutor:
    """Runs the CAD pipeline in the current process (CLI, tests, dev without queues)."""

    async def cancel_flag(self, job_id: str) -> bool:
        return False

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_execute_payload, payload)

    async def process_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_mesh_payload, payload)

    async def fem_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_fem_payload, payload)

    async def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(run_optimize_payload, payload)

    async def topopt_resample(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_topopt_resample_payload

        return await asyncio.to_thread(run_topopt_resample_payload, payload)

    async def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_refine_payload

        return await asyncio.to_thread(run_refine_payload, payload)

    async def organic(self, payload: dict[str, Any]) -> dict[str, Any]:
        from cadx.cad.tasks import run_organic_payload

        return await asyncio.to_thread(run_organic_payload, payload)

    async def export_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        import uuid as _uuid

        from cadx import storage
        from cadx.cad.engine import export_from_brep, export_from_mesh

        def _run() -> dict[str, Any]:
            pid = _uuid.UUID(payload["project_id"])
            vid = _uuid.UUID(payload["version_id"])
            art_dir = str(payload.get("art_dir") or "")
            it_dir = (
                storage.resolve(art_dir)
                if art_dir
                else storage.iteration_dir(pid, vid, int(payload["iteration"]))
            )
            brep = it_dir / "_run" / "model.brep"
            flat = it_dir / "_run" / "flat_profile.brep"
            stl = it_dir / "_run" / "model.stl"
            out_dir = storage.export_dir(pid, vid)
            out_dir.mkdir(parents=True, exist_ok=True)
            if brep.exists():
                exports = export_from_brep(
                    brep,
                    out_dir,
                    name=payload.get("name", "model"),
                    flat_brep=flat if flat.exists() else None,
                )
            elif stl.exists():
                exports = export_from_mesh(stl, out_dir, name=payload.get("name", "model"))
            else:
                return {"ok": False, "error": "no model.brep or model.stl found"}
            rel = storage.relpath(out_dir)
            return {
                "ok": True,
                "exports": {k: (f"{rel}/{v}" if v else None) for k, v in exports.items()},
            }

        return await asyncio.to_thread(_run)


class QueueExecutor:
    """Enqueues work on the 'cad' arq queue and polls the job until complete."""

    def __init__(
        self, pool: ArqRedis | None = None, *, runtime_settings: RuntimeSettings | None = None
    ) -> None:
        self._pool = pool
        self.runtime_settings = runtime_settings or RuntimeSettings()

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            self._pool = await create_pool(
                arq.connections.RedisSettings.from_dsn(get_settings().redis_url)
            )
        return self._pool

    async def _run_task(
        self, func: str, payload: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        job_id = f"{func}-{uuid.uuid4()}"
        await pool.enqueue_job(func, payload, _queue_name=CAD_QUEUE, _job_id=job_id)
        job = Job(job_id, pool)
        waited = 0.0
        while waited < timeout_s:
            status = await job.status()
            if status == JobStatus.complete:
                try:
                    result = await job.result()
                except Exception as e:
                    return {"ok": False, "error": f"executor task failed: {e}"}
                return result if isinstance(result, dict) else {"ok": False, "error": "bad result"}
            await asyncio.sleep(1.0)
            waited += 1.0
        return {"ok": False, "error": f"executor task '{func}' timed out after {timeout_s}s"}

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.exec_timeout_s * 2 + 120
        return await self._run_task("execute_version", payload, timeout)

    async def process_mesh(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.exec_timeout_s + self.runtime_settings.mesh_timeout_s + 120
        return await self._run_task("process_mesh", payload, timeout)

    async def fem_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("fem_analyze", payload, timeout)

    async def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.runtime_settings.topopt_timeout_s + 120
        return await self._run_task("optimize", payload, timeout)

    async def topopt_resample(self, payload: dict[str, Any]) -> dict[str, Any]:
        # re-extraction + FEM verify only — no SIMP loop, much cheaper
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("topopt_resample", payload, timeout)

    async def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        # booleans + optional FEM verify — same envelope as a FEM check
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("refine", payload, timeout)

    async def organic(self, payload: dict[str, Any]) -> dict[str, Any]:
        # voxelize + field gen + extraction — no FEM/SIMP, same envelope
        timeout = self.runtime_settings.fem_timeout_s + 120
        return await self._run_task("organic", payload, timeout)

    async def export_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._run_task("export_all", payload, 300)

    async def cancel_flag(self, job_id: str) -> bool:
        """The orchestrator checks cancellation via ctx.check_cancel; nothing extra here."""
        return False
