"""Worker-side workbench for the mesh-refine tool session.

Holds the in-process trimesh state the `mesh_*` tools act on: the raw
generator mesh is normalised through the shared `baseline_mesh` (same code
path as the executor pipeline — no preview/final drift), then the session's
recorded `MeshOp` list is applied on top. On `done` the ops list is the
result; the authoritative geometry is re-produced by a `process_mesh` run in
the executor.

Tool errors are returned as `{"ok": False, ...}` results and never mutate the
working mesh — the model can self-correct.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh

from cadx import storage
from cadx.cad.mesh_ops import (
    MeshOp,
    MeshPostParams,
    apply_op,
    apply_ops,
    baseline_mesh,
    flat_base_ratio,
    load_mesh,
    mesh_analysis,
    rest_on_xy_center,
)
from cadx.logging import get_logger

if TYPE_CHECKING:
    from cadx.agent.types import AgentCtx
    from cadx.llm.schemas import PartSpec

log = get_logger(__name__)

# tool name -> op name; boolean direction is injected into args
_OP_TOOLS: dict[str, str] = {
    "mesh_set_up": "set_up",
    "mesh_rotate": "rotate",
    "mesh_mirror": "mirror",
    "mesh_scale": "scale",
    "mesh_flatten_bottom": "flatten_bottom",
    "mesh_slice": "slice",
    "mesh_crop": "crop",
    "mesh_decimate": "decimate",
    "mesh_smooth": "smooth",
    "mesh_smooth_preserve": "smooth_preserve",
    "mesh_keep_largest": "keep_largest",
    "mesh_drop_small": "drop_small",
    "mesh_repair": "repair",
    "mesh_boolean_add": "boolean",
    "mesh_boolean_cut": "boolean",
}


class MeshWorkbench:
    """Session state for kind="mesh" tool sessions."""

    def __init__(
        self,
        ctx: AgentCtx,
        *,
        raw_path: Path,
        post: MeshPostParams,
        spec: PartSpec | None,
        iteration: int,
        initial_ops: list[MeshOp] | None = None,
        preview_budget: int = 4,
        max_ops: int = 24,
    ) -> None:
        self.ctx = ctx
        self.iteration = iteration
        self.preview_left = preview_budget
        self.max_ops = max_ops
        self.error: str | None = None
        self.ops: list[MeshOp] = list(initial_ops or [])
        self.baseline: trimesh.Trimesh | None = None
        self.work: trimesh.Trimesh | None = None
        self._work_counter = 0
        try:
            raw = load_mesh(raw_path)
            self.baseline = baseline_mesh(raw, post, spec)[0]
            self.work, _ = apply_ops(self.baseline, self.ops)
            if not self.ops:
                self.work = rest_on_xy_center(self.baseline.copy())
        except Exception as e:
            log.warning("mesh.workbench_init_failed", error=str(e))
            self.error = f"failed to load/baseline mesh: {e}"

    # -- helpers --------------------------------------------------------- #

    def _session_dir(self) -> Path:
        it_dir = storage.iteration_dir(self.ctx.project_id, self.ctx.version_id, self.iteration)
        return storage.ensure(it_dir / "_mesh_session")

    def _stats(self) -> dict[str, Any]:
        assert self.work is not None
        b = self.work.bounds
        return {
            "faces": len(self.work.faces),
            "watertight": bool(self.work.is_watertight),
            "bbox_size": [round(float(v), 3) for v in (b[1] - b[0])],
            "ops_count": len(self.ops),
        }

    def _require_work(self) -> dict[str, Any] | None:
        if self.work is None:
            return {"ok": False, "error": self.error or "no mesh loaded"}
        return None

    async def _rebuild(self) -> dict[str, Any]:
        """Re-apply the recorded ops from baseline (undo/reset path)."""
        assert self.baseline is not None
        try:
            self.work, _ = await asyncio.to_thread(apply_ops, self.baseline, self.ops)
            if not self.ops:
                self.work = rest_on_xy_center(self.baseline.copy())
        except Exception as e:
            return {"ok": False, "error": f"rebuild after undo failed: {e}"}
        return {"ok": True, **self._stats()}

    # -- tool handlers ----------------------------------------------------- #

    async def handle(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if err := self._require_work():
            return err
        if name == "mesh_inspect":
            return await asyncio.to_thread(self._inspect)
        if name == "mesh_preview":
            return await self._preview()
        if name == "mesh_undo":
            if not self.ops:
                return {"ok": False, "error": "nothing to undo — op list is empty"}
            popped = self.ops.pop()
            res = await self._rebuild()
            res["undone"] = popped.model_dump()
            return res
        if name == "mesh_reset":
            n = len(self.ops)
            self.ops.clear()
            res = await self._rebuild()
            res["cleared"] = n
            return res
        op_name = _OP_TOOLS.get(name)
        if op_name is None:
            return {"ok": False, "error": f"unknown mesh tool '{name}'"}
        return await self._transform(op_name, name, args)

    def _inspect(self) -> dict[str, Any]:
        assert self.work is not None
        ana = mesh_analysis(self.work).model_dump()
        try:
            n_bodies = len(self.work.split(only_watertight=False))
        except Exception:
            n_bodies = -1
        return {
            "ok": True,
            "analysis": ana,
            "flat_base_ratio": round(flat_base_ratio(self.work), 4),
            "body_count": n_bodies,
            "ops_applied": [o.model_dump() for o in self.ops],
            "note": "final pipeline re-runs the recorded ops deterministically — "
            "this state is the preview of the result",
        }

    async def _transform(self, op_name: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if len(self.ops) >= self.max_ops:
            return {
                "ok": False,
                "error": f"op budget reached ({self.max_ops}) — call done or mesh_undo first",
            }
        op_args = dict(args)
        if tool == "mesh_boolean_add":
            op_args["op"] = "union"
        elif tool == "mesh_boolean_cut":
            op_args["op"] = "difference"
        try:
            mo = MeshOp(op=op_name, args=op_args)
        except Exception as e:
            return {"ok": False, "error": f"invalid args: {e}"}
        assert self.work is not None
        try:
            applied = await asyncio.to_thread(
                lambda: rest_on_xy_center(apply_op(self.work, mo))  # type: ignore[arg-type]
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        self.work = applied
        self.ops.append(mo)
        return {"ok": True, "applied": mo.model_dump(), **self._stats()}

    async def _preview(self) -> dict[str, Any]:
        if self.preview_left <= 0:
            return {"ok": False, "error": "preview budget exhausted"}
        assert self.work is not None
        self.preview_left -= 1
        self._work_counter += 1
        out = self._session_dir() / f"work_{self._work_counter}.glb"
        try:
            await asyncio.to_thread(self.work.export, str(out))
        except Exception as e:
            return {"ok": False, "error": f"failed to write preview mesh: {e}"}
        res = await self.ctx.executor.process_mesh(
            {
                "job_id": self.ctx.job_id,
                "project_id": str(self.ctx.project_id),
                "version_id": str(self.ctx.version_id),
                "iteration": self.iteration,
                "mode": "preview",
                "mesh_rel": storage.relpath(out),
                "settings": self.ctx.settings.model_dump(),
                "runtime_settings": self.ctx.runtime_settings.executor_values(),
            }
        )
        if not res.get("ok"):
            return {
                "ok": False,
                "error": res.get("error") or "preview render failed",
                "stats": res.get("stats"),
            }
        images = res.get("images") or {}
        paths = [str(storage.resolve(rel)) for rel in images.values()]
        return {
            "ok": True,
            "images": images,
            "_images": paths,
            "stats": res.get("stats"),
            "preview_left": self.preview_left,
        }
