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

import numpy as np
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
    "mesh_clip_height": "clip_height",
    "mesh_symmetrize": "symmetrize",
    "mesh_remesh": "remesh",
    "mesh_shell": "shell",
    "mesh_drill_holes": "drill_holes",
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
        anchors: dict[str, Any] | None = None,
        measured_outline: list[list[float]] | None = None,
    ) -> None:
        self.ctx = ctx
        self.spec = spec
        self.anchors = anchors or {}
        self.measured_outline = measured_outline or []
        self.iteration = iteration
        self.preview_left = preview_budget
        self.max_ops = max_ops
        self.error: str | None = None
        self.ops: list[MeshOp] = list(initial_ops or [])
        self.baseline: trimesh.Trimesh | None = None
        self.work: trimesh.Trimesh | None = None
        self._work_counter = 0
        self._features: Any = None  # cached detect_mesh_features result
        try:
            raw = load_mesh(raw_path)
            self.baseline = baseline_mesh(raw, post, spec)[0]
            self.work, _ = apply_ops(self.baseline, self.ops, anchors=self.anchors)
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
            self.work, _ = await asyncio.to_thread(
                apply_ops, self.baseline, self.ops, anchors=self.anchors
            )
            if not self.ops:
                self.work = rest_on_xy_center(self.baseline.copy())
        except Exception as e:
            return {"ok": False, "error": f"rebuild after undo failed: {e}"}
        self._features = None
        return {"ok": True, **self._stats()}

    def _detect_features(self) -> Any:
        """Cached detect_mesh_features on the working mesh."""
        if self._features is None:
            from cadx.cad.mesh_features import detect_mesh_features

            step = max(float(self.work.extents.max()) / 150.0, 0.3)
            try:
                self._features = detect_mesh_features(self.work, step_mm=step)
            except Exception:
                self._features = None
        return self._features

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
        if name == "mesh_measure":
            return await asyncio.to_thread(self._measure)
        if name == "mesh_section":
            return await asyncio.to_thread(self._section, args)
        if name == "mesh_overlay":
            return await asyncio.to_thread(self._overlay)
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
        feats = self._detect_features()
        holes = [
            {"x": h.x, "y": h.y, "dia": h.dia, "depth": h.depth, "kind": h.kind}
            for h in (feats.holes if feats else [])
        ][:24]
        anchor_holes = self.anchors.get("holes") or []
        return {
            "ok": True,
            "analysis": ana,
            "flat_base_ratio": round(flat_base_ratio(self.work), 4),
            "body_count": n_bodies,
            "features": {
                "holes": holes,
                "outline_points": len(feats.outline_mm) if feats else 0,
            },
            "anchors": [
                {"index": a.get("index"), "x": a.get("x"), "y": a.get("y"), "dia": a.get("dia")}
                for a in anchor_holes
            ][:24],
            "ops_applied": [o.model_dump() for o in self.ops],
            "note": "final pipeline re-runs the recorded ops deterministically — "
            "this state is the preview of the result",
        }

    def _measure(self) -> dict[str, Any]:
        """Spec-vs-mesh facts: bbox deltas, anchor↔detected-hole matching."""
        assert self.work is not None
        b = self.work.bounds
        size = [float(v) for v in (b[1] - b[0])]
        out: dict[str, Any] = {"ok": True, "bbox_size": [round(v, 2) for v in size]}
        if self.spec is not None:
            deltas: dict[str, float] = {}
            for name, idx, dim in (
                ("length_x", 0, self.spec.length_x),
                ("width_y", 1, self.spec.width_y),
                ("height_z", 2, self.spec.height_z),
            ):
                if dim is not None and (dim.value or 0) > 0:
                    deltas[name] = round(size[idx] - float(dim.value), 2)
            out["spec_delta_mm"] = deltas
        feats = self._detect_features()
        tunnels = [(h.x, h.y, h.dia) for h in (feats.holes if feats else []) if h.kind == "tunnel"]
        anchor_holes = self.anchors.get("holes") or []
        if anchor_holes:
            matches = []
            for a in anchor_holes:
                ax, ay, adia = (
                    float(a.get("x") or 0),
                    float(a.get("y") or 0),
                    float(a.get("dia") or 0),
                )
                best = None
                for hx, hy, hdia in tunnels:
                    d = float(np.hypot(hx - ax, hy - ay))
                    if best is None or d < best[0]:
                        best = (d, hdia)
                matches.append(
                    {
                        "anchor_index": a.get("index"),
                        "anchor": {"x": ax, "y": ay, "dia": adia},
                        "nearest_tunnel": (
                            None
                            if best is None
                            else {"dist_mm": round(best[0], 2), "dia": round(best[1], 2)}
                        ),
                    }
                )
            out["anchor_matches"] = matches
            missing = [
                m
                for m in matches
                if m["nearest_tunnel"] is None or m["nearest_tunnel"]["dist_mm"] > 3.0
            ]
            out["missing_holes"] = len(missing)
            if missing:
                out["hint"] = (
                    f"{len(missing)} measured hole(s) have no tunnel in the mesh — "
                    "mesh_drill_holes(from_anchors=true) drills them at the "
                    "measured positions"
                )
        else:
            out["tunnels_detected"] = len(tunnels)
        out["flat_base_ratio"] = round(flat_base_ratio(self.work), 4)
        return out

    def _section(self, args: dict[str, Any]) -> dict[str, Any]:
        """Rasterised cross-section of the working mesh at an axis plane."""
        assert self.work is not None
        if self.preview_left <= 0:
            return {"ok": False, "error": "preview budget exhausted"}
        axis = str(args.get("axis", "z"))
        if axis not in ("x", "y", "z"):
            return {"ok": False, "error": f"axis must be x/y/z, got '{axis}'"}
        idx = "xyz".index(axis)
        lo, hi = float(self.work.bounds[0][idx]), float(self.work.bounds[1][idx])
        at = args.get("at_mm")
        at_mm = float(at) if at is not None else (lo + hi) / 2.0
        if not lo - 1e-4 < at_mm < hi + 1e-4:
            return {"ok": False, "error": f"at_mm {at_mm:g} outside [{lo:.1f}, {hi:.1f}] on {axis}"}
        normal = np.zeros(3)
        normal[idx] = 1.0
        origin = np.zeros(3)
        origin[idx] = at_mm
        try:
            sec = self.work.section(plane_normal=normal, plane_origin=origin)
            if sec is None or not sec.entities:
                return {"ok": False, "error": f"no intersection at {axis}={at_mm:g} mm"}
            path2d, _matrix = sec.to_2D()
            ext = self.work.extents
            span = max(float(ext.max()), 1.0)
            img = path2d.rasterize(
                pitch=max(span / 700.0, 0.05), origin=[0, 0], resolution=[720, 720], fill=True
            )
            out = self._session_dir() / f"section_{axis}_{at_mm:.1f}.png"
            img.save(str(out))
        except Exception as e:
            return {"ok": False, "error": f"section failed: {e}"}
        self.preview_left -= 1
        return {
            "ok": True,
            "images": {"section": storage.relpath(out)},
            "_images": [str(out)],
            "preview_left": self.preview_left,
            "note": f"cross-section at {axis}={at_mm:.1f} mm (hatch = solid material)",
        }

    def _overlay(self) -> dict[str, Any]:
        """Top-view overlay: mesh footprint vs measured outline + anchors."""
        assert self.work is not None
        if self.preview_left <= 0:
            return {"ok": False, "error": "preview budget exhausted"}
        try:
            from PIL import Image, ImageDraw

            feats = self._detect_features()
            mesh_outline = feats.outline_mm if feats else []
            measured = self.measured_outline
            anchor_holes = self.anchors.get("holes") or []
            all_pts = list(mesh_outline) + list(measured)
            for a in anchor_holes:
                all_pts.append([float(a.get("x") or 0), float(a.get("y") or 0)])
            if not all_pts:
                return {"ok": False, "error": "no outline or anchor data to overlay"}
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            pad = 5.0
            x0, x1 = min(xs) - pad, max(xs) + pad
            y0, y1 = min(ys) - pad, max(ys) + pad
            W = 720
            H = max(int(W * (y1 - y0) / max(x1 - x0, 1e-9)), 120)

            def px(p: list[float]) -> tuple[float, float]:
                return (
                    (p[0] - x0) / (x1 - x0) * W,
                    (1.0 - (p[1] - y0) / (y1 - y0)) * H,
                )

            img = Image.new("RGB", (W, H), "white")
            dr = ImageDraw.Draw(img)
            if len(measured) >= 3:
                dr.line([px(p) for p in measured], fill=(220, 40, 40), width=3)
            if len(mesh_outline) >= 3:
                dr.line([px(p) for p in mesh_outline], fill=(40, 90, 220), width=2)
            r = 6
            for a in anchor_holes:
                cx, cy = px([float(a.get("x") or 0), float(a.get("y") or 0)])
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(20, 160, 60), width=3)
                dr.line([cx - r, cy, cx + r, cy], fill=(20, 160, 60), width=1)
                dr.line([cx, cy - r, cx, cy + r], fill=(20, 160, 60), width=1)
            dr.text(
                (8, 8), "red = measured outline | blue = mesh | green = anchors", fill=(0, 0, 0)
            )
            out = self._session_dir() / "overlay.png"
            img.save(str(out))
        except Exception as e:
            return {"ok": False, "error": f"overlay failed: {e}"}
        self.preview_left -= 1
        return {
            "ok": True,
            "images": {"overlay": storage.relpath(out)},
            "_images": [str(out)],
            "preview_left": self.preview_left,
            "note": "red = measured outline, blue = mesh footprint, green = anchors",
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
                lambda: rest_on_xy_center(  # type: ignore[arg-type]
                    apply_op(self.work, mo, anchors=self.anchors)
                )
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        self.work = applied
        self.ops.append(mo)
        self._features = None
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
