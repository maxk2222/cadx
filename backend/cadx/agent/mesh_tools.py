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
import json
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
    decimate,
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


def _primitive_volume(args: dict[str, Any]) -> float | None:
    """Bounding volume of a boolean primitive in mm³ (None when unknown)."""
    import math

    shape = str(args.get("shape") or "box")
    if shape == "cylinder":
        r, h = args.get("radius"), args.get("height")
        if r and h:
            return math.pi * float(r) ** 2 * float(h)
    elif shape == "sphere":
        r = args.get("radius")
        if r:
            return 4.0 / 3.0 * math.pi * float(r) ** 3
    else:
        ext = args.get("size_xyz")
        if ext and len(ext) == 3:
            return float(ext[0]) * float(ext[1]) * float(ext[2])
        s = args.get("size")
        if s:
            return float(s) ** 3
    return None


def _union_volume(ops: list[MeshOp]) -> float:
    """Cumulative volume of union-boolean primitives in an op list."""
    total = 0.0
    for op in ops:
        if op.op == "boolean" and op.args.get("op") == "union":
            total += _primitive_volume(op.args) or 0.0
    return total


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
    "mesh_radial_profile": "radial_profile",
}

# rigid-orientation ops — refused outright when `orientation_locked` is set
# (the orient step already placed the part on measurement evidence)
_ORIENTATION_OPS = {"set_up", "rotate", "mirror"}

# ops exempt from the silhouette-drift guard (still get the coarse volume
# floor): rigid reorientation keeps the shape — blocking set_up on a lying
# part would prevent exactly the fix the guard protects; cut-class ops
# (slice/flatten/clip/crop), spec/measure-driven reshapes and structural
# cleanups (shell hollows by design, keep/drop fix fractured bodies, remesh
# is lossy on sharp geometry by construction) legitimately move the outline
_SHAPE_GUARD_EXEMPT = {
    "set_up",
    "rotate",
    "mirror",
    "drill_holes",
    "boolean",
    "clip_to_outline",
    "radial_profile",
    "slice",
    "flatten_bottom",
    "clip_height",
    "crop",
    "symmetrize",
    "shell",
    "keep_largest",
    "drop_small",
    "remesh",
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
        orientation_locked: bool = False,
    ) -> None:
        self.ctx = ctx
        self.spec = spec
        self.anchors = anchors or {}
        self.measured_outline = measured_outline or []
        self.iteration = iteration
        self.preview_left = preview_budget
        self.max_ops = max_ops
        # rigid-orientation ops are refused while True — set when the orient
        # step already placed the part on measurement evidence (silhouette /
        # plan-view / inherited). The knob run showed the refine LLM freely
        # re-trying set_up until it had rotated a correct ±X-up orientation
        # back onto its side.
        self.orientation_locked = orientation_locked
        self.error: str | None = None
        self.ops: list[MeshOp] = list(initial_ops or [])
        # cumulative volume added by union booleans — guards against the model
        # sculpting the part out of primitive blobs (grip iter2 added ~4
        # cylinders ≈ body volume after a clean bore cut, turning it into a tube)
        self._added_vol = _union_volume(self.ops)
        self.baseline: trimesh.Trimesh | None = None
        self.work: trimesh.Trimesh | None = None
        self._work_counter = 0
        self._features: Any = None  # cached detect_mesh_features result
        self._post = post  # kept for re-baselining after mesh_regen_edit
        self.new_mesh_rel: str | None = None  # set by mesh_regen_edit
        # undo/redo cycle guard: models with weak visual evidence loop on
        # "apply cut -> doubt it -> undo -> re-apply" — the third identical
        # application is refused so the session budget isn't burned
        self._op_apply_count: dict[str, int] = {}
        # sequence-level loop guard: fingerprints (op *names* — args differ on
        # purpose when the model retries a strategy) of complete ops lists.
        # Undo/reset cycles that rebuild the same strategy (observed:
        # reset -> scale -> flatten -> smooth x4) are warned then refused —
        # the per-op guard alone only catches literally identical args
        self._seq_count: dict[str, int] = {}
        # fingerprints of ops-states that deterministic evidence confirmed —
        # undoing a verified state is warned once, then refused
        self._verified_fps: set[str] = set()
        self._verified_undo_warned = False
        try:
            raw = load_mesh(raw_path)
            # workbench is preview-only and runs in the worker process — cap
            # faces so voxel/boolean ops can't OOM the worker (the executor
            # replays ops on its own baseline for the authoritative result)
            cap = (
                getattr(getattr(self.ctx, "runtime_settings", None), "mesh_decimate_faces", None)
                or 30_000
            )
            self.baseline = baseline_mesh(raw, post, spec)[0]
            if len(self.baseline.faces) > cap:
                self.baseline = decimate(self.baseline, cap)
                log.info("mesh.workbench_decimated", faces=len(self.baseline.faces), cap=cap)
            self.work, _ = apply_ops(
                self.baseline,
                self.ops,
                anchors=self.anchors,
                runtime=getattr(self.ctx, "runtime_settings", None),
            )
            if not self.ops:
                self.work = rest_on_xy_center(self.baseline.copy())
        except Exception as e:
            log.warning("mesh.workbench_init_failed", error=str(e))
            self.error = f"failed to load/baseline mesh: {e}"

    # -- helpers --------------------------------------------------------- #

    @staticmethod
    def _ops_fp(ops: list[MeshOp]) -> str:
        """Fingerprint of an ops list — identifies identical mesh states."""
        return json.dumps([o.model_dump() for o in ops], sort_keys=True, default=str)

    @staticmethod
    def _names_fp(ops: list[MeshOp]) -> str:
        """Op-name sequence fingerprint — same strategy regardless of args."""
        return "|".join(o.op for o in ops)

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
                apply_ops,
                self.baseline,
                self.ops,
                anchors=self.anchors,
                runtime=getattr(self.ctx, "runtime_settings", None),
            )
            if not self.ops:
                self.work = rest_on_xy_center(self.baseline.copy())
            cap = (
                getattr(getattr(self.ctx, "runtime_settings", None), "mesh_decimate_faces", None)
                or 30_000
            )
            if self.work is not None and len(self.work.faces) > cap:
                self.work = await asyncio.to_thread(decimate, self.work, cap)
        except Exception as e:
            return {"ok": False, "error": f"rebuild after undo failed: {e}"}
        self._features = None
        self._added_vol = _union_volume(self.ops)
        fp = self._names_fp(self.ops)
        self._seq_count[fp] = self._seq_count.get(fp, 0) + 1
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
            verified = self._ops_fp(self.ops) in self._verified_fps
            if verified and self._verified_undo_warned:
                return {
                    "ok": False,
                    "error": (
                        "refused: this state was verified by deterministic "
                        "measurement (tunnels/anchors matched) — undoing reverts "
                        "a confirmed feature. Call mesh_measure to see the "
                        "evidence, or done to keep it."
                    ),
                }
            popped = self.ops.pop()
            res = await self._rebuild()
            res["undone"] = popped.model_dump()
            if verified:
                self._verified_undo_warned = True
                res["warning"] = (
                    "the state you just reverted was verified correct by "
                    "deterministic measurement — undoing it removes a confirmed "
                    "feature"
                )
            return res
        if name == "mesh_reset":
            if self._ops_fp(self.ops) in self._verified_fps:
                return {
                    "ok": False,
                    "error": (
                        "refused: the current state was verified by deterministic "
                        "measurement — resetting would discard confirmed features. "
                        "Undo a single op instead, or call done."
                    ),
                }
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
        if name == "mesh_regen_edit":
            return await self._regen_edit(args)
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
            {"x": h.x, "y": h.y, "dia": h.dia, "depth": h.depth, "kind": h.kind, "axis": h.axis}
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
        all_tunnels = [h for h in (feats.holes if feats else []) if h.kind == "tunnel"]
        # anchor positions are plan-view (XY) — only z-axis tunnels can match
        tunnels = [(h.x, h.y, h.dia) for h in all_tunnels if h.axis == "z"]
        off_axis = [{"dia": h.dia, "axis": h.axis} for h in all_tunnels if h.axis != "z"]
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
            if not missing and matches:
                self._verified_fps.add(self._ops_fp(self.ops))
            if missing:
                out["hint"] = (
                    f"{len(missing)} measured hole(s) have no tunnel in the mesh — "
                    "mesh_drill_holes(from_anchors=true) drills them at the "
                    "measured positions"
                )
        else:
            out["tunnels_detected"] = len(all_tunnels)
        if off_axis:
            # lateral/longitudinal bores are invisible to the Z-only views —
            # surface them explicitly so the agent doesn't think a successful
            # axis bore "didn't take" and undo it
            out["tunnels_off_axis"] = off_axis
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
            # entities is a numpy array — `not arr` raises ambiguity for
            # multi-entity sections (e.g. section through a bored hole)
            if sec is None or len(sec.entities) == 0:
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

    async def _regen_edit(self, args: dict[str, Any]) -> dict[str, Any]:
        """Text-steered regeneration: edit the conditioning image with a local
        img2img/instruction-edit adapter, regenerate the mesh from the edited
        image (same model+seed), then re-baseline. Recorded ops are discarded
        — they were tuned for the previous shape."""
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return {"ok": False, "error": "prompt is required"}
        if self.ctx.image is None:
            return {"ok": False, "error": "no image adapter configured (ctx.image is None)"}
        if self.ctx.mesh is None:
            return {"ok": False, "error": "no mesh backend configured"}

        # 1. pick the conditioning image this mesh was generated from
        src = self._gen_source_image()
        if src is None:
            return {"ok": False, "error": "no source image available to edit"}

        # 2. image edit via a supports_edit adapter (server-side selection)
        from cadx.agent.image_client import ImageGenRequest

        s = self.ctx.runtime_settings
        tmp = self._session_dir() / "edit"
        tmp.mkdir(parents=True, exist_ok=True)
        strength = args.get("strength")
        try:
            edit_res = await self.ctx.image.generate(
                ImageGenRequest(
                    prompt=prompt,
                    model=str(args.get("image_model") or "auto"),
                    seed=int(args.get("seed") or 0),
                    count=1,
                    width=min(768, s.image_max_side),
                    height=min(768, s.image_max_side),
                    steps=getattr(s, "image_steps", None),
                    init_image=src,
                    strength=float(strength) if strength is not None else 0.6,
                    out_dir=tmp,
                ),
                None,
                self.ctx.check_cancel,
            )
        except Exception as e:
            return {"ok": False, "error": f"image edit failed: {e}"}
        if not edit_res.image_paths:
            return {"ok": False, "error": "image edit returned no image"}
        edited = edit_res.image_paths[0]

        # 3. regenerate the mesh from the edited image, same model+seed
        from cadx.agent.mesh_client import MeshGenRequest

        gen = self.ctx.mesh_info.get("generation") or {}
        self._edit_counter = getattr(self, "_edit_counter", 0) + 1
        out = (
            storage.iteration_dir(self.ctx.project_id, self.ctx.version_id, self.iteration)
            / f"mesh_raw_edit{self._edit_counter}.glb"
        )
        try:
            await self.ctx.mesh.generate(
                MeshGenRequest(
                    images=[edited],
                    image_roles=["front"],
                    subject=str(gen.get("subject") or self.ctx.prompt[:80]),
                    model=str(gen.get("model") or "auto"),
                    seed=int(gen.get("seed") or 0),
                    steps=self.ctx.settings.mesh_steps,
                    guidance=self.ctx.settings.mesh_guidance,
                    resolution=self.ctx.settings.mesh_resolution,
                    out_path=out,
                    job_id=self.ctx.job_id,
                ),
                None,
                self.ctx.check_cancel,
            )
        except Exception as e:
            return {"ok": False, "error": f"mesh regeneration failed: {e}"}
        if not out.exists():
            return {"ok": False, "error": "mesh regeneration produced no file"}

        # 4. re-baseline on the new raw mesh; previous ops no longer apply
        try:
            cap = (
                getattr(getattr(self.ctx, "runtime_settings", None), "mesh_decimate_faces", None)
                or 30_000
            )
            raw = await asyncio.to_thread(load_mesh, out)
            self.baseline = await asyncio.to_thread(
                lambda: baseline_mesh(raw, self._post, self.spec)[0]
            )
            if len(self.baseline.faces) > cap:
                self.baseline = await asyncio.to_thread(decimate, self.baseline, cap)
            dropped = len(self.ops)
            self.ops.clear()
            self.work = rest_on_xy_center(self.baseline.copy())
            self._features = None
            # the new raw mesh is in generator space — the orientation lock
            # belongs to the old baseline and would block re-orienting this one
            self.orientation_locked = False
        except Exception as e:
            return {"ok": False, "error": f"new mesh failed baseline processing: {e}"}
        self.new_mesh_rel = storage.relpath(out)
        gen["edit"] = {
            "prompt": prompt,
            "image_model": edit_res.model,
            "mesh_rel": self.new_mesh_rel,
        }
        self.ctx.mesh_info["generation"] = gen
        return {
            "ok": True,
            "new_mesh_rel": self.new_mesh_rel,
            "edited_image": str(edited),
            "ops_dropped": dropped,
            **self._stats(),
            "note": "mesh regenerated from the edited image — previous ops were "
            "discarded; inspect and re-apply transforms as needed",
        }

    def _gen_source_image(self) -> Path | None:
        """The conditioning image used for the current mesh generation —
        asset ids in ctx.mesh_info['generation']['images'] resolved against
        ctx.assets; falls back to the first input image."""
        from cadx.agent.types import IMAGE_INPUT_KINDS

        gen_ids = {str(i) for i in (self.ctx.mesh_info.get("generation") or {}).get("images") or []}
        for a in self.ctx.assets:
            if str(a.asset_id) in gen_ids and getattr(a, "path", None):
                p = Path(a.path)
                if p.exists():
                    return p
        for a in self.ctx.assets:
            if a.kind in IMAGE_INPUT_KINDS and getattr(a, "path", None):
                p = Path(a.path)
                if p.exists():
                    return p
        return None

    async def _transform(self, op_name: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.orientation_locked and op_name in _ORIENTATION_OPS:
            return {
                "ok": False,
                "error": (
                    "refused: orientation was fixed by the deterministic orient "
                    "step (measurement-backed) — it is not a refine parameter. "
                    "Fix local shape issues with scale/clip/boolean ops, call "
                    "mesh_regen_edit for a fundamentally wrong mesh, or done."
                ),
            }
        if len(self.ops) >= self.max_ops:
            return {
                "ok": False,
                "error": f"op budget reached ({self.max_ops}) — call done or mesh_undo first",
            }
        op_args = dict(args)
        if (
            tool == "mesh_boolean_add"
            and self.work is not None
            and (refusal := self._boolean_add_guard(op_args))
        ):
            return refusal
        if tool == "mesh_boolean_add":
            op_args["op"] = "union"
        elif tool == "mesh_boolean_cut":
            op_args["op"] = "difference"
        try:
            mo = MeshOp(op=op_name, args=op_args)
        except Exception as e:
            return {"ok": False, "error": f"invalid args: {e}"}
        # identical op applied a 3rd time means the model undid it twice —
        # a burn-the-budget cycle (observed: bore cut->undo->cut->undo x4).
        # Refuse so the session ends or the model changes strategy.
        op_fp = self._ops_fp([mo])
        if self._op_apply_count.get(op_fp, 0) >= 2:
            return {
                "ok": False,
                "error": (
                    f"refused: '{tool}' with identical arguments was already "
                    "applied twice — re-applying and undoing the same op is a "
                    "cycle. Call mesh_measure to verify the feature, adjust "
                    "the arguments, or call done."
                ),
            }
        # sequence-level loop guard: an op-name sequence reached a third time
        # via undo/reset cycles means the session is going in circles
        seq_fp = self._names_fp([*self.ops, mo])
        n_seq = self._seq_count.get(seq_fp, 0)
        if n_seq >= 2:
            return {
                "ok": False,
                "error": (
                    "refused: this exact op sequence was already reached "
                    "twice — the session is cycling. Call done with the "
                    "current state, or change strategy entirely."
                ),
            }
        assert self.work is not None
        try:
            applied = await asyncio.to_thread(
                lambda: rest_on_xy_center(  # type: ignore[arg-type]
                    apply_op(self.work, mo, anchors=self.anchors)
                )
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        # ops that densify (remesh/voxel rebuilds) must not push the session
        # mesh past the workbench cap — the worker eval was OOM-killed twice
        # by runaway face counts between init-time decimations
        cap = (
            getattr(getattr(self.ctx, "runtime_settings", None), "mesh_decimate_faces", None)
            or 30_000
        )
        if len(applied.faces) > cap:
            applied = await asyncio.to_thread(decimate, applied, cap)
        # deterministic shape guard — see _shape_guard
        if refusal := await self._shape_guard(op_name, applied):
            return refusal
        self.work = applied
        self.ops.append(mo)
        self._seq_count[seq_fp] = n_seq + 1
        res = {"ok": True, "applied": mo.model_dump(), **self._stats()}
        if n_seq == 1:
            res["warning"] = (
                "this op sequence was already reached once — a second "
                "undo/redo cycle on it will be refused"
            )
        if op_args.get("op") == "union":
            self._added_vol += _primitive_volume(op_args) or 0.0
        self._op_apply_count[op_fp] = self._op_apply_count.get(op_fp, 0) + 1
        self._features = None
        return res

    def _boolean_add_guard(self, op_args: dict[str, Any]) -> dict[str, Any] | None:
        """Refuse boolean-add primitives that would sculpt the part out of
        blobs instead of refining the generated shape."""
        assert self.work is not None
        prim_vol = _primitive_volume(op_args)
        # compare against the baseline volume, not the current body: every
        # accepted add inflates the body and would progressively weaken the
        # cumulative cap below
        base = self.baseline if self.baseline is not None else self.work
        body_vol = float(abs(base.volume)) if base.is_volume else 0.0
        if not (body_vol and prim_vol is not None):
            return None
        # single-add cap: one primitive must not replace the body
        if prim_vol > 0.5 * body_vol:
            return {
                "ok": False,
                "error": (
                    "refused: the added primitive is larger than half the body — "
                    "it would replace the organic shape. Use mesh_scale / "
                    "mesh_clip_height to adjust size; boolean_add is for small "
                    "flanges, bases and bosses only."
                ),
            }
        # cumulative cap: several sub-50% adds sculpt the part out of
        # primitives (observed: 4 cylinder blobs ≈ body volume turning
        # a grip into a tube) — unioning to reach target dims is the
        # documented anti-pattern
        if self._added_vol + prim_vol > 0.5 * body_vol:
            return {
                "ok": False,
                "error": (
                    "refused: cumulative boolean_add volume would exceed half "
                    "the body — unioning primitives to force proportions is "
                    "not allowed. Use mesh_scale / per-axis scale to reach "
                    "target dimensions instead."
                ),
            }
        return None

    async def _shape_guard(self, op_name: str, applied: trimesh.Trimesh) -> dict[str, Any] | None:
        """Refuse ops that destroy the part rather than refine it.

        Per-op drift is measured against the working state, not the baseline —
        legitimate prior ops (clip, shell) already changed the volume, so a
        baseline comparison misattributes cumulative drift to this op. An op
        that collapses the volume or drags the silhouette in one step is
        deterministic destruction evidence (observed: flatten_bottom on a
        lying knob, non-uniform squash to a capsule). Spec cuts and rigid
        reorientation are exempt from the silhouette check but still get a
        coarse volume floor. Returns the refusal result or None.
        """
        guard_iou = float(
            getattr(self.ctx.runtime_settings, "agent_mesh_shape_guard_iou", 0.8) or 0.0
        )
        if guard_iou <= 0 or self.work is None:
            return None
        try:
            bv = abs(float(self.work.volume)) if self.work.is_volume else 0.0
            av = abs(float(applied.volume)) if applied.is_volume else 0.0
        except Exception:
            bv = av = 0.0
        exempt = op_name in _SHAPE_GUARD_EXEMPT
        # exempt ops legitimately remove lots of material (shell on a thin
        # part drops >50%) — the floor only catches near-annihilation
        vol_floor = 0.25 if exempt else 0.85
        if bv > 0 and av > 0 and av < vol_floor * bv:
            return {
                "ok": False,
                "error": (
                    f"refused: the op removes {(1 - av / bv) * 100:.0f}% of "
                    "the part's volume — that reshapes the part, not "
                    "refines it. Undo to revert further if the shape is "
                    "wrong; use spec-driven cuts for material removal."
                ),
            }
        if exempt:
            return None
        from cadx.cad.silhouette import mesh_silhouette_iou

        drift = await asyncio.to_thread(mesh_silhouette_iou, self.work, applied)
        if drift < guard_iou:
            return {
                "ok": False,
                "error": (
                    f"refused: silhouette drifts to IoU {drift:.2f} vs "
                    f"the baseline (guard {guard_iou:.2f}) — the op "
                    "deforms the overall shape. If the part is lying "
                    "on its side fix orientation with mesh_set_up "
                    "FIRST, and prefer small local transforms over "
                    "flatten/squash."
                ),
            }
        return None

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
        ext = [float(v) for v in self.work.extents]
        # end views along the longest axis: a longitudinal bore/slot is
        # invisible in iso/front/top — without end views the agent couldn't
        # see a successful bore and kept undoing it
        end_views = (("left", "right"), ("front", "back"), ("top", "bottom"))[int(np.argmax(ext))]
        views = tuple(dict.fromkeys(("iso", "front", "top", *end_views)))
        res = await self.ctx.executor.process_mesh(
            {
                "job_id": self.ctx.job_id,
                "project_id": str(self.ctx.project_id),
                "version_id": str(self.ctx.version_id),
                "iteration": self.iteration,
                "mode": "preview",
                "views": list(views),
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
        out_res: dict[str, Any] = {
            "ok": True,
            "images": images,
            "_images": paths,
            "stats": res.get("stats"),
            "preview_left": self.preview_left,
        }
        # deterministic through-hole confirmation alongside the render —
        # the vision model once undid a correct longitudinal bore 4x because
        # no view and no metric confirmed it
        feats = self._detect_features()
        tunnels = [
            {"dia": h.dia, "axis": h.axis, "at": [h.x, h.y]}
            for h in (feats.holes if feats else [])
            if h.kind == "tunnel"
        ]
        if tunnels:
            out_res["tunnels_measured"] = tunnels
            # deterministic confirmation — protect this state from undo churn
            self._verified_fps.add(self._ops_fp(self.ops))
        return out_res
