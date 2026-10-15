"""Generation subprocess entry point.

Each job runs `python -m meshserver.gen_worker <request.json>` in its own
process so the parent can SIGKILL it on stall/cancel/RSS-cap — model weights
and device buffers live in the child and are fully reclaimed by the OS when
it dies (a frozen adapter *thread* would hold them forever).

Protocol (files in the job work dir):
- request.json  — written by the parent: kind, params, image paths
- progress.json — child writes {"progress": f, "message": msg} (atomic rename)
- cancel.flag   — parent touches it to ask the child to abort early
- result.json   — child writes on success (seed/steps/triangles/...)
- out.glb / img_*.png — binary payloads

Exit codes: 0 ok, 2 generation/cancel abort, 1 crash (stderr has the error).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from meshserver.adapters.base import GenOpts
from meshserver.adapters.image_base import ImageGenOpts
from meshserver.config import ServerConfig
from meshserver.postprocess import clean_mesh, export_glb
from meshserver.preprocess import prep_images
from meshserver.registry import Registry


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


class _Progress:
    """Adapter progress callback -> progress.json, throttled to ~0.5s."""

    def __init__(self, path: Path, cancel_flag: Path) -> None:
        self.path = path
        self.cancel_flag = cancel_flag
        self._last_write = 0.0

    def __call__(self, frac: float, msg: str) -> None:
        if self.cancel_flag.exists():
            raise _Cancelled()
        now = time.monotonic()
        if now - self._last_write < 0.5:
            return
        self._last_write = now
        _atomic_json(self.path, {"progress": round(min(frac, 1.0), 4), "message": msg})


class _Cancelled(Exception):
    pass


_current_model = ""  # resolved adapter id — reported back on failure


def _run_mesh(req: dict, cfg: ServerConfig, registry: Registry, work: Path) -> dict:
    global _current_model
    images = prep_images([Path(p) for p in req["image_paths"]], cfg.bg_removal)
    n = len(images)
    model = req["model"]
    cls = None
    attempted: set[str] = set()
    while cls is None or cls.id in attempted:
        cls = registry.select(model, n)
        if cls is None or cls.id in attempted:
            if model in registry._failed:
                raise RuntimeError(
                    f"adapter '{model}' failed earlier in this process: "
                    f"{registry._failed[model]}"
                )
            raise RuntimeError(f"no adapter available for model '{model}'")
        attempted.add(cls.id)
        try:
            adapter = registry.get(cls)
            _current_model = cls.id
            break
        except Exception:
            registry.mark_failed(cls.id, "load failed in gen worker")
            if model not in ("", "auto"):
                raise
            cls = None
            continue
    if len(images) > cls.max_images:
        images = images[: cls.max_images]
    progress = _Progress(work / "progress.json", work / "cancel.flag")
    result = adapter.generate(
        images,
        GenOpts(
            seed=req["params"].get("seed") or 0,
            steps=req["params"].get("steps"),
            guidance=req["params"].get("guidance"),
            resolution=req["params"].get("resolution"),
            subject=req["params"].get("subject", ""),
            image_roles=req["params"].get("image_roles") or [],
            work_dir=work,
        ),
        progress=progress,
        cancelled=lambda: (work / "cancel.flag").exists(),
    )
    _atomic_json(work / "progress.json", {"progress": 0.95, "message": "cleaning + exporting"})
    mesh = clean_mesh(result.mesh)
    glb = export_glb(mesh)
    (work / "out.glb").write_bytes(glb)
    return {
        "model": cls.id,
        "backend": cls.backend,
        "seed": result.seed,
        "steps": result.steps,
        "guidance": result.guidance,
        "resolution": result.resolution,
        "triangles": len(mesh.faces),
    }


def _run_image(req: dict, cfg: ServerConfig, registry: Registry, work: Path) -> dict:
    model = req["model"]
    init = Path(req["init_image_path"]) if req.get("init_image_path") else None
    cls = registry.select_image(model, need_edit=init is not None)
    if cls is None:
        raise RuntimeError(f"no image adapter available for '{model}'")
    global _current_model
    _current_model = cls.id
    adapter = registry.get(cls)
    params = req["params"]
    count = min(params.get("count") or 1, getattr(cls, "max_count", 4))
    side = getattr(cls, "max_side", 1024)
    progress = _Progress(work / "progress.json", work / "cancel.flag")
    result = adapter.generate(
        ImageGenOpts(
            prompt=params.get("prompt", ""),
            negative=params.get("negative", ""),
            seed=params.get("seed") or 0,
            count=count,
            width=min(params.get("width") or 1024, side),
            height=min(params.get("height") or 1024, side),
            steps=params.get("steps"),
            guidance=params.get("guidance"),
            work_dir=work,
            image_path=init,
            image_strength=params.get("strength"),
        ),
        progress=progress,
        cancelled=lambda: (work / "cancel.flag").exists(),
    )
    import io

    for i, img in enumerate(result.images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        (work / f"img_{i}.png").write_bytes(buf.getvalue())
    return {
        "model": cls.id,
        "backend": cls.backend,
        "seed": result.seed,
        "steps": result.steps,
        "guidance": result.guidance,
        "n_images": len(result.images),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m meshserver.gen_worker <request.json>", file=sys.stderr)
        return 2
    req_path = Path(sys.argv[1])
    req = json.loads(req_path.read_text())
    work = req_path.parent
    cfg = ServerConfig()
    # MPS shares unified memory with the host OS — an unbounded torch MPS
    # allocator can over-commit the machine and stall the Metal queue,
    # freezing macOS instead of raising OOM (observed: mv-turbo job hung in
    # mps_copy_ at 0% CPU → full system crash). Cap it BEFORE any torch/mps
    # use so a runaway job dies cleanly. setdefault: operator env wins.
    # ratio × Metal's recommendedMaxWorkingSetSize (~75% RAM ≈ 48GB @64GB):
    # 0.65 ≈ 31GB MPS budget — above mv-turbo's ~25GB peak, well below the
    # point where Metal page starvation froze the whole UI (observed crash)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.65")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")
    # config knobs the parent resolved explicitly (e.g. fake=True in tests
    # never reaches the child via env) — whitelist, no env passthrough needed
    for key in (
        "fake",
        "allow_cpu",
        "device",
        "model_dir",
        "bg_removal",
        "default_model",
        "image_default_model",
        "cuda_offload",
        "cuda_offload_vram_gb",
        "flux_mlx_quant",
        "mlx_mem_limit_gb",
    ):
        if key in req.get("cfg", {}):
            v = req["cfg"][key]
            setattr(cfg, key, Path(v) if key == "model_dir" else v)
    registry = Registry(cfg)
    for adapter_id, err in req.get("badlist", {}).items():
        registry.mark_failed(adapter_id, err)
    try:
        if req.get("kind") == "image":
            meta = _run_image(req, cfg, registry, work)
        else:
            meta = _run_mesh(req, cfg, registry, work)
    except _Cancelled:
        return 2
    except Exception as e:
        # the parent's poll loop only sees the exit code — persist the real
        # message (plus the resolved adapter id) so job.error is actionable
        # and the parent can badlist the failing model
        _atomic_json(
            work / "error.json",
            {"error": f"{type(e).__name__}: {e}", "model": _current_model},
        )
        return 1
    _atomic_json(work / "result.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
