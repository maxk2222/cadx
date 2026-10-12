"""OOM recovery: badlisting, unload_all, fallback selection, clear errors."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from PIL import Image

from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts, ImageGenResult
from meshserver.config import ServerConfig
from meshserver.jobs import JobManager, _is_oom
from meshserver.registry import Registry
from meshserver.schemas import ImageGenParams


class BoomLoadAdapter(ImageAdapter):
    """Always available but load() raises an OOM-style error."""

    id = "boom-load"
    name = "Boom (load OOM)"
    device = "cpu"
    backend = "test"
    license_note = "test"

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:
        raise RuntimeError("CUDA out of memory allocating weights")

    def generate(
        self,
        opts: ImageGenOpts,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        raise AssertionError("unreachable — load() fails first")


class BoomGenAdapter(ImageAdapter):
    """Loads fine but generate() raises an OOM-style error."""

    id = "boom-gen"
    name = "Boom (gen OOM)"
    device = "cpu"
    backend = "test"
    license_note = "test"

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate(
        self,
        opts: ImageGenOpts,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        raise RuntimeError("CUDA out of memory during attention")


class OkFallbackAdapter(ImageAdapter):
    id = "ok-fallback"
    name = "OK fallback"
    device = "cpu"
    backend = "test"
    license_note = "test"

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._loaded = False
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(
        self,
        opts: ImageGenOpts,
        progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ImageGenResult:
        return ImageGenResult(
            images=[Image.new("RGB", (32, 32))],
            model=self.id,
            backend="test",
            seed=opts.seed,
        )


def _select_pref(self, requested):
    """select_image stand-in with a test preference ladder."""
    all_img = {
        cls.id: cls for cls in self.available() if getattr(cls, "kind", "mesh") == "image"
    }
    if requested and requested != "auto":
        return all_img.get(requested)
    avail = {k: v for k, v in all_img.items() if k not in self._failed}
    for pid in ("boom-load", "boom-gen", "ok-fallback"):
        if cls := avail.get(pid):
            return cls
    return None


@pytest.fixture
def registry(tmp_path):
    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs")
    reg = Registry(cfg)
    reg._classes = [BoomLoadAdapter, BoomGenAdapter, OkFallbackAdapter]
    return reg


@pytest.fixture
def manager(registry, monkeypatch):
    monkeypatch.setattr(Registry, "select_image", _select_pref)
    return JobManager(registry.cfg, registry)


def _run_once(manager, model="auto"):
    """Submit one image job and drive it through _run_image_job + OOM handler
    synchronously (mirrors the worker loop's exception path)."""
    import asyncio

    async def go():
        job = await manager.submit_image(ImageGenParams(model=model, prompt="x"))
        try:
            manager._run_image_job(job)  # noqa: SLF001
        except Exception as e:
            job.status = "failed"
            if _is_oom(e):
                manager.registry.unload_all()
                if getattr(job, "model", "") not in ("", "auto"):
                    manager.registry.mark_failed(job.model, str(e))
                job.error = "VRAM exhausted"
            else:
                job.error = str(e)
        return manager.jobs[job.id]

    return asyncio.run(go())


def test_is_oom_detection():
    assert _is_oom(RuntimeError("CUDA out of memory")) is True
    assert _is_oom(RuntimeError("cublas allocation failed")) is True
    assert _is_oom(ValueError("bad prompt")) is False
    assert _is_oom(RuntimeError("network timeout")) is False


def test_load_failure_badlists_and_auto_falls_back(registry, manager):
    """auto picks boom-load → OOM on load → badlisted; next auto job skips it,
    hits boom-gen → OOM on generate → badlisted; third job succeeds."""
    j1 = _run_once(manager)
    assert j1.status == "failed"
    assert registry._failed.get("boom-load")

    j2 = _run_once(manager)
    assert j2.status == "failed"
    assert registry._failed.get("boom-gen")

    j3 = _run_once(manager)
    assert j3.status == "succeeded"
    assert j3.model == "ok-fallback"


def test_explicit_request_surfaces_error(registry, manager):
    """Explicitly requesting a badlisted model still attempts it (no silent
    downgrade) — the raw error reaches the client."""
    registry.mark_failed("boom-load", "earlier oom")
    j = _run_once(manager, model="boom-load")
    assert j.status == "failed"
    assert "vram" in j.error.lower() or "out of memory" in j.error.lower()


def test_unload_all_frees_instances(registry):
    ok = OkFallbackAdapter(registry.cfg)
    ok.load()
    registry._instances["ok-fallback"] = ok
    registry.unload_all()
    assert ok.loaded is False


def test_model_info_reports_last_error(registry):
    registry.mark_failed("boom-gen", "CUDA out of memory during attention")
    info = {m.id: m for m in registry.models()}
    assert info["boom-gen"].last_error == "CUDA out of memory during attention"
    assert info["ok-fallback"].kind == "image"
    assert info["ok-fallback"].last_error is None
