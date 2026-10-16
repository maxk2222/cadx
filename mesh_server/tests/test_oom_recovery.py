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


def _select_pref(self, requested, need_edit=False):  # noqa: ARG002
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
    """auto picks boom-load → OOM on load → badlisted → SAME job walks the
    ladder to boom-gen → OOM on generate → badlisted; the next job goes
    straight to ok-fallback."""
    j1 = _run_once(manager)
    assert j1.status == "failed"
    assert registry._failed.get("boom-load")
    assert registry._failed.get("boom-gen")  # generate-OOM also badlists

    j2 = _run_once(manager)
    assert j2.status == "succeeded"
    assert j2.model == "ok-fallback"


def test_auto_load_failure_retries_next_adapter_in_same_job(tmp_path, monkeypatch):
    """A load failure under 'auto' must not fail the job when another adapter
    can take over — the job itself walks the preference ladder."""
    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs")
    reg = Registry(cfg)
    reg._classes = [BoomLoadAdapter, OkFallbackAdapter]
    mgr = JobManager(cfg, reg)
    monkeypatch.setattr(Registry, "select_image", _select_pref)

    j = _run_once(mgr)
    assert j.status == "succeeded"
    assert j.model == "ok-fallback"
    assert reg._failed.get("boom-load")


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


# -- subprocess auto-retry ------------------------------------------------- #


class _StubProc:
    pid = 424242

    def __init__(self) -> None:
        self.returncode: int | None = None

    async def wait(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _run_proc_job(manager, watch, monkeypatch, model="auto"):
    """Submit one image job and drive it through _run_job_proc with a stubbed
    child — `watch` plays the gen_worker by writing error/result files."""
    import asyncio

    async def _spawn(*a, **k):  # noqa: ARG001
        return _StubProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    manager._watch_gen_proc = watch  # noqa: SLF001

    async def go():
        job = await manager.submit_image(ImageGenParams(model=model, prompt="x"))
        await manager._run_job_proc(job)  # noqa: SLF001
        return manager.jobs[job.id]

    return asyncio.run(go())


def test_proc_auto_retries_next_adapter_on_child_error(tmp_path, monkeypatch):
    """A deterministic adapter error inside the child (error.json names the
    adapter) must retry under 'auto' — _finish_proc_failed badlists it first,
    so the second dispatch lands on the next preference-ladder entry."""
    import json

    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs", jobs_subprocess=True)
    reg = Registry(cfg)
    reg._classes = [BoomLoadAdapter, OkFallbackAdapter]
    mgr = JobManager(cfg, reg)
    monkeypatch.setattr(Registry, "select_image", _select_pref)

    spawn_reqs: list[dict] = []

    async def _watch(proc, job, work):
        spawn_reqs.append(json.loads((work / "request.json").read_text()))
        if len(spawn_reqs) == 1:
            (work / "error.json").write_text(
                json.dumps({"error": "tokenizer files broken", "model": "boom-load"})
            )
            proc.returncode = 1
        else:
            (work / "img_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (work / "result.json").write_text(
                json.dumps({"model": "ok-fallback", "backend": "test", "seed": 1})
            )
            proc.returncode = 0
        return None, 10.0

    job = _run_proc_job(mgr, _watch, monkeypatch)

    assert job.status == "succeeded"
    assert job.model == "ok-fallback"
    assert len(spawn_reqs) == 2
    # the badlist from attempt 1 steered the retry's adapter pick
    assert "boom-load" in spawn_reqs[1]["badlist"]
    assert job.images


def test_proc_auto_no_retry_without_child_model(tmp_path, monkeypatch):
    """A generic child crash (error.json without a model) is not retryable —
    re-dispatching 'auto' would land on the same adapter and fail identically."""
    import json

    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs", jobs_subprocess=True)
    reg = Registry(cfg)
    reg._classes = [BoomLoadAdapter, OkFallbackAdapter]
    mgr = JobManager(cfg, reg)
    monkeypatch.setattr(Registry, "select_image", _select_pref)

    spawns = []

    async def _watch(proc, job, work):
        spawns.append(1)
        (work / "error.json").write_text(json.dumps({"error": "segfault", "model": ""}))
        proc.returncode = 1
        return None, 10.0

    job = _run_proc_job(mgr, _watch, monkeypatch)

    assert job.status == "failed"
    assert len(spawns) == 1


def test_proc_explicit_model_no_retry(tmp_path, monkeypatch):
    """An explicitly requested adapter gets a single attempt — a silent swap
    would be wrong even when the child reports a model-named error."""
    import json

    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs", jobs_subprocess=True)
    reg = Registry(cfg)
    reg._classes = [BoomLoadAdapter, OkFallbackAdapter]
    mgr = JobManager(cfg, reg)
    monkeypatch.setattr(Registry, "select_image", _select_pref)

    spawns = []

    async def _watch(proc, job, work):
        spawns.append(1)
        (work / "error.json").write_text(
            json.dumps({"error": "weights corrupt", "model": "boom-load"})
        )
        proc.returncode = 1
        return None, 10.0

    job = _run_proc_job(mgr, _watch, monkeypatch, model="boom-load")

    assert job.status == "failed"
    assert len(spawns) == 1
