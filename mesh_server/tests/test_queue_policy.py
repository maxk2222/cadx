"""Queue policy: model-affinity batching, fairness cap, memory admission."""

from __future__ import annotations

import time

import pytest

from meshserver.adapters.fake import FakeAdapter
from meshserver.adapters.fake_image import FakeImageAdapter
from meshserver.config import ServerConfig
from meshserver.jobs import JobManager
from meshserver.registry import Registry
from meshserver.schemas import GenParams, ImageGenParams


class FakeImage2(FakeImageAdapter):
    id = "fake-image-2"
    supports_edit = False


@pytest.fixture
def manager(tmp_path):
    cfg = ServerConfig(
        fake=True,
        device="cpu",
        job_dir=tmp_path / "jobs",
        model_dir=tmp_path / "models",
        jobs_subprocess=False,
        warm_workers=False,
    )
    reg = Registry(cfg)
    reg._classes = [FakeAdapter, FakeImageAdapter, FakeImage2]
    return JobManager(cfg, reg)


async def test_submit_resolves_explicit_model(manager):
    job = await manager.submit(GenParams(model="fake"), [])
    assert job.resolved_model == "fake"
    assert job.status == "queued"
    # 'auto' defers resolution to pick time — the ladder may shift meanwhile
    job2 = await manager.submit(GenParams(model="auto"), [])
    assert job2.resolved_model == ""


async def test_pick_is_fifo_without_affinity(manager):
    a = await manager.submit_image(ImageGenParams(model="fake-image", prompt="a"))
    b = await manager.submit_image(ImageGenParams(model="fake-image-2", prompt="b"))
    assert manager._pick_next().id == a.id  # noqa: SLF001
    assert manager._pick_next().id == b.id  # noqa: SLF001
    assert manager._pick_next() is None  # noqa: SLF001


async def test_affinity_batches_same_model(manager):
    """A fresh same-model job outranks an older different-model job while
    it's inside the fairness window — the resident weights get reused."""
    b = await manager.submit_image(ImageGenParams(model="fake-image-2", prompt="b"))
    a = await manager.submit_image(ImageGenParams(model="fake-image", prompt="a"))
    manager._last_model = "fake-image"  # noqa: SLF001
    pick = manager._pick_next()  # noqa: SLF001
    assert pick.id == a.id
    # and the remaining job still runs afterwards
    assert manager._pick_next().id == b.id  # noqa: SLF001


async def test_fairness_cap_overrides_affinity(manager):
    """After queue_fair_s of waiting, a non-affine job beats the batch."""
    b = await manager.submit_image(ImageGenParams(model="fake-image-2", prompt="b"))
    await manager.submit_image(ImageGenParams(model="fake-image", prompt="a"))
    manager._last_model = "fake-image"  # noqa: SLF001
    b.created = time.monotonic() - (manager.cfg.queue_fair_s + 60)
    pick = manager._pick_next()  # noqa: SLF001
    assert pick.id == b.id


async def test_memory_wait_then_timeout(manager, monkeypatch):
    """A job whose model doesn't fit waits in-queue and fails after
    queue_mem_wait_s instead of blocking the worker on it."""
    manager.cfg.queue_mem_wait_s = 1
    monkeypatch.setattr(manager.registry.ledger, "free_mb", lambda *a, **kw: 0.0)
    job = await manager.submit_image(ImageGenParams(model="fake-image", prompt="x"))
    assert job.resolved_model == "fake-image"
    assert manager._pick_next() is None  # noqa: SLF001
    assert job.status == "queued"
    assert job.message == "waiting for memory"
    job.created = time.monotonic() - 5
    assert manager._pick_next() is None  # noqa: SLF001
    assert job.status == "failed"
    assert "insufficient host memory" in (job.error or "")


async def test_unfit_job_does_not_block_fittable(manager, monkeypatch):
    """A queued job that can't fit must not head-of-line block the queue —
    a job whose model is already resident is admitted ahead of it."""
    manager.cfg.queue_mem_wait_s = 0  # never time out
    manager.cfg.mem_headroom_mb = 100
    big = await manager.submit_image(ImageGenParams(model="fake-image", prompt="big"))
    small = await manager.submit_image(ImageGenParams(model="fake-image-2", prompt="s"))
    # 50MB free fits neither declared 100MB model — but fake-image is
    # already resident (in-process mode), so its job needs no new memory
    monkeypatch.setattr(manager.registry.ledger, "free_mb", lambda *a, **kw: 50.0)
    manager.registry.ledger.reserve("fake-image", 100, pinned=False, kind="model")
    pick = manager._pick_next()  # noqa: SLF001
    assert pick.id == big.id  # resident model -> admitted ahead of small
    assert small.message == "waiting for memory"
    assert small.status == "queued"


async def test_vram_admission_evicts_lru_resident(manager, monkeypatch):
    """VRAM admission: a resident model pinning the card is evicted when the
    resolved adapter's declared VRAM peak doesn't fit — the ledger only
    accounts host RAM, so without this the worker dies on a real CUDA OOM."""
    evicted: list[str | None] = []
    free_levels = iter([1000.0, 30000.0])
    monkeypatch.setattr(
        "meshserver.jobs.free_vram_mb", lambda: next(free_levels, 30000.0)
    )
    monkeypatch.setattr(
        manager.registry,
        "evict_lru",
        lambda exclude=None: evicted.append("fake-image-2") or "fake-image-2",
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    manager.cfg.vram_headroom_mb = 1500
    job = await manager.submit_image(ImageGenParams(model="fake-image", prompt="x"))
    assert manager._admit(job) is True  # noqa: SLF001
    assert evicted == ["fake-image-2"]


async def test_vram_unfittable_waits_for_memory(manager, monkeypatch):
    """Nothing evictable + not enough VRAM -> job waits ('waiting for
    memory') instead of running into a CUDA OOM inside the worker."""
    monkeypatch.setattr("meshserver.jobs.free_vram_mb", lambda: 500.0)
    monkeypatch.setattr(manager.registry, "evict_lru", lambda exclude=None: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    manager.cfg.vram_headroom_mb = 1500
    job = await manager.submit_image(ImageGenParams(model="fake-image", prompt="x"))
    assert manager._admit(job) is False  # noqa: SLF001


async def test_vram_committed_spec_blocks_admission(manager, monkeypatch):
    """A model still LOADING (speculative reservation) already commits its
    declared VRAM — the llm-vs-flux race: the probe reads plenty free while
    the incoming allocation ramps, so only accounting can block admission."""
    ledger = manager.registry.ledger
    ledger.reserve(
        "fake-image-2", 100, pinned=False, kind="model", spec=True, vram_mb=33000
    )
    monkeypatch.setattr("meshserver.jobs.free_vram_mb", lambda: 33000.0)
    monkeypatch.setattr("meshserver.jobs.total_vram_mb", lambda: 34000.0)
    monkeypatch.setattr(manager.registry, "evict_lru", lambda exclude=None: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    manager.cfg.vram_headroom_mb = 1500
    job = await manager.submit_image(ImageGenParams(model="fake-image", prompt="x"))
    # accounted free = 34-33 = 1 GB < headroom → blocked although probe says 33 GB
    assert manager._admit(job) is False  # noqa: SLF001
    ledger.release("fake-image-2")
    assert manager._admit(job) is True  # noqa: SLF001


async def test_queue_snapshot_fields(manager, monkeypatch):
    monkeypatch.setattr(manager.registry.ledger, "free_mb", lambda *a, **kw: 12345.0)
    manager.registry.ledger.reserve("fake-image", 100, pinned=False, kind="model")
    job = await manager.submit_image(ImageGenParams(model="fake-image", prompt="x"))
    snap = manager.queue_snapshot()
    assert snap.free_mb == 12345.0
    res = snap.residents[0]
    assert res["id"] == "fake-image" and res["kind"] == "model"
    assert res["pinned"] is False and res["mb"] == 100
    e = snap.jobs[0]
    assert e.id == job.id
    assert e.kind == "image"
    assert e.model == "fake-image"
    assert e.resolved_model == "fake-image"
    assert e.status == "queued"
    assert e.waited_s >= 0
    assert e.est_mem_mb == 100
