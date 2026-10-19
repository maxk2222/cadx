"""Warm generation workers: gen_worker --serve reuse, idle exit, RSS kill."""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from meshserver.config import ServerConfig
from meshserver.jobs import JobManager
from meshserver.registry import Registry
from meshserver.schemas import GenParams


def _png(tmp_path: Path) -> Path:
    p = tmp_path / "in.png"
    Image.new("RGB", (32, 32), (200, 10, 10)).save(p)
    return p


def _manager(tmp_path: Path, **kw) -> tuple[JobManager, Registry]:
    cfg = ServerConfig(
        fake=True,
        device="cpu",
        job_dir=tmp_path / "jobs",
        model_dir=tmp_path / "models",
        jobs_subprocess=True,
        **{"warm_workers": True, "warm_max": 2, "warm_idle_s": 30} | kw,
    )
    reg = Registry(cfg)
    return JobManager(cfg, reg), reg


async def _wait_done(mgr: JobManager, job, timeout: float = 90.0):
    for _ in range(int(timeout * 10)):
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.1)
    return job


async def test_warm_worker_serves_two_jobs(tmp_path):
    """Two jobs for one model share a single --serve child: one spawn, the
    model loads once, both jobs succeed."""
    mgr, reg = _manager(tmp_path)
    mgr.start()
    try:
        img = _png(tmp_path)
        j1 = await mgr.submit(GenParams(model="fake"), [img])
        await _wait_done(mgr, j1)
        assert j1.status == "succeeded", j1.error
        assert j1.model == "fake"
        assert "fake" in mgr._warm  # noqa: SLF001
        assert "warm:fake" in {r["id"] for r in reg.ledger.residents()}

        j2 = await mgr.submit(GenParams(model="fake"), [img])
        await _wait_done(mgr, j2)
        assert j2.status == "succeeded", j2.error
        # same worker, still exactly one child process for the model
        assert len(mgr._warm) == 1  # noqa: SLF001
        warm = mgr._warm["fake"]  # noqa: SLF001
        assert warm.proc is not None and warm.proc.returncode is None
        assert warm.idle
    finally:
        await mgr.stop()
    assert mgr._warm == {}  # noqa: SLF001
    assert reg.ledger.residents() == []


async def test_warm_worker_idle_self_exit(tmp_path):
    """The child exits by itself after warm_idle_s of stdin silence; the
    parent's bookkeeping follows."""
    mgr, reg = _manager(tmp_path, warm_idle_s=1)
    mgr.start()
    try:
        j = await mgr.submit(GenParams(model="fake"), [_png(tmp_path)])
        await _wait_done(mgr, j)
        assert j.status == "succeeded", j.error
        assert "fake" in mgr._warm  # noqa: SLF001
        for _ in range(100):
            mgr._warm_gc()  # noqa: SLF001
            if "fake" not in mgr._warm:  # noqa: SLF001
                break
            await asyncio.sleep(0.2)
        assert "fake" not in mgr._warm  # noqa: SLF001
        assert "warm:fake" not in {r["id"] for r in reg.ledger.residents()}
    finally:
        await mgr.stop()


async def test_warm_worker_rss_cap_kill(tmp_path):
    """job_max_rss_mb applies to warm workers too — a runaway child is
    killed mid-job exactly like a cold subprocess."""
    mgr, _reg = _manager(tmp_path, job_max_rss_mb=1)  # any python child exceeds 1MB
    mgr.start()
    try:
        j = await mgr.submit(GenParams(model="fake"), [_png(tmp_path)])
        await _wait_done(mgr, j)
        assert j.status == "failed"
        assert "RSS" in (j.error or "")
        mgr._warm_gc()  # noqa: SLF001
        assert "fake" not in mgr._warm  # noqa: SLF001
    finally:
        await mgr.stop()


async def test_cold_path_when_warm_disabled(tmp_path):
    """warm_workers=False keeps the classic one-shot subprocess path."""
    mgr, _reg = _manager(tmp_path, warm_workers=False)
    mgr.start()
    try:
        j = await mgr.submit(GenParams(model="fake"), [_png(tmp_path)])
        await _wait_done(mgr, j)
        assert j.status == "succeeded", j.error
        assert mgr._warm == {}  # noqa: SLF001
    finally:
        await mgr.stop()


async def test_warm_eviction_frees_resident(tmp_path):
    """The admission path can evict an idle warm worker (LRU) to make room."""
    mgr, reg = _manager(tmp_path)
    mgr.start()
    try:
        j = await mgr.submit(GenParams(model="fake"), [_png(tmp_path)])
        await _wait_done(mgr, j)
        assert j.status == "succeeded", j.error
        warm = mgr._warm["fake"]  # noqa: SLF001
        assert warm.idle
        mgr._evict_warm(warm)  # noqa: SLF001
        assert "fake" not in mgr._warm  # noqa: SLF001
        assert "warm:fake" not in {r["id"] for r in reg.ledger.residents()}
        await asyncio.sleep(0.5)  # let the fire-and-forget reap task finish
    finally:
        await mgr.stop()
