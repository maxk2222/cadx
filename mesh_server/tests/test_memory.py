"""MemoryLedger: learned footprints, admission control, resident set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meshserver.config import ServerConfig
from meshserver.memory import MemoryLedger


def _ledger(tmp_path: Path, **kw) -> MemoryLedger:
    cfg = ServerConfig(model_dir=tmp_path, job_dir=tmp_path / "jobs", **kw)
    return MemoryLedger(cfg)


def test_est_declared_until_measured(tmp_path):
    led = _ledger(tmp_path)
    assert led.est_mb("m1", 1000) == 1000
    led.record_peak("m1", 2000)
    # measured wins once known: 1.1 * 2000 > declared 1000
    assert led.est_mb("m1", 1000) == pytest.approx(2200.0)
    # a lucky low-RSS run never shrinks the learned footprint
    led.record_peak("m1", 500)
    assert led.est_mb("m1", 1000) == pytest.approx(2200.0)


def test_est_no_data(tmp_path):
    led = _ledger(tmp_path)
    assert led.est_mb("nope", None) == 0.0


def test_record_peak_ignores_placeholders(tmp_path):
    led = _ledger(tmp_path)
    led.record_peak("", 5000)
    led.record_peak("auto", 5000)
    led.record_peak("m1", 0)
    assert led._stats == {}  # noqa: SLF001


def test_fits_headroom(tmp_path, monkeypatch):
    led = _ledger(tmp_path, mem_headroom_mb=1000)
    monkeypatch.setattr(led, "free_mb", lambda: 5000.0)
    assert led.fits(4000) is True
    assert led.fits(4001) is False
    # unknown free memory -> admit (RSS watchdog stays the last defence)
    monkeypatch.setattr(led, "free_mb", lambda: None)
    assert led.fits(10**9) is True


def test_residents_lifecycle(tmp_path):
    led = _ledger(tmp_path)
    led.reserve("a", 100, pinned=False, kind="model")
    led.reserve("warm:b", 200, pinned=False, kind="warm")
    led.reserve("c", 300, pinned=True, kind="llm")

    res = {r["id"]: r for r in led.residents()}
    assert set(res) == {"a", "warm:b", "c"}
    assert res["a"]["mb"] == 100 and res["a"]["kind"] == "model"
    assert res["warm:b"]["kind"] == "warm"
    assert res["c"]["pinned"] is True
    assert res["a"]["idle_s"] >= 0
    assert led.has("a") and not led.has("zz")

    led.release("a")
    assert not led.has("a")
    assert {r["id"] for r in led.residents()} == {"warm:b", "c"}


def test_evictable_lru_order_and_pin(tmp_path):
    led = _ledger(tmp_path)
    led.reserve("a", 100, pinned=False, kind="model")
    led.reserve("b", 100, pinned=False, kind="model")
    led.reserve("c", 100, pinned=True, kind="model")
    # reserved in order a < b -> a is the LRU candidate; pinned c never is
    assert led.evictable(set()) == ["a", "b"]
    assert led.evictable({"a"}) == ["b"]
    # touch makes b the older entry
    led.touch("a")
    assert led.evictable(set()) == ["b", "a"]
    led.set_pinned("b", True)
    assert led.evictable(set()) == ["a"]


def test_record_peak_persists_across_restart(tmp_path):
    led = _ledger(tmp_path)
    led.record_peak("m1", 1234)
    led2 = _ledger(tmp_path)
    assert led2.est_mb("m1", None) == pytest.approx(1.1 * 1234)
    stats = json.loads((tmp_path / "model_stats.json").read_text())
    assert stats["peak_rss_mb"]["m1"] == 1234


def test_corrupt_stats_file_tolerated(tmp_path):
    (tmp_path / "model_stats.json").write_text("{not json")
    led = _ledger(tmp_path)  # must not raise
    assert led.est_mb("m1", 500) == 500
    # a fresh write repairs the file for the next restart
    led.record_peak("m1", 100)
    assert _ledger(tmp_path).est_mb("m1", None) == pytest.approx(110.0)
