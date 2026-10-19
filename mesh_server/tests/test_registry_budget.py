"""Registry memory budget: multi-model residency, LRU eviction, pinning."""

from __future__ import annotations

import pytest
import trimesh
from PIL import Image

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter
from meshserver.config import ServerConfig
from meshserver.registry import Registry

# fake "host free memory" — unload() adds the adapter's footprint back so the
# admission loop sees memory actually returning
FREE = {"mb": 0.0}


class _TinyBase(MeshAdapter):
    device = "cpu"
    backend = "test"
    license_note = "test"
    est_mem_mb = 100

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:
        FREE["mb"] -= self.est_mem_mb  # model weights eat host RAM
        self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            FREE["mb"] += self.est_mem_mb
        self._loaded = False

    def generate(
        self,
        images: list[Image.Image],
        opts: GenOpts,
        progress=None,
        cancelled=None,
    ) -> AdapterGenResult:
        return AdapterGenResult(mesh=trimesh.creation.box())


class TinyA(_TinyBase):
    id = "tiny-a"


class TinyB(_TinyBase):
    id = "tiny-b"


class TinyC(_TinyBase):
    id = "tiny-c"


class BigC(_TinyBase):
    id = "big-c"
    est_mem_mb = 900


class FakeLlm(MeshAdapter):
    """Minimal kind="llm" stand-in for select_llm()/models() coverage."""

    id = "llm-fake"
    name = "Fake LLM"
    device = "cpu"
    backend = "test"
    kind = "llm"
    supports_vision = True
    supports_tools = True
    downloaded = True
    est_mem_mb = 4000
    custom = True

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def generate(self, images, opts, progress=None, cancelled=None):  # noqa: ARG002
        raise AssertionError("llm adapters don't implement generate()")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    cfg = ServerConfig(
        fake=False,
        job_dir=tmp_path / "jobs",
        model_dir=tmp_path / "models",
        mem_headroom_mb=100,
    )
    reg = Registry(cfg)
    reg._classes = [TinyA, TinyB, TinyC, BigC]
    # isolate from the host's builtin llm catalog — on darwin-arm64 with a
    # working .venv-llm the real mlx adapters probe as available and would
    # leak into select_llm()/models()
    reg._llm_classes = []
    FREE["mb"] = 1000.0
    monkeypatch.setattr(reg.ledger, "free_mb", lambda: FREE["mb"])
    return reg


def test_two_small_models_coexist(registry):
    """The old single-loaded-model invariant is gone: as long as the ledger
    fits, multiple adapters stay resident."""
    a = registry.get(TinyA)
    b = registry.get(TinyB)
    assert a.loaded and b.loaded
    residents = {r["id"] for r in registry.ledger.residents()}
    assert {"tiny-a", "tiny-b"} <= residents


def test_big_model_evicts_everything_evictable(registry):
    """A newcomer too big for the budget drains all non-pinned residents in
    LRU order, then loads anyway — eviction is best-effort, not a refusal."""
    registry.get(TinyA)
    registry.get(TinyB)
    FREE["mb"] = 150.0  # host got tight: big-c (900) can't fit either way
    registry.get(BigC)
    assert registry._instances["tiny-a"].loaded is False
    assert registry._instances["tiny-b"].loaded is False
    assert registry._instances["big-c"].loaded is True


def test_lru_eviction_keeps_newer_resident(registry):
    """Room for exactly one more model: only the oldest resident goes."""
    registry.get(TinyA)
    registry.get(TinyB)
    registry._last_used["tiny-a"] -= 50  # a is clearly the older resident
    # free=100, need=100, headroom=100: one 100MB eviction makes it fit
    FREE["mb"] = 100.0
    registry.get(TinyC)
    assert registry._instances["tiny-a"].loaded is False  # evicted (LRU)
    assert registry._instances["tiny-b"].loaded is True  # survived
    assert registry._instances["tiny-c"].loaded is True


def test_pinned_model_not_evicted(registry):
    registry.get(TinyA)
    registry.pin("tiny-a", True)
    FREE["mb"] = 50.0  # nowhere near enough for big-c
    registry.get(BigC)  # loads anyway — best-effort eviction, OOM if real
    assert registry._instances["tiny-a"].loaded is True
    assert registry.ledger.residents()
    pinned = {r["id"] for r in registry.ledger.residents() if r["pinned"]}
    assert pinned == {"tiny-a"}


def test_pin_updates_ledger_flag(registry):
    registry.get(TinyA)
    registry.pin("tiny-a", True)
    res = {r["id"]: r for r in registry.ledger.residents()}
    assert res["tiny-a"]["pinned"] is True
    registry.pin("tiny-a", False)
    res = {r["id"]: r for r in registry.ledger.residents()}
    assert res["tiny-a"]["pinned"] is False


def test_unload_releases_ledger_entry(registry):
    registry.get(TinyA)
    assert registry.ledger.has("tiny-a")
    registry.unload_all()
    assert not registry.ledger.has("tiny-a")
    assert registry.ledger.residents() == []


def test_maybe_unload_idle_skips_pinned(registry):
    registry.cfg.unload_idle_s = 1
    registry.get(TinyA)
    registry.get(TinyB)
    registry.pin("tiny-a", True)
    registry._last_used["tiny-a"] = 0.0
    registry._last_used["tiny-b"] = 0.0
    registry.maybe_unload_idle()
    assert registry._instances["tiny-a"].loaded is True
    assert registry._instances["tiny-b"].loaded is False


def test_models_reports_ledger_fields(registry):
    registry._llm_classes = [FakeLlm]
    registry.get(TinyA)
    registry.pin("tiny-a", True)
    info = {m.id: m for m in registry.models()}
    assert info["tiny-a"].est_mem_mb == 100
    assert info["tiny-a"].loaded is True
    assert info["tiny-a"].pinned is True
    assert info["tiny-a"].downloaded is True
    assert info["tiny-b"].loaded is False
    # llm class surfaced with its own fields
    assert info["llm-fake"].kind == "llm"
    assert info["llm-fake"].supports_vision is True
    assert info["llm-fake"].supports_tools is True
    assert info["llm-fake"].custom is True


def test_select_llm(registry):
    assert registry.select_llm("auto") is None  # no llm classes registered
    registry._llm_classes = [FakeLlm]
    assert registry.select_llm("auto") is FakeLlm
    assert registry.select_llm("llm-fake") is FakeLlm
    # unknown ids degrade to auto — a stale external model name shouldn't 404
    assert registry.select_llm("nope") is FakeLlm
    registry.mark_failed("llm-fake", "boom")
    assert registry.select_llm("auto") is None  # badlisted
    assert registry.select_llm("llm-fake") is FakeLlm  # explicit still allowed
    # auto-fallback of an unknown id honours the badlist too
    assert registry.select_llm("nope") is None
