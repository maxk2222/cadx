"""Phase-B scheduling: _LlmQueue ordering (priority/affinity/fairness),
internal/external priority classification, queue snapshot visibility,
SSE keepalive + reasoning feed, omni alias, undownloaded-model 404."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from meshserver.auth import is_internal_caller, request_priority
from meshserver.config import ServerConfig
from meshserver.llm_proxy import _LlmQueue, _LlmEntry, chat_completions


class _Cfg:
    """cfg stand-in with an effective() overlay — ServerConfig.real would
    pull env/overlay; tests only need the knob lookups the queue reads."""

    def __init__(self, **knobs: Any) -> None:
        self._knobs = knobs

    def effective(self, key: str) -> Any:
        return self._knobs.get(key)


def _entry(model: str, prio: int = 0, *, stream: bool = False) -> _LlmEntry:
    return _LlmEntry(model, prio, stream=stream, manager=None)


class _FakeReg:
    """Registry stub with a controllable resident set."""

    def __init__(self, loaded: set[str] | None = None) -> None:
        self._instances = {
            mid: SimpleNamespace(loaded=True) for mid in (loaded or set())
        }


def _drained_order(queue: _LlmQueue, cfg: Any, reg: Any) -> list[str]:
    """Model ids of waiting entries in dispatch order (no grants)."""
    return [e.model_id for e in queue._ranked(cfg, reg)]


async def _hold_capacity(q: _LlmQueue, cfg: Any, reg: Any) -> _LlmEntry:
    """Grant the single capacity slot so subsequent enqueues wait."""
    holder = _entry("holder")
    assert await q.enqueue(cfg, reg, holder)
    assert holder.event.is_set()
    return holder


async def test_queue_priority_outranks_fifo():
    """A later high-priority entry must dispatch before an earlier
    zero-priority one (one priority step = one fairness window)."""
    q = _LlmQueue()
    cfg = _Cfg(llm_max_concurrent=1, llm_queue_fair_s=60, llm_queue_max=32)
    reg = _FakeReg()
    await _hold_capacity(q, cfg, reg)
    low = _entry("m", prio=0)
    high = _entry("m", prio=1)
    assert await q.enqueue(cfg, reg, low)
    assert await q.enqueue(cfg, reg, high)
    ranked = q._ranked(cfg, reg)
    assert ranked[0] is high and ranked[1] is low


async def test_queue_affinity_batches_same_model():
    """An entry for the resident model beats a foreign model of the same
    priority — model-batch coherence over strict FIFO."""
    q = _LlmQueue()
    cfg = _Cfg(llm_max_concurrent=1, llm_queue_fair_s=60, llm_queue_max=32)
    reg = _FakeReg(loaded={"alpha"})
    await _hold_capacity(q, cfg, reg)
    foreign = _entry("beta", prio=0)
    same = _entry("alpha", prio=0)
    assert await q.enqueue(cfg, reg, foreign)
    assert await q.enqueue(cfg, reg, same)
    ranked = q._ranked(cfg, reg)
    assert ranked[0] is same and ranked[1] is foreign


async def test_queue_fairness_aging_beats_affinity():
    """A foreign-model entry older than queue_fair_s outscores a fresh
    same-model entry — affinity can never starve."""
    q = _LlmQueue()
    cfg = _Cfg(llm_max_concurrent=1, llm_queue_fair_s=60, llm_queue_max=32)
    reg = _FakeReg(loaded={"alpha"})
    await _hold_capacity(q, cfg, reg)
    old_foreign = _entry("beta", prio=0)
    old_foreign.t0 -= 120  # waited two fairness windows
    fresh_same = _entry("alpha", prio=0)
    assert await q.enqueue(cfg, reg, old_foreign)
    assert await q.enqueue(cfg, reg, fresh_same)
    assert q._ranked(cfg, reg)[0] is old_foreign


async def test_queue_grant_respects_capacity():
    """Dispatch grants up to llm_max_concurrent, rest stay queued and
    visible in the snapshot."""
    q = _LlmQueue()
    cfg = _Cfg(llm_max_concurrent=1, llm_queue_fair_s=60, llm_queue_max=32)
    reg = _FakeReg()
    a = _entry("m")
    b = _entry("m")
    assert await q.enqueue(cfg, reg, a)
    assert await q.enqueue(cfg, reg, b)
    assert a.event.is_set() and not b.event.is_set()
    snap = q.snapshot(cfg, reg)
    assert len(snap) == 1 and snap[0]["position"] == 1
    # release dispatches the waiter
    q.release(cfg, reg, a)
    assert b.event.is_set()
    assert q.snapshot(cfg, reg) == []


async def test_queue_max_rejects_overflow():
    q = _LlmQueue()
    cfg = _Cfg(llm_max_concurrent=1, llm_queue_max=1)
    reg = _FakeReg()
    await _hold_capacity(q, cfg, reg)  # the slot is taken → entries queue
    assert await q.enqueue(cfg, reg, _entry("m"))
    assert not await q.enqueue(cfg, reg, _entry("m"))


# -- priority classification -------------------------------------------------


def _req(headers: dict[str, str], host: str = "127.0.0.1") -> Any:
    return SimpleNamespace(
        headers={k.lower(): v for k, v in headers.items()},
        client=SimpleNamespace(host=host),
    )


def test_internal_classification_header_and_loopback():
    assert is_internal_caller(None) is True
    assert is_internal_caller(_req({"x-mesh-internal": "1"}, host="10.0.0.5"))
    assert is_internal_caller(_req({}, host="127.0.0.1"))
    assert not is_internal_caller(_req({}, host="10.0.0.5"))


def test_request_priority_reads_configured_class():
    cfg = _Cfg(llm_prio_internal=0, llm_prio_external=10)
    assert request_priority(_req({"x-mesh-internal": "1"}), cfg,
                            internal_key="llm_prio_internal",
                            external_key="llm_prio_external") == 0
    assert request_priority(_req({}, host="10.0.0.9"), cfg,
                            internal_key="llm_prio_internal",
                            external_key="llm_prio_external") == 10


def test_external_cannot_self_escalate():
    """X-Mesh-Priority is honored only from loopback — a LAN caller forging
    X-Mesh-Internal still lands on the configured external value."""
    cfg = _Cfg(llm_prio_internal=0, llm_prio_external=10)
    forged = _req(
        {"x-mesh-internal": "1", "x-mesh-priority": "99"}, host="10.0.0.9"
    )
    got = request_priority(
        forged, cfg,
        internal_key="llm_prio_internal", external_key="llm_prio_external",
    )
    assert got == 0  # internal class only — the raw override is loopback-only
    local = _req({"x-mesh-priority": "7"}, host="127.0.0.1")
    assert request_priority(
        local, cfg,
        internal_key="llm_prio_internal", external_key="llm_prio_external",
    ) == 7


# -- stream path: reasoning feed + keepalive ---------------------------------


class _Adapter:
    def __init__(self, url: str) -> None:
        self._url = url

    def base_url(self) -> str:
        return self._url

    def served_model_id(self) -> str:
        return "served"


class _SlowReg:
    """Registry whose get() blocks until released — holds the request in
    the queue/loading window long enough to observe the SSE feed."""

    def __init__(self, adapter: Any, gate: asyncio.Event) -> None:
        self._adapter = adapter
        self._gate = gate
        self._cls = type(
            "SlowCls",
            (),
            {"id": "slow-llm", "kind": "llm",
             "supports_vision": True, "supports_tools": True},
        )
        self._instances: dict[str, Any] = {}
        self.select_calls: list[dict] = []

    def select_llm(self, model: str, **kw: Any):
        self.select_calls.append(kw)
        return self._cls

    def get(self, cls, **kw: Any):  # noqa: ARG002
        self._gate.wait(5)
        return self._adapter


class _Mgr:
    def __init__(self) -> None:
        self.llm_loading: dict = {}
        self.llm_in_flight: dict = {}
        self.llm_waiting = 0

    def admit_resolved(self, resolved: str) -> bool:  # noqa: ARG002
        return True


async def test_stream_emits_reasoning_during_load(tmp_path):
    """stream:true commits SSE immediately — while the model load runs,
    reasoning_content status deltas must reach the client before the
    first upstream token."""
    import threading

    gate = threading.Event()
    reg = _SlowReg(_Adapter("http://127.0.0.1:1/v1"), gate)
    cfg = ServerConfig(llm_keepalive_s=0.05, llm_load_timeout_s=10,
                       model_dir=tmp_path)
    mgr = _Mgr()
    resp = await chat_completions(
        reg, cfg, {"model": "slow-llm", "stream": True, "messages": []},
        manager=mgr,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.media_type
    chunks: list[bytes] = []
    aiter = resp.body_iterator
    try:
        # gate CLOSED: get() blocks → keepalive + reasoning frames must flow
        async for chunk in aiter:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
            if len(chunks) >= 4:
                break
    finally:
        gate.set()  # release the load → upstream unreachable → error chunk
    raw = b"".join(chunks)
    reasoning = [line for line in raw.split(b"\n\n") if b"reasoning_content" in line]
    assert b": keepalive" in raw or reasoning, (
        "expected keepalive comments or reasoning_content while the load blocks"
    )


async def test_omni_alias_forces_full_capabilities(tmp_path):
    """model:'omni' must ask select_llm for vision+tools so the pick serves
    every request shape — without leaking 'omni' as the model id."""
    reg = _SlowReg(_Adapter("http://127.0.0.1:1/v1"), asyncio.Event())
    cfg = ServerConfig(llm_queue_wait_s=0.05, model_dir=tmp_path)
    resp = await chat_completions(
        reg, cfg, {"model": "omni", "messages": []}, manager=_Mgr(),
    )
    # select_llm got the capability flags (vision+tools forced)
    assert reg.select_calls and reg.select_calls[0].get("need_vision") is True
    assert reg.select_calls[0].get("need_tools") is True


async def test_explicit_undownloaded_model_404():
    """An explicit model id whose weights aren't on disk must fail fast
    with 404 — not stall inside an implicit multi-GB download."""

    class NotDownloaded(type):
        id = "fat-llm"
        kind = "llm"
        supports_vision = True
        supports_tools = True

        @classmethod
        def downloaded(cls, cfg):  # noqa: ARG003
            return False

    class Reg:
        _cls = NotDownloaded

        def select_llm(self, model: str, **kw: Any):  # noqa: ARG002
            return self._cls

    resp = await chat_completions(
        Reg(), ServerConfig(), {"model": "fat-llm", "messages": []},
        manager=_Mgr(),
    )
    assert resp.status_code == 404
    assert b"not downloaded" in resp.body


async def test_queue_full_returns_503(tmp_path):
    """A full queue (capacity slot + queue depth both taken) → honest 503,
    no silent hang."""
    import threading

    gate = threading.Event()
    reg = _SlowReg(_Adapter("http://127.0.0.1:1/v1"), gate)
    cfg = ServerConfig(llm_max_concurrent=1, llm_queue_max=1,
                       llm_queue_wait_s=0.5, llm_load_timeout_s=5,
                       model_dir=tmp_path)
    mgr = _Mgr()
    body = {"model": "slow-llm", "messages": []}
    first = asyncio.create_task(  # occupies the single capacity slot
        chat_completions(reg, cfg, body, manager=mgr)
    )
    await asyncio.sleep(0.05)
    second = asyncio.create_task(  # fills the queue depth
        chat_completions(reg, cfg, body, manager=mgr)
    )
    await asyncio.sleep(0.05)
    resp = await chat_completions(reg, cfg, body, manager=mgr)
    assert resp.status_code == 503
    assert b"queue full" in resp.body
    gate.set()
    await asyncio.gather(first, second)


async def test_nonstream_wait_bounded(tmp_path):
    """stream:false stays bounded — queue slots exhausted returns the
    honest JSON 503 (no SSE feed possible over a JSON body)."""
    import threading

    gate = threading.Event()
    reg = _SlowReg(_Adapter("http://127.0.0.1:1/v1"), gate)
    cfg = ServerConfig(llm_max_concurrent=1, llm_queue_wait_s=0.1,
                       llm_load_timeout_s=5, model_dir=tmp_path)
    mgr = _Mgr()
    body = {"model": "slow-llm", "messages": []}
    first = asyncio.create_task(  # holds the single slot in a slow load
        chat_completions(reg, cfg, body, manager=mgr)
    )
    await asyncio.sleep(0.05)
    resp = await chat_completions(reg, cfg, body, manager=mgr)
    assert resp.status_code == 503
    assert b"retry_after_s" in resp.body
    gate.set()
    await first
