"""OpenAI passthrough for the LLM adapter family.

``chat_completions``/``embeddings`` resolve body['model'] to an llm
adapter class (``registry.select_llm`` — capability-aware: vision/tool
requests only match adapters that declare the support), queue through the
priority/model-aware ``_LlmQueue``, pass the memory admission gate
(``manager.admit_resolved`` — the same LRU-eviction logic the job queue
uses, minus the "load anyway" fallback), then load through the registry's
single-flight ``get()`` and proxy to the managed server's endpoint.

Scheduling (``_LlmQueue``) — score =
``waited_s + llm_queue_fair_s * (affinity + priority)``: requests for the
already-loaded/loading model batch together (no model thrash), priority
steps earn a head start, and the growing ``waited`` term keeps every
request starvation-free. Priority is classified server-side
(``auth.request_priority``): internal callers (CADX backend /
``X-Mesh-Internal`` / loopback) read ``llm_prio_internal``, external LAN
clients read ``llm_prio_external``.

``stream: true`` commits an SSE response immediately: while queued or
loading, the client receives ``: keepalive`` comments every
``llm_keepalive_s`` plus ``reasoning_content`` deltas carrying queue
position / load status — a queued request is never a silent HTTP hang.
After dispatch the upstream stream relays verbatim. Errors discovered
post-commit arrive as ``data: {"error": …}`` + ``[DONE]`` (the OpenAI
mid-stream error convention); pre-commit failures stay real HTTP statuses.

Bounds (runtime-mutable via /v1/config): ``llm_max_concurrent`` in-flight
capacity, ``llm_queue_max`` queue depth, ``llm_queue_wait_s`` non-stream
wait bound, ``llm_stream_queue_wait_s`` stream wait bound, and the
upstream watchdogs ``llm_upstream_timeout_s`` / ``llm_stream_idle_s``.

Error contract (JSON, OpenAI-ish shape):
    unknown model              → 404 {"error": ...}
    capability mismatch        → 400 {"error": ...}  (vision/tools unsupported)
    registry lacks select_llm  → 503 {"error": ...}   (llm family not wired)
    queue full / wait timeout  → 503 {"error": ..., "loading": true, retry_after_s}
    load failed / timed out    → 503 {"error": str, "loading": true?}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

from meshserver.auth import request_priority

from meshserver.logging import get_logger
from meshserver.registry import InsufficientMemoryError

log = get_logger(__name__)

# no client timeout — generation latency is unbounded (streaming included)
_CLIENT_TIMEOUT = httpx.Timeout(None)

# admission poll cadence while a queued llm request waits for host memory
_ADMIT_POLL_S = 0.5

# OpenAI-style content-part types → the adapter capability the request
# needs. Video parts decode through the same mmproj vision path (mtmd
# handles mp4/webm natively in recent llama-server builds); audio needs a
# model with an audio encoder (Qwen-Omni/Ultravox family — NOT Qwen3.8).
_VISION_PART_TYPES = frozenset(
    {"image_url", "image", "input_image", "video_url", "input_video", "video"}
)
_AUDIO_PART_TYPES = frozenset({"input_audio", "audio_url", "audio"})


def _err(status: int, message: str, **extra: Any) -> JSONResponse:
    headers = None
    if retry := extra.get("retry_after_s"):
        headers = {"Retry-After": str(int(retry))}
    return JSONResponse(
        {"error": {"message": message, **extra}}, status_code=status, headers=headers
    )


def _loading_in_progress(registry: Any, model_id: str, manager: Any = None) -> bool:
    """'another request is still loading this model' — the manager tracks
    llm loads inside its admit+load window under 'model#task' keys;
    registry stubs may additionally expose is_loading(id)/llm_loading."""
    if manager is not None:
        for k in getattr(manager, "llm_loading", {}):
            if k == model_id or k.startswith(model_id + "#"):
                return True
    probe = getattr(registry, "is_loading", None)
    if callable(probe):
        try:
            return bool(probe(model_id))
        except Exception:
            return False
    flag = getattr(registry, "llm_loading", False)
    return bool(flag(model_id)) if callable(flag) else bool(flag)


def _part_types(body: dict) -> set[str]:
    """All content-part types in the request's messages."""
    out: set[str] = set()
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type"):
                    out.add(str(part["type"]))
    return out


def _wants_vision(body: dict) -> bool:
    """Any message carrying an image or video part (mmproj path)."""
    return bool(_part_types(body) & _VISION_PART_TYPES)


def _wants_audio(body: dict) -> bool:
    """Any message carrying an audio part — needs an audio-encoder model."""
    return bool(_part_types(body) & _AUDIO_PART_TYPES)


def _select_llm(registry: Any, body: dict) -> type | None:
    """select_llm with capability filters — tolerates registry stubs that
    only implement older/simpler signatures.

    ``"omni"`` is the one-model-fits-all alias: it resolves like ``"auto"``
    but forces the full capability set (vision + tools) so clients get a
    model that serves every request shape without knowing catalog ids."""
    select_llm = registry.select_llm
    model = str(body.get("model") or "")
    omni = model == "omni"
    if omni:
        model = ""
    need_vision = _wants_vision(body) or omni
    need_tools = bool(body.get("tools")) or omni
    need_audio = _wants_audio(body)
    try:
        return select_llm(
            model,
            need_vision=need_vision,
            need_tools=need_tools,
            need_audio=need_audio,
        )
    except TypeError:
        try:
            return select_llm(model, need_vision=need_vision, need_tools=need_tools)
        except TypeError:
            return select_llm(model)


def _resolve_llm_cls(registry: Any, body: dict) -> type | JSONResponse:
    """Model id → adapter class, or an error response to return verbatim."""
    if not callable(getattr(registry, "select_llm", None)):
        return _err(503, "llm adapter family is not enabled on this server")
    model = str(body.get("model") or "")
    cls = _select_llm(registry, body)
    if cls is not None:
        # explicit ids must have weights ON DISK — a load pointed at a
        # partially-downloaded repo would stall inside a giant implicit
        # fetch or die at the readiness timeout; tell the caller to fetch
        # first instead (the 'auto' ladder already skips non-downloaded)
        if model and model not in ("auto", "omni"):
            downloaded = getattr(cls, "downloaded", None)
            if callable(downloaded):
                try:
                    ok = bool(downloaded(getattr(registry, "cfg", None)))
                except Exception:
                    ok = True  # can't tell → let the load try
                if not ok:
                    return _err(
                        404,
                        f"llm model '{cls.id}' weights are not downloaded — "
                        f"POST /v1/models/{cls.id}/download first",
                    )
        return cls
    if model and model not in ("auto", "omni"):
        known = getattr(registry, "find_class", lambda _id: None)(model)
        if known is not None and getattr(known, "kind", "") == "llm":
            missing: list[str] = []
            if _wants_vision(body) and not getattr(known, "supports_vision", False):
                missing.append("image/video input")
            if _wants_audio(body) and not getattr(known, "supports_audio", False):
                missing.append("audio input")
            if body.get("tools") and not getattr(known, "supports_tools", True):
                missing.append("tools")
            if missing:
                return _err(
                    400, f"llm model '{model}' does not support {', '.join(missing)}"
                )
            return _err(
                404, f"llm model '{model}' is not available (not downloaded or failed to load)"
            )
        return _err(404, f"unknown llm model '{model}'")
    lacks = []
    if _wants_vision(body):
        lacks.append("vision")
    if _wants_audio(body):
        lacks.append("audio")
    if body.get("tools"):
        lacks.append("tools")
    suffix = f" supporting {', '.join(lacks)}" if lacks else ""
    return _err(404, f"no llm model available{suffix}")


class _LlmEntry:
    """One queued llm request — granted a slot when the dispatcher picks it."""

    def __init__(self, model_id: str, priority: int, *, stream: bool, manager: Any) -> None:
        self.model_id = model_id
        self.priority = priority
        self.stream = stream
        self.manager = manager
        self.t0 = time.monotonic()
        self.event = asyncio.Event()


class _LlmQueue:
    """Priority/model-aware dispatcher replacing the flat semaphore.

    Grant order: max score over waiting entries,
    ``waited_s + llm_queue_fair_s * (affinity + priority)`` — affinity is
    1 when the entry's model is already loaded or currently loading (the
    request rides the single-flight load), priority is the classified
    caller priority, and ``waited`` grows unbounded so nothing starves.
    Capacity is the runtime-mutable ``llm_max_concurrent`` (0 = unbounded),
    read live on every drain."""

    def __init__(self) -> None:
        self.entries: list[_LlmEntry] = []
        self.in_flight: list[_LlmEntry] = []
        self._cond = asyncio.Condition()

    @staticmethod
    def _capacity(cfg: Any) -> float:
        effective = getattr(cfg, "effective", None)
        limit = int(effective("llm_max_concurrent") if effective else 4) or 0
        return limit if limit > 0 else float("inf")

    def _affinity_models(self, registry: Any, manager: Any) -> set[str]:
        """Models a fresh entry can ride without a new load: loaded
        registry instances + models inside the single-flight load window."""
        out = {
            aid
            for aid, inst in getattr(registry, "_instances", {}).items()
            if getattr(inst, "loaded", False)
        }
        for token in getattr(manager, "llm_loading", {}) or {}:
            out.add(str(token).split("#", 1)[0])
        return out

    async def enqueue(self, cfg: Any, registry: Any, entry: _LlmEntry) -> bool:
        effective = getattr(cfg, "effective", None)
        limit = int(effective("llm_queue_max") if effective else 32) or 0
        async with self._cond:
            if limit and len(self.entries) >= limit:
                return False
            self.entries.append(entry)
            self._sync_waiting()
            self._drain(cfg, registry)
            return True

    def position(self, cfg: Any, registry: Any, entry: _LlmEntry) -> int:
        """1-based rank by current score — the reasoning-feed position."""
        ranked = self._ranked(cfg, registry)
        try:
            return ranked.index(entry) + 1
        except ValueError:
            return 0

    def _ranked(self, cfg: Any, registry: Any) -> list[_LlmEntry]:
        now = time.monotonic()
        affinity = self._affinity_models(registry, self._any_manager())
        fair = self._fair_s(cfg)

        def score(e: _LlmEntry) -> float:
            bonus = fair * (int(e.model_id in affinity) + e.priority)
            return (now - e.t0) + bonus

        return sorted(self.entries, key=score, reverse=True)

    def _any_manager(self) -> Any:
        for e in [*self.entries, *self.in_flight]:
            if e.manager is not None:
                return e.manager
        return None

    @staticmethod
    def _fair_s(cfg: Any) -> float:
        effective = getattr(cfg, "effective", None)
        v = effective("llm_queue_fair_s") if effective else None
        return float(v if v is not None else 60)

    def _drain(self, cfg: Any, registry: Any) -> None:
        """Grant slots while capacity and waiting entries exist — caller
        holds _cond (enqueue/release paths)."""
        while len(self.in_flight) < self._capacity(cfg) and self.entries:
            ranked = self._ranked(cfg, registry)
            if not ranked:
                break
            entry = ranked[0]
            self.entries.remove(entry)
            self.in_flight.append(entry)
            entry.event.set()
        self._sync_waiting()

    def _sync_waiting(self) -> None:
        n = len(self.entries)
        seen = set()
        for e in self.entries:
            if e.manager is not None:
                seen.add(id(e.manager))
                e.manager.llm_waiting = n
        # managers no longer referenced decay to zero — they can't know
        # the queue drained past them
        for e in self.in_flight:
            if e.manager is not None and id(e.manager) not in seen:
                e.manager.llm_waiting = 0

    def remove(self, entry: _LlmEntry) -> None:
        """Dequeue a still-waiting entry (timeout/cancel path)."""
        try:
            self.entries.remove(entry)
        except ValueError:
            pass
        self._sync_waiting()

    def release(self, cfg: Any, registry: Any, entry: _LlmEntry) -> None:
        """Free a granted slot and dispatch the next waiter."""
        try:
            self.in_flight.remove(entry)
        except ValueError:
            pass
        # _drain touches the shared lists — serialize via the condition
        # lock's acquire-free path: we're on the event loop, and entries/
        # in_flight are only mutated here and in enqueue/_drain under the
        # same loop, so a plain call is race-free.
        self._drain(cfg, registry)

    def snapshot(self, cfg: Any, registry: Any) -> list[dict[str, Any]]:
        """Queue view for /v1/queue — score-ranked waiting entries."""
        ranked = self._ranked(cfg, registry)
        now = time.monotonic()
        return [
            {
                "position": i + 1,
                "model": e.model_id,
                "priority": e.priority,
                "stream": e.stream,
                "waited_s": round(now - e.t0, 1),
            }
            for i, e in enumerate(ranked)
        ]


def _llm_queue(cfg: Any) -> _LlmQueue:
    """Lazily-created per-config queue singleton (like _llm_sem before)."""
    q = getattr(cfg, "_llm_queue", None)
    if q is None:
        q = _LlmQueue()
        cfg._llm_queue = q
    return q


class _LlmSlot:
    """One in-flight llm request: queue slot + /v1/queue lane counters.
    release() is idempotent so the streaming path can delegate it into
    the relay's finally without double-free risk."""

    def __init__(
        self,
        cfg: Any,
        registry: Any,
        manager: Any,
        entry: _LlmEntry,
        queue: _LlmQueue,
    ) -> None:
        self._cfg = cfg
        self._registry = registry
        self._manager = manager
        self._entry = entry
        self._queue = queue
        self._released = False
        if manager is not None:
            model_id = entry.model_id
            manager.llm_in_flight[model_id] = manager.llm_in_flight.get(model_id, 0) + 1

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._queue.release(self._cfg, self._registry, self._entry)
        if self._manager is not None:
            model_id = self._entry.model_id
            cur = self._manager.llm_in_flight.get(model_id, 0)
            if cur <= 1:
                self._manager.llm_in_flight.pop(model_id, None)
            else:
                self._manager.llm_in_flight[model_id] = cur - 1


async def _acquire_slot(
    cfg: Any,
    manager: Any,
    model_id: str,
    registry: Any,
    *,
    priority: int = 0,
    stream: bool = False,
    prog: dict | None = None,
    deadline: float | None = None,
) -> _LlmSlot | JSONResponse:
    """Enqueue + wait for a dispatcher grant. Non-stream callers bound the
    wait at llm_queue_wait_s; stream callers get llm_stream_queue_wait_s
    (their client sees keepalive progress the whole time)."""
    queue = _llm_queue(cfg)
    entry = _LlmEntry(model_id, priority, stream=stream, manager=manager)
    if not await queue.enqueue(cfg, registry, entry):
        return _err(
            503,
            "llm request queue full — retry shortly",
            loading=True,
            retry_after_s=5,
        )
    if entry.event.is_set():  # granted inside enqueue's drain
        return _LlmSlot(cfg, registry, manager, entry, queue)
    effective = getattr(cfg, "effective", None)
    if deadline is None:
        wait_v = effective(
            "llm_stream_queue_wait_s" if stream else "llm_queue_wait_s"
        ) if effective else None
        wait_s = float(wait_v if wait_v is not None else (600 if stream else 30))
        deadline = time.monotonic() + wait_s
    try:
        while not entry.event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            if prog is not None:
                prog["position"] = queue.position(cfg, registry, entry)
            try:
                await asyncio.wait_for(
                    entry.event.wait(), timeout=min(remaining, 0.5)
                )
            except TimeoutError:
                continue  # poll cadence — refresh position, re-check deadline
    except asyncio.CancelledError:
        # caller abandoned while queued (client disconnect / task cancel) —
        # drop the entry so it can't be granted into the void
        queue.remove(entry)
        raise
    except TimeoutError:
        # grant-vs-timeout race: the dispatcher may have moved the entry
        # into in_flight at the same moment the deadline hit — the slot is
        # real, serving the request beats a spurious 503 (and dropping it
        # would leak the capacity slot forever)
        if entry.event.is_set():
            return _LlmSlot(cfg, registry, manager, entry, queue)
        queue.remove(entry)
        return _err(
            503,
            "llm request slots exhausted — retry shortly",
            loading=True,
            retry_after_s=5,
        )
    return _LlmSlot(cfg, registry, manager, entry, queue)


async def _load_llm(
    registry: Any,
    cfg: Any,
    manager: Any,
    cls: type,
    *,
    prog: dict | None = None,
    deadline: float | None = None,
) -> Any | JSONResponse:
    """Memory admission + single-flight load. The admission loop gives the
    host until `deadline` (default: now + llm_queue_wait_s) to free enough
    memory (other loads finishing, idle unloads, evictions) — then reports
    a deferrable 503 instead of falling into registry.get()'s 'load
    anyway' OOM path."""
    # already resident → no new memory needed, skip admission entirely
    inst = getattr(registry, "_instances", {}).get(cls.id)
    resident = bool(getattr(inst, "loaded", False))
    effective = getattr(cfg, "effective", None)
    wait_v = effective("llm_queue_wait_s") if effective else None
    wait_s = float(wait_v if wait_v is not None else 30)
    if deadline is None:
        deadline = time.monotonic() + wait_s
    # the loading marker spans admit+load; keyed by request so concurrent
    # same-model requests don't clobber each other (a second request
    # waiting on the single-flight load is what "loading: true" reports)
    token = f"{cls.id}#{id(asyncio.current_task()):x}"
    if manager is not None and not resident:
        manager.llm_loading[token] = time.monotonic()
    try:
        if manager is not None and not resident:
            if prog is not None:
                prog["state"] = "loading"
            while not manager.admit_resolved(cls.id):
                if time.monotonic() >= deadline:
                    return _err(
                        503,
                        f"insufficient host memory to load '{cls.id}' — "
                        "another resident model holds the budget",
                        loading=True,
                        retry_after_s=5,
                    )
                await asyncio.sleep(_ADMIT_POLL_S)
        timeout = float(getattr(cfg, "llm_load_timeout_s", 300) or 300)
        try:
            # registry.get() blocks on the registry lock + adapter.load() —
            # run it in a thread so concurrent requests converge on the same
            # single-flight load instead of piling onto the event loop.
            # strict: a bounded caller defers honestly rather than falling
            # into the "load anyway" OOM path; in-flight llm models are
            # protected from eviction mid-stream.
            in_use = getattr(manager, "llm_models_in_use", None)
            protected = in_use() if callable(in_use) else set()
            get = getattr(registry, "get")
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        get, cls, protected=protected, strict=True
                    ),
                    timeout=timeout + 30,  # grace over the subprocess readiness bound
                )
            except TypeError:
                # registry stubs without the new kwargs — degrade to the
                # legacy call (tests/stubs only; real Registry is strict)
                return await asyncio.wait_for(
                    asyncio.to_thread(get, cls),
                    timeout=timeout + 30,
                )
        except Exception as e:
            if manager is not None:
                manager.llm_loading.pop(token, None)  # own load is over —
                # a surviving key belongs to a concurrent request still
                # inside the single-flight window
            # a wait_for TimeoutError stringifies to "" — name it instead of
            # shipping an empty error message
            message = str(e) or f"llm load of '{cls.id}' timed out"
            payload: dict[str, Any] = {"message": message}
            # a strict-fit rejection is retryable by definition — the caller
            # should come back once a resident finishes/unloads
            retryable = isinstance(e, InsufficientMemoryError)
            headers = None
            if retryable or _loading_in_progress(
                registry, getattr(cls, "id", ""), manager
            ):
                payload["loading"] = True
                payload["retry_after_s"] = 5
                headers = {"Retry-After": "5"}
            return JSONResponse(
                {"error": payload}, status_code=503, headers=headers
            )
    finally:
        if manager is not None:
            manager.llm_loading.pop(token, None)


async def _prepare_llm(
    registry: Any,
    cfg: Any,
    manager: Any,
    body: dict,
    *,
    priority: int = 0,
    stream: bool = False,
    prog: dict | None = None,
    deadline: float | None = None,
) -> tuple[Any, _LlmSlot | None] | JSONResponse:
    """resolve → queue slot → load — shared by chat completions and
    embeddings. Returns (adapter, slot) or the error response to send.
    `prog` (when given) receives {state, position} updates the caller's
    SSE keepalive loop reads."""
    cls_or_err = _resolve_llm_cls(registry, body)
    if isinstance(cls_or_err, JSONResponse):
        return cls_or_err
    cls = cls_or_err
    slot_or_err = await _acquire_slot(
        cfg,
        manager,
        cls.id,
        registry,
        priority=priority,
        stream=stream,
        prog=prog,
        deadline=deadline,
    )
    if isinstance(slot_or_err, JSONResponse):
        return slot_or_err
    slot = slot_or_err
    try:
        loaded = await _load_llm(registry, cfg, manager, cls, prog=prog, deadline=deadline)
    except BaseException:
        # CancelledError is NOT an Exception — a disconnect mid-admission/
        # mid-load would otherwise propagate past here holding a granted
        # capacity slot forever
        slot.release()
        raise
    if isinstance(loaded, JSONResponse):
        slot.release()
        return loaded
    return (loaded, slot)


def _sse_headers() -> dict[str, str]:
    """SSE responses must not be buffered — X-Accel-Buffering stops nginx-
    style proxies, Cache-Control stops generic caches."""
    return {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
    }


def _reasoning_chunk(model_id: str, text: str) -> bytes:
    """One SSE chunk carrying a status line as reasoning_content — the
    field llama.cpp/mlx thinking models already use, so OpenAI-compatible
    clients render queue/load progress natively."""
    chunk = {
        "id": "chatcmpl-meshqueue",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}
        ],
    }
    return f"data: {_dumps(chunk)}\n\n".encode()


def _error_chunk(message: str) -> bytes:
    return f'data: {_dumps({"error": {"message": message}})}\n\ndata: [DONE]\n\n'.encode()


def _dumps(obj: Any) -> str:
    """JSON wire format for relayed chunks — compact separators keep an
    untouched upstream chunk byte-identical; ensure_ascii=False keeps
    UTF-8 verbatim instead of inflating it into \\uXXXX escapes."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ── reasoning wire format ────────────────────────────────────────────────
# Thinking models (Qwen3.8-thinking, DeepSeek-R1, gpt-oss…) emit their
# chain-of-thought inside <think>…</think> tags. Adapters differ: llama.cpp
# can split it server-side (--reasoning-format), mlx_lm/mlx_vlm always
# stream the raw tags inside `content`. The OpenAI/DeepSeek wire format
# carries reasoning in a SEPARATE field — `delta.reasoning_content` /
# `message.reasoning_content` — so the proxy normalizes every lane: tags
# in content are split out into reasoning_content, chunks that already
# carry reasoning_content pass through untouched.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_reasoning_text(text: str) -> tuple[str | None, str]:
    """Non-stream variant: parse every <think>…</think> block out of a
    completion string → (reasoning, visible_content). An unterminated
    block counts as reasoning to the end."""
    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    rest = text
    while True:
        i = rest.find(_THINK_OPEN)
        if i < 0:
            visible_parts.append(rest)
            break
        visible_parts.append(rest[:i])
        rest = rest[i + len(_THINK_OPEN):]
        j = rest.find(_THINK_CLOSE)
        if j < 0:
            reasoning_parts.append(rest)
            rest = ""
            break
        reasoning_parts.append(rest[:j])
        rest = rest[j + len(_THINK_CLOSE):]
    reasoning = "".join(reasoning_parts).strip()
    return (reasoning or None), "".join(visible_parts).strip()


def transform_reasoning_json(obj: Any) -> bool:
    """Split think blocks out of `choices[i].message.content` into
    `message.reasoning_content` — the non-stream wire format llama.cpp /
    vLLM emit. Returns True when a choice was rewritten."""
    changed = False
    for choice in (obj or {}).get("choices") or []:
        msg = (choice or {}).get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or _THINK_OPEN not in content:
            continue
        reasoning, visible = split_reasoning_text(content)
        if reasoning is None:
            continue
        msg["reasoning_content"] = reasoning
        msg["content"] = visible
        changed = True
    return changed


class _SseEvents:
    """Raw upstream bytes → complete SSE events (split on blank lines)."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        """Complete events, WITHOUT the \\n\\n terminator. CRLF line
        endings (legal SSE) normalize to LF on ingest — a \\r\\n\\r\\n
        boundary would otherwise never match and the buffer would grow
        while the client sees nothing."""
        self._buf += chunk.replace(b"\r\n", b"\n")
        out: list[bytes] = []
        while (i := self._buf.find(b"\n\n")) >= 0:
            out.append(self._buf[:i])
            self._buf = self._buf[i + 2:]
        return out

    def flush(self) -> bytes:
        tail, self._buf = self._buf, b""
        return tail


class _ReasoningStream:
    """Stateful SSE transformer — moves think-tag regions out of
    `delta.content` into `delta.reasoning_content` across arbitrary
    chunk boundaries (a tag may be split mid-token by the upstream)."""

    def __init__(self) -> None:
        self.in_think = False
        self._tail = ""  # held-back partial tag prefix
        self._last_id = "chatcmpl-mesh"
        self._last_model = "mesh"

    @staticmethod
    def _partial_suffix(s: str, tag: str) -> int:
        """Longest proper prefix of `tag` that ends `s` — that many
        chars must be held back (they could complete the tag in the
        next chunk)."""
        for n in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:n]):
                return n
        return 0

    def _split(self, text: str) -> tuple[str, str]:
        s = self._tail + text
        self._tail = ""
        reasoning: list[str] = []
        visible: list[str] = []
        buf = s
        while buf:
            tag = _THINK_CLOSE if self.in_think else _THINK_OPEN
            target = reasoning if self.in_think else visible
            idx = buf.find(tag)
            if idx >= 0:
                target.append(buf[:idx])
                buf = buf[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            # no full tag — hold back a trailing partial prefix
            hold = self._partial_suffix(buf, tag)
            take = len(buf) - hold
            target.append(buf[:take])
            self._tail = buf[take:]
            buf = ""
        return "".join(reasoning), "".join(visible)

    def finish(self) -> bytes | None:
        """Flush a partial-tag tail left dangling at stream end — it goes
        to whichever channel was active."""
        if not self._tail:
            return None
        tail, self._tail = self._tail, ""
        delta = {"reasoning_content": tail} if self.in_think else {"content": tail}
        chunk = {
            "id": self._last_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self._last_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        return f"data: {_dumps(chunk)}\n\n".encode()

    def feed_event(self, event: bytes) -> list[bytes]:
        """One complete SSE event → one or more output events (bytes,
        each \\n\\n-terminated). [DONE] events get the tail flushed
        BEFORE the terminator."""
        if event.strip().removeprefix(b"data:").strip() == b"[DONE]":
            out = []
            flushed = self.finish()
            if flushed is not None:
                out.append(flushed)
            out.append(event + b"\n\n")
            return out
        lines = event.split(b"\n")
        out_lines: list[bytes] = []
        for ln in lines:
            if not ln.startswith(b"data:"):
                out_lines.append(ln)
                continue
            payload = ln[5:].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                out_lines.append(ln)  # not JSON — pass through verbatim
                continue
            if isinstance(obj, dict):
                self._transform(obj)
            # compact separators keep an untouched chunk byte-identical
            # to what the upstream emitted (minimal wire delta)
            out_lines.append(b"data: " + _dumps(obj).encode())
        return [b"\n".join(out_lines) + b"\n\n"]

    def _transform(self, obj: dict) -> None:
        self._last_id = obj.get("id", self._last_id)
        self._last_model = obj.get("model", self._last_model)
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                continue
            reasoning, visible = self._split(content)
            delta.pop("content", None)
            if reasoning:
                prev = delta.get("reasoning_content")
                delta["reasoning_content"] = (prev or "") + reasoning
            if visible:
                delta["content"] = visible


async def _chat_stream(
    registry: Any,
    cfg: Any,
    manager: Any,
    body: dict,
    cls: type,
    priority: int,
) -> Response:
    """stream:true path — the SSE response commits immediately; queue+load
    run inside the generator while keepalive comments and
    reasoning_content status deltas flow to the client."""
    payload = dict(body)

    async def gen():
        effective = getattr(cfg, "effective", None)
        keep_v = effective("llm_keepalive_s") if effective else None
        keep_s = max(float(keep_v if keep_v is not None else 5), 0.5)
        wait_v = effective("llm_stream_queue_wait_s") if effective else None
        wait_s = float(wait_v if wait_v is not None else 600)
        prog: dict[str, Any] = {"state": "queued", "position": None}
        task = asyncio.create_task(
            _prepare_llm(
                registry,
                cfg,
                manager,
                body,
                priority=priority,
                stream=True,
                prog=prog,
                deadline=time.monotonic() + wait_s,
            )
        )
        last_status = ""
        last_status_at = 0.0
        try:
            try:
                while True:
                    done, _pending = await asyncio.wait({task}, timeout=keep_s)
                    if done:
                        break
                    yield b": keepalive\n\n"
                    state = prog.get("state", "queued")
                    pos = prog.get("position")
                    if state == "loading":
                        status = f"loading model '{cls.id}'…"
                    elif pos:
                        status = f"queued #{pos} — waiting for a '{cls.id}' slot…"
                    else:
                        status = f"queued — waiting for a '{cls.id}' slot…"
                    now = time.monotonic()
                    # status changes emit immediately; steady-state repeats
                    # every ~10s so clients see a living position
                    if status != last_status or now - last_status_at > 10:
                        yield _reasoning_chunk(cls.id, status)
                        last_status = status
                        last_status_at = now
                prepared = task.result()
            except Exception as e:
                yield _error_chunk(f"llm prepare failed: {e}")
                return
        finally:
            # client disconnect mid-queue/load: cancel prepare — its own
            # cleanup drops the queue entry so no slot leaks into the void
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if isinstance(prepared, JSONResponse):
            try:
                err_body = json.loads(prepared.body).get("error", {})
                message = str(err_body.get("message") or "llm request failed")
            except Exception:
                message = "llm request failed"
            yield _error_chunk(message)
            return
        adapter, slot = prepared
        url = adapter.base_url() + "/chat/completions"
        payload["model"] = adapter.served_model_id()
        # upstream open + verbatim relay (idle watchdog bounds silent gaps)
        client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
        ctx = None
        try:
            ctx = client.stream("POST", url, json=payload)
            upstream = await ctx.__aenter__()
            if upstream.status_code >= 400:
                data = await upstream.aread()
                try:
                    message = str(
                        json.loads(data).get("error", {}).get("message")
                        or f"upstream error {upstream.status_code}"
                    )
                except Exception:
                    message = f"upstream error {upstream.status_code}"
                yield _error_chunk(message)
                return
            idle_v = effective("llm_stream_idle_s") if effective else None
            idle_s = float(idle_v) if idle_v else None
            # reasoning normalization: when the upstream streams raw
            # <think> tags inside content (mlx always, gguf without
            # --reasoning-format), split them into reasoning_content so
            # every client sees the OpenAI wire format — llama.cpp's own
            # server-side split passes through untouched (no tags left)
            rp = effective("llm_reasoning_parse") if effective else None
            reasoner = _ReasoningStream() if (rp is None or rp) else None
            sse = _SseEvents()
            aiter = upstream.aiter_raw()
            while True:
                try:
                    if idle_s:
                        chunk = await asyncio.wait_for(aiter.__anext__(), idle_s)
                    else:
                        chunk = await aiter.__anext__()
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    yield _error_chunk(
                        f"upstream idle for {idle_s:.0f}s — aborted"
                    )
                    return
                if reasoner is None:
                    yield chunk
                    continue
                for ev in sse.feed(chunk):
                    for out in reasoner.feed_event(ev):
                        yield out
            if reasoner is not None:
                # upstream closed without [DONE] — flush a held-back
                # partial-tag tail, then any unterminated buffer bytes
                flushed = reasoner.finish()
                if flushed is not None:
                    yield flushed
                tail = sse.flush()
                if tail:
                    yield tail
        except httpx.HTTPError as e:
            yield _error_chunk(f"llm upstream unreachable: {e}")
            return
        finally:
            if ctx is not None:
                with contextlib.suppress(Exception):
                    await ctx.__aexit__(None, None, None)
            await client.aclose()
            slot.release()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


async def chat_completions(
    registry: Any,
    cfg: Any,
    body: dict,
    manager: Any = None,
    request: Any = None,
) -> Response:
    """POST /v1/chat/completions — resolve, queue, load, proxy."""
    if not isinstance(body, dict):
        return _err(422, "request body must be a JSON object")
    priority = request_priority(
        request,
        cfg,
        internal_key="llm_prio_internal",
        external_key="llm_prio_external",
    )
    if body.get("stream"):
        # resolve first so capability/model errors keep their HTTP status;
        # queue+load then run inside the SSE generator (keepalives flow)
        cls_or_err = _resolve_llm_cls(registry, body)
        if isinstance(cls_or_err, JSONResponse):
            return cls_or_err
        return await _chat_stream(
            registry, cfg, manager, body, cls_or_err, priority
        )
    prepared = await _prepare_llm(registry, cfg, manager, body, priority=priority)
    if isinstance(prepared, JSONResponse):
        return prepared
    adapter, slot = prepared
    upstream_url = adapter.base_url() + "/chat/completions"
    payload = dict(body)
    payload["model"] = adapter.served_model_id()
    try:
        return await _proxy_json(upstream_url, payload, str(body.get("model") or ""), cfg)
    finally:
        slot.release()


async def embeddings(
    registry: Any,
    cfg: Any,
    body: dict,
    manager: Any = None,
    request: Any = None,
) -> Response:
    """POST /v1/embeddings — same resolve/admit/load path, JSON passthrough
    to the managed server's /v1/embeddings (llama.cpp + mlx_lm both serve
    it; a missing endpoint relays its own status code)."""
    if not isinstance(body, dict):
        return _err(422, "request body must be a JSON object")
    prepared = await _prepare_llm(
        registry,
        cfg,
        manager,
        body,
        priority=request_priority(
            request,
            cfg,
            internal_key="llm_prio_internal",
            external_key="llm_prio_external",
        ),
    )
    if isinstance(prepared, JSONResponse):
        return prepared
    adapter, slot = prepared
    try:
        return await _proxy_json(
            adapter.base_url() + "/embeddings",
            dict(body, model=adapter.served_model_id()),
            str(body.get("model") or ""),
            cfg,
        )
    finally:
        slot.release()


_json_client: httpx.AsyncClient | None = None


def _shared_json_client() -> httpx.AsyncClient:
    """One keep-alive client for non-stream upstream posts — per-request
    clients paid a fresh loopback TCP+header setup on every call."""
    global _json_client
    if _json_client is None or _json_client.is_closed:
        _json_client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
    return _json_client


async def close_shared_client() -> None:
    """Lifespan teardown — release the shared keep-alive client's connection
    pool instead of letting it outlive the app's event loop."""
    global _json_client
    client, _json_client = _json_client, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


async def _proxy_json(url: str, payload: dict, model: str, cfg: Any = None) -> Response:
    effective = getattr(cfg, "effective", None)
    up_v = effective("llm_upstream_timeout_s") if effective else None
    up_s = float(up_v) if up_v else None
    try:
        post = _shared_json_client().post(url, json=payload)
        r = await asyncio.wait_for(post, timeout=up_s) if up_s else await post
    except TimeoutError:
        log.warning("llm.proxy.upstream_timeout", model=model, timeout_s=up_s)
        return _err(504, f"llm upstream did not respond within {up_s:.0f}s")
    except RuntimeError:
        # a shared client outlives the loop it bound to (test loops, worker
        # restarts) — drop it once and retry on a fresh client
        global _json_client
        _json_client = None
        try:
            r = await _shared_json_client().post(url, json=payload)
        except (httpx.HTTPError, RuntimeError) as e:
            log.warning("llm.proxy.upstream_error", model=model, error=str(e))
            return _err(502, f"llm upstream unreachable: {e}")
    except httpx.HTTPError as e:
        log.warning("llm.proxy.upstream_error", model=model, error=str(e))
        return _err(502, f"llm upstream unreachable: {e}")
    content = r.content
    # reasoning normalization (non-stream): raw <think> blocks inside
    # message.content get split into message.reasoning_content so LAN
    # clients see the same wire format stream:true already delivers
    rp = effective("llm_reasoning_parse") if effective else None
    if (rp is None or rp) and b"<think>" in content:
        try:
            obj = json.loads(content)
        except Exception:
            obj = None
        if obj is not None and transform_reasoning_json(obj):
            content = _dumps(obj).encode()
    return Response(
        content=content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


async def _proxy_stream(url: str, payload: dict, on_done: Any = None) -> Response:
    """Open the upstream stream first (surfacing its real status code), then
    relay SSE chunks verbatim; client + stream close when the relay ends.
    ``on_done`` (the slot release) fires exactly once — in the relay's
    finally or on the early error paths before the stream exists."""
    client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
    ctx = client.stream("POST", url, json=payload)
    try:
        upstream = await ctx.__aenter__()
    except httpx.HTTPError as e:
        await client.aclose()
        if on_done is not None:
            on_done()
        return _err(502, f"llm upstream unreachable: {e}")
    if upstream.status_code >= 400:
        data = await upstream.aread()
        await ctx.__aexit__(None, None, None)
        await client.aclose()
        if on_done is not None:
            on_done()
        return Response(
            content=data,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await ctx.__aexit__(None, None, None)
            await client.aclose()
            if on_done is not None:
                on_done()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=_sse_headers(),
    )


def _llm_classes(registry: Any) -> list[type]:
    classes = getattr(registry, "_llm_classes", None)
    if classes is not None:
        return list(classes)
    classes = getattr(registry, "_classes", None)
    if classes is not None:
        return [c for c in classes if getattr(c, "kind", "") == "llm"]
    out = []
    models_fn = getattr(registry, "models", None)
    if callable(models_fn):
        for info in models_fn():
            if getattr(info, "kind", "") == "llm":
                out.append(info)
    return out


async def llm_models(registry: Any) -> dict:
    """OpenAI /v1/models-style listing for llm entries — the app merges this
    with the management /v1/models payload during integration."""
    infos: dict[str, Any] = {}
    models_fn = getattr(registry, "models", None)
    if callable(models_fn):
        infos = {m.id: m for m in models_fn()}
    # effective served ctx — the sidecar config override (llm_ctx) beats the
    # adapter's catalog max (same resolution as llm_gguf.server_argv); the
    # runtime overlay can raise it between model loads
    cfg = getattr(registry, "cfg", None)
    eff = getattr(cfg, "effective", None)
    cfg_ctx = int(
        (eff("llm_ctx") if callable(eff) else getattr(cfg, "llm_ctx", 0) if cfg else 0) or 0
    )
    data: list[dict[str, Any]] = []
    for cls in _llm_classes(registry):
        info = infos.get(getattr(cls, "id", ""))
        data.append(
            {
                "id": getattr(cls, "id", "?"),
                "object": "model",
                "created": 0,
                "owned_by": "meshserver",
                "supports_vision": getattr(cls, "supports_vision", False),
                "supports_tools": getattr(cls, "supports_tools", True),
                "context_len": cfg_ctx or getattr(cls, "context_len", None),
                "downloaded": getattr(info, "downloaded", True) if info else True,
                "loaded": getattr(info, "loaded", False) if info else False,
            }
        )
    if data:
        # discovery alias — clients point at "omni" and get the full-
        # capability auto-pick (vision + tools) without knowing catalog ids
        data.insert(
            0,
            {
                "id": "omni",
                "object": "model",
                "created": 0,
                "owned_by": "meshserver",
                "supports_vision": True,
                "supports_tools": True,
                "alias": True,
            },
        )
    return {"object": "list", "data": data}
