"""OpenAI passthrough for the LLM adapter family.

``chat_completions``/``embeddings`` resolve body['model'] to an llm
adapter class (``registry.select_llm`` — capability-aware: vision/tool
requests only match adapters that declare the support), pass the memory
admission gate (``manager.admit_resolved`` — the same LRU-eviction logic
the job queue uses, minus the "load anyway" fallback), then load through
the registry's single-flight ``get()`` and proxy to the managed server's
endpoint. ``stream: true`` bodies are relayed as verbatim SSE.

Concurrency is bounded two ways, both runtime-mutable via /v1/config:
``llm_max_concurrent`` — a proxy-side semaphore over the whole upstream
call (load + generation + stream), and ``llm_queue_wait_s`` — the bounded
wait for a semaphore slot or memory admission before an honest
``503 {loading: true, retry_after_s}`` instead of an unbounded stall.

Error contract (JSON, OpenAI-ish shape):
    unknown model              → 404 {"error": ...}
    capability mismatch        → 400 {"error": ...}  (vision/tools unsupported)
    registry lacks select_llm  → 503 {"error": ...}   (llm family not wired)
    admission/slot wait timeout→ 503 {"error": ..., "loading": true, retry_after_s}
    load failed / timed out    → 503 {"error": str, "loading": true?}
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

from meshserver.logging import get_logger

log = get_logger(__name__)

# no client timeout — generation latency is unbounded (streaming included)
_CLIENT_TIMEOUT = httpx.Timeout(None)

# admission poll cadence while a queued llm request waits for host memory
_ADMIT_POLL_S = 0.5

_IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})


def _err(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"message": message, **extra}}, status_code=status)


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


def _wants_vision(body: dict) -> bool:
    """Any message carrying an image part (OpenAI content-part list)."""
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    return True
    return False


def _select_llm(registry: Any, body: dict) -> type | None:
    """select_llm with capability filters — tolerates registry stubs that
    only implement the single-arg form."""
    select_llm = registry.select_llm
    model = str(body.get("model") or "")
    need_vision = _wants_vision(body)
    need_tools = bool(body.get("tools"))
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
        return cls
    if model and model != "auto":
        known = getattr(registry, "find_class", lambda _id: None)(model)
        if known is not None and getattr(known, "kind", "") == "llm":
            missing: list[str] = []
            if _wants_vision(body) and not getattr(known, "supports_vision", False):
                missing.append("image input")
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
    if body.get("tools"):
        lacks.append("tools")
    suffix = f" supporting {', '.join(lacks)}" if lacks else ""
    return _err(404, f"no llm model available{suffix}")


def _llm_sem(cfg: Any) -> asyncio.Semaphore | None:
    """Request-slot semaphore sized from the runtime-mutable
    llm_max_concurrent (0 = unbounded). A changed limit swaps in a fresh
    semaphore — in-flight requests finish on the old one."""
    effective = getattr(cfg, "effective", None)
    limit = int(effective("llm_max_concurrent") if effective else 4) or 0
    if limit <= 0:
        return None
    sem = getattr(cfg, "_llm_sem", None)
    if sem is None or getattr(sem, "_mesh_limit", 0) != limit:
        sem = asyncio.Semaphore(limit)
        sem._mesh_limit = limit  # type: ignore[attr-defined]
        cfg._llm_sem = sem
    return sem


class _LlmSlot:
    """One in-flight llm request: semaphore lease + /v1/queue lane
    counters. release() is idempotent so the streaming path can delegate
    it into the relay's finally without double-free risk."""

    def __init__(self, manager: Any, sem: asyncio.Semaphore | None, model_id: str) -> None:
        self._manager = manager
        self._sem = sem
        self._model_id = model_id
        self._released = False
        if manager is not None:
            manager.llm_in_flight[model_id] = manager.llm_in_flight.get(model_id, 0) + 1

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._sem is not None:
            self._sem.release()
        if self._manager is not None:
            cur = self._manager.llm_in_flight.get(self._model_id, 0)
            if cur <= 1:
                self._manager.llm_in_flight.pop(self._model_id, None)
            else:
                self._manager.llm_in_flight[self._model_id] = cur - 1


async def _acquire_slot(cfg: Any, manager: Any, model_id: str) -> _LlmSlot | JSONResponse:
    effective = getattr(cfg, "effective", None)
    wait_v = effective("llm_queue_wait_s") if effective else None
    wait_s = float(wait_v if wait_v is not None else 30)
    sem = _llm_sem(cfg)
    if sem is not None:
        if manager is not None:
            manager.llm_waiting += 1
        try:
            await asyncio.wait_for(sem.acquire(), timeout=wait_s)
        except (TimeoutError, asyncio.TimeoutError):
            return _err(
                503,
                f"llm request slots exhausted ({getattr(sem, '_mesh_limit', '?')} "
                "in flight) — retry shortly",
                loading=True,
                retry_after_s=5,
            )
        finally:
            if manager is not None:
                manager.llm_waiting = max(0, manager.llm_waiting - 1)
    return _LlmSlot(manager, sem, model_id)


async def _load_llm(
    registry: Any, cfg: Any, manager: Any, cls: type
) -> Any | JSONResponse:
    """Memory admission + single-flight load. The admission loop gives the
    host llm_queue_wait_s to free enough memory (other loads finishing,
    idle unloads, evictions) — then reports a deferrable 503 instead of
    falling into registry.get()'s 'load anyway' OOM path."""
    # already resident → no new memory needed, skip admission entirely
    inst = getattr(registry, "_instances", {}).get(cls.id)
    resident = bool(getattr(inst, "loaded", False))
    effective = getattr(cfg, "effective", None)
    wait_v = effective("llm_queue_wait_s") if effective else None
    wait_s = float(wait_v if wait_v is not None else 30)
    # the loading marker spans admit+load; keyed by request so concurrent
    # same-model requests don't clobber each other (a second request
    # waiting on the single-flight load is what "loading: true" reports)
    token = f"{cls.id}#{id(asyncio.current_task()):x}"
    if manager is not None and not resident:
        manager.llm_loading[token] = time.monotonic()
    try:
        if manager is not None and not resident:
            deadline = time.monotonic() + wait_s
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
            # single-flight load instead of piling onto the event loop
            return await asyncio.wait_for(
                asyncio.to_thread(registry.get, cls),
                timeout=timeout + 30,  # grace over the subprocess readiness bound
            )
        except Exception as e:
            if manager is not None:
                manager.llm_loading.pop(token, None)  # own load is over —
                # a surviving key belongs to a concurrent request still
                # inside the single-flight window
            payload: dict[str, Any] = {"message": str(e)}
            if _loading_in_progress(registry, getattr(cls, "id", ""), manager):
                payload["loading"] = True
            return JSONResponse({"error": payload}, status_code=503)
    finally:
        if manager is not None:
            manager.llm_loading.pop(token, None)


async def _prepare_llm(
    registry: Any, cfg: Any, manager: Any, body: dict
) -> tuple[Any, _LlmSlot | None] | JSONResponse:
    """resolve → slot → load — shared by chat completions and embeddings.
    Returns (adapter, slot) or the error response to send."""
    cls_or_err = _resolve_llm_cls(registry, body)
    if isinstance(cls_or_err, JSONResponse):
        return cls_or_err
    cls = cls_or_err
    slot_or_err = await _acquire_slot(cfg, manager, cls.id)
    if isinstance(slot_or_err, JSONResponse):
        return slot_or_err
    slot = slot_or_err
    loaded = await _load_llm(registry, cfg, manager, cls)
    if isinstance(loaded, JSONResponse):
        slot.release()
        return loaded
    return (loaded, slot)


async def chat_completions(
    registry: Any, cfg: Any, body: dict, manager: Any = None
) -> Response:
    """POST /v1/chat/completions — resolve, admit (bounded), load, proxy."""
    prepared = await _prepare_llm(registry, cfg, manager, body)
    if isinstance(prepared, JSONResponse):
        return prepared
    adapter, slot = prepared
    upstream_url = adapter.base_url() + "/chat/completions"
    payload = dict(body)
    payload["model"] = adapter.served_model_id()

    if body.get("stream"):
        # the slot lives until the relay finishes (or the client drops) —
        # release is delegated into the stream's finally
        resp = await _proxy_stream(upstream_url, payload, on_done=slot.release)
        if not isinstance(resp, StreamingResponse):
            slot.release()
        return resp
    try:
        return await _proxy_json(upstream_url, payload, str(body.get("model") or ""))
    finally:
        slot.release()


async def embeddings(
    registry: Any, cfg: Any, body: dict, manager: Any = None
) -> Response:
    """POST /v1/embeddings — same resolve/admit/load path, JSON passthrough
    to the managed server's /v1/embeddings (llama.cpp + mlx_lm both serve
    it; a missing endpoint relays its own status code)."""
    prepared = await _prepare_llm(registry, cfg, manager, body)
    if isinstance(prepared, JSONResponse):
        return prepared
    adapter, slot = prepared
    try:
        return await _proxy_json(
            adapter.base_url() + "/embeddings",
            dict(body, model=adapter.served_model_id()),
            str(body.get("model") or ""),
        )
    finally:
        slot.release()


async def _proxy_json(url: str, payload: dict, model: str) -> Response:
    try:
        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
            r = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        log.warning("llm.proxy.upstream_error", model=model, error=str(e))
        return _err(502, f"llm upstream unreachable: {e}")
    return Response(
        content=r.content,
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
    )


def _llm_classes(registry: Any) -> list[type]:
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
    # adapter's catalog max (same resolution as llm_gguf.server_argv)
    cfg_ctx = int(getattr(getattr(registry, "cfg", None), "llm_ctx", 0) or 0)
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
    return {"object": "list", "data": data}
