"""OpenAI passthrough for the LLM adapter family.

``chat_completions`` resolves body['model'] to an llm adapter class (via
``registry.select_llm`` — wired into Registry during integration), loads the
managed server through the registry's single-flight ``get()`` (the registry
lock already serializes concurrent loads — parallel requests while a model
spins up converge on ONE load), then proxies the request to the
subprocess's /v1/chat/completions with ``model`` rewritten to whatever id
the inner server actually serves. ``stream: true`` bodies are relayed as
verbatim SSE.

Error contract (JSON, OpenAI-ish shape):
    unknown model              → 404 {"error": ...}
    registry lacks select_llm  → 503 {"error": ...}   (llm family not wired)
    load failed / timed out    → 503 {"error": str, "loading": true?}
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

from meshserver.logging import get_logger

log = get_logger(__name__)

# no client timeout — generation latency is unbounded (streaming included)
_CLIENT_TIMEOUT = httpx.Timeout(None)


def _err(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"message": message, **extra}}, status_code=status)


def _loading_in_progress(registry: Any, model_id: str) -> bool:
    """Probe the registry for a 'load in flight' signal — integration adds
    ``select_llm``; a ``is_loading(id)`` method or ``llm_loading`` flag may
    ride along, both optional."""
    probe = getattr(registry, "is_loading", None)
    if callable(probe):
        try:
            return bool(probe(model_id))
        except Exception:
            return False
    flag = getattr(registry, "llm_loading", False)
    return bool(flag(model_id)) if callable(flag) else bool(flag)


async def chat_completions(registry: Any, cfg: Any, body: dict) -> Response:
    """POST /v1/chat/completions — resolve, load (bounded), proxy."""
    model = str(body.get("model") or "")
    select_llm = getattr(registry, "select_llm", None)
    if not callable(select_llm):
        return _err(503, "llm adapter family is not enabled on this server")
    cls = select_llm(model)
    if cls is None:
        return _err(404, f"unknown llm model '{model}'")

    timeout = float(getattr(cfg, "llm_load_timeout_s", 300) or 300)
    try:
        # registry.get() blocks on the registry lock + adapter.load() — run
        # it in a thread so concurrent /chat/completions requests converge on
        # the same single-flight load instead of piling onto the event loop
        adapter = await asyncio.wait_for(
            asyncio.to_thread(registry.get, cls),
            timeout=timeout + 30,  # grace over the subprocess readiness bound
        )
    except Exception as e:
        payload: dict[str, Any] = {"message": str(e)}
        if _loading_in_progress(registry, getattr(cls, "id", model)):
            payload["loading"] = True
        return JSONResponse({"error": payload}, status_code=503)

    upstream_url = adapter.base_url() + "/chat/completions"
    payload = dict(body)
    payload["model"] = adapter.served_model_id()

    if body.get("stream"):
        return await _proxy_stream(upstream_url, payload)
    return await _proxy_json(upstream_url, payload, model)


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


async def _proxy_stream(url: str, payload: dict) -> Response:
    """Open the upstream stream first (surfacing its real status code), then
    relay SSE chunks verbatim; client + stream close when the relay ends."""
    client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
    ctx = client.stream("POST", url, json=payload)
    try:
        upstream = await ctx.__aenter__()
    except httpx.HTTPError as e:
        await client.aclose()
        return _err(502, f"llm upstream unreachable: {e}")
    if upstream.status_code >= 400:
        data = await upstream.aread()
        await ctx.__aexit__(None, None, None)
        await client.aclose()
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
                "downloaded": getattr(info, "downloaded", True) if info else True,
                "loaded": getattr(info, "loaded", False) if info else False,
            }
        )
    return {"object": "list", "data": data}
