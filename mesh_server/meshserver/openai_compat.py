"""OpenAI-compatible shims over the internal job machinery.

`/v1/images/generations` + `/v1/images/edits` map onto the existing
`ImageJob` queue (`manager.submit_image`) and wait in-process for the
terminal state — no HTTP self-call. The wait is bounded by
``openai_image_wait_s`` (runtime-mutable); a timeout returns an honest
504 while the job keeps running and remains fetchable via the internal
job API. `X-Mesh-Job-Id` response headers keep every result auditable
back to the queued job.

`/v1/embeddings` lives in `llm_proxy.py` (shares the resolve→admit→load
path); `/v1/models` merges `openai_models()` here with the internal
listing (dual-shape: OpenAI clients read `data[]`, KIARA reads `models[]`).
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from meshserver.jobs import ImageJob
from meshserver.logging import get_logger
from meshserver.schemas import ImageGenParams

log = get_logger(__name__)

_POLL_S = 1.0


def _err(status: int, message: str, *, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": type_}}, status_code=status
    )


def _parse_size(raw: Any) -> tuple[int, int]:
    """OpenAI size strings → (w, h), clamped into the adapter range."""
    default = (1024, 1024)
    if not raw or str(raw).lower() == "auto":
        return default
    try:
        w_s, h_s = str(raw).lower().split("x", 1)
        w, h = int(w_s), int(h_s)
    except (ValueError, AttributeError):
        return default
    clamp = lambda v: max(256, min(2048, v))  # noqa: E731
    return clamp(w), clamp(h)


def _image_params(body: dict, *, need_edit: bool, registry: Any) -> tuple[Any, Any] | JSONResponse:
    """OpenAI images body → (ImageGenParams, resolved adapter class)."""
    model = str(body.get("model") or "auto")
    cls = registry.select_image(model, need_edit=need_edit)
    if cls is None:
        kind = "edit" if need_edit else "generation"
        if model != "auto":
            return _err(404, f"unknown or incapable image model '{model}' for {kind}")
        return _err(404, f"no image model available for {kind}")
    w, h = _parse_size(body.get("size"))
    n = body.get("n", 1)
    try:
        n = max(1, min(4, int(n)))
    except (TypeError, ValueError):
        n = 1
    steps = body.get("steps")
    # quality:"hd" bumps the adapter's own default ~1.5x (distilled models
    # like flux-schnell stay near their design point, never force-fed 50)
    if steps is None and str(body.get("quality", "")).lower() == "hd":
        default_steps = getattr(cls, "default_steps", None)
        if default_steps:
            steps = int(default_steps * 1.5)

    def _num(key: str, cast: type) -> Any:
        v = body.get(key)
        if v is None or v == "":
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            raise ValueError(key) from None

    try:
        seed = _num("seed", int) or 0
        steps_v = int(steps) if steps is not None else None
        guidance = _num("guidance", float)
        strength = _num("strength", float)
    except ValueError as e:
        return _err(422, f"'{e}' must be a number")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return _err(422, "prompt is required")
    params = ImageGenParams(
        model=cls.id,  # resolved at submit → keeps model-affinity scheduling
        prompt=prompt,
        negative=str(body.get("negative_prompt") or body.get("negative") or ""),
        seed=seed,
        count=n,
        width=w,
        height=h,
        steps=steps_v,
        guidance=guidance,
        strength=strength,
    )
    return (params, cls)


async def _wait_image_job(manager: Any, job: ImageJob, cfg: Any) -> ImageJob | JSONResponse:
    effective = getattr(cfg, "effective", None)
    wait_v = effective("openai_image_wait_s") if effective else None
    wait_s = float(wait_v if wait_v is not None else 600)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        current = manager.get(job.id)
        if current is None:
            return _err(500, "image job vanished mid-run")
        if current.status == "succeeded":
            return current
        if current.status in ("failed", "cancelled"):
            return _err(
                500,
                f"image job {current.status}: {current.error or 'unknown error'}",
                type_="server_error",
            )
        await asyncio.sleep(_POLL_S)
    # honest timeout — the job keeps running and stays fetchable via the
    # internal /v1/image-jobs API (id echoed in the message + header)
    return JSONResponse(
        {
            "error": {
                "message": f"image job '{job.id}' still running after "
                f"{wait_s:.0f}s — poll /v1/image-jobs/{job.id} for the result",
                "type": "server_error",
            }
        },
        status_code=504,
        headers={"X-Mesh-Job-Id": job.id},
    )


def _image_body(job: ImageJob, *, response_format: str, base_url: str) -> dict:
    if response_format == "url":
        data = [
            {"url": f"{base_url}/v1/image-jobs/{job.id}/image?i={i}"}
            for i in range(len(job.images))
        ]
    else:
        data = [
            {"b64_json": base64.b64encode(png).decode(), "revised_prompt": None}
            for png in job.images
        ]
    return {"created": int(time.time()), "data": data}


def _image_response(job: ImageJob, *, response_format: str, base_url: str) -> JSONResponse:
    return JSONResponse(
        _image_body(job, response_format=response_format, base_url=base_url),
        headers={"X-Mesh-Job-Id": job.id},
    )


def _job_queue_position(manager: Any, job: ImageJob) -> int | None:
    """1-based rank inside the score-ordered queued segment of /v1/queue —
    the position external clients see in status events."""
    snap = manager.queue_snapshot()
    pos = 0
    for entry in snap.jobs:
        if getattr(entry, "status", "") != "queued":
            continue
        pos += 1
        if entry.id == job.id:
            return pos
    return None


def _stream_image_job(
    manager: Any,
    cfg: Any,
    job: ImageJob,
    *,
    fmt: str,
    base_url: str,
    warning: str | None = None,
) -> Response:
    """stream:true path — SSE commits immediately. Status events
    ({"type":"status", state, position, progress, message}) carry queue
    rank and adapter progress while the job waits/runs; the terminal
    event is the standard OpenAI images payload, then [DONE]."""
    from meshserver.llm_proxy import _dumps, _error_chunk, _sse_headers

    async def gen():
        effective = getattr(cfg, "effective", None)
        keep_v = effective("llm_keepalive_s") if effective else None
        keep_s = max(float(keep_v if keep_v is not None else 5), 0.5)
        wait_v = effective("openai_image_wait_s") if effective else None
        wait_s = float(wait_v if wait_v is not None else 600)
        deadline = time.monotonic() + wait_s
        if warning is not None:
            yield f"data: {_dumps({'type': 'warning', 'message': warning})}\n\n".encode()
        last_status = ""
        last_status_at = 0.0
        since_keepalive = 0.0
        while time.monotonic() < deadline:
            current = manager.get(job.id)
            if current is None:
                yield _error_chunk("image job vanished mid-run")
                return
            if current.status == "succeeded":
                payload = _image_body(current, response_format=fmt, base_url=base_url)
                yield f"data: {_dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            if current.status in ("failed", "cancelled"):
                yield _error_chunk(
                    f"image job {current.status}: {current.error or 'unknown error'}"
                )
                return
            pos = _job_queue_position(manager, current)
            status_obj = {
                "type": "status",
                "state": current.status,
                "position": pos,
                "progress": round(current.progress, 3),
                "message": current.message,
                "model": current.resolved_model or current.params.model,
                "job_id": current.id,
            }
            status = _dumps(status_obj)
            now = time.monotonic()
            # emit on change or every ~10s steady-state; SSE comments keep
            # proxies/clients alive independently of event cadence
            if status != last_status or now - last_status_at > 10:
                yield f"data: {status}\n\n".encode()
                last_status = status
                last_status_at = now
            if now - since_keepalive >= keep_s:
                yield b": keepalive\n\n"
                since_keepalive = now
            await asyncio.sleep(_POLL_S)
        yield _error_chunk(
            f"image job '{job.id}' still running after {wait_s:.0f}s — "
            f"poll /v1/image-jobs/{job.id} for the result"
        )

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers=_sse_headers()
    )


def _wants_stream(body: dict) -> bool:
    """Opt-in SSE for image endpoints — an extension field OpenAI clients
    simply ignore (unknown keys fall through the spec silently)."""
    return str(body.get("stream") or "").lower() in ("1", "true", "yes")


async def images_generations(
    manager: Any, cfg: Any, body: dict, *, base_url: str, priority: int = 0
) -> Response:
    """POST /v1/images/generations — OpenAI images API → ImageJob → b64/url."""
    if not isinstance(body, dict):
        return _err(422, "request body must be a JSON object")
    resolved = _image_params(body, need_edit=False, registry=manager.registry)
    if isinstance(resolved, JSONResponse):
        return resolved
    params, _cls = resolved
    job = await manager.submit_image(params, priority=priority)
    if job is None:
        return JSONResponse(
            {
                "error": {
                    "message": "job queue full or host memory too low — retry shortly",
                    "type": "rate_limit_error",
                }
            },
            status_code=429,
            headers={"Retry-After": "10"},
        )
    fmt = str(body.get("response_format") or "b64_json").lower()
    if _wants_stream(body):
        return _stream_image_job(manager, cfg, job, fmt=fmt, base_url=base_url)
    done = await _wait_image_job(manager, job, cfg)
    if isinstance(done, JSONResponse):
        return done
    return _image_response(done, response_format=fmt, base_url=base_url)


async def images_edits(
    manager: Any,
    cfg: Any,
    body: dict,
    *,
    image_name: str,
    image_bytes: bytes,
    has_mask: bool,
    base_url: str,
    priority: int = 0,
) -> Response:
    """POST /v1/images/edits — multipart init image → edit-ladder job."""
    if not isinstance(body, dict):
        return _err(422, "request body must be a JSON object")
    resolved = _image_params(body, need_edit=True, registry=manager.registry)
    if isinstance(resolved, JSONResponse):
        return resolved
    params, _cls = resolved
    # same staging contract as /v1/image-edit-jobs: uuid-prefixed sanitized
    # name in job_dir/incoming (cleaned up by job GC)
    staging = cfg.job_dir / "incoming"
    staging.mkdir(parents=True, exist_ok=True)
    safe_name = Path(image_name or "init.png").name or "init.png"
    init_path = staging / f"edit_{uuid.uuid4().hex[:8]}_{safe_name}"
    init_path.write_bytes(image_bytes)
    job = await manager.submit_image(params, init_image_path=init_path, priority=priority)
    if job is None:
        init_path.unlink(missing_ok=True)  # no job owns the staged file
        return JSONResponse(
            {
                "error": {
                    "message": "job queue full or host memory too low — retry shortly",
                    "type": "rate_limit_error",
                }
            },
            status_code=429,
            headers={"Retry-After": "10"},
        )
    fmt = str(body.get("response_format") or "b64_json").lower()
    warning = (
        "mask input ignored — adapter edits whole image" if has_mask else None
    )
    if _wants_stream(body):
        return _stream_image_job(
            manager, cfg, job, fmt=fmt, base_url=base_url, warning=warning
        )
    done = await _wait_image_job(manager, job, cfg)
    if isinstance(done, JSONResponse):
        return done
    resp = _image_response(done, response_format=fmt, base_url=base_url)
    if warning:
        resp.headers["X-Mesh-Warning"] = warning
    return resp


async def openai_models(registry: Any) -> dict:
    """OpenAI `data[]` listing: llm entries from llm_proxy.llm_models plus
    image adapters so /v1/images/* clients can discover model ids."""
    from meshserver.llm_proxy import llm_models

    base = await llm_models(registry)
    infos = {m.id: m for m in registry.models()} if callable(
        getattr(registry, "models", None)
    ) else {}
    have = {e["id"] for e in base["data"]}
    for cls in getattr(registry, "_classes", []):
        if getattr(cls, "kind", "") != "image" or cls.id in have:
            continue
        info = infos.get(cls.id)
        if info is not None and not (info.available or getattr(info, "downloaded", True)):
            continue
        base["data"].append(
            {
                "id": cls.id,
                "object": "model",
                "created": 0,
                "owned_by": "meshserver",
                "supports_edit": getattr(cls, "supports_edit", False),
                "max_count": getattr(cls, "max_count", None),
                "default_steps": getattr(cls, "default_steps", None),
                "loaded": getattr(info, "loaded", False) if info else False,
            }
        )
    return base
