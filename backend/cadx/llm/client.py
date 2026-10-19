"""Thin OpenAI-compatible Chat Completions client (httpx, no SDK).

Works with OpenAI, OpenRouter, Ollama, LM Studio, vLLM and any compatible
endpoint. Structured output is requested via response_format where supported
and verified/parsed by structured.py either way.

Requests use SSE streaming by default: the idle timeout applies between
chunks, so slow models can take a long time as long as tokens keep flowing,
while a stalled connection dies fast.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from cadx.llm.structured import LLMFormatError, parse_model
from cadx.llm.tools import (
    ToolCall,
    _StreamToolAcc,
    parse_content_tool_call,
    parse_tool_calls,
    sanitize_text,
)
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

Role = Literal["vision", "code"]


@dataclass
class LLMResponse:
    text: str
    parsed: BaseModel | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = ""
    # vLLM automatic prefix caching reports reused KV tokens via
    # usage.prompt_tokens_details.cached_tokens (OpenAI-compatible); 0 = miss
    cached_tokens: int = 0


class LLMError(Exception):
    pass


def _cached_tokens(usage: dict[str, Any]) -> int:
    """Prompt tokens served from the provider's prefix cache.

    OpenAI-compatible shape: ``usage.prompt_tokens_details.cached_tokens``
    (vLLM fills it when automatic prefix caching hits). Missing or malformed
    details read as a miss — the counter stays informational, never fatal."""
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


class _RetryNoStream(Exception):
    """Streaming request was rejected (e.g. 400) -> caller falls back to _post."""


# payload keys this client sets itself — anything beyond these came from
# llm_extra_body / per-call extra_body (e.g. thinking dialects like
# `chat_template_kwargs` or `reasoning_effort`) and is droppable on a 400
_CORE_PAYLOAD_KEYS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
    }
)


_STRUCT_CHARS = '\n"{}[]()<>0123456789'

_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "as",
        "at",
        "by",
        "from",
        "not",
        "but",
        "we",
        "you",
        "they",
        "he",
        "she",
        "if",
        "then",
        "else",
        "return",
        "def",
        "class",
        "import",
        "true",
        "false",
        "none",
        "null",
        "self",
    ]
)


def _looks_degenerate(tail: str) -> bool:
    """Detect word-salad degeneration: a long flat run of bare words with
    almost no repetition and no sentence glue — weak models fall into
    synonym/category-listing loops (`hunger_appetite_thirst_need_want`,
    `scand titanium vanadium chromium manganese`, emoji bullet spam) that
    burn the entire max_tokens budget. Real code/JSON/prose tails always
    carry structural characters, repeated stop-words, or normal
    punctuation/capitalisation."""
    if len(tail) < 128:
        return False
    # whitespace-only run: tab/newline spam that carries zero content —
    # must be checked before the _STRUCT_CHARS bail-out (\n is structural)
    non_ws = sum(1 for c in tail if not c.isspace())
    if non_ws <= len(tail) * 0.05:
        return True
    if any(c in tail for c in _STRUCT_CHARS):
        return False
    # classic style: long runs of tokens joined by `_`/`.` with few spaces
    if tail.count(" ") <= len(tail) / 30:
        words = [w for w in re.split(r"[_.]+", tail.strip("_. ")) if w]
        if len(words) >= 20:
            return True
    # space-separated / emoji-bullet salad: many distinct "vocabulary-list"
    # tokens with no sentence glue — real prose repeats function words
    tokens = [t.lower() for t in re.split(r"[_\W]+", tail) if t]
    if len(tokens) < 20:
        return False
    distinct = len(set(tokens))
    stop_ratio = sum(t in _STOPWORDS for t in tokens) / len(tokens)
    return distinct >= 8 and stop_ratio <= 0.1


def _trim_degenerate(text: str) -> str:
    """Cut a degenerate word-salad suffix back to the last structural
    character (newline, bracket, quote) — the salad only ever appends."""
    if not _looks_degenerate(text[-384:]):
        return text
    cut = max(text.rfind(c) for c in _STRUCT_CHARS)
    return text[: cut + 1] if cut >= 0 else ""


def _stream_tail_degenerate(text_parts: list[str], tool_acc: _StreamToolAcc) -> bool:
    """Degeneration can hide in streamed text OR inside tool-call arguments
    (text_parts stays empty while args fill with salad)."""
    args_tail = tool_acc.debug_tail()
    if "(" in args_tail and args_tail.endswith(")"):
        args_tail = args_tail[args_tail.index("(") + 1 : -1]  # unwrap name(args)
    return _looks_degenerate("".join(text_parts)[-384:]) or _looks_degenerate(args_tail[-384:])


def _chunk_has_reasoning(chunk: dict[str, Any]) -> bool:
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if delta.get("reasoning_content") or delta.get("reasoning"):
            return True
    return False


def _absorb_chunk(
    chunk: dict[str, Any], text_parts: list[str], tool_acc: _StreamToolAcc, prod: list[int]
) -> str | None:
    """Merge one SSE chunk. ``prod[0]`` accumulates meaningful progress:
    non-whitespace content/tool-arg chars, +1 per reasoning fragment."""
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        piece = delta.get("content")
        if piece:
            text_parts.append(piece)
            prod[0] += sum(1 for c in piece if not c.isspace())
        if delta.get("tool_calls"):
            prod[0] += tool_acc.feed(delta["tool_calls"])
        if delta.get("reasoning_content") or delta.get("reasoning"):
            prod[0] += 1
        if ch.get("finish_reason"):
            return ch["finish_reason"]
    return None


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_vision: str | None = None,
        model_code: str | None = None,
        json_mode: str | None = None,
        timeout_s: float | None = None,
        runtime_settings: RuntimeSettings | None = None,
    ) -> None:
        settings = runtime_settings or RuntimeSettings()
        self.runtime_settings = settings
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key.get_secret_value()
        self.models: dict[Role, str] = {
            "vision": model_vision or settings.llm_model_vision,
            "code": model_code or settings.llm_model_code,
        }
        self.json_mode = json_mode or settings.llm_json_mode
        self.timeout_s = timeout_s or settings.llm_timeout_s
        self.stream = settings.llm_stream
        self.idle_timeout_s = settings.llm_idle_timeout_s
        self.total_timeout_s = settings.llm_total_timeout_s
        self.degenerate_stall_s = settings.llm_degenerate_stall_s
        self.tools_mode = settings.llm_tools  # auto|on|off
        # per-role probe state: absent = unprobed; False = role's model rejects
        # `tools` payloads (e.g. the vision model can lack tool support while
        # the code model still has it — the fallback must not bleed across)
        self.tools_ok: dict[Role, bool] = {}
        self.extra_body: dict[str, Any] = {}
        if settings.llm_extra_body:
            try:
                parsed = json.loads(settings.llm_extra_body)
                if isinstance(parsed, dict):
                    self.extra_body = parsed
                else:
                    log.warning("llm.extra_body.not_object")
            except json.JSONDecodeError:
                log.warning("llm.extra_body.invalid_json")
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
        }

    def _response_format(self, schema: type[BaseModel] | None) -> dict[str, Any] | None:
        if schema is None or self.json_mode == "none":
            return None
        if self.json_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            }
        return {"type": "json_object"}

    async def _post(self, payload: dict[str, Any], role: Role) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        last_err: Exception | None = None
        for _ in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                    resp, payload = await self._handle_400(client, resp, payload, headers, role)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data: dict[str, Any] = resp.json()
                return data
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = LLMError(f"transport: {type(e).__name__}: {e}")
                await asyncio.sleep(delay)
                delay *= 2
        raise last_err or LLMError("request failed")

    async def _handle_400(
        self,
        client: httpx.AsyncClient,
        resp: httpx.Response,
        payload: dict[str, Any],
        headers: dict[str, str],
        role: Role,
    ) -> tuple[httpx.Response, dict[str, Any]]:
        """Retry a 400 by stripping optional payload keys providers may reject.

        Order: `response_format` first (some builds reject it together with
        tools), then `tools`+`tool_choice`. Stripping tools marks
        tools_ok[role]=False so later calls for THIS role skip them entirely.
        With tools_mode="on" tools are never stripped — the 400 propagates.
        Last resort: provider-dialect extras (llm_extra_body / per-call
        extra_body / thinking bodies) — one retry without any non-core key.
        """
        for key in ("response_format", "tools"):
            if resp.status_code != 400 or key not in payload:
                continue
            if key == "tools" and self.tools_mode == "on":
                continue  # strict mode: let the 400 surface as LLMError
            log.warning("llm.payload_key.unsupported", key=key, status=resp.status_code)
            drop: list[str] = [key, "tool_choice"] if key == "tools" else [key]
            if key == "tools":
                self.tools_ok[role] = False
            payload = {k: v for k, v in payload.items() if k not in drop}
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
        if resp.status_code == 400:
            extras = sorted(k for k in payload if k not in _CORE_PAYLOAD_KEYS)
            if extras:
                log.warning(
                    "llm.payload_key.unsupported",
                    key=",".join(extras),
                    status=resp.status_code,
                )
                payload = {k: v for k, v in payload.items() if k in _CORE_PAYLOAD_KEYS}
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
        return resp, payload

    async def _consume_stream(  # noqa: C901
        self, resp: httpx.Response, on_delta: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """Read SSE lines and assemble them into a normal response dict.

        ``on_delta`` (when given) is called with the *cumulative* text so far,
        throttled to ~10 Hz inside the loop and once with the final text —
        snapshot semantics keep it correct across mid-stream retries."""
        t0 = time.monotonic()
        last_emit = 0.0
        text_parts: list[str] = []
        reasoning_parts = 0
        usage: dict[str, Any] = {}
        finish: str | None = None
        chunks = 0
        tool_acc = _StreamToolAcc()
        # content-stall watchdog: proxies/keepalives can keep the socket
        # "alive" while the model emits nothing — abort when no real
        # content arrived for idle_timeout_s
        last_progress = t0
        last_content_size = -1
        # meaningful-progress watchdog: non-whitespace content/tool-arg chars
        # and reasoning activity — a stream of pure whitespace/indentation is
        # degeneration even though bytes keep flowing
        prod = [0]
        last_prod = 0
        last_meaningful = t0
        async for line in resp.aiter_lines():
            chunks += 1
            content_size = sum(map(len, text_parts)) + reasoning_parts + tool_acc.size
            if content_size != last_content_size:
                last_content_size = content_size
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > self.idle_timeout_s:
                raise LLMError(f"stream stalled: no content for {self.idle_timeout_s:.0f}s")
            if prod[0] != last_prod:
                last_prod = prod[0]
                last_meaningful = time.monotonic()
            elif time.monotonic() - last_meaningful > self.degenerate_stall_s:
                log.warning("llm.stream.degenerate_stall", chunks=chunks)
                finish = "degenerate"
                break
            if chunks == 1 or chunks % 200 == 0:
                tail_src = "".join(text_parts) or tool_acc.debug_tail()
                log.debug(
                    "llm.stream.progress",
                    chunks=chunks,
                    elapsed_s=round(time.monotonic() - t0, 1),
                    reasoning_parts=reasoning_parts,
                    tail=tail_src[-120:],
                )
            if time.monotonic() - t0 > self.total_timeout_s:
                raise LLMError(f"stream exceeded total timeout {self.total_timeout_s}s")
            if chunks % 50 == 0 and _stream_tail_degenerate(text_parts, tool_acc):
                log.warning("llm.stream.degenerate", chunks=chunks)
                finish = "degenerate"
                break
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            finish = _absorb_chunk(chunk, text_parts, tool_acc, prod) or finish
            reasoning_parts += 1 if _chunk_has_reasoning(chunk) else 0
            if chunk.get("usage"):
                usage = chunk["usage"]
            if on_delta is not None and time.monotonic() - last_emit >= 0.1:
                last_emit = time.monotonic()
                on_delta("".join(text_parts))
        calls = tool_acc.result()
        text = "".join(text_parts)
        if finish == "degenerate":
            text = _trim_degenerate(text)
        if on_delta is not None:
            on_delta(text)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.raw_arguments},
                            }
                            for c in calls
                        ]
                        if calls
                        else None,
                    },
                    "finish_reason": finish or ("tool_calls" if calls else None),
                }
            ],
            "usage": usage,
        }

    async def _post_stream(
        self, payload: dict[str, Any], role: Role, on_delta: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """POST with stream=true; idle timeout applies between chunks so a
        slow-but-progressing call keeps going while a stalled one aborts."""
        payload = dict(payload, stream=True, stream_options={"include_usage": True})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        last_err: Exception | None = None
        for _ in range(3):
            try:
                async with (
                    httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=30.0,
                            read=self.idle_timeout_s,
                            write=60.0,
                            pool=60.0,
                        )
                    ) as client,
                    client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp,
                ):
                    if resp.status_code == 400:
                        raise _RetryNoStream()
                    if resp.status_code in (429, 500, 502, 503, 504):
                        last_err = LLMError(f"HTTP {resp.status_code}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    if resp.status_code != 200:
                        body = (await resp.aread())[:400]
                        raise LLMError(f"HTTP {resp.status_code}: {body!r}")
                    return await self._consume_stream(resp, on_delta=on_delta)
            except _RetryNoStream:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = LLMError(f"transport: {type(e).__name__}: {e}")
                await asyncio.sleep(delay)
                delay *= 2
            except LLMError:
                raise
        raise last_err or LLMError("stream request failed")

    def _send_tools(self, tools: list[dict[str, Any]] | None, role: Role) -> bool:
        """Whether to attach `tools` to the next request for this role."""
        return bool(tools) and self.tools_mode != "off" and self.tools_ok.get(role) is not False

    async def chat(
        self,
        role: Role,
        messages: list[dict[str, Any]],
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        on_delta: Callable[[str], None] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = self.models[role]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": (
                self.runtime_settings.llm_temperature_code if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.runtime_settings.llm_max_tokens,
        }
        payload.update(self.extra_body)
        # per-call body (e.g. the adaptive thinking dialect) is merged LAST —
        # on key conflicts it wins over the client-wide llm_extra_body
        payload.update(extra_body or {})
        send_tools = self._send_tools(tools, role)
        if send_tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        rf = self._response_format(json_schema)
        if rf:
            payload["response_format"] = rf

        t0 = time.monotonic()
        if self.stream:
            try:
                data = await self._post_stream(payload, role, on_delta=on_delta)
            except _RetryNoStream:
                log.warning("llm.stream.unsupported")
                data = await self._post(payload, role)
        else:
            data = await self._post(payload, role)
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            msg = data["choices"][0]["message"]
            text = sanitize_text(msg.get("content") or "")
            tool_calls = parse_tool_calls(msg)
            finish = data["choices"][0].get("finish_reason") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {data}") from e

        trimmed = _trim_degenerate(text)
        if len(trimmed) != len(text):
            log.warning("llm.response.degenerate_tail_trimmed", dropped=len(text) - len(trimmed))
            text = trimmed

        if not tool_calls and tools and text:
            # model smuggled the call into content -> synthetic tool call.
            # `tools` (not send_tools): sessions must keep working after a
            # 400-strip removed tools from the wire.
            synthetic = parse_content_tool_call(text)
            if synthetic and any(
                t.get("function", {}).get("name") == synthetic.name for t in tools
            ):
                tool_calls = [synthetic]
                finish = "tool_calls"

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cached_tokens = _cached_tokens(usage)
        self.total_usage["prompt_tokens"] += prompt_tokens
        self.total_usage["completion_tokens"] += completion_tokens
        self.total_usage["cached_tokens"] += cached_tokens
        self.total_usage["calls"] += 1

        parsed: BaseModel | None = None
        if json_schema is not None and text and not tool_calls:
            try:
                parsed = parse_model(text, json_schema)
            except LLMFormatError:
                # one re-ask with the validation error
                retry_messages = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Your reply could not be parsed as the required JSON schema. "
                            "Return ONLY the corrected JSON object, no markdown, no text."
                        ),
                    },
                ]
                # re-ask must not offer tools — a tool call here would fail
                # the schema parse again. Route through the same transport as
                # the first call: non-stream `_post` caps at timeout_s (180s),
                # which kills heavy vision re-asks that a stream survives.
                payload2 = {k: v for k, v in payload.items() if k not in ("tools", "tool_choice")}
                payload2["messages"] = retry_messages
                if self.stream:
                    try:
                        data2 = await self._post_stream(payload2, role, on_delta=on_delta)
                    except _RetryNoStream:
                        data2 = await self._post(payload2, role)
                else:
                    data2 = await self._post(payload2, role)
                try:
                    text2 = sanitize_text(data2["choices"][0]["message"]["content"] or "")
                except (KeyError, IndexError, TypeError) as e:
                    raise LLMFormatError("re-ask produced unusable response") from e
                u2 = data2.get("usage") or {}
                self.total_usage["prompt_tokens"] += int(u2.get("prompt_tokens", 0))
                self.total_usage["completion_tokens"] += int(u2.get("completion_tokens", 0))
                self.total_usage["cached_tokens"] += _cached_tokens(u2)
                self.total_usage["calls"] += 1
                # attribute the re-ask's usage to this call too — the returned
                # response feeds per-step trace/token accounting
                prompt_tokens += int(u2.get("prompt_tokens", 0))
                completion_tokens += int(u2.get("completion_tokens", 0))
                cached_tokens += _cached_tokens(u2)
                parsed = parse_model(text2, json_schema)  # raises if still bad
                text = text2

        # authoritative final snapshot — covers the non-stream path and any
        # degenerate trim / re-ask replacement
        if on_delta is not None:
            on_delta(text)

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=model,
            tool_calls=tool_calls or None,
            finish_reason=finish,
            cached_tokens=cached_tokens,
        )

    async def test(self) -> dict[str, Any]:
        """Connectivity + vision capability probe used by the settings page."""
        out: dict[str, Any] = {"ok": False, "vision_ok": False, "error": None, "latency_ms": 0}
        tiny_png = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUl"
            "EQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        try:
            r = await self.chat(
                "code",
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=16,
                temperature=0.0,
            )
            out["latency_ms"] = r.latency_ms
            out["ok"] = len(r.text) > 0
        except Exception as e:
            out["error"] = str(e)
            return out
        try:
            await self.chat(
                "vision",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in one word."},
                            {"type": "image_url", "image_url": {"url": tiny_png}},
                        ],
                    }
                ],
                max_tokens=16,
                temperature=0.0,
            )
            out["vision_ok"] = True
        except Exception as e:
            out["vision_error"] = str(e)
        return out


def make_llm(runtime: RuntimeSettings, *, backend: str | None = None) -> LLMClient:
    """Central LLM factory — resolves the configured provider backend.

    ``backend`` overrides ``runtime.llm_backend`` (per-job RunOptions):

    - ``"external"`` (default): ``llm_base_url`` + ``llm_api_key`` — OpenAI,
      OpenRouter, vLLM, ... — the historical behaviour.
    - ``"sidecar"``: the OpenAI-compatible chat endpoint hosted by the local
      mesh-server sidecar at ``runtime.mesh_base_url`` (no auth). When
      ``llm_sidecar_model`` is set it pins BOTH roles to that one local
      model; otherwise the configured role models apply there too.
    """
    mode = backend or runtime.llm_backend
    if mode == "sidecar":
        base = runtime.mesh_base_url.strip().rstrip("/")
        if not base:
            raise ValueError("local sidecar not configured")
        # the sidecar serves chat under /v1/chat/completions (mesh_base_url
        # itself is version-agnostic — HttpMeshClient appends /v1 the same way)
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        model = runtime.llm_sidecar_model.strip() or None
        return LLMClient(
            base_url=base,
            api_key="",
            model_vision=model,
            model_code=model,
            runtime_settings=runtime,
        )
    return LLMClient(runtime_settings=runtime)
