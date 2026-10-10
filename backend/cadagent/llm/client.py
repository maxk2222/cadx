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
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from cadagent.config import get_settings
from cadagent.llm.structured import LLMFormatError, parse_model
from cadagent.logging import get_logger

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


class LLMError(Exception):
    pass


class _RetryNoStream(Exception):
    """Streaming request was rejected (e.g. 400) -> caller falls back to _post."""


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
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.models: dict[Role, str] = {
            "vision": model_vision or s.llm_model_vision,
            "code": model_code or s.llm_model_code,
        }
        self.json_mode = json_mode or s.llm_json_mode
        self.timeout_s = timeout_s or s.llm_timeout_s
        self.stream = s.llm_stream
        self.idle_timeout_s = s.llm_idle_timeout_s
        self.total_timeout_s = s.llm_total_timeout_s
        self.extra_body: dict[str, Any] = {}
        if s.llm_extra_body:
            try:
                parsed = json.loads(s.llm_extra_body)
                if isinstance(parsed, dict):
                    self.extra_body = parsed
                else:
                    log.warning("llm.extra_body.not_object")
            except json.JSONDecodeError:
                log.warning("llm.extra_body.invalid_json")
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

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

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                    if resp.status_code == 400 and "response_format" in payload:
                        # provider does not support response_format -> retry without it
                        log.warning("llm.response_format.unsupported", status=resp.status_code)
                        payload = {k: v for k, v in payload.items() if k != "response_format"}
                        resp = await client.post(
                            f"{self.base_url}/chat/completions", json=payload, headers=headers
                        )
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
                last_err = LLMError(f"transport: {e}")
                await asyncio.sleep(delay)
                delay *= 2
        raise last_err or LLMError("request failed")

    async def _consume_stream(self, resp: httpx.Response) -> dict[str, Any]:
        """Read SSE lines and assemble them into a normal response dict."""
        t0 = time.monotonic()
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        finish: str | None = None
        chunks = 0
        async for line in resp.aiter_lines():
            chunks += 1
            if chunks == 1 or chunks % 200 == 0:
                log.debug(
                    "llm.stream.progress",
                    chunks=chunks,
                    elapsed_s=round(time.monotonic() - t0, 1),
                )
            if time.monotonic() - t0 > self.total_timeout_s:
                raise LLMError(f"stream exceeded total timeout {self.total_timeout_s}s")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    text_parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }

    async def _post_stream(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                    return await self._consume_stream(resp)
            except _RetryNoStream:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = LLMError(f"transport: {e}")
                await asyncio.sleep(delay)
                delay *= 2
            except LLMError:
                raise
        raise last_err or LLMError("stream request failed")

    async def chat(
        self,
        role: Role,
        messages: list[dict[str, Any]],
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        s = get_settings()
        model = self.models[role]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": s.llm_temperature_code if temperature is None else temperature,
            "max_tokens": max_tokens or s.llm_max_tokens,
        }
        payload.update(self.extra_body)
        rf = self._response_format(json_schema)
        if rf:
            payload["response_format"] = rf

        t0 = time.monotonic()
        if self.stream:
            try:
                data = await self._post_stream(payload)
            except _RetryNoStream:
                log.warning("llm.stream.unsupported")
                data = await self._post(payload)
        else:
            data = await self._post(payload)
        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {data}") from e

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self.total_usage["prompt_tokens"] += prompt_tokens
        self.total_usage["completion_tokens"] += completion_tokens
        self.total_usage["calls"] += 1

        parsed: BaseModel | None = None
        if json_schema is not None and text:
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
                payload2 = dict(payload, messages=retry_messages)
                data2 = await self._post(payload2)
                try:
                    text2 = data2["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as e:
                    raise LLMFormatError("re-ask produced unusable response") from e
                u2 = data2.get("usage") or {}
                self.total_usage["prompt_tokens"] += int(u2.get("prompt_tokens", 0))
                self.total_usage["completion_tokens"] += int(u2.get("completion_tokens", 0))
                self.total_usage["calls"] += 1
                parsed = parse_model(text2, json_schema)  # raises if still bad
                text = text2

        return LLMResponse(
            text=text,
            parsed=parsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=model,
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
