"""Streaming LLM client tests: SSE chunk assembly, fallback, idle timeout."""

from __future__ import annotations

import httpx
import pytest

from cadagent.llm import client as client_mod
from cadagent.llm.client import LLMClient


def _sse(*chunks: dict, usage: dict | None = None) -> bytes:
    lines = []
    for c in chunks:
        lines.append(f"data: {__import__('json').dumps(c)}")
    if usage:
        lines.append(f"data: {__import__('json').dumps({'choices': [], 'usage': usage})}")
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode()


def _patch_transport(monkeypatch, handler):
    orig = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return orig(*a, **kw)

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", patched)


def _mk(**kw) -> LLMClient:
    c = LLMClient(api_key="k", base_url="http://llm.test/v1")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


async def test_stream_assembles_chunks(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "stream=true" in request.content.decode() or b'"stream":true' in request.content
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            ),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    c = _mk()
    r = await c.chat("code", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert r.text == "Hello"
    assert r.prompt_tokens == 5
    assert r.completion_tokens == 2


async def test_stream_400_falls_back_to_post(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        calls.append(body.get("stream", False))
        if body.get("stream"):
            return httpx.Response(400, json={"error": "no stream"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "non-stream"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    _patch_transport(monkeypatch, handler)
    c = _mk()
    r = await c.chat("code", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert r.text == "non-stream"
    assert calls == [True, False]


async def test_non_stream_when_disabled(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        assert not body.get("stream")
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    _patch_transport(monkeypatch, handler)
    c = _mk(stream=False)
    r = await c.chat("code", [{"role": "user", "content": "hi"}])
    assert r.text == "x"


async def test_stream_error_raises_llmerror(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    _patch_transport(monkeypatch, handler)
    c = _mk()
    with pytest.raises(client_mod.LLMError, match="403"):
        await c.chat("code", [{"role": "user", "content": "hi"}])
