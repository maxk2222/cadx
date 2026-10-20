"""Streaming LLM client tests: SSE chunk assembly, fallback, idle timeout."""

from __future__ import annotations

import httpx
import pytest

from cadx.llm import client as client_mod
from cadx.llm.client import LLMClient


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
    # capability probing is off-topic here — pre-mark roles probed so the
    # request log only sees chat traffic
    c._probed_roles = {"vision", "code"}
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


async def test_cached_tokens_extracted_from_usage(monkeypatch):
    """vLLM's APC report (usage.prompt_tokens_details.cached_tokens) lands on
    LLMResponse.cached_tokens and accumulates into total_usage."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )

    _patch_transport(monkeypatch, handler)
    c = _mk(stream=False)
    r = await c.chat("code", [{"role": "user", "content": "hi"}])
    assert r.cached_tokens == 80
    assert c.total_usage["cached_tokens"] == 80


async def test_cached_tokens_zero_when_absent(monkeypatch):
    """Endpoints without prompt_tokens_details read as a cache miss."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    _patch_transport(monkeypatch, handler)
    c = _mk(stream=False)
    r = await c.chat("code", [{"role": "user", "content": "hi"}])
    assert r.cached_tokens == 0


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


def test_looks_degenerate_word_salad():
    from cadx.llm.client import _looks_degenerate

    salad = "_".join(
        ["hunger", "appetite", "thirst", "need", "want", "wish", "aim", "end", "goal", "object"]
        * 15
    )
    assert _looks_degenerate(salad)
    code = "part -= Pos(x, 0, 4) * Cylinder(radius=2, height=20)\n" * 10
    assert not _looks_degenerate(code)
    prose = "the back face is flat and the top has pockets. " * 10
    assert not _looks_degenerate(prose)
    # space-separated category listing — what real weak models emit
    space_salad = " ".join(
        [
            "titanium",
            "vanadium",
            "chromium",
            "manganese",
            "cobalt",
            "nickel",
            "copper",
            "zinc",
            "gallium",
            "germanium",
            "arsenic",
            "selenium",
        ]
        * 4
    )
    assert _looks_degenerate(space_salad)


def test_trim_degenerate_cuts_salad_suffix():
    from cadx.llm.client import _trim_degenerate

    salad = "_" + "_".join(["location", "position", "site", "setting", "layout"] * 20)
    text = "x = 1\ny = 2\n" + salad
    assert _trim_degenerate(text) == "x = 1\ny = 2\n"
    assert _trim_degenerate("x = 1\n") == "x = 1\n"


async def test_stream_aborts_on_word_salad(monkeypatch):
    salad_piece = "_".join(
        ["hunger", "appetite", "thirst", "need", "want", "wish", "aim", "end", "goal", "object"]
    )
    chunks = [{"choices": [{"delta": {"content": "x = 1\n"}}]}]
    chunks += [{"choices": [{"delta": {"content": salad_piece}}]} for _ in range(80)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(*chunks),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    c = _mk()
    r = await c.chat("code", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert r.text == "x = 1\n"
    assert r.finish_reason == "degenerate"


async def test_stream_aborts_on_tool_arg_salad(monkeypatch):
    """Degeneration inside streamed tool-call arguments also aborts early."""
    salad_piece = " " + " ".join(
        [
            "titanium",
            "vanadium",
            "chromium",
            "manganese",
            "cobalt",
            "nickel",
            "copper",
            "zinc",
            "gallium",
            "germanium",
        ]
        * 2
    )
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "done", "arguments": salad_piece},
                            }
                        ]
                    }
                }
            ]
        }
        for _ in range(80)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(*chunks),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    c = _mk()
    r = await c.chat("code", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert r.finish_reason == "degenerate"


def test_looks_degenerate_whitespace_tail():
    from cadx.llm.client import _looks_degenerate

    # the real-eval failure mode: \t\n spam — must NOT be shielded by \n
    # being a structural char
    assert _looks_degenerate("\t \n   \t\t\n" * 30)
    assert not _looks_degenerate("def f():\n    return 1\n" * 10)


async def test_stream_aborts_on_whitespace_flood(monkeypatch):
    chunks = [{"choices": [{"delta": {"content": "result = {"}}]}]
    chunks += [{"choices": [{"delta": {"content": " \n\t  "}}]} for _ in range(80)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(*chunks),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    c = _mk(degenerate_stall_s=0.0)
    r = await c.chat("code", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert r.finish_reason == "degenerate"


async def test_on_delta_cumulative_snapshots(monkeypatch):
    """on_delta receives cumulative text snapshots — throttled mid-stream and
    always once with the final text (chat UI renders deltas as replacements)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            ),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    deltas: list[str] = []
    c = _mk()
    r = await c.chat(
        "code",
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        on_delta=deltas.append,
    )
    assert r.text == "Hello"
    assert deltas and deltas[-1] == "Hello"
    # every snapshot is a prefix-consistent cumulative value
    assert all("Hello".startswith(d) for d in deltas)


async def test_schema_reask_uses_stream(monkeypatch):
    """After a schema parse failure the re-ask must keep streaming —
    non-stream posts cap at timeout_s and kill slow vision replies."""
    from pydantic import BaseModel

    class Out(BaseModel):
        value: int

    calls: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        calls.append(bool(body.get("stream")))
        if not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        n = len(calls)
        if n == 1:
            chunk = {
                "choices": [{"delta": {"content": "not json at all"}, "finish_reason": "stop"}]
            }
        else:
            chunk = {"choices": [{"delta": {"content": '{"value": 3}'}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            content=_sse(chunk),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    c = _mk()
    r = await c.chat("code", [{"role": "user", "content": "hi"}], json_schema=Out)
    assert r.parsed and r.parsed.value == 3
    assert calls == [True, True]
