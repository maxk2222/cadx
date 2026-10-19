"""llm_proxy: model routing, served-model rewrite, SSE passthrough, errors.

The "upstream LLM server" is a stdlib HTTP stub on a real loopback port;
the registry is a hand-rolled stub exposing select_llm/get/_classes.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request

from meshserver.config import ServerConfig
from meshserver.llm_proxy import chat_completions, llm_models


class _UpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("explode"):
            data = json.dumps({"error": {"message": "upstream exploded"}}).encode()
            self._reply(500, data, "application/json")
            return
        if body.get("stream"):
            data = (
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self._reply(200, data, "text/event-stream")
            return
        data = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "echo_model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self._reply(200, data, "application/json")

    def _reply(self, code: int, data: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()
    srv.server_close()


class _Adapter:
    def __init__(self, url: str) -> None:
        self._url = url
        self.loaded = True

    def base_url(self) -> str:
        return self._url

    def served_model_id(self) -> str:
        return "served-xyz"


class _Reg:
    """Minimal registry stub — select_llm/get/is_loading/_classes."""

    def __init__(self, adapter: _Adapter | None = None, boom: str = "") -> None:
        self._adapter = adapter
        self._boom = boom
        self._loading = False
        self._cls = type(
            "FakeLlmCls",
            (),
            {"id": "my-llm", "kind": "llm", "supports_vision": True, "supports_tools": True},
        )
        self._classes = [self._cls]

    def select_llm(self, model: str):
        return self._cls if model == "my-llm" else None

    def get(self, cls):
        if self._boom:
            raise RuntimeError(self._boom)
        return self._adapter

    def is_loading(self, model_id: str) -> bool:  # noqa: ARG002
        return self._loading


class _BareReg:
    """Registry without the llm wiring — select_llm absent by design."""

    _classes: list = []


def _app(reg: Any) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await chat_completions(reg, request.app.state.cfg, await request.json())

    @app.get("/v1/llm-models")
    async def models():
        return await llm_models(reg)

    app.state.cfg = ServerConfig(llm_load_timeout_s=5)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_passthrough_rewrites_model(upstream):
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "my-llm", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["echo_model"] == "served-xyz"  # rewritten to the served id
    assert body["choices"][0]["message"]["content"] == "ok"


async def test_unknown_model_404(upstream):
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404
    assert "unknown llm model" in r.json()["error"]["message"]


async def test_streaming_passthrough(upstream):
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "my-llm", "stream": True, "messages": []},
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data: [DONE]" in r.text
    assert '"content":"hi"' in r.text


async def test_upstream_error_status_propagates(upstream):
    """A non-2xx upstream response is relayed with its own status code."""
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "my-llm", "explode": True, "messages": []},
        )
    assert r.status_code == 500
    assert "exploded" in r.json()["error"]["message"]


async def test_upstream_stream_error_status_propagates(upstream):
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "my-llm", "explode": True, "stream": True, "messages": []},
        )
    assert r.status_code == 500


async def test_load_failed_503(upstream):
    reg = _Reg(_Adapter(upstream), boom="weights missing")
    async with _client(_app(reg)) as c:
        r = await c.post("/v1/chat/completions", json={"model": "my-llm", "messages": []})
    assert r.status_code == 503
    assert "weights missing" in r.json()["error"]["message"]
    assert "loading" not in r.json()["error"]


async def test_load_failed_503_with_loading_flag(upstream):
    reg = _Reg(_Adapter(upstream), boom="still warming up")
    reg._loading = True
    async with _client(_app(reg)) as c:
        r = await c.post("/v1/chat/completions", json={"model": "my-llm", "messages": []})
    assert r.status_code == 503
    assert r.json()["error"]["loading"] is True


async def test_no_select_llm_503():
    async with _client(_app(_BareReg())) as c:
        r = await c.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 503
    assert "not enabled" in r.json()["error"]["message"]


async def test_llm_models_listing(upstream):
    reg = _Reg(_Adapter(upstream))
    async with _client(_app(reg)) as c:
        r = await c.get("/v1/llm-models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data[0]["id"] == "my-llm"
    assert data[0]["object"] == "model"
    assert data[0]["supports_vision"] is True


def test_upstream_stub_direct(upstream):
    """Sanity: the stub server itself answers OpenAI-shaped requests."""
    r = httpx.post(
        f"{upstream}/chat/completions",
        json={"model": "anything", "messages": []},
        timeout=5,
    )
    assert r.status_code == 200
    assert r.json()["echo_model"] == "anything"

