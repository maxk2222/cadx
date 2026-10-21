"""External provider clients: OpenAIImageClient + TripoMeshClient (MockTransport)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cadx.agent.image_client import (
    ImageGenError,
    ImageGenRequest,
    OpenAIImageClient,
    make_image_client,
)
from cadx.agent.mesh_client import (
    HttpMeshClient,
    MeshGenError,
    MeshGenRequest,
    make_mesh_client,
)
from cadx.agent.tripo_client import TripoMeshClient
from cadx.runtime_config import RuntimeSettings


class _Transport:
    """MockTransport wrapper whose handler the test can swap per-case."""

    def __init__(self) -> None:
        self.handler: Any = lambda request: httpx.Response(404)
        self.transport = httpx.MockTransport(lambda request: self.handler(request))


@pytest.fixture()
def http(monkeypatch):
    """Redirect every httpx.AsyncClient the providers construct to a mock."""
    t = _Transport()
    real_async = httpx.AsyncClient

    def _async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("timeout", None)
        return real_async(transport=t.transport)

    monkeypatch.setattr(httpx, "AsyncClient", _async_client)
    return t


# --------------------------------------------------------------------------- #
# TripoMeshClient


def _mesh_req(tmp_path: Path, images: list[Path] | None = None) -> MeshGenRequest:
    img = tmp_path / "hero.png"
    img.write_bytes(b"\x89PNG fake")
    return MeshGenRequest(
        images=images or [img],
        subject="knob",
        seed=7,
        out_path=tmp_path / "out.glb",
    )


def _tripo(state: dict[str, Any]):
    """Tripo v2 openapi stand-in. ``state`` lets tests steer status/errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=b"GLB-BYTES")
        assert request.headers.get("authorization") == "Bearer sk-tripo"
        if path.endswith("/balance"):
            return httpx.Response(200, json=state.get("balance", {"code": 0, "data": {}}))
        if path.endswith("/upload"):
            state["uploads"] = state.get("uploads", 0) + 1
            return httpx.Response(200, json={"code": 0, "data": {"image_token": "tok-1"}})
        if path.endswith("/task") and request.method == "POST":
            state["task_body"] = json.loads(request.content)
            if "submit_error" in state:
                return httpx.Response(200, json={"code": 1001, "message": state["submit_error"]})
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "task-1"}})
        if path.endswith("/task/task-1/cancel"):
            state["cancelled"] = True
            return httpx.Response(200, json={"code": 0})
        if path.endswith("/task/task-1"):
            state["polls"] = state.get("polls", 0) + 1
            if state.get("fail_task"):
                return httpx.Response(
                    200, json={"code": 0, "data": {"status": "failed", "message": "boom"}}
                )
            if state["polls"] >= state.get("success_after", 1):
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "status": "success",
                            "progress": 100,
                            "result": {"model": {"url": "https://cdn.test/model.glb"}},
                        },
                    },
                )
            return httpx.Response(
                200, json={"code": 0, "data": {"status": "running", "progress": 50}}
            )
        return httpx.Response(404)

    return handler


async def test_tripo_generate_full_flow(http, tmp_path):
    state: dict[str, Any] = {"success_after": 2}
    http.handler = _tripo(state)
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi", poll_s=0.01)
    progress: list[float] = []

    async def on_progress(msg: str, p: float) -> None:
        progress.append(p)

    res = await c.generate(_mesh_req(tmp_path), on_progress)
    assert res.backend == "tripo3d"
    assert res.mesh_path.read_bytes() == b"GLB-BYTES"
    # upload -> task -> poll -> download happened
    assert state["uploads"] == 1
    body = state["task_body"]
    assert body["type"] == "image_to_model"
    assert body["file"]["image_token"] == "tok-1"
    assert body["file"]["type"] == "png"
    assert "model_version" in body
    # percent progress normalised to 0..1
    assert all(0.0 <= p <= 1.0 for p in progress)


async def test_tripo_prehosted_url_skips_upload(http, tmp_path):
    state: dict[str, Any] = {}
    http.handler = _tripo(state)
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi", poll_s=0.01)
    req = _mesh_req(tmp_path, images=[Path("https://img.test/a.jpg")])
    await c.generate(req, None)
    assert "uploads" not in state
    assert state["task_body"]["file"] == {"type": "jpg", "url": "https://img.test/a.jpg"}


async def test_tripo_abort_cancels_task(http, tmp_path):
    state: dict[str, Any] = {"success_after": 99}  # never finishes
    http.handler = _tripo(state)
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi", poll_s=0.01)

    async def abort() -> bool:
        return state.get("polls", 0) >= 2

    with pytest.raises(MeshGenError, match="cancelled"):
        await c.generate(_mesh_req(tmp_path), None, should_abort=abort)
    assert state.get("cancelled") is True


async def test_tripo_submit_error_mapped(http, tmp_path):
    state: dict[str, Any] = {"submit_error": "quota exceeded"}
    http.handler = _tripo(state)
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi", poll_s=0.01)
    with pytest.raises(MeshGenError, match="quota exceeded"):
        await c.generate(_mesh_req(tmp_path), None)


async def test_tripo_task_failed_mapped(http, tmp_path):
    state: dict[str, Any] = {"fail_task": True}
    http.handler = _tripo(state)
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi", poll_s=0.01)
    with pytest.raises(MeshGenError, match="boom"):
        await c.generate(_mesh_req(tmp_path), None)


async def test_tripo_missing_key(tmp_path):
    c = TripoMeshClient("", base_url="https://tripo.test")
    h = await c.health()
    assert h.ok is False and h.error == "no tripo_api_key"
    with pytest.raises(MeshGenError, match="tripo_api_key"):
        await c.generate(_mesh_req(tmp_path), None)


async def test_tripo_health_and_models(http):
    http.handler = _tripo({})
    c = TripoMeshClient("sk-tripo", base_url="https://tripo.test/v2/openapi")
    h = await c.health()
    assert h.ok and h.backend == "tripo3d"
    models = await c.models()
    assert models[0].kind == "mesh" and models[0].backend == "tripo3d"


def test_make_mesh_client_backend_switch():
    # tripo needs no mesh_base_url — a client is always constructible
    rt = RuntimeSettings(mesh_backend="tripo", tripo_api_key=SecretStr("k"))
    assert isinstance(make_mesh_client(runtime_settings=rt), TripoMeshClient)
    # per-job override beats the runtime default
    rt2 = RuntimeSettings(mesh_base_url="http://s:1")  # sidecar default
    assert isinstance(make_mesh_client(runtime_settings=rt2, backend="tripo"), TripoMeshClient)
    assert isinstance(make_mesh_client(runtime_settings=rt2), HttpMeshClient)
    assert make_mesh_client() is None  # default sidecar, no mesh_base_url


# --------------------------------------------------------------------------- #
# OpenAIImageClient


def _img_req(tmp_path: Path, **kw: Any) -> ImageGenRequest:
    return ImageGenRequest(prompt="a red gear", seed=7, out_dir=tmp_path, **kw)


_PNG = base64.b64encode(b"\x89PNG fake-image").decode()


def _openai(state: dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=b"PNG-FROM-URL")
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-image-1"}]})
        if path.endswith("/images/generations"):
            state["gen_body"] = json.loads(request.content)
            if "gen_error" in state:
                return httpx.Response(401, json={"error": {"message": state["gen_error"]}})
            n = state["gen_body"].get("n", 1)
            if state.get("return_url"):
                data = [{"url": "https://cdn.test/i.png"} for _ in range(n)]
            else:
                data = [{"b64_json": _PNG} for _ in range(n)]
            return httpx.Response(200, json={"data": data})
        if path.endswith("/images/edits"):
            state["edit_content_type"] = request.headers.get("content-type", "")
            state["edit_calls"] = state.get("edit_calls", 0) + 1
            return httpx.Response(200, json={"data": [{"b64_json": _PNG}]})
        return httpx.Response(404)

    return handler


async def test_openai_image_generate(http, tmp_path):
    state: dict[str, Any] = {}
    http.handler = _openai(state)
    c = OpenAIImageClient("https://api.test/v1", "sk-img", model="gpt-image-1")
    res = await c.generate(_img_req(tmp_path, count=2), None)
    assert res.backend == "openai" and res.model == "gpt-image-1"
    assert len(res.image_paths) == 2
    assert res.image_paths[0].read_bytes() == base64.b64decode(_PNG)
    body = state["gen_body"]
    assert body["model"] == "gpt-image-1"
    assert body["size"] == "768x768" and body["n"] == 2
    # gpt-image-1 rejects response_format — must not be sent
    assert "response_format" not in body


async def test_openai_image_dalle_requests_b64(http, tmp_path):
    state: dict[str, Any] = {}
    http.handler = _openai(state)
    c = OpenAIImageClient("https://api.test/v1", "sk-img")
    await c.generate(_img_req(tmp_path, model="dall-e-3"), None)
    assert state["gen_body"]["response_format"] == "b64_json"


async def test_openai_image_edit_uses_edits_endpoint(http, tmp_path):
    state: dict[str, Any] = {}
    http.handler = _openai(state)
    init = tmp_path / "init.png"
    init.write_bytes(b"\x89PNG init")
    c = OpenAIImageClient("https://api.test/v1", "sk-img")
    res = await c.generate(_img_req(tmp_path, init_image=init), None)
    assert state["edit_calls"] == 1
    assert "multipart/form-data" in state["edit_content_type"]
    assert res.image_paths[0].exists()


async def test_openai_image_url_fallback(http, tmp_path):
    state: dict[str, Any] = {"return_url": True}
    http.handler = _openai(state)
    c = OpenAIImageClient("https://api.test/v1", "sk-img")
    res = await c.generate(_img_req(tmp_path), None)
    assert res.image_paths[0].read_bytes() == b"PNG-FROM-URL"


async def test_openai_image_error_mapped(http, tmp_path):
    state: dict[str, Any] = {"gen_error": "bad key"}
    http.handler = _openai(state)
    c = OpenAIImageClient("https://api.test/v1", "sk-img")
    with pytest.raises(ImageGenError, match="401"):
        await c.generate(_img_req(tmp_path), None)


async def test_openai_image_missing_key(tmp_path):
    c = OpenAIImageClient("https://api.test/v1", "")
    h = await c.health()
    assert h.ok is False
    with pytest.raises(ImageGenError, match="api key"):
        await c.generate(_img_req(tmp_path), None)


async def test_openai_image_models_synthetic():
    models = await OpenAIImageClient("https://x", "k").models()
    ids = {m.id for m in models}
    assert {"gpt-image-1", "dall-e-3", "dall-e-2"} <= ids
    by_id = {m.id: m for m in models}
    assert by_id["gpt-image-1"].supports_edit and by_id["dall-e-2"].supports_edit
    assert not by_id["dall-e-3"].supports_edit
    # a custom configured model is surfaced as an entry too
    custom = await OpenAIImageClient("https://x", "k", model="flux-pro").models()
    assert custom[0].id == "flux-pro"


def test_make_image_client_backend_switch():
    # openai falls back to llm_base_url + llm_api_key
    rt = RuntimeSettings(
        image_backend="openai",
        llm_base_url="https://llm.test/v1",
        llm_api_key=SecretStr("sk-llm"),
    )
    c = make_image_client(runtime_settings=rt)
    assert isinstance(c, OpenAIImageClient)
    assert c.base_url == "https://llm.test/v1" and c.api_key == "sk-llm"
    # dedicated image_api_* wins over the llm fallback
    rt2 = RuntimeSettings(
        image_backend="openai",
        llm_base_url="https://llm.test/v1",
        llm_api_key=SecretStr("sk-llm"),
        image_api_base="https://img.test/v1",
        image_api_key=SecretStr("sk-img"),
        image_openai_model="dall-e-3",
    )
    c2 = make_image_client(runtime_settings=rt2)
    assert c2.base_url == "https://img.test/v1" and c2.api_key == "sk-img"  # type: ignore[union-attr]
    assert c2.model == "dall-e-3"  # type: ignore[union-attr]
    # per-job override + unchanged default path
    rt3 = RuntimeSettings(mesh_base_url="http://sidecar:8765")
    assert isinstance(make_image_client(runtime_settings=rt3), object)
    assert isinstance(make_image_client(runtime_settings=rt3, backend="openai"), OpenAIImageClient)
