"""ImagePort clients: FakeImageClient, make_image_client factory, HttpImageClient (mocked)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from PIL import Image

from cadx.agent.image_client import (
    FakeImageClient,
    HttpImageClient,
    ImageGenError,
    ImageGenRequest,
    make_image_client,
)


def _req(tmp_path, seed=7, count=1) -> ImageGenRequest:
    return ImageGenRequest(
        prompt="a red gear",
        seed=seed,
        count=count,
        out_dir=tmp_path,
    )


async def test_fake_client_generates_png(tmp_path):
    c = FakeImageClient()
    res = await c.generate(_req(tmp_path), None)
    assert res.model == "fake-image"
    assert len(res.image_paths) == 1
    im = Image.open(res.image_paths[0])
    assert im.format == "PNG"
    assert c.calls[0]["seed"] == 7


async def test_fake_client_count_and_determinism(tmp_path):
    c = FakeImageClient()
    res = await c.generate(_req(tmp_path, count=2), None)
    assert len(res.image_paths) == 2
    b1 = res.image_paths[0].read_bytes()
    c2 = FakeImageClient()
    res2 = await c2.generate(_req(tmp_path / "b", count=2, seed=8), None)
    assert res2.image_paths[0].read_bytes() != b1  # different seed → different image


async def test_fake_client_health_lists_image_kind():
    c = FakeImageClient()
    h = await c.health()
    assert h.ok
    models = await c.models()
    assert models[0].kind == "image"
    assert models[0].id == "fake-image"


def test_make_image_client_none_without_url():
    assert make_image_client() is None
    assert isinstance(make_image_client(fake=True), FakeImageClient)
    assert isinstance(make_image_client(base_url="http://x:8765"), HttpImageClient)


# --------------------------------------------------------------------------- #
# HttpImageClient against a mocked transport


class _FakeServer:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.posted: list[dict] = []
        self.fail_with: str | None = None  # if set, submitted jobs fail with this error

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"id": "fake", "kind": "mesh"},
                        {"id": "fake-image", "kind": "image", "available": True},
                    ]
                },
            )
        if path == "/v1/image-jobs" and request.method == "POST":
            import json as _json

            self.posted.append(_json.loads(request.content))
            if self.fail_with:
                self.jobs["i1"] = {"status": "failed", "error": self.fail_with}
            else:
                self.jobs["i1"] = {
                    "status": "succeeded",
                    "model": "flux-schnell-mlx",
                    "backend": "mlx",
                    "seed": 42,
                    "steps": 4,
                    "images": 2,
                }
            return httpx.Response(200, json={"job_id": "i1"})
        if path.startswith("/v1/image-jobs/"):
            job_id = path.split("/")[3]
            if request.method == "DELETE":
                self.deleted.append(job_id)
                return httpx.Response(200, json={"ok": True})
            if path.endswith("/image"):
                import io

                buf = io.BytesIO()
                Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
                return httpx.Response(200, content=buf.getvalue())
            job = self.jobs.get(job_id)
            return httpx.Response(200, json=job) if job else httpx.Response(404)
        return httpx.Response(404)


@pytest.fixture
def fake_server(monkeypatch):
    srv = _FakeServer()
    transport = httpx.MockTransport(srv.handler)
    real_async, real_sync = httpx.AsyncClient, httpx.Client

    monkeypatch.setattr(
        "cadx.agent.image_client.httpx.AsyncClient",
        lambda *a, **kw: real_async(transport=transport),
    )
    monkeypatch.setattr(
        "cadx.agent.image_client.httpx.Client",
        lambda *a, **kw: real_sync(transport=transport),
    )

    async def _sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return srv


async def test_http_client_models_filters_image_kind(fake_server):
    c = HttpImageClient("http://mesh.test")
    models = await c.models()
    assert [m.id for m in models] == ["fake-image"]


async def test_http_client_generate(fake_server, tmp_path):
    c = HttpImageClient("http://mesh.test", timeout_s=30)
    res = await c.generate(_req(tmp_path, count=2), None)
    assert res.model == "flux-schnell-mlx"
    assert res.backend == "mlx"
    assert res.seed == 42
    assert len(res.image_paths) == 2
    assert all(p.exists() for p in res.image_paths)
    # client-local field must NOT be serialised to the sidecar
    assert "out_dir" not in fake_server.posted[0]
    assert fake_server.posted[0]["prompt"] == "a red gear"
    assert "i1" in fake_server.deleted  # cleanup


async def test_http_client_job_failed(fake_server, tmp_path):
    fake_server.fail_with = "VRAM exhausted — try a smaller model"
    with pytest.raises(ImageGenError, match="VRAM exhausted"):
        await HttpImageClient("http://mesh.test", timeout_s=30).generate(_req(tmp_path), None)
