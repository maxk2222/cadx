"""MeshPort clients: FakeMeshClient, make_mesh_client factory, HttpMeshClient (mocked HTTP)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import trimesh

from cadx.agent.mesh_client import (
    CachedMeshAvailability,
    FakeMeshClient,
    HttpMeshClient,
    MeshGenRequest,
    make_mesh_client,
)


def _req(tmp_path, seed=7) -> MeshGenRequest:
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG fake")
    return MeshGenRequest(
        images=[img],
        subject="knob",
        seed=seed,
        out_path=tmp_path / "out.glb",
    )


async def test_fake_client_generates_glb(tmp_path):
    c = FakeMeshClient()
    res = await c.generate(_req(tmp_path), None)
    assert res.model == "fake"
    assert res.mesh_path.exists()
    m = trimesh.load(str(res.mesh_path), force="mesh")
    assert len(m.faces) == res.triangles > 0
    # normalized space (like real generators)
    assert max(m.extents) <= 1.01
    assert c.calls[0]["seed"] == 7


async def test_fake_client_deterministic_seed(tmp_path):
    c1, c2 = FakeMeshClient(), FakeMeshClient()
    (tmp_path / "b").mkdir()
    r1 = await c1.generate(_req(tmp_path, seed=1), None)
    r2 = await c2.generate(_req(tmp_path / "b", seed=2), None)
    b1 = r1.mesh_path.read_bytes()
    b2 = r2.mesh_path.read_bytes()
    assert b1 == r1.mesh_path.read_bytes()
    # different seeds give different geometry
    assert b1 != b2


async def test_fake_client_health():
    h = await FakeMeshClient().health()
    assert h.ok
    assert h.device == "fake"
    assert h.models[0].id == "fake"


def test_make_mesh_client_none_without_url():
    # default RuntimeSettings have no mesh_base_url → unconfigured
    assert make_mesh_client() is None
    assert isinstance(make_mesh_client(fake=True), FakeMeshClient)
    assert isinstance(make_mesh_client(base_url="http://x:8765"), HttpMeshClient)


async def test_cached_availability_single_probe():
    class C(FakeMeshClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def health(self):
            self.n += 1
            return await super().health()

    c = CachedMeshAvailability(C())
    assert await c.available()
    assert await c.available()
    assert c.client.n == 1  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# HttpMeshClient against a mocked transport


class _FakeServer:
    """In-memory mesh-server used via httpx.MockTransport."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.submit_status = 200
        self.override = None  # callable(request) -> Response | None, checked first

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.override is not None:
            r = self.override(request)
            if r is not None:
                return r
        return self._dispatch(request)

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/health":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "device": "cpu",
                    "backend": "fake",
                    "models": [{"id": "fake", "device": "cpu"}],
                },
            )
        if path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "fake", "device": "cpu"}]})
        if path == "/v1/jobs" and request.method == "POST":
            if self.submit_status != 200:
                return httpx.Response(self.submit_status, text="busy")
            job_id = "j1"
            self.jobs[job_id] = {"status": "succeeded", "model": "fake", "seed": 7, "triangles": 12}
            return httpx.Response(200, json={"job_id": job_id})
        if path.startswith("/v1/jobs/"):
            parts = path.split("/")
            job_id = parts[3]
            if request.method == "DELETE":
                self.deleted.append(job_id)
                return httpx.Response(200, json={"ok": True, "deleted": True})
            if path.endswith("/mesh"):
                m = trimesh.creation.box()
                import io as _io

                buf = _io.BytesIO()
                m.export(buf, file_type="glb")
                return httpx.Response(
                    200, content=buf.getvalue(), headers={"content-type": "model/gltf-binary"}
                )
            job = self.jobs.get(job_id)
            if job is None:
                return httpx.Response(404)
            return httpx.Response(200, json=job)
        return httpx.Response(404)


@pytest.fixture
def fake_server(monkeypatch):
    srv = _FakeServer()
    transport = httpx.MockTransport(srv.handler)
    real_async, real_sync = httpx.AsyncClient, httpx.Client

    def _async_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_async(transport=transport)

    def _sync_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_sync(transport=transport)

    async def _sleep(_):
        return None

    monkeypatch.setattr("cadx.agent.mesh_client.httpx.AsyncClient", _async_client)
    monkeypatch.setattr("cadx.agent.mesh_client.httpx.Client", _sync_client)
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return srv


async def test_http_client_health(fake_server):
    c = HttpMeshClient("http://mesh.test")
    h = await c.health()
    assert h.ok and h.device == "cpu"
    models = await c.models()
    assert models[0].id == "fake"


async def test_http_client_generate(fake_server, tmp_path):
    c = HttpMeshClient("http://mesh.test", timeout_s=30)
    res = await c.generate(_req(tmp_path), None)
    assert res.model == "fake"
    assert res.mesh_path.exists()
    m = trimesh.load(str(res.mesh_path), force="mesh")
    assert len(m.faces) == 12
    # cleanup: server job deleted after download
    assert "j1" in fake_server.deleted


async def test_http_client_submit_error(fake_server, tmp_path):
    fake_server.submit_status = 429
    with pytest.raises(Exception, match="429"):
        await HttpMeshClient("http://mesh.test").generate(_req(tmp_path), None)


async def test_http_client_job_failed(fake_server, tmp_path):
    srv = fake_server

    def failing(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/v1/jobs" and request.method == "POST":
            srv.jobs["j9"] = {"status": "failed", "error": "no adapter"}
            return httpx.Response(200, json={"job_id": "j9"})
        return None

    srv.override = failing
    with pytest.raises(Exception, match="no adapter"):
        await HttpMeshClient("http://mesh.test").generate(_req(tmp_path), None)


async def test_http_client_stalled_job_aborts(fake_server, tmp_path):
    """A 'running' job whose progress never moves is aborted, not polled
    until the full timeout (observed live: hunyuan3d-2mv froze at 9.5%
    with 0% CPU for >10min)."""
    srv = fake_server

    def frozen(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/v1/jobs" and request.method == "POST":
            srv.jobs["js"] = {"status": "running", "progress": 0.095, "message": "generating"}
            return httpx.Response(200, json={"job_id": "js"})
        return None

    srv.override = frozen
    c = HttpMeshClient("http://mesh.test", timeout_s=900, stall_timeout_s=6)
    with pytest.raises(Exception, match="stalled"):
        await c.generate(_req(tmp_path), None)


async def test_http_client_memory_gate_refuses_low_host_ram(fake_server, tmp_path):
    """B.2: when the sidecar reports less free RAM than configured, submit is
    refused before a job is even created — a job started under memory
    pressure stalls at 0% CPU and swaps the host."""
    srv = fake_server

    def low_mem(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/v1/health":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "device": "mps",
                    "backend": "torch-mps",
                    "models": [],
                    "mem_free_mb": 800,
                },
            )
        return None

    srv.override = low_mem
    c = HttpMeshClient("http://mesh.test", min_free_mb=4096)
    with pytest.raises(Exception, match="only 800MB free"):
        await c.generate(_req(tmp_path), None)
    assert srv.jobs == {}  # nothing was submitted

    # gate off / enough memory → normal flow
    srv.override = lambda r: None
    c2 = HttpMeshClient("http://mesh.test", min_free_mb=4096)
    # still below threshold? no — override cleared, health has no mem_free_mb
    res = await c2.generate(_req(tmp_path), None)
    assert res.model == "fake"


async def test_http_client_progress_resets_stall_clock(fake_server, tmp_path):
    """Progress that keeps advancing must not trip the stall watchdog."""
    srv = fake_server
    state = {"n": 0}

    def slow(request: httpx.Request) -> httpx.Response | None:
        if request.url.path == "/v1/jobs" and request.method == "POST":
            srv.jobs["jp"] = {"status": "running", "progress": 0.1, "message": "generating"}
            return httpx.Response(200, json={"job_id": "jp"})
        if request.url.path == "/v1/jobs/jp":
            state["n"] += 1
            # progress advances every second poll — inside stall window
            if state["n"] % 2 == 0:
                srv.jobs["jp"]["progress"] = min(0.1 + state["n"] * 0.05, 0.9)
            if state["n"] > 10:
                srv.jobs["jp"] = {"status": "succeeded", "model": "fake", "triangles": 5}
        return None

    srv.override = slow
    c = HttpMeshClient("http://mesh.test", timeout_s=900, stall_timeout_s=5)
    res = await c.generate(_req(tmp_path), None)
    assert res.model == "fake"


def test_request_schema_defaults(tmp_path):
    r = _req(tmp_path)
    d = json.loads(r.model_dump_json())
    assert d["model"] == "auto"
    assert d["seed"] == 7
    assert d["image_roles"] == []
