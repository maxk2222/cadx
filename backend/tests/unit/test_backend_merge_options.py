"""Backend toggles: RunOptions merge, settings schema roundtrip, mesh proxies."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from cadx.agent.worker_jobs import _job_runtime, _merge_settings
from cadx.api.schemas import RunOptions, SettingsUpdate
from cadx.db.models import AppSetting
from cadx.db.session import create_all, session_scope
from cadx.runtime_config import RuntimeSettings, save_runtime_settings

# --------------------------------------------------------------------------- #
# RunOptions -> ProjectSettings merge


def test_backend_runoptions_reach_project_settings():
    opts = RunOptions(
        llm_backend="sidecar",
        mesh_backend="tripo",
        image_backend="openai",
        mesh_model="tripo-v2",
        image_model="gpt-image-1",
        image_edit_model="flux-kontext",
    ).model_dump(exclude_none=True)
    merged = _merge_settings({}, opts)
    assert merged.llm_backend == "sidecar"
    assert merged.mesh_backend == "tripo"
    assert merged.image_backend == "openai"
    assert merged.mesh_model == "tripo-v2"
    assert merged.image_model == "gpt-image-1"
    assert merged.image_edit_model == "flux-kontext"


def test_version_options_win_over_project_settings():
    merged = _merge_settings(
        {"mesh_backend": "tripo", "material": "pla"}, {"mesh_backend": "sidecar"}
    )
    assert merged.mesh_backend == "sidecar"  # version options win
    assert merged.material == "pla"  # untouched project keys survive


def test_ui_keys_still_stripped():
    merged = _merge_settings({}, {"ui_editor": {"x": 1}, "mesh_backend": "tripo"})
    assert merged.mesh_backend == "tripo"
    assert not hasattr(merged, "ui_editor")


def test_job_runtime_applies_overrides():
    rt = RuntimeSettings(mesh_model="global-mesh", image_model="global-img")
    merged = _merge_settings(
        {}, {"mesh_model": "job-mesh", "image_model": "job-img", "mesh_backend": "tripo"}
    )
    jr = _job_runtime(rt, merged)
    assert jr.mesh_model == "job-mesh"
    assert jr.image_model == "job-img"
    assert jr.image_edit_model == "auto"  # no override -> runtime default
    assert jr.mesh_backend == "tripo"
    # the global snapshot is untouched
    assert rt.mesh_model == "global-mesh" and rt.mesh_backend == "sidecar"


def test_job_runtime_no_overrides_returns_same_object():
    rt = RuntimeSettings()
    assert _job_runtime(rt, _merge_settings({}, {})) is rt


# --------------------------------------------------------------------------- #
# settings schema roundtrip / secrets


def test_new_secret_fields_not_public():
    s = RuntimeSettings(
        llm_api_key=SecretStr("sk-llm-1"),
        tripo_api_key=SecretStr("sk-tripo-1"),
        image_api_key=SecretStr("sk-img-1"),
    )
    pub = s.public_values()
    for leaked in ("llm_api_key", "tripo_api_key", "image_api_key"):
        assert leaked not in pub
    dumped = json.dumps(pub)
    for secret in ("sk-llm-1", "sk-tripo-1", "sk-img-1"):
        assert secret not in dumped
    assert pub["llm_api_key_set"] and pub["tripo_api_key_set"] and pub["image_api_key_set"]
    # storage roundtrip keeps the raw secrets for internal use
    stored = s.to_storage()
    assert stored["tripo_api_key"] == "sk-tripo-1"
    assert stored["image_api_key"] == "sk-img-1"
    rt2 = RuntimeSettings.model_validate(stored)
    assert rt2.tripo_api_key.get_secret_value() == "sk-tripo-1"
    assert rt2.image_api_key.get_secret_value() == "sk-img-1"


def test_settings_update_accepts_new_fields():
    body = SettingsUpdate(
        llm_backend="sidecar",
        llm_sidecar_model="qwen3",
        mesh_backend="tripo",
        image_backend="openai",
        tripo_api_key="tk",
        image_api_key="ik",
        clear_llm_api_key=True,
        image_openai_model="dall-e-3",
    )
    fields = body.model_fields_set
    for f in (
        "llm_backend",
        "mesh_backend",
        "image_backend",
        "tripo_api_key",
        "image_api_key",
        "image_openai_model",
    ):
        assert f in fields


# --------------------------------------------------------------------------- #
# /api/mesh/* sidecar-management proxies


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _set_runtime(**kw: Any) -> None:
    async with session_scope() as db:
        row = await db.get(AppSetting, "runtime")
        if row is not None:
            await db.delete(row)
        await save_runtime_settings(db, RuntimeSettings(**kw))
        await db.commit()


async def _clear_runtime() -> None:
    async with session_scope() as db:
        row = await db.get(AppSetting, "runtime")
        if row is not None:
            await db.delete(row)
            await db.commit()


def _mock_httpx(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_async = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real_async(transport=transport),
    )


async def test_mesh_proxies_503_when_unconfigured(client):
    await _clear_runtime()  # default runtime: mesh_base_url == ""
    for method, path in (
        ("get", "/api/mesh/queue"),
        ("post", "/api/mesh/models/m1/download"),
        ("get", "/api/mesh/models/m1/download"),
        ("delete", "/api/mesh/models/m1/download"),
        ("delete", "/api/mesh/models/m1/weights"),
        ("post", "/api/mesh/models/register"),
        ("delete", "/api/mesh/models/m1/custom"),
    ):
        kwargs = {"json": {"repo": "org/m", "kind": "llm"}} if path.endswith("register") else {}
        r = await getattr(client, method)(path, **kwargs)
        assert r.status_code == 503, (method, path, r.text)
        assert r.json()["detail"] == "mesh server not configured"


async def test_mesh_proxy_passthroughs(client, monkeypatch):
    await _set_runtime(mesh_base_url="http://sidecar.test")
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        p = request.url.path
        if p == "/v1/queue":
            return httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": "j1",
                            "kind": "mesh",
                            "model": "hunyuan",
                            "resolved_model": "hunyuan3d-2mv",
                            "status": "running",
                            "waited_s": 1.5,
                            "est_mem_mb": 8000,
                            "message": "generating",
                        }
                    ],
                    "free_mb": 2048.0,
                    "residents": [
                        {"id": "hunyuan", "mb": 8000, "kind": "mesh", "pinned": True}
                    ],
                },
            )
        if p == "/v1/models/m1/download":
            if request.method == "POST":
                return httpx.Response(
                    202, json={"status": "downloading", "bytes_done": 10}
                )
            if request.method == "DELETE":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                200, json={"status": "downloading", "bytes_done": 42, "bytes_total": 100}
            )
        if p == "/v1/models/m1/weights":
            return httpx.Response(200, json={"deleted": True, "error": None})
        if p == "/v1/models/register":
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "custom:m",
                    "name": body.get("name") or "m",
                    "kind": body["kind"],
                    "custom": True,
                    "supports_vision": bool(body.get("vision")),
                },
            )
        if p == "/v1/models/m1/custom":
            return httpx.Response(200, json={"ok": True})
        if p == "/v1/models/gone/download":
            return httpx.Response(404, json={"detail": "no such model"})
        return httpx.Response(404, json={"detail": "nope"})

    try:
        _mock_httpx(monkeypatch, handler)

        r = await client.get("/api/mesh/queue")
        assert r.status_code == 200, r.text
        q = r.json()
        assert q["jobs"][0]["resolved_model"] == "hunyuan3d-2mv"
        assert q["free_mb"] == 2048.0
        assert q["residents"][0]["pinned"] is True

        r = await client.post("/api/mesh/models/m1/download")
        assert r.status_code == 200 and r.json()["status"] == "downloading"
        assert ("POST", "/v1/models/m1/download") in calls

        r = await client.get("/api/mesh/models/m1/download")
        assert r.json()["bytes_done"] == 42 and r.json()["bytes_total"] == 100

        r = await client.delete("/api/mesh/models/m1/download")
        assert r.json()["ok"] is True

        r = await client.delete("/api/mesh/models/m1/weights")
        assert r.json() == {"deleted": True, "error": None}

        r = await client.post(
            "/api/mesh/models/register",
            json={"repo": "org/m", "kind": "llm", "name": "M", "vision": True},
        )
        assert r.status_code == 200, r.text
        m = r.json()
        assert m["custom"] is True and m["kind"] == "llm" and m["supports_vision"] is True

        r = await client.delete("/api/mesh/models/m1/custom")
        assert r.json()["ok"] is True

        # sidecar errors are forwarded with their status + body
        r = await client.get("/api/mesh/models/gone/download")
        assert r.status_code == 404 and r.json()["detail"] == "no such model"
    finally:
        await _clear_runtime()


async def test_llm_models_endpoint_external(client, monkeypatch):
    await _set_runtime(llm_backend="external", llm_base_url="https://llm.test/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "x"}]})
        return httpx.Response(404)

    try:
        _mock_httpx(monkeypatch, handler)
        r = await client.get("/api/settings/llm-models")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["backend"] == "external"
        assert body["error"] is None
        assert body["models"] == [{"id": "gpt-4o", "vision": False}, {"id": "x", "vision": False}]
    finally:
        await _clear_runtime()


async def test_llm_models_endpoint_sidecar(client, monkeypatch):
    await _set_runtime(llm_backend="sidecar", mesh_base_url="http://sidecar.test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "id": "qwen3",
                            "kind": "llm",
                            "supports_vision": True,
                            "downloaded": True,
                        },
                        {"id": "hunyuan", "kind": "mesh"},
                    ]
                },
            )
        return httpx.Response(404)

    try:
        _mock_httpx(monkeypatch, handler)
        r = await client.get("/api/settings/llm-models")
        body = r.json()
        assert body["backend"] == "sidecar"
        # only kind=="llm" listed, vision+downloaded flags mapped through
        assert body["models"] == [
            {"id": "qwen3", "vision": True, "downloaded": True}
        ]
    finally:
        await _clear_runtime()


async def test_llm_models_endpoint_sidecar_unconfigured(client):
    # sidecar backend selected but no mesh_base_url -> clean error, no 500
    await _set_runtime(llm_backend="sidecar")
    try:
        r = await client.get("/api/settings/llm-models")
        body = r.json()
        assert body["backend"] == "sidecar"
        assert body["models"] == [] and "not configured" in body["error"]
    finally:
        await _clear_runtime()


async def test_settings_out_new_fields(client):
    await _set_runtime(
        mesh_backend="tripo",
        tripo_api_key=SecretStr("tk-secret"),
        image_backend="openai",
        llm_backend="sidecar",
        mesh_base_url="http://sidecar.test",
    )
    try:
        r = await client.get("/api/settings")
        assert r.status_code == 200
        body = r.json()
        assert body["llm_backend"] == "sidecar"
        assert body["mesh_backend"] == "tripo"
        assert body["image_backend"] == "openai"
        assert body["tripo_api_key_set"] is True
        assert body["image_api_key_set"] is False
        assert "tripo_api_key" not in body and "image_api_key" not in body
        assert "tk-secret" not in json.dumps(body)
    finally:
        await _clear_runtime()


async def test_settings_put_backends_and_secret_roundtrip(client):
    try:
        r = await client.put(
            "/api/settings",
            json={
                "mesh_backend": "tripo",
                "tripo_api_key": "tk-1",
                "image_backend": "openai",
                "image_api_key": "ik-1",
            },
        )
        body = r.json()
        assert r.status_code == 200, r.text
        assert body["mesh_backend"] == "tripo" and body["image_backend"] == "openai"
        assert body["tripo_api_key_set"] is True and body["image_api_key_set"] is True
        from cadx.runtime_config import load_runtime_settings

        async with session_scope() as db:
            loaded = await load_runtime_settings(db)
        assert loaded.tripo_api_key.get_secret_value() == "tk-1"
        assert loaded.image_api_key.get_secret_value() == "ik-1"
        # clear flag resets the stored secret
        r = await client.put("/api/settings", json={"clear_tripo_api_key": True})
        assert r.json()["tripo_api_key_set"] is False
        async with session_scope() as db:
            loaded = await load_runtime_settings(db)
        assert loaded.tripo_api_key.get_secret_value() == ""
    finally:
        await _clear_runtime()
