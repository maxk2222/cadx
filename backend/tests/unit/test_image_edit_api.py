"""Image edit API: reference-image generation, edit provenance, role_hint."""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadx.agent.image_client import FakeImageClient
from cadx.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture()
def fake_image_client(monkeypatch):
    """Route POST /projects/{id}/images through the deterministic fake client."""
    fake = FakeImageClient()
    monkeypatch.setattr("cadx.api.routers.images.make_image_client", lambda **_: fake)
    return fake


def _img_bytes(size=(32, 32), color=(10, 200, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


async def _mk_project_with_image(client) -> tuple[str, str]:
    pid = (await client.post("/api/projects", json={"name": "img-edit"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("ref.png", _img_bytes(), "image/png"))],
    )
    assert r.status_code == 200, r.text
    return pid, r.json()[0]["id"]


async def test_generate_with_reference(client, fake_image_client):
    pid, aid = await _mk_project_with_image(client)
    r = await client.post(
        f"/api/projects/{pid}/images",
        json={
            "prompt": "make it matte black",
            "reference_asset_id": aid,
            "strength": 0.45,
        },
    )
    assert r.status_code == 201, r.text
    a = r.json()[0]
    assert a["kind"] == "generated_image"
    assert a["derived_from"] == aid
    gp = a["gen_params"]
    assert gp["edit"] is True
    assert gp["reference_asset_id"] == aid
    assert gp["strength"] == 0.45
    assert gp["model"] == "fake-image-edit"
    # the request carried the init image to the client
    assert fake_image_client.calls[0]["init_image"]


async def test_generate_reference_uses_edit_default_model(client, fake_image_client):
    pid, aid = await _mk_project_with_image(client)
    r = await client.post(
        f"/api/projects/{pid}/images",
        json={"prompt": "vary it", "reference_asset_id": aid},
    )
    assert r.status_code == 201, r.text
    # no explicit model → edit jobs default to image_edit_model ("auto")
    assert r.json()[0]["gen_params"]["requested_model"] == "auto"


async def test_generate_reference_cross_project_404(client, fake_image_client):
    _pid, aid = await _mk_project_with_image(client)
    other = (await client.post("/api/projects", json={"name": "other"})).json()["id"]
    r = await client.post(
        f"/api/projects/{other}/images",
        json={"prompt": "x", "reference_asset_id": aid},
    )
    assert r.status_code == 404


async def test_generate_reference_non_image_422(client, fake_image_client):
    pid = (await client.post("/api/projects", json={"name": "stl-ref"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("part.stl", b"solid x endsolid x", "application/octet-stream"))],
    )
    aid = r.json()[0]["id"]
    r = await client.post(
        f"/api/projects/{pid}/images",
        json={"prompt": "x", "reference_asset_id": aid},
    )
    assert r.status_code == 422


async def test_generate_reference_missing_404(client, fake_image_client):
    pid = (await client.post("/api/projects", json={"name": "no-ref"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/images",
        json={"prompt": "x", "reference_asset_id": str(uuid.uuid4())},
    )
    assert r.status_code == 404


async def test_mark_sketch_sets_role_hint(client, fake_image_client):
    pid, aid = await _mk_project_with_image(client)
    r = await client.post(
        f"/api/projects/{pid}/images",
        json={
            "prompt": "turn into a clean silhouette sketch",
            "reference_asset_id": aid,
            "mark_sketch": True,
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()[0]["gen_params"]["role_hint"] == "sketch"


async def test_asset_patch_role_hint(client):
    _pid, aid = await _mk_project_with_image(client)
    r = await client.patch(f"/api/assets/{aid}", json={"role_hint": "sketch"})
    assert r.status_code == 200, r.text
    assert r.json()["gen_params"]["role_hint"] == "sketch"
    # caption still patchable alongside
    r = await client.patch(f"/api/assets/{aid}", json={"caption": "my drawing", "role_hint": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["caption"] == "my drawing"
    assert (body["gen_params"] or {}).get("role_hint") is None


async def test_settings_expose_edit_fields(client):
    r = await client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["image_edit_model"] == "auto"
    assert body["image_default_strength"] == 0.6
    # round-trip
    r = await client.put(
        "/api/settings",
        json={"image_edit_model": "qwen-image-edit", "image_default_strength": 0.35},
    )
    assert r.status_code == 200, r.text
    r = await client.get("/api/settings")
    body = r.json()
    assert body["image_edit_model"] == "qwen-image-edit"
    assert body["image_default_strength"] == 0.35
    await client.put(
        "/api/settings", json={"image_edit_model": "auto", "image_default_strength": 0.6}
    )
