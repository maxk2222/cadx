"""Generated-image persistence: Asset rows + provenance + the /images API."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from cadx import storage
from cadx.agent.image_assets import persist_generated_images
from cadx.db.models import AssetKind, Project
from cadx.db.session import create_all, session_scope


def _png(tmp_path, name="g0.png", size=(32, 24)) -> Path:
    p = Path(tmp_path) / name
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
async def db():
    await create_all()
    async with session_scope() as s:
        yield s


async def test_persist_generated_images(db: AsyncSession, tmp_path):
    project = Project(name="p", description="")
    db.add(project)
    await db.flush()
    src = _png(tmp_path)
    assets = persist_generated_images(
        db,
        project.id,
        [src],
        prompt="a brass gear",
        gen_params={"prompt": "a brass gear", "model": "flux-schnell-mlx", "seed": 5},
    )
    await db.commit()
    assert len(assets) == 1
    a = assets[0]
    assert a.kind == AssetKind.generated_image
    assert a.mime == "image/png"
    assert (a.width, a.height) == (32, 24)
    assert a.gen_params["model"] == "flux-schnell-mlx"
    assert a.gen_params["seed"] == 5
    assert a.gen_params["image_index"] == 0
    # file landed inside DATA_DIR via storage helpers
    p = storage.resolve(a.path)
    assert p.exists()
    assert Image.open(p).format == "PNG"
    assert a.caption == "a brass gear"


async def test_persist_multiple_and_caption_override(db: AsyncSession, tmp_path):
    project = Project(name="p2", description="")
    db.add(project)
    await db.flush()
    assets = persist_generated_images(
        db,
        project.id,
        [_png(tmp_path, "a.png"), _png(tmp_path, "b.png")],
        prompt="x",
        caption="hero candidate",
        gen_params={"prompt": "x", "model": "fake-image", "seed": 1},
    )
    await db.commit()
    assert [a.gen_params["image_index"] for a in assets] == [0, 1]
    assert all(a.caption == "hero candidate" for a in assets)


# --------------------------------------------------------------------------- #
# API: POST /projects/{id}/images with a fake image client


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_generate_images_api(client, monkeypatch):
    from cadx.agent.image_client import FakeImageClient
    from cadx.api.routers import images as images_router

    monkeypatch.setattr(images_router, "make_image_client", lambda **kw: FakeImageClient())
    # project
    r = await client.post("/api/projects", json={"name": "imgtest", "description": ""})
    assert r.status_code in (200, 201)
    pid = r.json()["id"]

    r = await client.post(
        f"/api/projects/{pid}/images",
        json={"prompt": "a steel bracket", "seed": 9, "count": 2, "width": 64, "height": 64},
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert len(out) == 2
    a = out[0]
    assert a["kind"] == "generated_image"
    assert a["gen_params"]["prompt"] == "a steel bracket"
    assert a["gen_params"]["model"] == "fake-image"
    assert a["gen_params"]["seed"] == 9
    # file served back
    f = await client.get(a["url"])
    assert f.status_code == 200
    assert Image.open(io.BytesIO(f.content)).format == "PNG"


async def test_generate_images_503_without_adapter(client, monkeypatch):
    from cadx.agent.image_client import FakeImageClient
    from cadx.api.routers import images as images_router

    class NoModels(FakeImageClient):
        async def models(self):
            return []

    monkeypatch.setattr(images_router, "make_image_client", lambda **kw: NoModels())
    r = await client.post("/api/projects", json={"name": "imgtest2", "description": ""})
    pid = r.json()["id"]
    r = await client.post(f"/api/projects/{pid}/images", json={"prompt": "x"})
    assert r.status_code == 503


async def test_generate_images_503_unconfigured(client, monkeypatch):
    from cadx.api.routers import images as images_router

    monkeypatch.setattr(images_router, "make_image_client", lambda **kw: None)
    r = await client.post("/api/projects", json={"name": "imgtest3", "description": ""})
    pid = r.json()["id"]
    r = await client.post(f"/api/projects/{pid}/images", json={"prompt": "x"})
    assert r.status_code == 503


async def test_generated_image_derivable(client, monkeypatch):
    """generated_image assets can be edited like regular images (derivative)."""
    from cadx.agent.image_client import FakeImageClient
    from cadx.api.routers import images as images_router

    monkeypatch.setattr(images_router, "make_image_client", lambda **kw: FakeImageClient())
    r = await client.post("/api/projects", json={"name": "imgtest4", "description": ""})
    pid = r.json()["id"]
    r = await client.post(f"/api/projects/{pid}/images", json={"prompt": "x"})
    aid = r.json()[0]["id"]

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(buf, format="PNG")
    r = await client.post(
        f"/api/assets/{aid}/derivative",
        files={"file": ("edit.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["derived_from"] == aid
    assert d["kind"] == "image"  # edited copy → regular image asset
