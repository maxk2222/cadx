"""Image editing: derived-asset endpoint, rembg endpoint, alpha-safe LLM prep."""

from __future__ import annotations

import io
import sys

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadx.db.session import create_all
from cadx.llm.images import prepare


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _png(size=(24, 16), mode="RGBA") -> bytes:
    buf = io.BytesIO()
    img = Image.new(mode, size, (10, 20, 30, 255))
    if mode == "RGBA":
        # transparent corner so alpha survives roundtrip detectably
        for x in range(6):
            for y in range(6):
                img.putpixel((x, y), (0, 0, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _upload(client, name="src.png") -> dict:
    pid = (await client.post("/api/projects", json={"name": "edit"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", (name, _png(), "image/png"))],
    )
    assert r.status_code == 200, r.text
    return r.json()[0]


# --- derivative ----------------------------------------------------------------


async def test_derivative_stores_png_and_keeps_alpha(client):
    src = await _upload(client)
    r = await client.post(
        f"/api/assets/{src['id']}/derivative",
        files=[("file", ("edit.png", _png(size=(30, 20)), "image/png"))],
    )
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["id"] != src["id"]
    assert a["derived_from"] == src["id"]
    assert a["mime"] == "image/png"
    assert a["filename"].endswith("-edit.png")
    assert (a["width"], a["height"]) == (30, 20)

    r = await client.get(f"/api/assets/{a['id']}/file")
    stored = Image.open(io.BytesIO(r.content))
    assert stored.format == "PNG" and stored.mode == "RGBA"
    assert stored.getpixel((0, 0))[3] == 0  # alpha survived

    # original untouched
    r = await client.get(f"/api/assets/{src['id']}/file")
    assert Image.open(io.BytesIO(r.content)).format == "JPEG"


async def test_derivative_rejects_garbage(client):
    src = await _upload(client)
    r = await client.post(
        f"/api/assets/{src['id']}/derivative",
        files=[("file", ("edit.png", b"not a png", "image/png"))],
    )
    assert r.status_code == 415


async def test_derivative_rejects_jpeg_payload(client):
    src = await _upload(client)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    r = await client.post(
        f"/api/assets/{src['id']}/derivative",
        files=[("file", ("edit.png", buf.getvalue(), "image/png"))],
    )
    assert r.status_code == 415


async def test_derivative_404_and_non_image(client):
    r = await client.post(
        "/api/assets/00000000-0000-0000-0000-000000000000/derivative",
        files=[("file", ("e.png", _png(), "image/png"))],
    )
    assert r.status_code == 404

    pid = (await client.post("/api/projects", json={"name": "step"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("p.step", b"ISO-10303-21;", "application/octet-stream"))],
    )
    step = r.json()[0]
    r = await client.post(
        f"/api/assets/{step['id']}/derivative",
        files=[("file", ("e.png", _png(), "image/png"))],
    )
    assert r.status_code == 400


# --- remove-bg -----------------------------------------------------------------


async def test_remove_bg_returns_png(client, monkeypatch):
    src = await _upload(client)

    class _FakeRembg:
        @staticmethod
        def new_session(model):
            return object()

        @staticmethod
        def remove(img, session=None):
            out = img.convert("RGBA")
            out.putpixel((0, 0), (0, 0, 0, 0))
            return out

    monkeypatch.setitem(sys.modules, "rembg", _FakeRembg)
    monkeypatch.setattr("cadx.api.routers.assets._rembg_sess", None)
    r = await client.post(f"/api/assets/{src['id']}/remove-bg")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "RGBA"
    # no asset created — editor persists via /derivative on save
    listed = await client.get(f"/api/projects/{src['project_id']}/assets")
    assert len(listed.json()) == 1


async def test_remove_bg_503_without_rembg(client, monkeypatch):
    src = await _upload(client)
    monkeypatch.setitem(sys.modules, "rembg", None)
    r = await client.post(f"/api/assets/{src['id']}/remove-bg")
    assert r.status_code == 503


# --- alpha handling for the LLM path -------------------------------------------


def test_prepare_flattens_alpha_onto_white(tmp_path):
    p = tmp_path / "cutout.png"
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))  # fully transparent
    img.putpixel((5, 5), (255, 0, 0, 255))
    img.save(p)
    out = Image.open(io.BytesIO(prepare(p)))
    corner = out.getpixel((0, 0))
    assert corner[0] > 240 and corner[1] > 240  # white, not black


def test_prepare_max_edge_override(tmp_path):
    """prepare(max_edge=N) downscales the long edge to N — the runtime
    llm_image_max_edge knob rides on this parameter."""
    p = tmp_path / "big.png"
    Image.new("RGB", (1200, 600), (40, 80, 120)).save(p)
    out = Image.open(io.BytesIO(prepare(p, max_edge=512)))
    assert max(out.size) <= 512
    assert out.size == (512, 256)


def test_image_part_uses_runtime_defaults(tmp_path):
    """image_part() picks detail/max-edge from RuntimeSettings when not
    given explicitly — explicit args still win."""
    from cadx.llm.images import image_part

    p = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(p)
    part = image_part(p)
    assert part["type"] == "image_url"
    assert part["image_url"]["detail"] == "high"  # code default
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert image_part(p, detail="low")["image_url"]["detail"] == "low"
