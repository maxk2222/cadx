"""Asset upload: server-side image normalization (EXIF/resize/JPEG)."""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadagent.db.session import create_all
from cadagent.llm.images import normalize


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadagent.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _img_bytes(size=(16, 16), mode="RGB", fmt="PNG", color=(128, 64, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


# --- unit: normalize() --------------------------------------------------------


def test_normalize_downscales_and_flattens():
    buf = io.BytesIO()
    Image.effect_noise((4000, 3000), 128).convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()
    out = normalize(data)
    assert out is not None
    jpeg, w, h = out
    assert (w, h) == (2048, 1536)
    assert len(jpeg) < len(data)
    im = Image.open(io.BytesIO(jpeg))
    assert im.format == "JPEG" and im.mode == "RGB"


def test_normalize_never_upscales():
    out = normalize(_img_bytes(size=(100, 50)))
    assert out is not None
    assert out[1:] == (100, 50)


def test_normalize_alpha_on_white():
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(buf, format="PNG")  # fully transparent
    out = normalize(buf.getvalue())
    assert out is not None
    px = Image.open(io.BytesIO(out[0])).getpixel((32, 32))
    assert min(px) > 240  # white background, not black


def test_normalize_gif_first_frame():
    buf = io.BytesIO()
    f1 = Image.new("RGB", (40, 30), (255, 0, 0))
    f2 = Image.new("RGB", (40, 30), (0, 255, 0))
    f1.save(buf, format="GIF", save_all=True, append_images=[f2])
    out = normalize(buf.getvalue())
    assert out is not None
    assert out[1:] == (40, 30)
    px = Image.open(io.BytesIO(out[0])).getpixel((20, 15))
    assert px[0] > 200 and px[1] < 80  # first (red) frame


def test_normalize_undecodable_returns_none():
    assert normalize(b"not an image at all") is None


# --- api: upload ---------------------------------------------------------------


async def test_upload_normalizes_large_image(client):
    pid = (await client.post("/api/projects", json={"name": "img-norm"})).json()["id"]
    raw = _img_bytes(size=(4000, 3000))
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("photo.png", raw, "image/png"))],
    )
    assert r.status_code == 200, r.text
    a = r.json()[0]
    assert a["width"] == 2048 and a["height"] == 1536
    assert a["mime"] == "image/jpeg"
    assert a["filename"].endswith(".jpg")

    r = await client.get(f"/api/assets/{a['id']}/file")
    assert r.status_code == 200
    stored = Image.open(io.BytesIO(r.content))
    assert stored.format == "JPEG" and stored.size == (2048, 1536)


async def test_upload_step_passthrough(client):
    pid = (await client.post("/api/projects", json={"name": "step-past"})).json()["id"]
    raw = b"ISO-10303-21; fake step body " * 100
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("bracket.step", raw, "application/octet-stream"))],
    )
    assert r.status_code == 200, r.text
    a = r.json()[0]
    assert a["filename"] == "bracket.step"
    assert a["width"] is None
    r = await client.get(f"/api/assets/{a['id']}/file")
    assert r.content == raw


async def test_upload_undecodable_image_falls_back_to_raw(client):
    pid = (await client.post("/api/projects", json={"name": "raw-fb"})).json()["id"]
    raw = b"\xff\xd8\xff garbage that is not a real jpeg"
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("scan.jpg", raw, "image/jpeg"))],
    )
    assert r.status_code == 200, r.text
    a = r.json()[0]
    assert a["filename"] == "scan.jpg" and a["width"] is None
    r = await client.get(f"/api/assets/{a['id']}/file")
    assert r.content == raw


async def test_upload_heic_accepted_as_raw(client):
    pid = (await client.post("/api/projects", json={"name": "heic"})).json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/assets",
        files=[("files", ("shot.heic", b"\x00fake-heic", "image/heic"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()[0]["kind"] == "image"
