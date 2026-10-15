"""Image-job API contract tests using the fake-image adapter (no GPU needed)."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pytest

os.environ["MESH_FAKE"] = "1"

from meshserver.app import create_app
from meshserver.config import ServerConfig


@pytest.fixture
def app(tmp_path):
    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", bg_removal="never")
    return create_app(cfg)


def _run_image_job(c, body: dict) -> dict:
    r = c.post("/v1/image-jobs", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    status = {}
    for _ in range(200):
        r = c.get(f"/v1/image-jobs/{job_id}")
        status = r.json()
        if status["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.1)
    status["job_id"] = job_id
    return status


def test_image_job_flow(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        # health lists the image adapter with kind=image
        h = c.get("/v1/health").json()
        fake = next(m for m in h["models"] if m["id"] == "fake-image")
        assert fake["kind"] == "image"
        assert fake["available"] is True

        st = _run_image_job(c, {"model": "fake-image", "prompt": "a red mug", "seed": 7})
        assert st["status"] == "succeeded", st
        assert st["images"] == 1
        assert st["model"] == "fake-image"

        r = c.get(f"/v1/image-jobs/{st['job_id']}/image")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

        r = c.delete(f"/v1/image-jobs/{st['job_id']}")
        assert r.status_code == 200
        assert c.get(f"/v1/image-jobs/{st['job_id']}").status_code == 404


def test_image_job_multi(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        st = _run_image_job(
            c, {"model": "fake-image", "prompt": "x", "count": 3, "seed": 5}
        )
        assert st["status"] == "succeeded", st
        assert st["images"] == 3
        seen = set()
        for i in range(3):
            r = c.get(f"/v1/image-jobs/{st['job_id']}/image?i={i}")
            assert r.status_code == 200
            seen.add(r.content)
        assert len(seen) == 3  # distinct seeds -> distinct images


def test_image_deterministic_seed(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        a = _run_image_job(c, {"model": "fake-image", "prompt": "same", "seed": 42})
        b = _run_image_job(c, {"model": "fake-image", "prompt": "same", "seed": 42})
        ra = c.get(f"/v1/image-jobs/{a['job_id']}/image")
        rb = c.get(f"/v1/image-jobs/{b['job_id']}/image")
        assert ra.content == rb.content


def test_image_job_index_errors(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        st = _run_image_job(c, {"model": "fake-image", "prompt": "x"})
        assert st["status"] == "succeeded"
        assert c.get(f"/v1/image-jobs/{st['job_id']}/image?i=5").status_code == 400
        assert c.get(f"/v1/image-jobs/{st['job_id']}/image?i=-1").status_code == 400


def test_image_job_validation(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        assert c.post("/v1/image-jobs", json={"prompt": ""}).status_code == 422
        assert c.post("/v1/image-jobs", json={"prompt": "x", "count": 9}).status_code == 422
        assert (
            c.post("/v1/image-jobs", json={"prompt": "x", "width": 64}).status_code == 422
        )


def test_unavailable_image_model_fails(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        st = _run_image_job(c, {"model": "flux-schnell", "prompt": "x"})
        assert st["status"] == "failed"
        assert "no image adapter" in st["error"]


def test_fake_image_hidden_without_flag(tmp_path):
    """fake adapters are only listed under MESH_FAKE."""
    from fastapi.testclient import TestClient

    cfg = ServerConfig(fake=False, job_dir=tmp_path / "jobs")
    with TestClient(create_app(cfg)) as c:
        ids = [m["id"] for m in c.get("/v1/models").json()["models"]]
        fake_models = [m for m in c.get("/v1/models").json()["models"] if m["id"].startswith("fake")]
        assert "fake-image" in ids  # registered but ...
        assert all(m["available"] is False for m in fake_models)  # ... unavailable


def test_mesh_and_image_jobs_share_queue(app):
    """An image job and a mesh job serialize on the same worker thread."""
    import json as _json

    from fastapi.testclient import TestClient

    def _png() -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, "PNG")
        return buf.getvalue()

    with TestClient(app) as c:
        r = c.post("/v1/image-jobs", json={"model": "fake-image", "prompt": "a"})
        img_job = r.json()["job_id"]
        files = [
            ("json", ("p.json", _json.dumps({"model": "fake", "seed": 1}), "application/json")),
            ("images", ("a.png", _png(), "image/png")),
        ]
        r = c.post("/v1/jobs", files=files)
        mesh_job = r.json()["job_id"]

        results = {}
        for _ in range(300):
            m = c.get(f"/v1/jobs/{mesh_job}").json()
            i = c.get(f"/v1/image-jobs/{img_job}").json()
            results["mesh"] = m["status"]
            results["image"] = i["status"]
            if m["status"] in ("succeeded", "failed") and i["status"] in (
                "succeeded",
                "failed",
            ):
                break
            time.sleep(0.1)
        assert results["mesh"] == "succeeded"
        assert results["image"] == "succeeded"


def test_image_edit_job_roundtrip(app, tmp_path):
    """POST /v1/image-edit-jobs (multipart json+image) → fake adapter echoes
    the init image back as the result."""
    import json as _json

    from fastapi.testclient import TestClient
    from PIL import Image

    src = tmp_path / "init.png"
    img = Image.new("RGB", (40, 40), (10, 200, 30))
    img.save(src)

    with TestClient(app) as c:
        r = c.post(
            "/v1/image-edit-jobs",
            files={
                "json": (
                    "p.json",
                    _json.dumps({"prompt": "add grip", "seed": 3}),
                    "application/json",
                ),
                "image": ("init.png", src.read_bytes(), "image/png"),
            },
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        st = {}
        for _ in range(200):
            st = c.get(f"/v1/image-jobs/{job_id}").json()
            if st["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert st["status"] == "succeeded", st
        dl = c.get(f"/v1/image-jobs/{job_id}/image")
        assert dl.status_code == 200
        out = Image.open(io.BytesIO(dl.content))
        assert out.size == (40, 40)


def test_select_image_need_edit(tmp_path):
    """Edit jobs only match supports_edit adapters; txt2img jobs never pick
    a requires_image adapter."""
    from meshserver.adapters.fake_image import FakeImageAdapter
    from meshserver.registry import Registry

    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs")
    reg = Registry(cfg)

    class TxtOnly(FakeImageAdapter):
        id = "txt-only"
        supports_edit = False

    class EditOnly(FakeImageAdapter):
        id = "edit-only"
        supports_edit = True
        requires_image = True

    reg._classes = [TxtOnly, EditOnly]
    assert reg.select_image("auto").id == "txt-only"  # edit-only hidden
    assert reg.select_image("auto", need_edit=True).id == "edit-only"
    assert reg.select_image("txt-only", need_edit=True) is None
    assert reg.select_image("edit-only", need_edit=True).id == "edit-only"
