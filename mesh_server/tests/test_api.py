"""API contract tests using the fake adapter (no GPU needed)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

os.environ["MESH_FAKE"] = "1"

from meshserver.app import create_app
from meshserver.config import ServerConfig


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def app(tmp_path):
    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", bg_removal="never")
    return create_app(cfg)


def test_full_job_flow(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.get("/v1/health")
        assert r.status_code == 200
        h = r.json()
        assert h["ok"] is True
        assert any(m["id"] == "fake" for m in h["models"])

        files = [
            ("json", ("params.json", json.dumps({"model": "fake", "seed": 7}), "application/json")),
            ("images", ("a.png", _png(), "image/png")),
        ]
        r = c.post("/v1/jobs", files=files)
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        # poll until done
        import time

        status = ""
        for _ in range(50):
            r = c.get(f"/v1/jobs/{job_id}")
            status = r.json()["status"]
            if status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert status == "succeeded", r.json()
        assert r.json()["triangles"] > 0

        r = c.get(f"/v1/jobs/{job_id}/mesh")
        assert r.status_code == 200
        assert r.content[:4] == b"glTF"

        r = c.delete(f"/v1/jobs/{job_id}")
        assert r.status_code == 200
        r = c.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 404


def test_stalled_running_job_marked_failed(tmp_path):
    """A 'running' job whose progress hasn't moved past job_stall_s is
    reported as failed (frozen MPS/MLX adapter) instead of polling forever."""
    import time

    from meshserver.jobs import JobManager, MeshJob
    from meshserver.registry import Registry
    from meshserver.schemas import GenParams

    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", job_stall_s=5)
    mgr = JobManager(cfg, Registry(cfg))
    job = MeshJob("j-stall", GenParams(model="fake"), [])
    job.status = "running"
    job.last_progress_at = time.monotonic() - 10  # stalled 10s > 5s budget
    mgr.jobs[job.id] = job

    got = mgr.get(job.id)
    assert got is not None
    assert got.status == "failed"
    assert "stalled" in (got.error or "")
    assert got.cancelled is True  # adapter thread should see the cancel flag


def test_fresh_running_job_not_stalled(tmp_path):
    import time

    from meshserver.jobs import JobManager, MeshJob
    from meshserver.registry import Registry
    from meshserver.schemas import GenParams

    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", job_stall_s=60)
    mgr = JobManager(cfg, Registry(cfg))
    job = MeshJob("j-fresh", GenParams(model="fake"), [])
    job.status = "running"
    job.last_progress_at = time.monotonic()
    mgr.jobs[job.id] = job

    assert mgr.get(job.id).status == "running"


def test_no_image_rejected(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        files = [
            ("json", ("p.json", json.dumps({"model": "fake"}), "application/json")),
        ]
        r = c.post("/v1/jobs", files=files)
        assert r.status_code == 422


def test_unavailable_model_fails_job(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        files = [
            ("json", ("p.json", json.dumps({"model": "trellis2", "seed": 1}), "application/json")),
            ("images", ("a.png", _png(), "image/png")),
        ]
        r = c.post("/v1/jobs", files=files)
        job_id = r.json()["job_id"]
        import time

        status = ""
        for _ in range(50):
            r = c.get(f"/v1/jobs/{job_id}")
            status = r.json()["status"]
            if status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert status == "failed"
        assert "no adapter" in r.json()["error"]
