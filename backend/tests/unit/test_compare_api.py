"""GET /versions/{vid}/compare/{other_id} — version geometry + params diff."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _export_dir(pid: str, vid: str) -> Path:
    from cadx import storage

    out = storage.export_dir(pid, vid)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _export_step(pid: str, vid: str, shape: Any) -> Path:
    from build123d import export_step

    step = _export_dir(pid, vid) / "model.step"
    export_step(shape, str(step))
    return step


async def _new_version(client: AsyncClient, pid: str, prompt: str) -> str:
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": prompt})
    assert r.status_code == 200, r.text
    return r.json()["version_id"]


async def test_compare_two_versions(client):
    """STEP exports diff: offset boxes, added == removed == 500 mm³."""
    from build123d import Box, Pos

    pid = (await client.post("/api/projects", json={"name": "cmp"})).json()["id"]
    va = await _new_version(client, pid, "box a")
    vb = await _new_version(client, pid, "box b")
    _export_step(pid, va, Box(10, 10, 10))
    _export_step(pid, vb, Pos(5, 0, 0) * Box(10, 10, 10))

    r = await client.get(f"/api/versions/{va}/compare/{vb}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a"] == va and body["b"] == vb
    assert body["ok"] is True
    assert body["error"] is None
    assert body["params_diff"] == []
    shape = body["shape"]
    assert shape["boolean"] == "ok"
    assert shape["volume_a"] == pytest.approx(1000.0)
    assert shape["volume_b"] == pytest.approx(1000.0)
    assert shape["added_volume"] == pytest.approx(500.0)
    assert shape["removed_volume"] == pytest.approx(500.0)
    assert "+500" in body["diff_summary"] and "added" in body["diff_summary"]


async def test_compare_params_diff(client):
    """Union of both params dicts, differing keys only, absent -> null,
    sorted by key."""
    from build123d import Box

    from cadx.db.models import Version
    from cadx.db.session import session_scope

    pid = (await client.post("/api/projects", json={"name": "cmp-params"})).json()["id"]
    va = await _new_version(client, pid, "a")
    vb = await _new_version(client, pid, "b")
    _export_step(pid, va, Box(10, 10, 10))
    _export_step(pid, vb, Box(10, 10, 10))

    async with session_scope() as db:
        v_a = await db.get(Version, uuid.UUID(va))
        v_b = await db.get(Version, uuid.UUID(vb))
        assert v_a is not None and v_b is not None
        v_a.params = {"w": 10, "same": 1, "gone": "x"}
        v_b.params = {"w": 20, "same": 1, "new": True}
        await db.commit()

    r = await client.get(f"/api/versions/{va}/compare/{vb}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["params_diff"] == [
        {"key": "gone", "a": "x", "b": None},
        {"key": "new", "a": None, "b": True},
        {"key": "w", "a": 10, "b": 20},
    ]
    assert "no material change" in body["diff_summary"]


async def test_compare_mesh_fallback(client):
    """No model.step -> best STL is sewn into a solid for the diff."""
    import trimesh
    from build123d import Box

    pid = (await client.post("/api/projects", json={"name": "cmp-mesh"})).json()["id"]
    va = await _new_version(client, pid, "a")
    vb = await _new_version(client, pid, "b")
    _export_step(pid, va, Box(10, 10, 10))
    trimesh.creation.box(extents=(20, 10, 10)).export(str(_export_dir(pid, vb) / "model.stl"))

    r = await client.get(f"/api/versions/{va}/compare/{vb}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["shape"]["volume_a"] == pytest.approx(1000.0)
    assert body["shape"]["volume_b"] == pytest.approx(2000.0)


async def test_compare_404_and_409(client):
    from build123d import Box

    pid = (await client.post("/api/projects", json={"name": "cmp-err"})).json()["id"]
    va = await _new_version(client, pid, "a")
    vb = await _new_version(client, pid, "b")  # no geometry on disk
    _export_step(pid, va, Box(5, 5, 5))

    z = uuid.uuid4()
    assert (await client.get(f"/api/versions/{z}/compare/{va}")).status_code == 404
    assert (await client.get(f"/api/versions/{va}/compare/{z}")).status_code == 404
    assert (await client.get(f"/api/versions/{va}/compare/{vb}")).status_code == 409
    assert (await client.get(f"/api/versions/{vb}/compare/{va}")).status_code == 409


async def test_compare_unreadable_geometry_ok_false(client):
    """A corrupt STEP artifact degrades to ok=False + error, never a 500."""
    from build123d import Box

    pid = (await client.post("/api/projects", json={"name": "cmp-bad"})).json()["id"]
    va = await _new_version(client, pid, "a")
    vb = await _new_version(client, pid, "b")
    _export_step(pid, va, Box(5, 5, 5))
    (_export_dir(pid, vb) / "model.step").write_bytes(b"not a step file")

    r = await client.get(f"/api/versions/{va}/compare/{vb}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["error"]
    # shape is the raw shape_compare() dict — an import that doesn't raise
    # surfaces as {"error": ...} inside it; a hard load failure gives None
    assert body["shape"] is None or "error" in body["shape"]
