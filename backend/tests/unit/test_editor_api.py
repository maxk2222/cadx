"""API tests for the visual mesh editor + structural editor endpoints."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.session import create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _succeeded_version_with_stl(client, name: str = "editor") -> tuple[str, str]:
    """Create a project + version marked succeeded with an exports/model.stl."""
    from cadx import storage
    from cadx.db.models import Version, VersionStatus
    from cadx.db.session import session_scope

    pid = (await client.post("/api/projects", json={"name": name})).json()["id"]
    vid = (await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a block"})).json()[
        "version_id"
    ]
    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(vid))
        assert v is not None
        v.status = VersionStatus.succeeded
        await db.commit()
    out = storage.version_dir(pid, vid) / "exports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.stl").write_bytes(b"solid fake\nendsolid fake\n")
    return pid, vid


def _case() -> dict:
    return {
        "name": "c1",
        "fixed": [{"kind": "points", "points": [[0, 0, 0]], "radius_mm": 4}],
        "loads": [
            {
                "region": {"kind": "points", "points": [[10, 0, 5]], "radius_mm": 4},
                "force_n": [0, 0, -50],
            }
        ],
    }


async def test_mesh_preview_validation(client):
    _, vid = await _succeeded_version_with_stl(client, "mp-val")

    # unknown op → 422 listing valid ops
    r = await client.post(f"/api/versions/{vid}/mesh-preview", json={"ops": [{"op": "explode"}]})
    assert r.status_code == 422
    assert "explode" in r.json()["detail"]

    # version without any mesh artifact → 409
    pid2 = (await client.post("/api/projects", json={"name": "mp-nomesh"})).json()["id"]
    vid2 = (await client.post(f"/api/projects/{pid2}/generate", json={"prompt": "x"})).json()[
        "version_id"
    ]
    from cadx.db.models import Version, VersionStatus
    from cadx.db.session import session_scope

    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(vid2))
        assert v is not None
        v.status = VersionStatus.succeeded
        await db.commit()
    r = await client.post(f"/api/versions/{vid2}/mesh-preview", json={"ops": []})
    assert r.status_code == 409

    assert (
        await client.post(f"/api/versions/{uuid.uuid4()}/mesh-preview", json={})
    ).status_code == 404


async def test_mesh_preview_ok_and_scratch_file(client, monkeypatch):
    from cadx import storage
    from cadx.api.routers import editor

    pid, vid = await _succeeded_version_with_stl(client, "mp-ok")

    async def fake_run_cad(func: str, payload: dict, timeout_s: float) -> dict:
        assert func == "process_mesh" and payload["mode"] == "ops_preview"
        sdir = storage.ensure(storage.scratch_dir(pid, vid))
        (sdir / "preview_test.glb").write_bytes(b"glTFfake")
        return {
            "ok": True,
            "glb": storage.relpath(sdir / "preview_test.glb"),
            "stats": {"faces": 12, "watertight": True, "bbox_size": [10, 10, 10]},
        }

    monkeypatch.setattr(editor, "_run_cad", fake_run_cad)
    r = await client.post(
        f"/api/versions/{vid}/mesh-preview",
        json={"ops": [{"op": "rotate", "args": {"axis": "z", "deg": 45}}]},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["stats"]["faces"] == 12
    assert d["glb_url"] == f"/api/versions/{vid}/scratch/preview_test.glb"

    # the scratch route serves whitelisted files only
    r = await client.get(d["glb_url"])
    assert r.status_code == 200 and r.content == b"glTFfake"
    assert r.headers["cache-control"] == "no-store"

    sdir = storage.scratch_dir(pid, vid)
    (sdir / "evil.txt").write_text("nope")
    assert (await client.get(f"/api/versions/{vid}/scratch/evil.txt")).status_code == 403
    assert (await client.get(f"/api/versions/{vid}/scratch/missing.glb")).status_code == 404
    assert (await client.get(f"/api/versions/{vid}/scratch/../model.stl")).status_code in (404, 422)


async def test_mesh_edit_creates_child(client):
    _, vid = await _succeeded_version_with_stl(client, "me-child")

    # running versions reject edits
    pid2 = (await client.post("/api/projects", json={"name": "me-run"})).json()["id"]
    vid2 = (await client.post(f"/api/projects/{pid2}/generate", json={"prompt": "x"})).json()[
        "version_id"
    ]
    assert (
        await client.post(f"/api/versions/{vid2}/mesh-edit", json={"ops": []})
    ).status_code == 409

    r = await client.post(
        f"/api/versions/{vid}/mesh-edit",
        json={
            "ops": [
                {"op": "rotate", "args": {"axis": "z", "deg": 90}},
                {"op": "decimate", "args": {"faces": 1000}},
            ],
            "note": "gui edit",
        },
    )
    assert r.status_code == 200, r.text
    child = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert child["kind"] == "mesh_edit"
    assert child["parent_id"] == vid
    job = (await client.get(f"/api/jobs/{r.json()['job_id']}")).json()
    assert job["type"] == "mesh_edit"


async def test_fem_preview_and_check(client, monkeypatch):
    from cadx.api.routers import editor

    _, vid = await _succeeded_version_with_stl(client, "fem-api")

    # needs at least one load case
    r = await client.post(f"/api/versions/{vid}/fem-preview", json={})
    assert r.status_code == 422
    r = await client.post(f"/api/versions/{vid}/fem", json={})
    assert r.status_code == 422

    async def fake_run_cad(func: str, payload: dict, timeout_s: float) -> dict:
        assert func == "fem_analyze" and payload["preview_only"] is True
        assert payload["load_cases"][0]["fixed"][0]["kind"] == "points"
        return {
            "ok": True,
            "preview": {
                "pitch_mm": 1.0,
                "dims": [10, 10, 10],
                "solid_elements": 800,
                "cases": [
                    {
                        "case": "c1",
                        "fixed_nodes": 12,
                        "loaded_nodes": 8,
                        "total_force_n": 50.0,
                        "warnings": [],
                    }
                ],
                "regions_png": ["projects/x/_scratch/fem/regions_iso.png"],
            },
        }

    monkeypatch.setattr(editor, "_run_cad", fake_run_cad)
    r = await client.post(
        f"/api/versions/{vid}/fem-preview",
        json={"load_cases": [_case()], "keep_regions": [{"kind": "all"}]},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] and d["solid_elements"] == 800
    assert d["cases"][0]["fixed_nodes"] == 12
    assert d["pngs"] and d["pngs"][0].startswith(f"/api/versions/{vid}/scratch/")

    # FEM check creates a fem-type job on the SAME version (enqueue may fail
    # without redis — the job row is still created)
    r = await client.post(
        f"/api/versions/{vid}/fem",
        json={"load_cases": [_case()], "material": "pla"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["version_id"] == vid
    job = (await client.get(f"/api/jobs/{r.json()['job_id']}")).json()
    assert job["type"] == "fem"


async def test_editor_state_roundtrip(client):
    _, vid = await _succeeded_version_with_stl(client, "editor-state")

    payload = {
        "ui_editor": {
            "ops": [{"op": "rotate", "args": {"axis": "z", "deg": 30}}],
            "load_cases": [_case()],
            "material": "petg",
            "volfrac": 0.45,
        }
    }
    r = await client.patch(f"/api/versions/{vid}/editor-state", json=payload)
    assert r.status_code == 200

    v = (await client.get(f"/api/versions/{vid}")).json()
    assert v["options"]["ui_editor"]["ops"][0]["op"] == "rotate"
    assert v["options"]["ui_editor"]["load_cases"][0]["fixed"][0]["kind"] == "points"

    # unknown fields are rejected (forbid extra)
    r = await client.patch(f"/api/versions/{vid}/editor-state", json={"bogus": 1})
    assert r.status_code == 422
