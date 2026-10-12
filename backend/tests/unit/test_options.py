"""Run-options panel: persistence, settings merge, prompt injection."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.agent.context import user_block
from cadx.agent.worker_jobs import _merge_settings
from cadx.db.models import ProjectSettings, Version, VersionStatus
from cadx.db.session import create_all, session_scope


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_generate_options_persisted(client):
    r = await client.post("/api/projects", json={"name": "opts"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={
            "prompt": "bracket 60x40x6mm",
            "options": {
                "process": "cnc3",
                "material": "aluminum",
                "include_fasteners": True,
                "fem": True,
                "fem_note": "fixed at M4 holes, 100 N on top face",
            },
        },
    )
    assert r.status_code == 200, r.text
    v = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    opts = v["options"]
    assert opts["process"] == "cnc3" and opts["material"] == "aluminum"
    assert opts["fem"] is True and opts["fem_note"].startswith("fixed at M4")
    assert v["gen_mode"] == "auto"  # no topopt → mode untouched


async def test_topopt_option_forces_topopt_mode(client):
    r = await client.post("/api/projects", json={"name": "topopt"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={
            "prompt": "lightweight bracket",
            "mode": "cad",
            "options": {"topopt": True, "topopt_volfrac": 0.4},
        },
    )
    assert r.status_code == 200, r.text
    v = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert v["gen_mode"] == "topopt"
    assert v["options"]["topopt_volfrac"] == 0.4


async def test_edit_inherits_parent_options(client):
    r = await client.post("/api/projects", json={"name": "inherit"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={
            "prompt": "plate",
            "options": {"process": "cnc3", "material": "aluminum"},
        },
    )
    vid = r.json()["version_id"]
    # mark parent editable
    async with session_scope() as db:
        v = await db.get(Version, uuid.UUID(vid))
        assert v is not None
        v.status = VersionStatus.succeeded
        v.code = "result = None"
        await db.commit()
    r = await client.post(f"/api/versions/{vid}/edit", json={"prompt": "thicker"})
    assert r.status_code == 200, r.text
    child = (await client.get(f"/api/versions/{r.json()['version_id']}")).json()
    assert child["options"]["process"] == "cnc3"
    assert child["options"]["material"] == "aluminum"


async def test_invalid_options_rejected(client):
    r = await client.post("/api/projects", json={"name": "bad-opts"})
    pid = r.json()["id"]
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "options": {"topopt_volfrac": 1.5}},
    )
    assert r.status_code == 422
    r = await client.post(
        f"/api/projects/{pid}/generate",
        json={"prompt": "x", "options": {"process": "bogus"}},
    )
    assert r.status_code == 422


def test_merge_settings_override_and_fem_fallback():
    s = _merge_settings(
        {"process": "fdm", "material": "pla"},
        {"process": "cnc3", "fem": True},
    )
    assert s.process == "cnc3"
    assert s.fem is True
    assert s.fem_material == "pla"  # falls back to part material
    # explicit fem_material wins over fallback
    s = _merge_settings({}, {"fem": True, "material": "steel", "fem_material": "pa"})
    assert s.fem_material == "pa"


def test_user_block_requirements():
    s = ProjectSettings(fem=True, fem_note="fixed at holes", topopt=True, topopt_volfrac=0.4)
    text = user_block("a bracket", s, "generate")
    assert "USER-SELECTED REQUIREMENTS" in text
    assert "FEM structural analysis (fixed at holes)" in text
    assert "volume fraction 0.40" in text
    # defaults → no requirements block, prompt still injected
    base = user_block("a bracket", ProjectSettings(), "generate")
    assert "USER-SELECTED REQUIREMENTS" not in base
    assert "USER REQUEST: a bracket" in base
