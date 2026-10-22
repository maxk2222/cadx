"""New API surface for evalkit: POST /projects/{id}/meshgen + GET /jobs/{id}/trace."""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from cadx.db.models import AgentTrace
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


async def test_meshgen_route_enqueues_version_and_job(client):
    pid = (await client.post("/api/projects", json={"name": "meshgen-eval"})).json()["id"]
    assets = (
        await client.post(
            f"/api/projects/{pid}/assets",
            files=[("files", ("ref.png", _png(), "image/png"))],
        )
    ).json()
    aid = assets[0]["id"]

    # missing asset -> 422
    r = await client.post(
        f"/api/projects/{pid}/meshgen",
        json={"asset_ids": [str(uuid.uuid4())]},
    )
    assert r.status_code == 422
    # empty asset_ids -> validation error
    r = await client.post(f"/api/projects/{pid}/meshgen", json={"asset_ids": []})
    assert r.status_code == 422
    # unknown project -> 404
    r = await client.post(f"/api/projects/{uuid.uuid4()}/meshgen", json={"asset_ids": [aid]})
    assert r.status_code == 404
    # happy path: version+job created (enqueue fails w/o redis — same as generate)
    r = await client.post(
        f"/api/projects/{pid}/meshgen",
        json={"asset_ids": [aid], "subject": "a gear", "seed": 7},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_id"] and body["job_id"]
    v = (await client.get(f"/api/versions/{body['version_id']}")).json()
    assert v["kind"] == "meshgen"
    j = (await client.get(f"/api/jobs/{body['job_id']}")).json()
    assert j["type"] == "meshgen"


async def test_job_trace_endpoint(client):
    pid = (await client.post("/api/projects", json={"name": "trace-eval"})).json()["id"]
    r = await client.post(f"/api/projects/{pid}/generate", json={"prompt": "a cube"})
    jid = r.json()["job_id"]

    r = await client.get(f"/api/jobs/{jid}/trace")
    assert r.status_code == 200 and r.json() == []

    async with async_session() as db:
        db.add(
            AgentTrace(
                job_id=uuid.UUID(jid),
                step="codegen",
                iteration=1,
                model="m",
                messages=[{"role": "user", "content": "secret-ish"}],
                response="code...",
                parsed={"thinking": True},
                prompt_tokens=11,
                completion_tokens=7,
                latency_ms=42,
            )
        )
        db.add(
            AgentTrace(
                job_id=uuid.UUID(jid),
                step="critique",
                iteration=1,
                model="m",
                prompt_tokens=5,
                completion_tokens=2,
                latency_ms=10,
            )
        )
        await db.commit()

    rows = (await client.get(f"/api/jobs/{jid}/trace")).json()
    assert len(rows) == 2
    assert rows[0]["step"] == "codegen"
    assert rows[0]["prompt_tokens"] == 11
    assert rows[0]["parsed"] == {"thinking": True}
    assert "messages" not in rows[0] and "response" not in rows[0]  # slim by default
    full = (await client.get(f"/api/jobs/{jid}/trace?full=1")).json()
    assert full[0]["messages"][0]["role"] == "user"
    assert full[0]["response"] == "code..."
    # unknown job -> 404
    assert (await client.get(f"/api/jobs/{uuid.uuid4()}/trace")).status_code == 404


async def test_meshgen_rejects_non_image_asset(client):
    from cadx.db.models import Asset, AssetKind

    pid = (await client.post("/api/projects", json={"name": "meshgen-bad"})).json()["id"]
    async with async_session() as db:
        a = Asset(
            project_id=uuid.UUID(pid),
            kind=AssetKind.reference_step,
            filename="x.step",
            path="projects/x/x.step",
            mime="application/step",
        )
        db.add(a)
        await db.commit()
        aid = str(a.id)
    r = await client.post(f"/api/projects/{pid}/meshgen", json={"asset_ids": [aid]})
    assert r.status_code == 422 and "not images" in r.text
