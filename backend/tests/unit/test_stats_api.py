"""GET /api/stats — aggregated generation statistics for the Stats page.

The test DB is shared across modules, so assertions are structural or scoped
to uniquely-named seeds rather than exact global totals.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from cadx.db.models import (
    AgentTrace,
    Artifact,
    ArtifactKind,
    Job,
    JobStatus,
    JobType,
    Project,
    Version,
    VersionKind,
    VersionStatus,
)
from cadx.db.session import async_session, create_all


@pytest.fixture(scope="module", autouse=True)
async def _db():
    await create_all()


@pytest.fixture(scope="module")
async def client():
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_stats_shape_and_days(client):
    r = await client.get("/api/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["days"] == 0
    for k in (
        "totals",
        "versions_by_status",
        "versions_by_mode_resolved",
        "jobs_by_status",
        "job_types",
        "score",
        "activity",
        "models",
        "steps",
        "top_errors",
        "failed_steps",
        "mesh",
        "fem",
        "mode_stats",
        "issue_sources",
        "assets_by_kind",
        "artifacts_by_kind",
        "projects",
    ):
        assert k in d, k
    # activity is gapless and ends today
    assert d["activity"][-1]["date"] == datetime.now(UTC).date().isoformat()

    r = await client.get("/api/stats", params={"days": 30})
    assert r.status_code == 200
    assert r.json()["days"] == 30
    assert len(r.json()["activity"]) == 31  # cutoff day … today inclusive

    # validation bounds
    assert (await client.get("/api/stats", params={"days": -1})).status_code == 422
    assert (await client.get("/api/stats", params={"days": 99999})).status_code == 422


async def test_stats_seeded(client):
    """Seed a uniquely-named project with versions/job/trace/artifact and check
    the aggregates land in the right buckets."""
    tag = uuid.uuid4().hex[:8]
    pname = f"stats-{tag}"
    old = datetime.now(UTC) - timedelta(days=60)

    async with async_session() as db:
        p = Project(name=pname)
        db.add(p)
        await db.flush()
        v1 = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="plate",
            status=VersionStatus.succeeded,
            gen_mode="auto",
            resolved_mode="cad",
            score=85,
            iterations=6,
            usage={"prompt_tokens": 500, "completion_tokens": 100, "calls": 4},
            mesh={"generation": {"model": f"meshm-{tag}", "duration_s": 12.0, "cached": False}},
            issues=[
                {"source": "dims", "severity": "major", "what": f"dim-{tag}"},
                {"source": "lint", "severity": "minor", "what": f"lint-{tag}"},
                {"source": "lint", "severity": "minor", "what": f"lint2-{tag}"},
            ],
            fem={"cases": [{"safety_factor": 2.5}, {"safety_factor": 1.8}]},
        )
        v2 = Version(
            project_id=p.id,
            number=2,
            kind=VersionKind.edit,
            prompt="plate v2",
            status=VersionStatus.failed,
            gen_mode="mesh",
            resolved_mode="mesh",
            score=30,
            iterations=2,
            error=f"boom-{tag}: something broke",
        )
        v_old = Version(
            project_id=p.id,
            number=3,
            kind=VersionKind.generate,
            prompt="ancient",
            status=VersionStatus.succeeded,
            created_at=old,
        )
        for v in (v1, v2, v_old):
            db.add(v)
        await db.flush()
        j = Job(
            version_id=v1.id,
            type=JobType.generate,
            status=JobStatus.failed,
            step="codegen",
            started_at=datetime.now(UTC) - timedelta(minutes=10),
            finished_at=datetime.now(UTC) - timedelta(minutes=2),
            live={"step_history": [{"step": "codegen", "it": 1, "dt_s": 42.0}]},
        )
        db.add(j)
        j2 = Job(
            version_id=v1.id,
            type=JobType.optimize,
            status=JobStatus.succeeded,
            started_at=datetime.now(UTC) - timedelta(minutes=20),
            finished_at=datetime.now(UTC) - timedelta(minutes=5),
        )
        db.add(j2)
        await db.flush()
        db.add(
            AgentTrace(
                job_id=j.id,
                step="codegen",
                model=f"llm-{tag}",
                prompt_tokens=700,
                completion_tokens=200,
                latency_ms=1500,
            )
        )
        db.add(
            Artifact(
                version_id=v1.id,
                kind=ArtifactKind.stl,
                path=f"projects/{p.id}/versions/{v1.id}/out.stl",
                size=2048,
            )
        )
        await db.commit()

    d = (await client.get("/api/stats")).json()

    # project row aggregates our versions only
    pr = next(x for x in d["projects"] if x["name"] == pname)
    assert pr["versions"] == 3
    assert pr["succeeded"] == 2
    assert pr["avg_score"] == pytest.approx((85 + 30) / 2, abs=0.05)
    assert pr["tokens"] == 600

    # model + step rows exist
    m = next(x for x in d["models"] if x["model"] == f"llm-{tag}")
    assert m["calls"] == 1 and m["prompt_tokens"] == 700 and m["completion_tokens"] == 200
    assert m["avg_latency_ms"] == pytest.approx(1500)
    st = next(x for x in d["steps"] if x["step"] == "codegen")
    assert st["tokens"] >= 900 and st["wall_seconds"] >= 42.0

    # failure tracking
    assert any(f"boom-{tag}" in e["name"] for e in d["top_errors"])
    assert any(f["name"] == "codegen" for f in d["failed_steps"])
    jt = next(x for x in d["job_types"] if x["type"] == "generate")
    assert jt["avg_duration_s"] is not None and jt["avg_duration_s"] >= 60

    # mesh generation stats picked up from Version.mesh.generation
    mm = next(x for x in d["mesh"]["models"] if x["name"] == f"meshm-{tag}")
    assert mm["count"] == 1
    assert d["mesh"]["generations"] >= 1
    assert d["mesh"]["avg_duration_s"] is not None

    # artifact bytes counted
    assert d["totals"]["artifact_bytes"] >= 2048
    assert any(a["name"] == "stl" for a in d["artifacts_by_kind"])

    # score buckets contain our two scored versions
    buckets = {b["name"]: b["count"] for b in d["score"]["buckets"]}
    assert buckets["80-100"] >= 1 and buckets["<40"] >= 1

    # per-mode stats: cad has the succeeded v1, mesh has the failed v2
    modes = {m["mode"]: m for m in d["mode_stats"]}
    assert modes["cad"]["succeeded"] >= 1 and modes["cad"]["avg_score"] is not None
    assert modes["mesh"]["versions"] >= 1
    assert modes["cad"]["tokens"] >= 600

    # issue sources aggregated from Version.issues
    iss = {i["source"]: i for i in d["issue_sources"]}
    assert iss["dims"]["major"] >= 1
    assert iss["lint"]["minor"] >= 2 and iss["lint"]["total"] >= 2

    # fem stats: v1's worst case SF is min(2.5, 1.8) = 1.8
    assert d["fem"]["versions"] >= 1
    assert d["fem"]["avg_worst_sf"] is not None

    # job type percentiles + today's outcome bucket
    jt_opt = next(x for x in d["job_types"] if x["type"] == "optimize")
    assert jt_opt["p50_duration_s"] is not None and jt_opt["max_duration_s"] is not None
    today = datetime.now(UTC).date().isoformat()
    today_bucket = next(b for b in d["activity"] if b["date"] == today)
    assert today_bucket["succeeded"] >= 1 and today_bucket["failed"] >= 1
    assert today_bucket["score_avg"] is not None

    # days=7 filter: the 60-day-old version drops out, project keeps 2 versions
    d7 = (await client.get("/api/stats", params={"days": 7})).json()
    pr7 = next(x for x in d7["projects"] if x["name"] == pname)
    assert pr7["versions"] == 2
