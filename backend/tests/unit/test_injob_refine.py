"""In-job topopt auto-refine.

A one-shot generate->topopt run (`run_agent_job` -> resolved_mode="topopt")
must enqueue the same feature-finishing 'refine' child the standalone
optimize job spawns via `_maybe_spawn_refine` — bores come out voxel-rough
and threads destroyed otherwise. Gated by runtime `topopt_auto_refine` and
by having something to re-cut (spec features or measured topopt anchors).
"""

from __future__ import annotations

import uuid

import pytest
from sqlmodel import select

from cadx.agent.orchestrator import RunOutcome
from cadx.db.models import Job, JobType, Project, Version, VersionKind, VersionStatus
from cadx.db.session import async_session, create_all
from cadx.llm.schemas import PartSpec
from cadx.runtime_config import RuntimeSettings, save_runtime_settings


@pytest.fixture(autouse=True)
async def _runtime_defaults():
    """Pin the runtime row to code defaults — earlier test files may leave
    one seeded (e.g. a mesh_base_url that would make client probes slow)."""
    await create_all()
    async with async_session() as db:
        await save_runtime_settings(db, RuntimeSettings())
        await db.commit()


def _spec_dict(*, with_hole: bool = True) -> dict:
    return {
        "name": "bracket",
        "summary": "40x20x10 bracket",
        "units": "mm",
        "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
        "width_y": {"value": 20.0, "unit": "mm", "source": "stated"},
        "height_z": {"value": 10.0, "unit": "mm", "source": "stated"},
        "features": (
            [
                {
                    "name": "m4 hole",
                    "type": "hole",
                    "description": "M4 through-hole",
                    "dims": {"diameter": 4.0},
                    "position": "center",
                    "count": 1,
                    "source": "stated",
                }
            ]
            if with_hole
            else []
        ),
        "load_cases": [
            {
                "name": "cantilever",
                "fixed": [{"kind": "face", "axis": "-x"}],
                "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -50]}],
            }
        ],
    }


def _topopt_outcome(*, with_hole: bool = True, anchors: list | None = None) -> RunOutcome:
    return RunOutcome(
        status="succeeded",
        resolved_mode="topopt",
        spec=_spec_dict(with_hole=with_hole),
        mesh={
            "pipeline": "topopt",
            "topopt": {"anchors": anchors or [], "volfrac_achieved": 0.4},
        },
        fem={"cases": [{"safety_factor": 2.0}]},
    )


async def _mk_agent_version() -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session() as db:
        p = Project(name="injob-refine")
        db.add(p)
        await db.flush()
        v = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="lightweight bracket with M4 hole",
            gen_mode="topopt",
        )
        db.add(v)
        await db.flush()
        j = Job(version_id=v.id, type=JobType.generate)
        db.add(j)
        await db.commit()
        return j.id, v.id


async def _refine_children(parent_id: uuid.UUID) -> list[Job]:
    async with async_session() as db:
        kids = (
            (await db.execute(select(Version).where(Version.parent_id == parent_id)))
            .scalars()
            .all()
        )
        jobs: list[Job] = []
        for kid in kids:
            jobs += list(
                (
                    await db.execute(
                        select(Job).where(Job.version_id == kid.id, Job.type == JobType.refine)
                    )
                )
                .scalars()
                .all()
            )
        return jobs


def _patch(monkeypatch, outcome: RunOutcome) -> list[dict]:
    import cadx.api.deps as deps
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "fake-arq-id"

    async def fake_run_generation(actx):
        return outcome

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)
    monkeypatch.setattr(wj, "run_generation", fake_run_generation)
    return enqueued


async def test_topopt_outcome_enqueues_refine_child(monkeypatch):
    """resolved_mode='topopt' + spec hole feature -> refine child spawned."""
    from cadx.agent import worker_jobs as wj

    enqueued = _patch(monkeypatch, _topopt_outcome(with_hole=True))
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}

    assert enqueued and enqueued[0]["func"] == "run_refine_job"
    payload = enqueued[0]["payload"]
    assert payload["load_cases"] and payload["load_cases"][0]["name"] == "cantilever"
    async with async_session() as db:
        child = await db.get(Version, uuid.UUID(payload["version_id"]))
        assert child is not None and child.kind == VersionKind.refine
        assert child.parent_id == version_id
        rjob = (await db.execute(select(Job).where(Job.version_id == child.id))).scalars().one()
        assert rjob.type == JobType.refine


async def test_topopt_outcome_without_features_enqueues_nothing(monkeypatch):
    """Guard: empty spec.features AND no topopt anchors -> no useless child."""
    from cadx.agent import worker_jobs as wj

    enqueued = _patch(monkeypatch, _topopt_outcome(with_hole=False, anchors=[]))
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}
    assert not enqueued
    assert await _refine_children(version_id) == []


async def test_topopt_outcome_refine_disabled_by_runtime(monkeypatch):
    """topopt_auto_refine=False suppresses the spawn even with features."""
    from cadx.agent import worker_jobs as wj

    async with async_session() as db:
        await save_runtime_settings(db, RuntimeSettings(topopt_auto_refine=False))
        await db.commit()

    enqueued = _patch(monkeypatch, _topopt_outcome(with_hole=True))
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}
    assert not enqueued
    assert await _refine_children(version_id) == []


async def test_non_topopt_outcome_enqueues_nothing(monkeypatch):
    """CAD outcomes never reach the refine hook."""
    from cadx.agent import worker_jobs as wj

    outcome = RunOutcome(
        status="succeeded",
        resolved_mode="cad",
        spec=_spec_dict(with_hole=True),
        code="result = None\n",
    )
    enqueued = _patch(monkeypatch, outcome)
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}
    assert not enqueued


async def test_runtime_flag_alone_triggers_spawn(monkeypatch):
    """`_maybe_spawn_refine` fires on runtime topopt_auto_refine without the
    OptimizeSpec/request flag — anchors alone satisfy the guard."""
    import cadx.api.deps as deps
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "x"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    async with async_session() as db:
        p = Project(name="injob-refine-anchors")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            mesh={"topopt": {"anchors": [{"x": 1.0, "y": 2.0, "dia": 4.0}]}},
        )
        db.add(parent)
        await db.commit()

        ok = await wj._maybe_spawn_refine(
            db,
            parent,
            spec=None,
            optimize={},
            load_cases=[],
            material="",
            runtime=RuntimeSettings(),
        )
        assert ok is True
    assert enqueued and enqueued[0]["func"] == "run_refine_job"


async def test_maybe_spawn_refine_guard_blocks_featureless(monkeypatch):
    """Explicit flag set but nothing to re-cut -> blocked (info log), no child."""
    import cadx.api.deps as deps
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "x"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    spec = PartSpec(**_spec_dict(with_hole=False))
    async with async_session() as db:
        p = Project(name="injob-refine-guard")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
        )
        db.add(parent)
        await db.commit()

        ok = await wj._maybe_spawn_refine(
            db,
            parent,
            spec=spec,
            optimize={"auto_refine_features": True},
            load_cases=[],
            material="",
            runtime=RuntimeSettings(),
        )
        assert ok is False
    assert not enqueued
