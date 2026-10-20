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

from cadx import storage
from cadx.agent.orchestrator import RunOutcome
from cadx.db.models import Artifact, Job, JobType, Project, Version, VersionKind, VersionStatus
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


async def test_topopt_outcome_with_committed_injob_refine_skips_spawn(monkeypatch):
    """Dedup: the agent optimize session already committed a `refine_features`
    run (agent_runs entry with params.refine_of marked chosen) — the
    standalone refine child must NOT be spawned on top of it."""
    from cadx.agent import worker_jobs as wj

    outcome = _topopt_outcome(with_hole=True)
    outcome.mesh["topopt"]["agent_runs"] = [
        {"index": 0, "chosen": False, "params": {"target_volume_fraction": 0.4}},
        {"index": 1, "chosen": True, "params": {"refine_of": 0, "features": "auto"}},
    ]
    enqueued = _patch(monkeypatch, outcome)
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}
    assert not [e for e in enqueued if e["func"] == "run_refine_job"]
    assert await _refine_children(version_id) == []
    async with async_session() as db:
        job = await db.get(Job, job_id)
        assert "refine already applied in generation job" in (job.log or "")


async def test_topopt_outcome_with_unchosen_refine_run_still_spawns(monkeypatch):
    """A refine run the agent did NOT choose leaves the committed result
    unrefined — the standalone child still spawns."""
    from cadx.agent import worker_jobs as wj

    outcome = _topopt_outcome(with_hole=True)
    outcome.mesh["topopt"]["agent_runs"] = [
        {"index": 0, "chosen": True, "params": {"target_volume_fraction": 0.4}},
        {"index": 1, "chosen": False, "params": {"refine_of": 0, "features": "auto"}},
    ]
    enqueued = _patch(monkeypatch, outcome)
    job_id, version_id = await _mk_agent_version()

    result = await wj.run_agent_job({}, {"job_id": str(job_id), "version_id": str(version_id)})
    assert result == {"ok": True, "status": "succeeded"}
    assert [e for e in enqueued if e["func"] == "run_refine_job"]


async def test_maybe_spawn_refine_skips_committed_injob_refine(monkeypatch):
    """Same dedup on the standalone-optimize path via _maybe_spawn_refine:
    persisted agent_runs on the parent version block the child."""
    import cadx.api.deps as deps
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "x"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    async with async_session() as db:
        p = Project(name="injob-refine-dedup")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            mesh={
                "topopt": {
                    "anchors": list(_REFINE_ANCHORS),
                    "agent_runs": [
                        {"index": 0, "chosen": False, "params": {}},
                        {"index": 1, "chosen": True, "params": {"refine_of": 0}},
                    ],
                }
            },
        )
        db.add(parent)
        await db.commit()

        ok = await wj._maybe_spawn_refine(
            db,
            parent,
            spec=None,
            optimize={"auto_refine_features": True},
            load_cases=[],
            material="",
            runtime=RuntimeSettings(),
        )
        assert ok is False
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


# --------------------------------------------------------------------------- #
# run_refine_job: version-level exports + post-refine tunnel verification       #
# --------------------------------------------------------------------------- #

_REFINE_ANCHORS = [
    {
        "index": 1,
        "kind": "hole",
        "axis": "+z",
        "x": 5.0,
        "y": 0.0,
        "dia": 4.0,
        "pos": [5.0, 0.0, 5.0],
    }
]
_REFINE_APPLIED = [
    {
        "name": "anchor 1",
        "kind": "hole",
        "axis": "+z",
        "dia": 4.0,
        "depth_mm": None,
        "source": "anchor",
    }
]


class _FakeRefineExec:
    """QueueExecutor stand-in: writes the iteration artifact set and returns
    a canned refine result (no real executor / LLM)."""

    def __init__(self, refine_report: dict) -> None:
        self.report = refine_report
        self.payload: dict | None = None

    async def refine(self, payload: dict) -> dict:
        self.payload = payload
        it = storage.iteration_dir(
            uuid.UUID(payload["project_id"]), uuid.UUID(payload["version_id"]), 1
        )
        run = storage.ensure(it / "_run")
        (run / "model.stl").write_text("solid m\nendsolid m\n")
        exp = storage.ensure(it / "exports")
        for fmt in ("stl", "step", "3mf", "glb"):
            (exp / f"model.{fmt}").write_text(fmt)
        it_rel = storage.relpath(it)
        return {
            "ok": True,
            "exports": {f: f"{it_rel}/exports/model.{f}" for f in ("stl", "step", "3mf", "glb")},
            "features": {"refine": self.report},
            "fem": {"cases": [{"safety_factor": 2.0}]},
            "mesh": {"pipeline": "refine", "triangles": 12},
            "analysis": None,
            "renders": {},
        }


async def _mk_refine_child(
    *, anchors: list[dict] | None = None
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """topopt parent (with a real exports/model.stl for _parent_stl) + refine
    child version + refine job. Returns (project_id, child_id, job_id)."""
    async with async_session() as db:
        p = Project(name="refine-job")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="bracket",
            status=VersionStatus.succeeded,
            gen_mode="topopt",
            resolved_mode="topopt",
            mesh={"topopt": {"anchors": anchors or []}},
        )
        db.add(parent)
        await db.flush()
        exp = storage.ensure(storage.version_dir(p.id, parent.id) / "exports")
        (exp / "model.stl").write_text("solid p\nendsolid p\n")
        child = Version(
            project_id=p.id,
            parent_id=parent.id,
            number=2,
            kind=VersionKind.refine,
            prompt="refine bracket",
        )
        db.add(child)
        await db.flush()
        job = Job(version_id=child.id, type=JobType.refine)
        db.add(job)
        await db.commit()
        return p.id, child.id, job.id


def _patch_refine(monkeypatch, report: dict, scanned: list[dict]) -> _FakeRefineExec:
    from cadx.agent import worker_jobs as wj

    fake = _FakeRefineExec(report)
    monkeypatch.setattr(wj, "QueueExecutor", lambda *a, **k: fake)
    monkeypatch.setattr("cadx.cad.fem.anchors3d.scan_mesh_anchors", lambda _path: scanned)
    return fake


async def test_refine_job_promotes_version_level_exports(monkeypatch):
    """Child refine writes iterations/1/exports — the job must promote them to
    versions/<vid>/exports/ so API/UI + best_mesh_relpath serve the REFINED
    files instead of the parent's unrefined output."""
    from cadx.agent import worker_jobs as wj

    report = {
        "applied": list(_REFINE_APPLIED),
        "skipped": [],
        "restored_regions": 0,
        "warnings": [],
    }
    fake = _patch_refine(
        monkeypatch,
        report,
        [{"axis": "+z", "kind": "hole", "x": 5.1, "y": 0.0, "dia": 4.1}],
    )
    pid, vid, jid = await _mk_refine_child(anchors=_REFINE_ANCHORS)

    res = await wj.run_refine_job(
        {},
        {"job_id": str(jid), "version_id": str(vid), "auto_keep_anchors": True},
    )
    assert res["ok"] is True
    assert fake.payload is not None
    assert fake.payload["auto_keep_anchors"] is True

    v_dir = storage.version_dir(pid, vid)
    for fmt in ("stl", "step", "3mf", "glb"):
        assert (v_dir / "exports" / f"model.{fmt}").exists()

    async with async_session() as db:
        v = await db.get(Version, vid)
        assert v.status == VersionStatus.succeeded
        arts = (
            (await db.execute(select(Artifact).where(Artifact.version_id == vid))).scalars().all()
        )
        exp_paths = {a.path for a in arts if a.kind.value in {"step", "stl", "3mf", "glb"}}
        assert exp_paths
        assert all(f"versions/{vid}/exports/" in p and "iterations" not in p for p in exp_paths)


async def test_refine_job_missing_tunnel_needs_review(monkeypatch):
    """Applied cut landed in a SIMP void — no tunnel on rescan: major issue +
    needs_review (never a silent success)."""
    from cadx.agent import worker_jobs as wj

    report = {
        "applied": list(_REFINE_APPLIED),
        "skipped": [],
        "restored_regions": 0,
        "warnings": [
            "feature 'anchor 1': measured dia 43.00 mm deviates 39.00 mm from requested 4.00 mm"
        ],
    }
    _patch_refine(monkeypatch, report, scanned=[])
    _pid, vid, jid = await _mk_refine_child(anchors=_REFINE_ANCHORS)

    res = await wj.run_refine_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True

    async with async_session() as db:
        v = await db.get(Version, vid)
        assert v.status == VersionStatus.needs_review
        severities = {(i["source"], i["severity"]) for i in (v.issues or [])}
        assert ("refine_verify", "major") in severities
        # the refine report warning is surfaced as a minor issue
        assert ("refine", "minor") in severities


async def test_refine_job_wrong_dia_tunnel_needs_review(monkeypatch):
    """A tunnel at the right spot but >25% off diameter counts as missing —
    the 'giant void' failure mode."""
    from cadx.agent import worker_jobs as wj

    report = {
        "applied": list(_REFINE_APPLIED),
        "skipped": [],
        "restored_regions": 0,
        "warnings": [],
    }
    _patch_refine(
        monkeypatch,
        report,
        [{"axis": "+z", "kind": "hole", "x": 5.0, "y": 0.0, "dia": 43.0}],
    )
    _pid, vid, jid = await _mk_refine_child(anchors=_REFINE_ANCHORS)

    res = await wj.run_refine_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True

    async with async_session() as db:
        v = await db.get(Version, vid)
        assert v.status == VersionStatus.needs_review
        assert any(i["severity"] == "major" for i in (v.issues or []))


async def test_refine_job_all_tunnels_present_succeeds(monkeypatch):
    """Every applied feature has a matching tunnel -> clean success, warnings
    still surface as minor issues."""
    from cadx.agent import worker_jobs as wj

    report = {
        "applied": list(_REFINE_APPLIED),
        "skipped": [],
        "restored_regions": 1,
        "warnings": ["some soft warning"],
    }
    _patch_refine(
        monkeypatch,
        report,
        [{"axis": "+z", "kind": "hole", "x": 5.0, "y": 0.0, "dia": 4.4}],
    )
    _pid, vid, jid = await _mk_refine_child(anchors=_REFINE_ANCHORS)

    res = await wj.run_refine_job({}, {"job_id": str(jid), "version_id": str(vid)})
    assert res["ok"] is True

    async with async_session() as db:
        v = await db.get(Version, vid)
        assert v.status == VersionStatus.succeeded
        assert all(i["severity"] != "major" for i in (v.issues or []))
        assert any(i["source"] == "refine" for i in (v.issues or []))


async def test_maybe_spawn_refine_auto_keep_anchors_flag(monkeypatch):
    """Empty keep_regions + measured topopt anchors -> auto_keep_anchors=True
    reaches the refine task payload; explicit keep_regions suppress it."""
    import cadx.api.deps as deps
    from cadx.agent import worker_jobs as wj

    enqueued: list[dict] = []

    async def fake_enqueue(func, payload):
        enqueued.append({"func": func, "payload": payload})
        return "x"

    monkeypatch.setattr(deps, "enqueue", fake_enqueue)

    async with async_session() as db:
        p = Project(name="auto-keep")
        db.add(p)
        await db.flush()
        parent = Version(
            project_id=p.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            mesh={"topopt": {"anchors": list(_REFINE_ANCHORS)}},
        )
        db.add(parent)
        await db.commit()

        ok = await wj._maybe_spawn_refine(
            db,
            parent,
            spec=None,
            optimize={"auto_refine_features": True, "keep_regions": []},
            load_cases=[],
            material="",
            runtime=RuntimeSettings(),
        )
        assert ok is True
    assert enqueued[0]["payload"]["auto_keep_anchors"] is True

    enqueued.clear()
    async with async_session() as db:
        p2 = Project(name="auto-keep-explicit")
        db.add(p2)
        await db.flush()
        parent2 = Version(
            project_id=p2.id,
            number=1,
            kind=VersionKind.generate,
            prompt="block",
            status=VersionStatus.succeeded,
            mesh={"topopt": {"anchors": list(_REFINE_ANCHORS)}},
        )
        db.add(parent2)
        await db.commit()

        ok = await wj._maybe_spawn_refine(
            db,
            parent2,
            spec=None,
            optimize={
                "auto_refine_features": True,
                "keep_regions": [{"kind": "all"}],
            },
            load_cases=[],
            material="",
            runtime=RuntimeSettings(),
        )
        assert ok is True
    assert enqueued[0]["payload"]["auto_keep_anchors"] is False


def test_refine_payload_forwards_auto_keep_anchors(monkeypatch, tmp_path):
    """run_refine_payload maps the payload flag onto run_refine's kwarg."""
    import cadx.cad.tasks as tasks
    from cadx.cad.engine import ExecResult

    captured: dict = {}

    def fake_run_refine(**kw):
        captured.update(kw)
        r = ExecResult()
        r.ok = True
        return r

    monkeypatch.setattr("cadx.cad.engine.run_refine", fake_run_refine)
    mesh = storage.tmp_dir() / "in.stl"
    mesh.write_text("solid m\nendsolid m\n")
    d = tasks.run_refine_payload(
        {
            "project_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "iteration": 1,
            "mesh_rel": storage.relpath(mesh),
            "auto_keep_anchors": True,
        }
    )
    assert d["ok"] is True
    assert captured["auto_keep_anchors"] is True


def test_refine_payload_forwards_spec(monkeypatch, tmp_path):
    """The serialized PartSpec must reach run_refine so its canonical
    verify_spec report receives the design intent (dims, features)."""
    import cadx.cad.tasks as tasks
    from cadx.cad.engine import ExecResult

    captured: dict = {}

    def fake_run_refine(**kw):
        captured.update(kw)
        r = ExecResult()
        r.ok = True
        return r

    monkeypatch.setattr("cadx.cad.engine.run_refine", fake_run_refine)
    mesh = storage.tmp_dir() / "in.stl"
    mesh.write_text("solid m\nendsolid m\n")
    spec = {"name": "p", "length_x": {"value": 100.0, "source": "stated"}, "features": []}
    d = tasks.run_refine_payload(
        {
            "project_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "iteration": 1,
            "mesh_rel": storage.relpath(mesh),
            "spec": spec,
        }
    )
    assert d["ok"] is True
    assert captured["spec"] == spec


def test_optimize_payload_forwards_spec(monkeypatch, tmp_path):
    """run_optimize_payload must forward the serialized PartSpec so the
    topopt pipeline can attach a spec-aware canonical verify_spec report."""
    import cadx.cad.tasks as tasks
    from cadx.cad.engine import ExecResult

    captured: dict = {}

    def fake_pipeline(**kw):
        captured.update(kw)
        r = ExecResult()
        r.ok = True
        return r

    # bound at module import time — patch the tasks-module name, not engine's
    monkeypatch.setattr(tasks, "run_topopt_pipeline", fake_pipeline)
    mesh = storage.tmp_dir() / "in.stl"
    mesh.write_text("solid m\nendsolid m\n")
    spec = {"name": "p", "length_x": {"value": 100.0, "source": "stated"}, "features": []}
    d = tasks.run_optimize_payload(
        {
            "project_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "iteration": 1,
            "mesh_rel": storage.relpath(mesh),
            "spec": spec,
        }
    )
    assert d["ok"] is True
    assert captured["spec"] == spec


def test_run_refine_synthesizes_auto_keep_regions(monkeypatch, tmp_path):
    """auto_keep_anchors + empty keep_regions -> one hole keep-region per
    fastener anchor (hole_index = 1-based list position); explicit
    keep_regions suppress synthesis."""
    import trimesh

    import cadx.cad.engine as engine
    import cadx.cad.fem.refine as refine_mod

    captured: dict = {}

    def fake_apply(mesh, feats, **kw):
        captured.update(kw)
        rep = refine_mod.RefineReport()
        rep.applied = [{"name": "a", "kind": "hole", "axis": "+z", "dia": 4.0}]
        return mesh, rep

    monkeypatch.setattr(refine_mod, "apply_refine", fake_apply)
    monkeypatch.setattr(refine_mod, "resolve_refine_features", lambda *a, **k: ([], []))
    monkeypatch.setattr(engine, "render_views", lambda *a, **k: {})
    monkeypatch.setattr(engine, "render_thumbnail", lambda *a, **k: None)

    stl = tmp_path / "in.stl"
    trimesh.creation.box(extents=[40.0, 20.0, 10.0]).export(str(stl))
    anchors = [
        {"index": 1, "kind": "hole", "axis": "+z", "x": 5.0, "y": 0.0, "dia": 4.0},
        {"index": 2, "kind": "dimple", "axis": "+z", "x": -5.0, "y": 0.0, "dia": 2.0},
    ]
    res = engine.run_refine(
        mesh_path=stl,
        out_dir=tmp_path,
        anchors=anchors,
        auto_keep_anchors=True,
        runtime=RuntimeSettings(),
    )
    assert res.ok
    # only the fastener anchor keeps material — the dimple is not a bore
    assert captured["keep_regions"] == [{"kind": "hole", "hole_index": 1, "margin_mm": 3.0}]

    captured.clear()
    res = engine.run_refine(
        mesh_path=stl,
        out_dir=tmp_path,
        anchors=anchors,
        keep_regions=[{"kind": "all"}],
        auto_keep_anchors=True,
        runtime=RuntimeSettings(),
    )
    assert res.ok
    assert captured["keep_regions"] == [{"kind": "all"}]
