"""use_parent_fallbacks — the evalkit driver-side spec fallback for stages
whose load_cases are left empty (agent declares BCs in the spec instead of
the suite), plus the production spawn_fem spec cascade underneath it."""

from __future__ import annotations

import uuid

import pytest

from cadx import storage
from cadx.db.models import Job, Part, Project, Version, VersionKind, VersionStatus
from cadx.db.session import async_session, create_all
from cadx.evalkit.local import LocalCtx, _spawn
from cadx.evalkit.suite import EvalStage

_SPEC_LCS = [{"name": "spec_case", "fixed": [{"kind": "surface"}], "loads": []}]
_PARAMS_LCS = [{"name": "params_case", "fixed": [{"kind": "surface"}], "loads": []}]


@pytest.fixture(autouse=True)
async def _tables():
    await create_all()


async def _mk_parent(
    *, spec_lcs=None, params_lcs=None, with_density: bool = False
) -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session() as db:
        p = Project(name="fb-eval")
        db.add(p)
        await db.flush()
        part = Part(project_id=p.id, number=1, name="bracket")
        db.add(part)
        await db.flush()
        v = Version(
            project_id=p.id,
            part_id=part.id,
            number=1,
            kind=VersionKind.generate,
            prompt="cantilever",
            status=VersionStatus.succeeded,
            spec={"load_cases": spec_lcs} if spec_lcs else {},
            params={"load_cases": params_lcs} if params_lcs else {},
            mesh=(
                {"topopt": {"density_relpath": "d.npz", "solver": "oc"}} if with_density else None
            ),
        )
        db.add(v)
        await db.commit()
        # fem spawns require a mesh artifact on disk
        exp = storage.export_dir(p.id, v.id)
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "model.stl").write_text("solid m\nendsolid m\n")
        return p.id, v.id


def _ctx(tmp_path, pid, vid) -> LocalCtx:
    ctx = LocalCtx(tmp_path)
    ctx.project_id = pid
    ctx.parent_version_id = vid
    return ctx


async def test_fem_stage_falls_back_to_parent_spec(tmp_path):
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS)
    stage = EvalStage(id="fem", type="fem", use_parent_fallbacks=True)
    out_vid, jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert out_vid == vid  # same-version stage
    assert meta["load_cases_source"] == "parent_spec"
    async with async_session() as db:
        job = await db.get(Job, jid)
        req = (job.resume_state or {})["fem_request"]
        # the driver injected the cases explicitly — spawn recorded them as
        # a request; the meta carries the real provenance for the extras
        assert req["load_cases"] == _SPEC_LCS
        assert req["load_cases_source"] == "request"
        args = (job.payload or {}).get("args") or {}
        assert args.get("load_cases") == _SPEC_LCS


async def test_fem_stage_explicit_cases_stay_stage_sourced(tmp_path):
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS)
    own = [{"name": "own", "fixed": [{"kind": "surface"}], "loads": []}]
    stage = EvalStage(id="fem", type="fem", load_cases=own)
    _, jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert "load_cases_source" not in meta  # stage-declared — nothing injected
    async with async_session() as db:
        job = await db.get(Job, jid)
        assert job.resume_state["fem_request"]["load_cases"] == own
        assert job.resume_state["fem_request"]["load_cases_source"] == "request"


async def test_fem_stage_flag_off_uses_product_spec_fallback(tmp_path):
    """use_parent_fallbacks=False only skips DRIVER injection — the product
    spawn_fem still resolves spec cases (same cascade the API applies), and
    records it so the extras mapping reports parent_spec provenance."""
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS)
    stage = EvalStage(id="fem", type="fem", use_parent_fallbacks=False)
    _, jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert "load_cases_source" not in meta
    async with async_session() as db:
        job = await db.get(Job, jid)
        req = job.resume_state["fem_request"]
        assert req["load_cases"] == _SPEC_LCS
        assert req["load_cases_source"] == "spec"


async def test_fem_stage_without_any_cases_still_422s(tmp_path):
    pid, vid = await _mk_parent()  # no spec cases anywhere
    stage = EvalStage(id="fem", type="fem", use_parent_fallbacks=True)
    with pytest.raises(Exception, match="load case"):
        await _spawn(stage, _ctx(tmp_path, pid, vid))


async def test_optimize_stage_uses_parent_spec_not_params(tmp_path):
    """run_optimize_job's cascade is payload -> spec (no params hop) — the
    driver mirrors it so parent params cases do NOT leak into optimize."""
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS, params_lcs=_PARAMS_LCS)
    stage = EvalStage(id="opt", type="optimize")
    _, jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert meta["load_cases_source"] == "parent_spec"
    async with async_session() as db:
        job = await db.get(Job, jid)
        assert (job.payload or {}).get("args", {}).get("load_cases") == _SPEC_LCS


async def test_resample_stage_prefers_parent_params(tmp_path):
    """run_*resample/refine* jobs cascade payload -> parent params -> spec —
    the driver mirrors that ordering (params are the previous run's actual
    inputs, closer to truth than the original spec)."""
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS, params_lcs=_PARAMS_LCS, with_density=True)
    stage = EvalStage(id="rs", type="resample")
    _, jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert meta["load_cases_source"] == "parent_params"
    async with async_session() as db:
        job = await db.get(Job, jid)
        assert (job.payload or {}).get("args", {}).get("load_cases") == _PARAMS_LCS


async def test_resample_stage_falls_through_to_spec(tmp_path):
    pid, vid = await _mk_parent(spec_lcs=_SPEC_LCS, with_density=True)  # no params cases
    stage = EvalStage(id="rs", type="resample")
    _, _jid, meta = await _spawn(stage, _ctx(tmp_path, pid, vid))
    assert meta["load_cases_source"] == "parent_spec"
