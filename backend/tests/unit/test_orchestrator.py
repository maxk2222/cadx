def test_build_issues_solids_gate():
    """Disconnected solids are a major deterministic issue regardless of
    what the critique model scored."""
    from cadx.agent.orchestrator import build_issues

    issues = build_issues(None, [], [], None, {"solid_count": 3})
    assert issues[0]["source"] == "solids"
    assert issues[0]["severity"] == "major"
    assert "3 disconnected solids" in issues[0]["what"]

    assert build_issues(None, [], [], None, {"solid_count": 1}) == []
    assert build_issues(None, [], [], None, None) == []


def test_copy_best_artifacts_uses_art_dir(tmp_path):
    """T8.7: when several pipeline runs share an iteration dir, the winner's
    renders must come from its own attempt subdir — the shared dir holds the
    LAST write, which may be a worse attempt (stale-score bug)."""
    import uuid

    from cadx import storage
    from cadx.agent.executor_client import InProcessExecutor
    from cadx.agent.orchestrator import _copy_best_artifacts
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import ProjectSettings
    from cadx.llm.fake import FakeLLM

    async def emit(*a, **kw):
        pass

    async def no_cancel():
        return False

    pid, vid = uuid.uuid4(), uuid.uuid4()
    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=pid,
        version_id=vid,
        prompt="t",
        mode="generate",
        settings=ProjectSettings(),
    )
    it_dir = storage.iteration_dir(pid, vid, 3)
    # attempt a1 (the winner) wrote a render; a later attempt overwrote the
    # shared iteration dir with different content
    winner_dir = it_dir / "a1"
    (winner_dir / "renders").mkdir(parents=True)
    (winner_dir / "renders" / "iso.png").write_bytes(b"winner")
    (winner_dir / "contact_sheet.png").write_bytes(b"winner-sheet")
    stale_dir = it_dir / "a2"
    (stale_dir / "renders").mkdir(parents=True)
    (stale_dir / "renders" / "iso.png").write_bytes(b"stale")
    (stale_dir / "contact_sheet.png").write_bytes(b"stale-sheet")

    exec_result = {
        "art_dir": storage.relpath(winner_dir),
        "renders": {"iso": f"{storage.relpath(winner_dir)}/renders/iso.png"},
    }
    out = _copy_best_artifacts(ctx, 3, exec_result)
    dst = storage.resolve(out["render:iso"])
    assert dst.read_bytes() == b"winner"
    assert storage.resolve(out["contact_sheet"]).read_bytes() == b"winner-sheet"


async def test_process_mesh_warns_on_ops_version_drift():
    """T8.8: an executor built from an older mesh_ops revision echoes a lower
    MESH_OPS_VERSION — the worker must surface it, not silently replay."""
    import uuid

    from cadx.agent.orchestrator import _process_mesh
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.cad.mesh_ops import MESH_OPS_VERSION, MeshPostParams
    from cadx.db.models import ProjectSettings
    from cadx.llm.fake import FakeLLM

    class DriftedExecutor:
        async def process_mesh(self, payload):
            return {
                "ok": True,
                "mesh": {"mesh_ops_version": MESH_OPS_VERSION - 1},
            }

    emitted: list[str] = []

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        emitted.append(msg)

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=DriftedExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="t",
        mode="generate",
        settings=ProjectSettings(),
    )
    res = await _process_mesh(ctx, 1, "x.glb", spec=None, post=MeshPostParams())
    assert res["ok"]
    assert any("ops" in m and "v" in m for m in emitted), emitted


async def test_clip_rejected_when_it_shrinks_below_stated_dims():
    """A footprint clip can only shrink the part — when the generated mesh is
    locally undersized vs the measured outline, clipping pulled a spec-fitted
    119x34x8 scale down to 115x28.5 (observed in knife_scale_hybrid). The IoU
    improved, so the clip used to be kept; now a dim regression rejects it."""
    import uuid

    from cadx.agent.orchestrator import _clip_mesh_to_measured
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.cad.mesh_ops import MeshPostParams
    from cadx.db.models import ProjectSettings
    from cadx.llm.fake import FakeLLM
    from cadx.llm.schemas import Dim, PartSpec

    async def emit(*a, **kw):
        pass

    async def no_cancel():
        return False

    class ShrinkingExecutor:
        async def process_mesh(self, payload):
            # clip applied: IoU improved but the bbox shrank below spec
            return {
                "ok": True,
                "analysis": {"bbox_size": [115.0, 28.5, 8.0]},
                "mesh": {"features": {"outline_mm": [[-10, -8], [10, -8], [10, 8], [-10, 8]]}},
            }

    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=ShrinkingExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="t",
        mode="generate",
        settings=ProjectSettings(),
    )
    ctx.measurements = {
        "m1": {
            "authoritative": True,
            "confidence": 1.0,
            # measured outline spanning the full stated 119x34
            "outline_mm": [[-59.5, -17.0], [59.5, -17.0], [59.5, 17.0], [-59.5, 17.0]],
        }
    }
    spec = PartSpec(
        name="scale",
        summary="knife handle scale",
        length_x=Dim(value=119.0, source="stated"),
        width_y=Dim(value=34.0, source="stated"),
        height_z=Dim(value=8.0, source="stated"),
    )
    # pre-clip result: spec-fitted, footprint differs from the measured outline
    exec_result = {
        "ok": True,
        "analysis": {"bbox_size": [119.0, 34.0, 8.0]},
        "mesh": {"features": {"outline_mm": [[-40, -10], [40, -10], [40, 10], [-40, 10]]}},
    }
    post = MeshPostParams()
    out_res, out_post = await _clip_mesh_to_measured(
        ctx, 1, "x.glb", spec, post, exec_result, iter([1])
    )
    # rejected: returned the pre-clip result, not the shrunken one
    assert out_res is exec_result
    assert out_res["analysis"]["bbox_size"] == [119.0, 34.0, 8.0]
    assert not out_post.ops  # clip op was not adopted


def test_orientation_ranking_support_picks_standing():
    """Lying knob (Hunyuan Y-up convention): aspect/dims cannot separate
    'standing on skirt' from 'lying on side' (40x40 vs ~46x46 footprint) —
    the support term must rank a y-axis up orientation first."""
    import numpy as np
    import trimesh

    from cadx import storage
    from cadx.agent.orchestrator import _orientation_ranking
    from cadx.llm.schemas import Dim, PartSpec

    skirt = trimesh.creation.cylinder(radius=18.0, height=8.0, sections=48)
    skirt.apply_translation([0, 0, 4.0])
    dome = trimesh.creation.icosphere(subdivisions=2, radius=20.0)
    dome.apply_scale([1, 1, 0.6])
    dome.apply_translation([0, 0, 12.0])
    m = trimesh.util.concatenate([skirt, dome])
    # author it Y-up (dome axis along +Y) like raw Hunyuan output
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    p = storage.resolve("t_orient_rank.glb")
    m.export(str(p), file_type="glb")

    spec = PartSpec(
        name="knob",
        summary="knob",
        length_x=Dim(value=40.0, source="stated"),
        width_y=Dim(value=40.0, source="stated"),
        # height not stated -> no dim violation separates standing from lying;
        # only the support term can pick the skirt-side-down orientation
        height_z=Dim(value=24.0, source="assumed"),
    )
    table, rank_fn = _orientation_ranking("t_orient_rank.glb", spec, 0.05, 1.0, canonical_up="y")
    assert table and rank_fn
    assert table[0][1]["up"].lstrip("+-") == "y"
    # a lying pick must not be ranked first
    lying = rank_fn({"up": "+x", "front": "+y"})
    standing = rank_fn({"up": "+y", "front": "+x"})
    assert lying[0] >= standing[0] and standing <= lying


# --------------------------------------------------------------------------- #
# _run_cad stall/progress loop (dq-based, F1 regression)
# --------------------------------------------------------------------------- #

_SPEC = {
    "name": "block",
    "summary": "40x20x10 block",
    "units": "mm",
    "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
    "width_y": {"value": 20.0, "unit": "mm", "source": "stated"},
    "height_z": {"value": 10.0, "unit": "mm", "source": "stated"},
    "features": [],
    "open_questions": [],
}


def _exec_res(np_pass: int, n_fail: int, bbox=None) -> dict:
    checks = [{"name": f"c{i}", "status": "PASS"} for i in range(np_pass)]
    checks += [{"name": f"f{i}", "status": "FAIL"} for i in range(n_fail)]
    return {
        "ok": True,
        "analysis": {"bbox_size": bbox or [40.0, 20.0, 10.0], "solid_count": 1},
        "verify_spec": {"status": "FAIL" if n_fail else "PASS", "checks": checks},
        "features": {},
        "params_used": {},
    }


class _ScriptedExec:
    """Executor port stub: pops scripted execute() results, exports nothing."""

    def __init__(self, results: list[dict]) -> None:
        self._script = list(results)
        self.exec_calls = 0

    async def execute(self, payload):
        self.exec_calls += 1
        if len(self._script) > 1:
            return self._script.pop(0)
        return self._script[0]

    async def export_all(self, payload):
        return {"exports": {}}


def _box_code() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "fixtures" / "simple_box.py").read_text()


def _fail_critique():
    return {
        "score": 0,
        "passed": False,
        "discrepancies": [
            {"what": f"issue {i}", "where": "part", "severity": "major", "source": "spec"}
            for i in range(4)
        ],
        "matches": [],
        "confidence": 0.9,
    }


def _pass_critique():
    return {
        "score": 95,
        "passed": True,
        "discrepancies": [],
        "matches": ["block"],
        "confidence": 0.9,
    }


async def _drive_cad(llm, results, emitted, *, max_iterations, patience):
    """Run _run_cad with a scripted FakeLLM + scripted executor."""
    import time
    import uuid as _uuid

    from cadx.agent.orchestrator import RunOutcome, _run_cad
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import ProjectSettings
    from cadx.llm.schemas import ModelingPlan, PartSpec
    from cadx.runtime_config import RuntimeSettings

    async def emit(step, it, msg, **_f):
        emitted.append((step, it, msg))

    async def no_cancel():
        return False

    ctx = AgentCtx(
        llm=llm,
        executor=_ScriptedExec(results),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=_uuid.uuid4(),
        version_id=_uuid.uuid4(),
        prompt="block 40x20x10",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(
            agent_max_wall_s=86_400,
            agent_param_autofix=False,
            agent_stall_patience=patience,
        ),
    )
    spec = PartSpec.model_validate(_SPEC)
    plan = ModelingPlan(approach="cad", approach_reason="r")
    outcome = RunOutcome()
    return await _run_cad(
        ctx,
        outcome,
        spec,
        plan,
        time.monotonic(),
        no_cancel,
        no_cancel,
        max_iterations=max_iterations,
        max_repairs=0,
        pass_score=80,
        tol_pct=2.0,
        tol_mm=0.5,
    ), ctx


async def test_stall_progress_via_dq_not_score():
    """F1: critique score pinned at 0 used to stall-abort at iteration 2 —
    but a rising verify_spec pass count (dq) is progress and must keep the
    loop going until the pass lands."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()
    llm = FakeLLM(
        [
            {"reasoning": "v1", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            _fail_critique(),
            {"reasoning": "imp", "params": [], "edits": [], "full_code": box_code},
            _fail_critique(),
            {"reasoning": "imp2", "params": [], "edits": [], "full_code": box_code},
            _pass_critique(),
        ]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [
            _exec_res(2, 4),  # it1: dq 33, score 0 -> new best, progress
            _exec_res(3, 3, bbox=[45.0, 20.0, 10.0]),  # it2: dq 40, dim issue -> not best, dq rises
            _exec_res(6, 0),  # it3: clean pass
        ],
        emitted,
        max_iterations=4,
        patience=2,
    )
    # the old score-only tracker would have stalled out at it2
    assert outcome.iterations == 3
    assert outcome.status == "succeeded"
    # no fresh-codegen escape was needed — dq progress kept the loop alive
    assert not any("fresh codegen" in m for _s, _i, m in emitted)


async def test_stall_triggers_one_fresh_codegen_then_finalize():
    """A truly flat run gets exactly ONE fresh-codegen escape (force_fresh
    skips the improve path once), then finalises on the next stall."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()
    code_resp = {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": box_code}
    patch = {"reasoning": "imp", "params": [], "edits": [], "full_code": box_code}
    llm = FakeLLM(
        [code_resp, _fail_critique(), patch, _fail_critique(), code_resp, _fail_critique()]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [_exec_res(2, 4)],
        emitted,
        max_iterations=5,
        patience=1,
    )
    assert outcome.iterations == 3  # it1 + improve it2 + fresh it3, then break
    fresh = [m for s, _i, m in emitted if s == "plan" and "fresh codegen" in m]
    assert len(fresh) == 1
    assert any(s == "codegen" and i == 3 for s, i, _m in emitted)
    assert not any(s == "improve" and i == 3 and "fixing" in m for s, i, m in emitted)
    assert any(s == "finalize" and "no improvement" in m for s, _i, m in emitted)


def test_regression_critique_fires_on_collapse():
    """Iter-6 handle: improve rebuilt a 30x20x6 plate (~10% volume) where the
    best candidate had the full mesh body — the vision critique just restated
    the obvious at full reasoning cost. Synthesize the verdict instead."""
    from cadx.agent.orchestrator import Candidate, _regression_critique
    from cadx.llm.schemas import CodeResponse

    best = Candidate(
        1,
        CodeResponse(reasoning="", params=[], expected_bbox=[], code="result=1"),
        {"analysis": {"volume": 40000.0, "bbox_size": [120, 40, 30]}},
        None,
        [],
    )
    collapsed = _regression_critique(
        best, {"volume": 3600.0, "bbox_size": [30, 20, 6]}
    )
    assert collapsed is not None
    assert collapsed.passed is False and collapsed.score <= 30
    assert collapsed.discrepancies[0].severity == "major"
    assert "minimal change" in collapsed.discrepancies[0].fix_hint


def test_regression_critique_ignores_normal_candidates():
    from cadx.agent.orchestrator import Candidate, _regression_critique
    from cadx.llm.schemas import CodeResponse

    best = Candidate(
        1,
        CodeResponse(reasoning="", params=[], expected_bbox=[], code="result=1"),
        {"analysis": {"volume": 40000.0, "bbox_size": [120, 40, 30]}},
        None,
        [],
    )
    # small volume reduction from a feature cut is not a collapse
    assert _regression_critique(best, {"volume": 36000.0, "bbox_size": [120, 40, 30]}) is None
    # no best candidate -> never fire
    assert _regression_critique(None, {"volume": 100.0, "bbox_size": [5, 5, 5]}) is None


def test_gate_generated_images_drops_bad_hero(tmp_path):
    """A persisted generated hero from an earlier version must not feed the
    mesh generator when its silhouette contradicts the spec — selection-time
    gate (creation-time verify only covers THIS job's images)."""
    from cadx.agent.orchestrator import _gate_generated_images
    from cadx.agent.types import InputAsset

    def _png(w: int, h: int):
        from pathlib import Path as _P  # noqa: F401
        from PIL import Image

        p = tmp_path / f"img_{w}x{h}.png"
        img = Image.new("RGB", (400, 400), (250, 250, 250))
        # dark subject block centred, w x h pixels
        for x in range(200 - w // 2, 200 + w // 2):
            for y in range(200 - h // 2, 200 + h // 2):
                img.putpixel((x, y), (20, 20, 20))
        img.save(p)
        return p

    def _dim(v):
        return {"value": v, "unit": "mm", "source": "stated"}

    from cadx.llm.schemas import PartSpec

    spec = PartSpec(
        name="t",
        summary="t",
        units="mm",
        length_x=_dim(120),
        width_y=_dim(40),
        height_z=_dim(30),
        features=[],
        open_questions=[],
    )
    bad = InputAsset(
        asset_id="bad1", kind="generated_image", path=_png(60, 200), caption="", filename="b.png"
    )
    good = InputAsset(
        asset_id="good1", kind="generated_image", path=_png(300, 80), caption="", filename="g.png"
    )
    user = InputAsset(
        asset_id="user1", kind="image", path=_png(60, 200), caption="", filename="u.png"
    )
    out = _gate_generated_images([bad, good, user], spec)
    ids = [a.asset_id for a in out]
    assert "bad1" not in ids  # tall image vs 4:1-flat spec -> dropped
    assert "good1" in ids
    assert "user1" in ids  # user photos are never gated
    assert _gate_generated_images([bad], None) == [bad]
