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


def test_params_vs_spec_only_checks_parametric_cad_envelopes():
    from cadx.agent.orchestrator import _params_vs_spec
    from cadx.llm.schemas import PartSpec

    spec = PartSpec(
        name="mesh-backed knob",
        summary="40 mm knob with a centered bore",
        length_x={"value": 40.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
        features=[],
        image_roles=[],
        verification_questions=[],
    )
    params = {"hole_d": 4.2}

    assert _params_vs_spec(spec, params, mode="hybrid") == []
    assert _params_vs_spec(spec, params, mode="mesh") == []
    assert len(_params_vs_spec(spec, params, mode="cad")) == 3


def test_best_candidate_keeps_its_own_issue_list():
    from cadx.agent.orchestrator import Candidate, _select_best_candidate
    from cadx.llm.schemas import Critique

    old = Candidate(1, None, {}, Critique(score=40, passed=False), [])
    current = Candidate(2, None, {}, Critique(score=80, passed=True), [])
    old_issues = [{"what": "old oversized-rib claim"}]
    current_issues = [{"what": "current slot tolerance note"}]

    best, issues = _select_best_candidate(old, old_issues, current, current_issues)

    assert best is current
    assert issues is current_issues


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

    def __init__(self, results: list[dict], exports: dict | None = None) -> None:
        self._script = list(results)
        self._exports = exports or {}
        self.exec_calls = 0

    async def execute(self, payload):
        self.exec_calls += 1
        if len(self._script) > 1:
            return self._script.pop(0)
        return self._script[0]

    async def export_all(self, payload):
        return {"exports": self._exports}


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


async def _drive_cad(
    llm, results, emitted, *, max_iterations, patience, hybrid=False, max_repairs=0, spec=None
):
    """Run _run_cad with a scripted FakeLLM + scripted executor."""
    import time
    import uuid as _uuid
    from pathlib import Path

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
        executor=_ScriptedExec(
            results,
            exports={fmt: "model." + fmt for fmt in ("step", "stl", "3mf", "glb")}
            if spec
            else None,
        ),
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
    if hybrid:
        # stand-in for the prepared hybrid mesh body — only `is not None`
        # gates the hybrid guard, the scripted executor never reads it
        ctx.hybrid_solid = Path("/tmp/fake_solid.brep")
        ctx.resolved_mode = "hybrid"
    spec = PartSpec.model_validate(spec or _SPEC)
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
        max_repairs=max_repairs,
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
    collapsed = _regression_critique(best, {"volume": 3600.0, "bbox_size": [30, 20, 6]})
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


def test_gate_generated_images_drops_major_verdict(tmp_path):
    """A generated image whose vision verdict is `major` (certified wrong
    part) must not condition the mesh generator — the drop falls through to
    the concept bootstrap, like the aspect gate."""
    from cadx.agent.orchestrator import _gate_generated_images
    from cadx.agent.types import InputAsset

    p = tmp_path / "hero.png"
    from PIL import Image

    Image.new("RGB", (400, 400), (120, 120, 120)).save(p)

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
        asset_id="bad1",
        kind="generated_image",
        path=p,
        caption="",
        filename="b.png",
        verify_severity="major",
    )
    fine = InputAsset(
        asset_id="fine1",
        kind="generated_image",
        path=p,
        caption="",
        filename="f.png",
        verify_severity="ok",
    )
    unver = InputAsset(
        asset_id="unver1",
        kind="generated_image",
        path=p,
        caption="",
        filename="u.png",
        verify_severity="unverifiable",
    )
    user = InputAsset(asset_id="user1", kind="image", path=p, caption="", filename="i.png")
    ids = [a.asset_id for a in _gate_generated_images([bad, fine, unver, user], spec)]
    assert "bad1" not in ids
    assert "fine1" in ids
    assert "unver1" in ids  # unverifiable never blocks generation
    assert "user1" in ids


# --------------------------------------------------------------------------- #
# B20b: declared CAD fallback in hybrid mode
# --------------------------------------------------------------------------- #


async def test_cad_fallback_marker_resolves_honest_cad():
    """`cad_fallback = "<reason>"` in hybrid mode: the code openly declares a
    parametric replacement for the mesh body — not a contract violation, so
    no hybrid major issue; when that candidate wins the outcome reports
    mode "cad" instead of claiming a hybrid part."""
    from cadx.llm.fake import FakeLLM

    fb_code = _box_code() + 'cad_fallback = "mesh booleans collapsed the organic body"\n'
    llm = FakeLLM(
        [
            {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": fb_code},
            _pass_critique(),
        ]
    )
    emitted: list = []
    outcome, ctx = await _drive_cad(
        llm,
        [_exec_res(6, 0)],
        emitted,
        max_iterations=2,
        patience=50,
        hybrid=True,
    )
    assert outcome.status == "succeeded"
    assert outcome.resolved_mode == "cad"
    assert ctx.resolved_mode == "cad"
    assert "declared CAD fallback" in outcome.mode_note
    assert not any(i["source"] == "hybrid" for i in outcome.issues)


async def test_hybrid_missing_without_marker_stays_major():
    """Same overwrite-without-declaration code in hybrid mode keeps the
    hybrid contract violation as a major issue."""
    from cadx.llm.fake import FakeLLM

    llm = FakeLLM(
        [
            {
                "reasoning": "v",
                "params": [],
                "expected_bbox": [40, 20, 10],
                "code": _box_code(),
            },
            _pass_critique(),
        ]
    )
    emitted: list = []
    outcome, ctx = await _drive_cad(
        llm,
        [_exec_res(6, 0)],
        emitted,
        max_iterations=1,
        patience=50,
        hybrid=True,
    )
    assert any(i["source"] == "hybrid" and i["severity"] == "major" for i in outcome.issues)
    # no declared fallback — the mode must not be rewritten to "cad"
    assert outcome.resolved_mode is None
    assert ctx.resolved_mode == "hybrid"


# --------------------------------------------------------------------------- #
# B22: issue-conflict pass — mesh-only anchor demands vs spec
# --------------------------------------------------------------------------- #


def _spec_two_bores():
    from cadx.llm.schemas import PartSpec

    return PartSpec.model_validate(
        {
            **_SPEC,
            "features": [
                {
                    "name": "M6 Durchgangsbohrung",
                    "type": "through_hole",
                    "count": 2,
                    "dims": {"diameter": 7.0},
                    "position": "y durchgehend",
                    "description": "two M6 through bores",
                }
            ],
        }
    )


def _anchor_finding(idx, source, *, actual="missing", severity="major"):
    return {
        "feature": f"hole {idx}",
        "expected": "at anchored position Ø7.0",
        "actual": actual,
        "severity": severity,
        "fix_hint": "cut the hole at the anchored position — use anchor_holes()",
        "anchor_source": source,
    }


def test_build_issues_mesh_only_anchor_downgraded():
    """A mesh-scan-only anchor demanding a hole the spec never declared must
    not appear as a major — the spec's declared feature set wins."""
    from cadx.agent.orchestrator import build_issues

    findings = [
        _anchor_finding(1, "spec+mesh"),
        _anchor_finding(3, "mesh_scan"),
    ]
    issues = build_issues(None, [], findings, None, {"solid_count": 1}, spec=_spec_two_bores())
    spec_issue = [i for i in issues if "hole 1" in i["what"]]
    mesh_issue = [i for i in issues if "hole 3" in i["what"]]
    assert spec_issue and spec_issue[0]["severity"] == "major"
    assert mesh_issue and mesh_issue[0]["severity"] == "minor"
    assert mesh_issue[0]["what"].startswith("mesh-detected (not spec-declared):")


def test_build_issues_drops_contradicted_mesh_only_demand():
    """Direct contradiction: a critique/spec-side issue already demands the
    extra hole NOT exist — the mesh-only 'cut a third hole' demand is
    dropped, not merely softened."""
    from cadx.agent.orchestrator import build_issues
    from cadx.llm.schemas import Critique, Discrepancy

    critique = Critique(
        score=50,
        passed=False,
        discrepancies=[
            Discrepancy(
                what="the part shows three holes but the spec declares exactly 2 "
                "— remove the extra hole",
                where="top face",
                severity="major",
                source="spec",
                fix_hint="",
            )
        ],
    )
    findings = [_anchor_finding(3, "mesh_scan")]
    issues = build_issues(critique, [], findings, None, {"solid_count": 1}, spec=_spec_two_bores())
    assert not any("hole 3" in i["what"] for i in issues)
    assert any("exactly 2" in i["what"] for i in issues)


def test_build_issues_no_spec_keeps_mesh_findings():
    """Without a spec there is no authoritative declaration to conflict
    with — mesh-only anchor findings keep their original severity."""
    from cadx.agent.orchestrator import build_issues

    findings = [_anchor_finding(3, "mesh_scan")]
    issues = build_issues(None, [], findings, None, {"solid_count": 1})
    assert issues[0]["severity"] == "major"
    assert not issues[0]["what"].startswith("mesh-detected")


def test_annotate_anchor_sources():
    """The `hole <index>` finding text maps back to the anchor's source so
    the conflict pass can see mesh-only provenance."""
    from cadx.agent.orchestrator import _annotate_anchor_sources

    anchors = {
        "holes": [
            {"index": 1, "source": "spec+mesh"},
            {"index": 2, "source": "mesh_scan"},
        ]
    }
    findings = [
        {"feature": "hole 1"},
        {"feature": "hole 2"},
        {"feature": "slot width"},  # not an anchor finding — untouched
    ]
    _annotate_anchor_sources(findings, anchors)
    assert findings[0]["anchor_source"] == "spec+mesh"
    assert findings[1]["anchor_source"] == "mesh_scan"
    assert "anchor_source" not in findings[2]
    # empty/absent anchors are a no-op
    findings2 = [{"feature": "hole 9"}]
    _annotate_anchor_sources(findings2, None)
    assert "anchor_source" not in findings2[0]


def test_gate_generated_images_logs_drop_once(tmp_path, monkeypatch):
    """B24: the gate runs per selection pass (hero list, combined list) —
    a dropped asset must be logged once per job, not once per pass."""
    import cadx.agent.orchestrator as orch
    from cadx.agent.types import InputAsset

    events: list = []

    class _Log:
        def info(self, ev, **kw):
            events.append((ev, kw))

        def warning(self, *a, **kw):
            pass

        def error(self, *a, **kw):
            pass

    monkeypatch.setattr(orch, "log", _Log())

    def _png(w: int, h: int):
        from PIL import Image

        p = tmp_path / f"img_{w}x{h}.png"
        img = Image.new("RGB", (400, 400), (250, 250, 250))
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
    logged: set[str] = set()
    orch._gate_generated_images([bad, good], spec, logged)
    orch._gate_generated_images([bad, good], spec, logged)
    drops = [kw["assets"] for ev, kw in events if ev == "mesh.aspect_gate_dropped"]
    assert drops == [["bad1"]]
    # and the asset is still dropped on every pass — only the log is deduped
    assert "bad1" not in [a.asset_id for a in orch._gate_generated_images([bad], spec, logged)]


# --------------------------------------------------------------------------- #
# P10: issue-fingerprint stall — repeat issues on unchanged geometry
# --------------------------------------------------------------------------- #


async def test_issue_fingerprint_flags_artifacts_and_skips_improve():
    """The same number-normalised issue text surviving 2+ iterations while
    the measured features did not move is a measurement artifact: flagged
    on the persisted list (honesty) but never fed to the patcher. When
    every remaining issue is an artifact the loop converges instead of
    burning another improve round."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()
    llm = FakeLLM(
        [
            {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            _fail_critique(),  # it1: 4x "issue N" — one fingerprint after #-normalisation
            {"reasoning": "imp", "params": [], "edits": [], "full_code": box_code},
            _fail_critique(),  # it2: identical texts on identical features -> artifacts
        ]
    )
    emitted: list = []
    outcome, ctx = await _drive_cad(
        llm,
        [
            _exec_res(2, 4),  # it1: dq 33 -> best
            _exec_res(6, 0),  # it2: dq 100 -> best replaced, its (flagged) issues feed back
        ],
        emitted,
        max_iterations=5,
        patience=50,
    )
    # the stall emit fired once, on the second sighting (iteration 2)
    stall_emits = [
        (i, m) for s, i, m in emitted if s == "improve" and "unchanged 2+ iterations" in m
    ]
    assert len(stall_emits) == 1 and stall_emits[0][0] == 2
    # flagged issues stay on the persisted list — honest, just not actionable
    assert any(i.get("artifact") for i in outcome.issues)
    critique_issues = [i for i in outcome.issues if i.get("source") == "critique"]
    assert critique_issues and all(i.get("artifact") for i in critique_issues)
    # all-artifact issue list -> converged before improve round 3: no
    # further LLM calls (2 codegen/improve + 2 critique) and no it3 exec
    assert any(s == "improve" and "converged" in m for s, _i, m in emitted)
    assert len(llm.calls) == 4
    assert ctx.executor.exec_calls == 2


async def test_issue_fingerprint_resets_when_issue_changes():
    """A different issue text restarting the streak must NOT flag — the
    fingerprint gate only fires on a genuinely repeated finding."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()

    def _crit(what):
        return {
            "score": 40,
            "passed": False,
            "discrepancies": [
                {"what": what, "where": "part", "severity": "major", "source": "spec"}
            ],
            "matches": [],
            "confidence": 0.9,
        }

    llm = FakeLLM(
        [
            {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            _crit("edge too sharp"),
            {"reasoning": "i", "params": [], "edits": [], "full_code": box_code},
            _crit("top face scratched"),  # different text -> no artifact
        ]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [_exec_res(2, 4), _exec_res(6, 0)],
        emitted,
        max_iterations=3,
        patience=1,
    )
    assert not any("unchanged 2+ iterations" in m for _s, _i, m in emitted)
    assert not any(i.get("artifact") for i in outcome.issues)


async def test_infra_exec_error_skips_repair(monkeypatch):
    """Executor-side failures (schema skew, sandbox outage) are not code
    bugs — the repair loop must not spend an LLM call on them, and the
    fabricated issue is recorded as source 'infra'."""
    import cadx.agent.orchestrator as orch
    from cadx.llm.fake import FakeLLM

    monkeypatch.setattr(
        orch,
        "_exec_error_signature",
        lambda _res: "infra:validation error for RuntimeSettings",
    )
    llm = FakeLLM(
        [
            {
                "reasoning": "v",
                "params": [],
                "expected_bbox": [40, 20, 10],
                "code": _box_code(),
            },
        ]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [
            {
                "ok": False,
                "error": "executor task failed: validation error for RuntimeSettings",
                "traceback": None,
                "stdout": "",
            }
        ],
        emitted,
        max_iterations=4,
        patience=1,
        max_repairs=1,
    )
    # repair skipped — no LLM call beyond the single codegen
    assert any(
        s == "repair" and "infrastructure error — skipping model repair" in m
        for s, _i, m in emitted
    )
    assert len(llm.calls) == 1
    # nothing executable ever landed -> honest failure, not a code verdict
    assert outcome.error == "all iterations failed to produce executable code"


def test_merge_features_appends_only_new_pairs():
    """P1: retry extraction merges into the spec's existing features —
    the old wholesale replace dropped entries intake got right."""
    from cadx.agent.orchestrator import _merge_features
    from cadx.llm.schemas import Feature

    old = [
        Feature(name="base plate", type="plate"),
        Feature(name="M6 bore", type="through_hole"),
    ]
    new = [
        Feature(name="m6 BORE", type="Through_Hole"),  # case-insensitive dup
        Feature(name="stiffening rib", type="rib"),
    ]
    merged = _merge_features(old, new)
    assert [f.name for f in merged] == ["base plate", "M6 bore", "stiffening rib"]
    # a wholly-new list merges cleanly; empty new keeps old untouched
    assert _merge_features([], new) == new
    assert _merge_features(old, []) == old


# --------------------------------------------------------------------------- #
# N1 — forced-hybrid conversion failure → CAD fallback for dimensioned specs  #
# --------------------------------------------------------------------------- #


def test_hybrid_prefers_cad_guard():
    """The stated-dimension/optimize guard: mesh-only rescue is only honest
    for organic (image-driven) requests."""
    from cadx.agent.orchestrator import _hybrid_prefers_cad
    from cadx.llm.schemas import PartSpec

    dim_spec = PartSpec.model_validate(_SPEC)  # 3 stated dims
    assert _hybrid_prefers_cad(dim_spec)

    organic = PartSpec.model_validate(
        {
            "name": "toy",
            "summary": "organic figurine",
            "units": "mm",
            "length_x": {"value": 80.0, "unit": "mm", "source": "inferred"},
            "width_y": {"value": 60.0, "unit": "mm", "source": "inferred"},
            "height_z": {"value": 90.0, "unit": "mm", "source": "inferred"},
            "features": [],
        }
    )
    assert not _hybrid_prefers_cad(organic)

    opt_spec = PartSpec.model_validate(
        {
            "name": "bracket",
            "summary": "organic-ish bracket with one dim",
            "units": "mm",
            "length_x": {"value": 80.0, "unit": "mm", "source": "stated"},
            "width_y": {"value": 60.0, "unit": "mm", "source": "inferred"},
            "height_z": {"value": 90.0, "unit": "mm", "source": "inferred"},
            "features": [],
            "optimize": {"enabled": True},
        }
    )
    assert _hybrid_prefers_cad(opt_spec)  # one stated dim + optimize → CAD


async def _drive_hybrid_approach(monkeypatch, spec):
    """_run_approach('hybrid') with a failed _prepare_hybrid_mesh and
    stubbed _run_cad/_run_mesh — records which leg ran."""
    import time
    import uuid as _uuid

    import cadx.agent.orchestrator as orc
    from cadx.agent.orchestrator import RunOutcome, _run_approach
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import ProjectSettings
    from cadx.llm.fake import FakeLLM
    from cadx.llm.schemas import ModelingPlan
    from cadx.runtime_config import RuntimeSettings

    async def emit(step, it, msg, **_f):
        pass

    async def no_cancel():
        return False

    ctx = AgentCtx(
        llm=FakeLLM([{"text": "x"}]),
        executor=None,  # type: ignore[arg-type]
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=_uuid.uuid4(),
        version_id=_uuid.uuid4(),
        prompt="t",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(),
    )
    ctx.mesh_info = {"_pre_result": {"pipeline": "mesh"}, "pipeline": "mesh"}
    ctx.resolved_mode = "hybrid"

    async def fail_prepare(*a, **k):
        return False, "mesh->solid conversion failed"

    calls: list[str] = []

    async def fake_run_cad(ctx, outcome, *a, **k):
        calls.append("cad")
        outcome.resolved_mode = "cad"
        return outcome

    async def fake_run_mesh(ctx, outcome, *a, **k):
        calls.append("mesh")
        outcome.resolved_mode = "mesh"
        return outcome

    monkeypatch.setattr(orc, "_prepare_hybrid_mesh", fail_prepare)
    monkeypatch.setattr(orc, "_run_cad", fake_run_cad)
    monkeypatch.setattr(orc, "_run_mesh", fake_run_mesh)

    plan = ModelingPlan(approach="hybrid", approach_reason="r")
    outcome = await _run_approach(
        ctx,
        RunOutcome(),
        spec,
        plan,
        time.monotonic(),
        no_cancel,
        no_cancel,
        approach="hybrid",
        max_repairs=0,
        tol_pct=2.0,
        tol_mm=0.5,
    )
    return outcome, calls


async def test_hybrid_failure_falls_back_to_cad_for_dimensioned_spec(monkeypatch):
    """N1: forced-hybrid conversion failure on a spec with stated dims must
    take the CAD fallback — a lattice that won't sew is not the answer."""
    from cadx.llm.schemas import PartSpec

    outcome, calls = await _drive_hybrid_approach(monkeypatch, PartSpec.model_validate(_SPEC))
    assert calls == ["cad"]
    assert outcome.resolved_mode == "cad"
    assert "CAD fallback" in outcome.mode_note
    # cad_fallback_plan stripped mesh provenance + steps
    assert (outcome.plan or {}).get("approach") == "cad"


async def test_hybrid_failure_keeps_mesh_rescue_for_organic(monkeypatch):
    """Organic (image-driven) requests keep the mesh-only rescue — there is
    no parametric spec to rebuild from."""
    from cadx.llm.schemas import PartSpec

    organic = PartSpec.model_validate(
        {
            "name": "toy",
            "summary": "organic figurine",
            "units": "mm",
            "length_x": {"value": 80.0, "unit": "mm", "source": "inferred"},
            "width_y": {"value": 60.0, "unit": "mm", "source": "inferred"},
            "height_z": {"value": 90.0, "unit": "mm", "source": "inferred"},
            "features": [],
        }
    )
    outcome, calls = await _drive_hybrid_approach(monkeypatch, organic)
    assert calls == ["mesh"]
    assert outcome.resolved_mode == "mesh"
    assert "mesh-only result" in outcome.mode_note


async def test_export_repair_rescues_failed_thread_rebuild():
    """Finalize: the preview candidate passed but the export-detail rebuild
    failed (observed: m8 plate — `edges() >> Axis.Z` kept the helical groove
    edges and the fillet died). One bounded improve pass must get a chance to
    repair the code before the run is declared failed."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()
    spec = {
        **_SPEC,
        "features": [
            {
                "name": "M8 tapped hole",
                "type": "thread",
                "thread_spec": "M8x1.25-6H",
            }
        ],
    }
    llm = FakeLLM(
        [
            {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            _pass_critique(),
            {"reasoning": "fix", "params": [], "edits": [], "full_code": box_code},
        ]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [
            _exec_res(6, 0),  # it1 preview candidate passes
            {"ok": False, "error": "safe_fillet returned 0 mm"},  # export rebuild fails
            {"ok": True, "analysis": {"bbox_size": [40.0, 20.0, 10.0]}},  # sanity preview
            {  # export retry: real helix present
                **_exec_res(6, 0),
                "thread_count": 1,
                "detail_mode": "export",
            },
        ],
        emitted,
        max_iterations=3,
        patience=1,
        spec=spec,
    )
    assert outcome.status == "succeeded", outcome.error
    assert outcome.code == box_code
    assert any(
        s == "improve" and "export-detail rebuild failed" in m for s, _i, m in emitted
    )
    assert any(
        s == "finalize" and "export repair applied" in m for s, _i, m in emitted
    )


async def test_export_repair_still_fails_honestly_when_unfixable():
    """If the repair pass cannot produce export-detail geometry, the run must
    still fail — preview geometry must never ship as the 'real thread' export."""
    from cadx.llm.fake import FakeLLM

    box_code = _box_code()
    spec = {
        **_SPEC,
        "features": [{"name": "M8 tapped hole", "type": "thread", "thread_spec": "M8x1.25"}],
    }
    llm = FakeLLM(
        [
            {"reasoning": "v", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
            _pass_critique(),
            {"reasoning": "fix", "params": [], "edits": [], "full_code": box_code},
            {"reasoning": "fix2", "params": [], "edits": [], "full_code": box_code},
        ]
    )
    emitted: list = []
    outcome, _ctx = await _drive_cad(
        llm,
        [
            _exec_res(6, 0),
            {"ok": True, "thread_count": 0, "detail_mode": "export"},  # no helix built
            {"ok": True, "analysis": {"bbox_size": [40.0, 20.0, 10.0]}},  # sanity
            {"ok": True, "thread_count": 0, "detail_mode": "export"},  # retry still 0
        ],
        emitted,
        max_iterations=3,
        patience=1,
        spec=spec,
    )
    assert outcome.status == "failed"
    assert "real thread" in outcome.error
