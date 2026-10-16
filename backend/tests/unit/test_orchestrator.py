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
