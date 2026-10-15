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
