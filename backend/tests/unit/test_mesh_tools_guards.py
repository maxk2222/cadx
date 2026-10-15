"""Volume guard for mesh_boolean_add (T4.3): a giant primitive must not
replace the organic body."""

from __future__ import annotations

import uuid
from pathlib import Path

import trimesh

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.mesh_tools import MeshWorkbench, _primitive_volume
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.cad.mesh_ops import MeshPostParams
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings


def _ctx(**kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=FakeLLM([]),
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="test",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        **kw,
    )


def _wb(tmp_path: Path) -> MeshWorkbench:
    # 130x30x30 grip-like cylinder ≈ 92k mm³
    raw = tmp_path / "raw.glb"
    trimesh.creation.cylinder(radius=15, height=130, sections=32).export(str(raw), file_type="glb")
    return MeshWorkbench(_ctx(), raw_path=raw, post=MeshPostParams(), spec=None, iteration=1)


def test_primitive_volume():
    assert _primitive_volume({"shape": "cylinder", "radius": 5, "height": 10}) > 700
    assert _primitive_volume({"shape": "box", "size": 10}) == 1000.0
    assert _primitive_volume({"shape": "box", "size_xyz": [2, 3, 4]}) == 24.0
    assert _primitive_volume({"shape": "sphere", "radius": 3}) > 100
    assert _primitive_volume({"shape": "cylinder"}) is None


async def test_boolean_add_giant_primitive_refused(tmp_path):
    wb = _wb(tmp_path)
    await wb.handle("mesh_inspect", {})
    # a r=16 h=130 cylinder is ~50% larger than the body → refused
    res = await wb.handle("mesh_boolean_add", {"shape": "cylinder", "radius": 16, "height": 130})
    assert res["ok"] is False
    assert "refused" in res["error"]
    assert not wb.ops  # nothing recorded


def test_workbench_decimates_dense_baseline(tmp_path):
    """T0.2: the in-worker workbench caps baseline faces so voxel/boolean ops
    can't exhaust worker memory (the executor replay keeps full detail)."""
    raw = tmp_path / "dense.glb"
    trimesh.creation.icosphere(subdivisions=4, radius=20).export(str(raw), file_type="glb")
    wb = MeshWorkbench(
        _ctx(runtime_settings=RuntimeSettings(mesh_decimate_faces=2_000)),
        raw_path=raw,
        post=MeshPostParams(),
        spec=None,
        iteration=1,
    )
    assert wb.error is None
    assert len(wb.baseline.faces) <= 2_000
    assert len(wb.work.faces) <= 2_000


async def test_boolean_add_small_boss_allowed(tmp_path):
    wb = _wb(tmp_path)
    await wb.handle("mesh_inspect", {})
    res = await wb.handle(
        "mesh_boolean_add",
        {"shape": "cylinder", "radius": 4, "height": 10, "at": [0, 0, 0]},
    )
    assert res["ok"], res.get("error")
    assert len(wb.ops) == 1


async def test_op_cycle_detector_refuses_third_identical_op(tmp_path):
    """T8.5: apply → undo → apply → undo → apply must refuse the third
    application — the observed bore-cut undo loop burned the whole session."""
    wb = _wb(tmp_path)
    args = {"axis": "z", "deg": 15.0}
    assert (await wb.handle("mesh_rotate", args))["ok"]
    assert (await wb.handle("mesh_undo", {}))["ok"]
    assert (await wb.handle("mesh_rotate", args))["ok"]
    assert (await wb.handle("mesh_undo", {}))["ok"]
    res = await wb.handle("mesh_rotate", args)
    assert res["ok"] is False
    assert "cycle" in res["error"]
    assert len(wb.ops) == 0  # nothing was appended


async def test_undo_of_verified_state_warns_then_refuses(tmp_path):
    """T8.5: deterministic confirmation (preview/measure marked the state
    verified) protects it — first undo warns, second is refused."""
    wb = _wb(tmp_path)
    assert (await wb.handle("mesh_rotate", {"axis": "z", "deg": 10.0}))["ok"]
    # deterministic evidence marked this exact ops state as verified
    wb._verified_fps.add(wb._ops_fp(wb.ops))
    res = await wb.handle("mesh_undo", {})
    assert res["ok"]
    assert "verified" in res.get("warning", "")
    # re-apply the same op → ops fingerprint matches the verified state again
    assert (await wb.handle("mesh_rotate", {"axis": "z", "deg": 10.0}))["ok"]
    res = await wb.handle("mesh_undo", {})
    assert res["ok"] is False
    assert "verified" in res["error"]
    assert len(wb.ops) == 1  # op list untouched


async def test_section_through_bore_multi_entity(tmp_path):
    """T8.3: a section through a through-bore produces multiple entities —
    numpy truth-value testing used to crash on exactly this."""
    raw = tmp_path / "bored.glb"
    body = trimesh.creation.cylinder(radius=12, height=60, sections=32)
    bore = trimesh.creation.cylinder(radius=5, height=80, sections=24)
    mesh = body.difference(bore, engine="manifold")
    mesh.export(str(raw), file_type="glb")
    wb = MeshWorkbench(_ctx(), raw_path=raw, post=MeshPostParams(), spec=None, iteration=1)
    assert wb.error is None
    res = await wb.handle("mesh_section", {"axis": "y", "at_mm": 0.0})
    assert res["ok"], res.get("error")
    assert "section" in res["images"]
