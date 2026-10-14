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


def _ctx() -> AgentCtx:
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
    )


def _wb(tmp_path: Path) -> MeshWorkbench:
    # 130x30x30 grip-like cylinder ≈ 92k mm³
    raw = tmp_path / "raw.glb"
    trimesh.creation.cylinder(radius=15, height=130, sections=32).export(
        str(raw), file_type="glb"
    )
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
    res = await wb.handle(
        "mesh_boolean_add", {"shape": "cylinder", "radius": 16, "height": 130}
    )
    assert res["ok"] is False
    assert "refused" in res["error"]
    assert not wb.ops  # nothing recorded


async def test_boolean_add_small_boss_allowed(tmp_path):
    wb = _wb(tmp_path)
    await wb.handle("mesh_inspect", {})
    res = await wb.handle(
        "mesh_boolean_add",
        {"shape": "cylinder", "radius": 4, "height": 10, "at": [0, 0, 0]},
    )
    assert res["ok"], res.get("error")
    assert len(wb.ops) == 1
