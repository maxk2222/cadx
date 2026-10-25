"""machine_feature / describe_faces tools — semantic machining on the
verified B-Rep, canonical `## block:` emission and auto-verify."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from build123d import Box, Sphere, export_brep

from cadx import storage
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

SEED = '''"""seed scaffold."""

from build123d import *

PARAMS = {}

result = Box(60, 40, 20)
'''


def _ctx(**settings_kw) -> AgentCtx:
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
        runtime_settings=RuntimeSettings(**settings_kw),
    )


async def _real_verify(code: str):
    """Runs the buffer through the actual sandbox — the emitted block only
    'verifies' when the generated code really executes."""
    from cadx.cad.sandbox import run_user_code

    out = run_user_code(code, Path(tempfile.mkdtemp()))
    return {
        "ok": out.ok,
        "error": out.error,
        "analysis": {"is_valid": out.ok},
        "features": {"design_features": out.design_features},
    }


def _write_session_brep(ctx: AgentCtx, part) -> None:
    it_dir = storage.version_dir(ctx.project_id, ctx.version_id) / "iterations" / "0" / "_run"
    it_dir.mkdir(parents=True, exist_ok=True)
    export_brep(part, str(it_dir / "model.brep"))


async def test_machine_feature_emits_block_and_auto_verifies():
    ctx = _ctx(agent_build_verify_max=4)
    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=_real_verify)
    _write_session_brep(ctx, Box(60, 40, 20))
    res = await session.execute(
        "machine_feature",
        {"face": "top", "op": "drill", "at": [10, 5], "dims": {"dia": 5}},
    )
    assert res["ok"], res
    assert res["lands_at_mm"] == pytest.approx([10, 5, 10], abs=1e-3)
    assert "machine_on_face" in res["emitted_code"]
    assert "face='top'" in res["emitted_code"]
    assert res["verify"]["ok"], res["verify"]
    assert "## block: machine_drill" in session.buffer
    assert "CADX_FEATURES = [_feat_drill_0]" in session.buffer


async def test_machine_feature_tap_registers_thread_in_cadx_features():
    ctx = _ctx(agent_build_verify_max=4)
    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=_real_verify)
    _write_session_brep(ctx, Box(60, 40, 20))
    res = await session.execute(
        "machine_feature",
        {
            "face": "left",
            "op": "tap",
            "dims": {"major_d": 8, "pitch": 1.25, "depth": 15},
        },
    )
    assert res["ok"], res
    assert res["approach_normal"] == pytest.approx([-1, 0, 0], abs=1e-3)
    assert "CADX_FEATURES = [_feat_tap_0]" in session.buffer
    # the emitted record carries the thread fields the export gate counts
    assert "kind': 'thread'" in res["emitted_code"] or "machine_on_face" in res["emitted_code"]
    verify = res["verify"]
    assert verify["ok"], verify
    feats = (verify.get("features") or {}).get("design_features") or []
    assert any(f.get("kind") == "thread" and f.get("external") is False for f in feats)


async def test_machine_feature_rejects_bad_op_and_bad_face():
    ctx = _ctx(agent_build_verify_max=4)
    session = ToolSession(ctx, kind="build", code=SEED, verify_fn=_real_verify)
    _write_session_brep(ctx, Box(60, 40, 20))
    res = await session.execute("machine_feature", {"face": "top", "op": "weld", "dims": {}})
    assert not res["ok"] and "one of" in res["error"]
    res = await session.execute(
        "machine_feature",
        {"face": "no_such_face", "op": "drill", "dims": {"dia": 4}},
    )
    assert not res["ok"] and "did not resolve" in res["error"]
    # nothing was appended on the rejections
    assert "## block: machine_" not in session.buffer


async def test_describe_faces_lists_the_verified_body():
    ctx = _ctx()
    session = ToolSession(ctx, kind="build", code=SEED)
    _write_session_brep(ctx, Sphere(radius=30))
    res = await session.execute("describe_faces", {})
    assert res["ok"], res
    assert len(res["faces"]) == 1
    assert res["faces"][0]["kind"] == "spherical"
    assert res["faces"][0]["area_mm2"] == pytest.approx(4 * 3.14159265 * 900, rel=0.05)


async def test_describe_faces_without_verified_brep_is_actionable_error():
    ctx = _ctx()
    session = ToolSession(ctx, kind="build", code=SEED)
    res = await session.execute("describe_faces", {})
    assert not res["ok"]
    assert "verify" in res["error"]
