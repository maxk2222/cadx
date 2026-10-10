"""Mode resolution + mesh/hybrid agent loop tests (FakeLLM + FakeMeshClient)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from cadagent.agent.executor_client import InProcessExecutor
from cadagent.agent.mesh_client import FakeMeshClient
from cadagent.agent.orchestrator import resolve_mode, run_generation
from cadagent.agent.trace import MemoryTrace
from cadagent.agent.types import AgentCtx, InputAsset
from cadagent.db.models import ProjectSettings
from cadagent.llm.fake import FakeLLM
from cadagent.llm.schemas import ModelingPlan

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _ctx(llm: FakeLLM | None = None, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=1))
    kw.setdefault("mode", "generate")
    kw.setdefault("prompt", "x")
    return AgentCtx(
        llm=llm or FakeLLM([]),
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        **kw,
    )


def _img() -> InputAsset:
    return InputAsset(
        asset_id=str(uuid.uuid4()),
        kind="image",
        path=FIXTURES / "images" / "knob.png",
        filename="knob.png",
    )


def _plan(approach: str) -> ModelingPlan:
    return ModelingPlan(approach=approach, approach_reason="r", mesh_subject="knob")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# resolve_mode


def test_resolve_explicit_cad():
    ctx = _ctx(gen_mode="cad", assets=[_img()], mesh_available=True)
    assert resolve_mode(ctx) == ("cad", "CAD mode selected")


def test_resolve_mesh_needs_images():
    ctx = _ctx(gen_mode="mesh", assets=[], mesh_available=True)
    mode, note = resolve_mode(ctx)
    assert mode == "cad"
    assert "image" in note


def test_resolve_mesh_needs_server():
    ctx = _ctx(gen_mode="mesh", assets=[_img()], mesh_available=False)
    mode, note = resolve_mode(ctx)
    assert mode == "cad"
    assert "unreachable" in note


def test_resolve_mesh_ok():
    ctx = _ctx(gen_mode="mesh", assets=[_img()], mesh_available=True)
    assert resolve_mode(ctx)[0] == "mesh"


def test_resolve_hybrid_ok():
    ctx = _ctx(gen_mode="hybrid", assets=[_img()], mesh_available=True)
    assert resolve_mode(ctx)[0] == "hybrid"


def test_resolve_auto_no_images_is_cad():
    ctx = _ctx(gen_mode="auto", assets=[], mesh_available=True, plan=_plan("mesh"))
    assert resolve_mode(ctx)[0] == "cad"


def test_resolve_auto_planner_mesh():
    ctx = _ctx(gen_mode="auto", assets=[_img()], mesh_available=True, plan=_plan("mesh"))
    assert resolve_mode(ctx)[0] == "mesh"


def test_resolve_auto_planner_mesh_no_server():
    ctx = _ctx(gen_mode="auto", assets=[_img()], mesh_available=False, plan=_plan("hybrid"))
    mode, note = resolve_mode(ctx)
    assert mode == "cad"
    assert "unreachable" in note


def test_resolve_auto_planner_cad():
    ctx = _ctx(gen_mode="auto", assets=[_img()], mesh_available=True, plan=_plan("cad"))
    assert resolve_mode(ctx)[0] == "cad"


# --------------------------------------------------------------------------- #
# full mesh-mode run

MESH_SPEC = {
    "name": "knob",
    "summary": "organic knob ~40mm",
    "units": "mm",
    "length_x": {"value": 40.0, "unit": "mm", "source": "stated"},
    "width_y": {"value": 40.0, "unit": "mm", "source": "assumed"},
    "height_z": {"value": 30.0, "unit": "mm", "source": "assumed"},
    "features": [],
    "open_questions": [],
}
MESH_PLAN = {
    "approach": "mesh",
    "approach_reason": "organic",
    "mesh_subject": "round domed knob",
    "steps": [],
    "params": [],
    "standard_parts": [],
    "notes": "",
}
MESH_ORIENT = {"up": "+z", "front": "+y", "confidence": 0.9, "reason": "dome up"}
MESH_CRITIQUE = {
    "score": 90,
    "passed": True,
    "discrepancies": [],
    "matches": ["domed"],
    "confidence": 0.9,
}


async def test_mesh_mode_generation(tmp_path):
    llm = FakeLLM([MESH_SPEC, MESH_PLAN, MESH_ORIENT, MESH_CRITIQUE])
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
    )
    outcome = await run_generation(ctx)
    assert outcome.status in ("succeeded", "needs_review")
    assert outcome.resolved_mode == "mesh"
    assert outcome.code is None
    assert outcome.mesh.get("generation", {}).get("model") == "fake"
    assert mesh.calls, "fake mesh client was not invoked"
    assert outcome.analysis is not None
    assert outcome.analysis["bbox_size"][0] == pytest.approx(40, abs=1.0)
    # mesh artifacts registered
    assert outcome.artifacts.get("mesh_raw")
    assert outcome.artifacts.get("mesh_clean")


async def test_mesh_mode_falls_back_without_server():
    """Requested mesh, no server → resolves to CAD (FakeLLM code path)."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,  # planner still says mesh; guard falls back to CAD
            {
                "reasoning": "box",
                "params": [],
                "expected_bbox": [40, 20, 10],
                "code": box_code,
            },
            MESH_CRITIQUE,
        ]
    )
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img()],
        mesh=None,
        mesh_available=False,
        prompt="40x20x10 block",
    )
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode == "cad"
    assert outcome.status in ("succeeded", "needs_review")
    assert outcome.code is not None


async def test_hybrid_mode_generation(tmp_path):
    code = (
        "from build123d import *\n\n"
        'PARAMS = {"hole_d": 4.2}\n\n'
        "base = load_mesh_solid()\n"
        'hole = Pos(0, 0, -100) * Cylinder(radius=PARAMS["hole_d"] / 2, height=300)\n'
        "result = base - hole\n"
    )
    llm = FakeLLM(
        [
            MESH_SPEC,
            {
                "approach": "hybrid",
                "approach_reason": "organic + precise hole",
                "mesh_subject": "round domed knob",
                "hybrid_cad_features": ["M4 through-hole"],
                "steps": ["mesh", "hole"],
                "params": [],
                "standard_parts": [],
                "notes": "",
            },
            MESH_ORIENT,
            {
                "reasoning": "mesh + hole",
                "params": [{"name": "hole_d", "value": 4.2, "unit": "mm"}],
                "expected_bbox": [40, 40, 30],
                "code": code,
            },
            MESH_CRITIQUE,
        ]
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="hybrid",
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        prompt="knob with M4 hole",
    )
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode == "hybrid"
    assert outcome.status in ("succeeded", "needs_review"), outcome.error
    assert outcome.code is not None
    assert "load_mesh_solid" in outcome.code
    # faceted solid was built and kept for the version
    assert ctx.hybrid_solid is not None and ctx.hybrid_solid.exists()
    assert outcome.mesh.get("generation", {}).get("model") == "fake"
