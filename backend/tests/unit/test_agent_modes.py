"""Mode resolution + mesh/hybrid agent loop tests (FakeLLM + FakeMeshClient)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.mesh_client import FakeMeshClient, MeshHealth, MeshModelInfo
from cadx.agent.orchestrator import _resolve_mesh_adapter, resolve_mode, run_generation
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import ModelingPlan, PartSpec
from cadx.runtime_config import RuntimeSettings

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


async def test_topopt_mode_runs_optimization_after_base_generation(monkeypatch):
    """TopOpt mode generates a base CAD part, then optimizes it with FEM feedback."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    spec = {
        **MESH_SPEC,
        "load_cases": [
            {
                "name": "cantilever",
                "fixed": [{"kind": "face", "axis": "-x"}],
                "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -50]}],
            }
        ],
        "optimize": {"enabled": True, "target_volume_fraction": 0.5},
    }
    code = {
        "reasoning": "base bracket",
        "params": [],
        "expected_bbox": [40, 20, 10],
        "code": box_code,
    }
    llm = FakeLLM([spec, {**MESH_PLAN, "approach": "cad"}, code, MESH_CRITIQUE])
    ctx = _ctx(
        llm=llm,
        gen_mode="topopt",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1, topopt_volfrac=0.3),
        prompt="lightweight 40x20x10 bracket, fixed at left, 50 N down at right",
    )
    calls: list[dict] = []

    async def fake_optimize(payload: dict) -> dict:
        calls.append(payload)
        return {
            "ok": True,
            "fem": {
                "material": {"name": "PLA"},
                "grid": {"elements": 100},
                "cases": [
                    {
                        "name": "cantilever",
                        "max_vm_mpa": 12.0,
                        "max_disp_mm": 0.3,
                        "safety_factor": 2.0,
                        "compliance_j": 1.0,
                        "converged": True,
                        "solver": "direct",
                        "warnings": [],
                    }
                ],
                "warnings": [],
            },
            "topopt": {"iterations": 12, "converged": True, "volfrac_achieved": 0.3},
            "analysis": {"is_valid": True, "volume": 3000.0},
            "exports": {},
        }

    monkeypatch.setattr(ctx.executor, "optimize", fake_optimize)
    outcome = await run_generation(ctx)

    assert outcome.status == "succeeded"
    assert outcome.resolved_mode == "topopt"
    assert outcome.mesh["pipeline"] == "topopt"
    assert outcome.fem["cases"][0]["safety_factor"] == 2.0
    assert len(calls) == 1
    assert calls[0]["load_cases"][0]["name"] == "cantilever"
    assert calls[0]["optimize"]["target_volume_fraction"] == 0.3
    assert calls[0]["mesh_rel"].endswith("renders/mesh.stl")


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


# --------------------------------------------------------------------------- #
# _resolve_mesh_adapter — client-side model/image-cap resolution


class _StubMesh(FakeMeshClient):
    """FakeMeshClient whose health reports a custom device + model list."""

    def __init__(self, models: list[MeshModelInfo], device: str = "mlx") -> None:
        super().__init__()
        self._h = MeshHealth(ok=True, device=device, backend="stub", models=models)

    async def health(self) -> MeshHealth:
        return self._h


def _model(mid: str, device: str, *, mv: bool = False, max_img: int = 1) -> MeshModelInfo:
    return MeshModelInfo(
        id=mid,
        name=mid,
        device=device,
        backend="stub",
        supports_multiview=mv,
        max_images=max_img,
        available=True,
    )


async def test_resolve_adapter_multiview_cap():
    """multi-image request picks the multiview adapter and its image cap."""
    mesh = _StubMesh(
        [
            _model("single-mlx", "mlx"),
            _model("mv-4", "mps", mv=True, max_img=4),
            _model("mv-8-cuda", "cuda", mv=True, max_img=8),
        ]
    )
    ctx = _ctx(mesh=mesh)
    model, cap = await _resolve_mesh_adapter(ctx, 5)
    assert model == "mv-8-cuda" and cap == 8  # highest max_images wins


async def test_resolve_adapter_degrades_to_hero():
    """mlx case: only a single-image adapter exists → cap=1, explicit model."""
    mesh = _StubMesh([_model("hunyuan3d-2.1-mlx", "mlx")], device="mlx")
    ctx = _ctx(mesh=mesh)
    model, cap = await _resolve_mesh_adapter(ctx, 4)
    assert model == "hunyuan3d-2.1-mlx" and cap == 1


async def test_resolve_adapter_single_image_stays_auto():
    mesh = _StubMesh([_model("hunyuan3d-2.1-mlx", "mlx")], device="mlx")
    ctx = _ctx(mesh=mesh)
    model, cap = await _resolve_mesh_adapter(ctx, 1)
    assert model == "auto" and cap == 1


async def test_resolve_adapter_explicit_model_caps_images():
    mesh = _StubMesh([_model("chosen", "mlx", max_img=2)], device="mlx")
    ctx = _ctx(mesh=mesh, settings=ProjectSettings(mesh_model="chosen"))
    model, cap = await _resolve_mesh_adapter(ctx, 5)
    assert model == "chosen" and cap == 2


# --------------------------------------------------------------------------- #
# MESH_REFINE tool sessions (agent_mesh_tools)


def _tc(name: str, args: dict | None = None) -> dict:
    return {"tool_calls": [{"name": name, "arguments": args or {}}]}


async def test_mesh_refine_session_applies_ops(tmp_path):
    """The shape session's recorded ops replay into the final post params."""
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            _tc("mesh_inspect"),
            _tc("mesh_rotate", {"axis": "z", "deg": 90}),
            _tc("done", {"reasoning": "aligned"}),
            MESH_CRITIQUE,
        ]
    )
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
    ops = outcome.params.get("ops") or []
    assert [o["op"] for o in ops] == ["rotate"]
    assert outcome.mesh["ops"][0]["args"]["deg"] == 90


async def test_mesh_refine_regenerate_triggers_new_seed(tmp_path):
    """`regenerate` in the shape session re-rolls the mesh with a new seed."""
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            _tc("regenerate", {"subject_hint": "flatter knob", "note": "wrong shape"}),
            # iteration 2: orientation is cached, refine session accepts as-is
            _tc("done", {"reasoning": "baseline is fine"}),
            MESH_CRITIQUE,
        ]
    )
    from cadx.runtime_config import RuntimeSettings

    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1),  # no best-of-N
        settings=ProjectSettings(clarification_mode="assume", max_iterations=2),
    )
    outcome = await run_generation(ctx)
    assert outcome.status in ("succeeded", "needs_review")
    assert len(mesh.calls) == 2
    assert mesh.calls[1]["seed"] != mesh.calls[0]["seed"]
    # accepted as-is → no ops on the final params
    assert not (outcome.params.get("ops") or [])


async def test_mesh_refine_repair_round_after_failed_critique(tmp_path):
    """A failing critique seeds an issues-driven repair session; its ops fix
    the candidate in place instead of re-rolling the seed."""
    fail_critique = {
        "score": 40,
        "passed": False,
        "discrepancies": [{"feature": "base", "issue": "wobbly underside", "severity": "major"}],
        "matches": [],
        "confidence": 0.8,
    }
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            _tc("done", {"reasoning": "shape ok"}),  # shape session: accept
            fail_critique,
            _tc("mesh_flatten_bottom", {"height_mm": 2.0}),  # repair session
            _tc("done", {"reasoning": "flattened the base"}),
            MESH_CRITIQUE,  # re-evaluation passes
        ]
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1),  # no best-of-N
    )
    outcome = await run_generation(ctx)
    assert outcome.status == "succeeded"
    assert len(mesh.calls) == 1  # fixed in place — no re-generation
    assert [o["op"] for o in outcome.params.get("ops", [])] == ["flatten_bottom"]


async def test_mesh_tools_disabled_keeps_seed_loop(tmp_path):
    """agent_mesh_tools=False → no sessions, previous behaviour unchanged."""
    from cadx.runtime_config import RuntimeSettings

    llm = FakeLLM([MESH_SPEC, MESH_PLAN, MESH_ORIENT, MESH_CRITIQUE])
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(agent_mesh_tools=False),
    )
    outcome = await run_generation(ctx)
    assert outcome.status in ("succeeded", "needs_review")
    assert not (outcome.params.get("ops") or [])
    # no refine-session LLM calls: spec, plan, orient, critique = 4 calls
    assert llm.total_usage["calls"] == 4


async def test_mesh_refine_unavailable_falls_back(tmp_path, monkeypatch):
    """Refine returning None (tools stripped mid-job, workbench failed) keeps
    the loop on the plain critique path."""
    from cadx.agent.steps import mesh_refine

    async def no_session(*a, **k):
        return None

    monkeypatch.setattr(mesh_refine, "run", no_session)
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
    assert not (outcome.params.get("ops") or [])


async def test_mesh_regen_exhaustion_falls_back_to_cad(monkeypatch, tmp_path):
    """Every refine session rejects the mesh (regenerate) → iterations burn →
    with ≥2 stated dims the outcome must request the CAD fallback."""
    from cadx.agent.steps import mesh_refine as mesh_refine_step

    async def always_regen(*a, **kw):
        return mesh_refine_step.RefineResult(regenerate=True, reasoning="flat slab")

    monkeypatch.setattr(mesh_refine_step, "run", always_regen)

    spec = dict(MESH_SPEC)
    spec["width_y"] = {"value": 40.0, "unit": "mm", "source": "stated"}
    spec["height_z"] = {"value": 30.0, "unit": "mm", "source": "stated"}
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            spec,
            MESH_PLAN,
            MESH_ORIENT,
            # CAD fallback: codegen + critique
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
        mesh=FakeMeshClient(),
        mesh_available=True,
        prompt="40x40x30 block",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=2),
    )
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode == "cad"
    assert "fallback" in (outcome.mode_note or "")
    assert outcome.code is not None
    assert outcome.status in ("succeeded", "needs_review")


async def test_topopt_agent_session_selects_run(monkeypatch):
    """Agent session drives two SIMP runs with varied volfrac and commits the
    second (higher SF) — the deterministic single-shot path is bypassed."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    spec = {
        **MESH_SPEC,
        "load_cases": [
            {
                "name": "cantilever",
                "fixed": [{"kind": "face", "axis": "-x"}],
                "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -50]}],
            }
        ],
        "optimize": {"enabled": True, "target_volume_fraction": 0.5},
    }
    code = {
        "reasoning": "base bracket",
        "params": [],
        "expected_bbox": [40, 20, 10],
        "code": box_code,
    }
    llm = FakeLLM(
        [
            spec,
            {**MESH_PLAN, "approach": "cad"},
            code,
            MESH_CRITIQUE,
            # optimize session: two runs + done on the second
            {"tool_calls": [{"name": "topopt_run", "arguments": {"target_volume_fraction": 0.3}}]},
            {"tool_calls": [{"name": "topopt_run", "arguments": {"target_volume_fraction": 0.7}}]},
            {"tool_calls": [{"name": "done", "arguments": {"run_index": 1, "reasoning": "SF ok"}}]},
        ]
    )
    ctx = _ctx(
        llm=llm,
        gen_mode="topopt",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        prompt="lightweight 40x20x10 bracket, fixed left, 50 N down at right",
    )
    calls: list[dict] = []

    async def fake_optimize(payload: dict) -> dict:
        calls.append(payload)
        vf = payload["optimize"]["target_volume_fraction"]
        sf = 0.9 if vf < 0.5 else 2.1
        return {
            "ok": True,
            "fem": {
                "material": {"name": "PLA"},
                "grid": {"elements": 100},
                "cases": [
                    {
                        "name": "cantilever",
                        "max_vm_mpa": 12.0,
                        "max_disp_mm": 0.3,
                        "safety_factor": sf,
                        "compliance_j": 1.0,
                        "converged": True,
                        "solver": "direct",
                        "warnings": [],
                    }
                ],
                "warnings": [],
            },
            "topopt": {
                "iterations": 12,
                "converged": True,
                "volfrac_target": vf,
                "volfrac_achieved": vf,
            },
            "analysis": {"is_valid": True, "volume": 3000.0},
            "exports": {},
        }

    monkeypatch.setattr(ctx.executor, "optimize", fake_optimize)
    outcome = await run_generation(ctx)

    assert len(calls) == 2
    assert [c["optimize"]["target_volume_fraction"] for c in calls] == [0.3, 0.7]
    # each run occupied its own iteration dir
    assert calls[0]["iteration"] != calls[1]["iteration"]
    assert outcome.resolved_mode == "topopt"
    assert outcome.fem["cases"][0]["safety_factor"] == 2.1
    assert outcome.status in ("succeeded", "needs_review")


async def test_topopt_agent_declines_keeps_base(monkeypatch):
    """done(run_index=-1) → no optimization, no extra executor call."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    spec = {
        **MESH_SPEC,
        "load_cases": [
            {
                "name": "cantilever",
                "fixed": [{"kind": "face", "axis": "-x"}],
                "loads": [{"region": {"kind": "face", "axis": "+x"}, "force_n": [0, 0, -50]}],
            }
        ],
    }
    code = {
        "reasoning": "base bracket",
        "params": [],
        "expected_bbox": [40, 20, 10],
        "code": box_code,
    }
    llm = FakeLLM(
        [
            spec,
            {**MESH_PLAN, "approach": "cad"},
            code,
            MESH_CRITIQUE,
            {
                "tool_calls": [
                    {"name": "done", "arguments": {"run_index": -1, "reasoning": "strong enough"}}
                ]
            },
        ]
    )
    ctx = _ctx(
        llm=llm,
        gen_mode="topopt",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        prompt="40x20x10 bracket",
    )

    async def fake_optimize(payload: dict) -> dict:
        raise AssertionError("optimize must not be called")

    monkeypatch.setattr(ctx.executor, "optimize", fake_optimize)
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode != "topopt"
    assert "kept the base part" in outcome.mode_note


# --------------------------------------------------------------------------- #
# T21: intake role mapping for mesh-source images
# T26: meshgen disk cache


def test_mesh_source_images_maps_intake_roles():
    """Flat part with a top view: the top view becomes the generator's hero
    (role 'front'), a second view degrades to 'other'."""
    from cadx.agent.orchestrator import _mesh_source_images

    top = _img()
    iso = _img()
    spec = PartSpec(
        name="handle",
        length_x={"value": 120.0, "source": "stated"},
        width_y={"value": 38.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[
            {"asset_id": top.asset_id, "role": "top"},
            {"asset_id": iso.asset_id, "role": "iso"},
        ],
    )
    ctx = _ctx(assets=[top, iso])
    ordered, roles = _mesh_source_images(ctx, 1, spec)
    assert ordered[0].asset_id == top.asset_id  # hero = the plan view
    assert roles == ["front", "other"]


async def test_generate_mesh_cache_hit_skips_server(tmp_path):
    """Second identical generation is served from the disk cache."""
    from cadx.agent.orchestrator import _generate_mesh
    from cadx.runtime_config import RuntimeSettings

    spec = PartSpec(
        name="knob",
        length_x={"value": 40.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(mesh_prep_images=False),
    )
    rel1 = await _generate_mesh(ctx, 1, spec, None)
    rel2 = await _generate_mesh(ctx, 1, spec, None)  # same request → cache hit
    assert rel1 and rel2
    assert len(mesh.calls) == 1  # second call served from cache
    assert ctx.mesh_info["generation"]["cached"] is True
