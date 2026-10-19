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
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=1))
    kw.setdefault("mode", "generate")
    kw.setdefault("prompt", "x")
    # deterministic mesh seed — the default derives from uuid4 project/asset
    # ids, which made mesh outcome assertions depend on hash luck (flaky)
    if kw["settings"].mesh_seed is None:
        kw["settings"].mesh_seed = 4242
    # wall-clock guards must never fire in tests — a loaded CI machine made
    # run_generation bail before the re-critique (agent_max_wall_s=900s)
    rs = kw.get("runtime_settings")
    kw["runtime_settings"] = (
        rs.model_copy(update={"agent_max_wall_s": 86_400})
        if rs is not None
        else RuntimeSettings(agent_max_wall_s=86_400)
    )
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


_IMG_TMP = Path(__file__).resolve().parent / "_img_tmp"


def _img(seed: int | None = None) -> InputAsset:
    """Distinct pixels per asset — the multiview aHash dedup would merge
    identical fixture copies into one. `seed` fixes the pixels for tests
    whose image-derived measurements (rembg mask → silhouette IoU) must be
    deterministic."""
    import tempfile

    from PIL import Image, ImageDraw

    aid = str(uuid.uuid4())
    p = Path(tempfile.mkdtemp(prefix="cadx-img-")) / f"{aid[:8]}.png"
    # deterministic subject-on-transparent: the alpha channel doubles as the
    # reference mask — rembg never runs in unit tests and the mask is noise-
    # free. The subject is deliberately TINY (~4-5 px): the _ref_side_mask
    # coverage gate (<3 % is speckle/junk) discards it, so the deterministic
    # spec ops (silhouette carve/profile) stay out of these tests entirely —
    # whatever aspect the fake blob happens to have.
    import random

    rng = random.Random(seed if seed is not None else int(aid[:8], 16))
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    # low-alpha pixel noise: unique bytes per asset (the disk mesh cache keys
    # on image CONTENT — a real cross-test cache hit was observed), invisible
    # to the reference mask which thresholds alpha at >128
    for _ in range(300):
        v = rng.randrange(256)
        dr.point(
            (rng.randrange(64), rng.randrange(64)),
            fill=(v, v, v, 40 + rng.randrange(60)),
        )
    r = 2 + rng.randrange(2)  # tiny subject, position varies per asset
    x, y = 20 + rng.randrange(25), 20 + rng.randrange(25)
    dr.ellipse([x - r, y - r, x + r, y + r], fill=(120, 120, 120, 255))
    img.save(p)
    return InputAsset(asset_id=aid, kind="image", path=p, filename=p.name)


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
        # pin the pure-mesh outcome: parametrize would legitimately set code
        runtime_settings=RuntimeSettings(agent_parametrize=False),
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
            # intake gate: prompt says "M4 hole" but spec.features is empty →
            # the focused re-ask recovers it before planning
            {
                "name": "knob",
                "summary": "organic knob with M4 hole",
                "features": [
                    {
                        "name": "m4 hole",
                        "type": "hole",
                        "description": "centered M4 through-hole",
                        "dims": {"diameter": 4.2},
                        "position": "center",
                        "count": 1,
                        "source": "stated",
                    }
                ],
            },
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
    """The repair session's recorded ops replay into the final post params."""
    fail_critique = {
        "score": 40,
        "passed": False,
        "discrepancies": [
            {
                "feature": "base",
                "issue": "rough underside",
                "severity": "major",
                "source": "spec",
            }
        ],
        "matches": [],
        "confidence": 0.8,
    }
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            fail_critique,  # baseline is critiqued BEFORE any refine session
            _tc("mesh_inspect"),
            # identity-scale: a repair op that cannot drop dims/features —
            # the regression gate would legitimately revert a real wreck
            _tc("mesh_scale", {"factor": 1.0}),
            _tc("done", {"reasoning": "touched up"}),
            MESH_CRITIQUE,
        ]
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img(seed=1)],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(agent_parametrize=False),
    )
    outcome = await run_generation(ctx)
    assert outcome.status in ("succeeded", "needs_review")
    ops = outcome.params.get("ops") or []
    # the repair session's ops land at the tail — deterministic spec ops
    # (spec_drill/spec_silhouette) may legitimately precede them
    assert ops and ops[-1]["op"] == "scale"
    scale = [o for o in outcome.mesh["ops"] if o["op"] == "scale"]
    assert scale and scale[0]["args"]["factor"] == 1.0


async def test_mesh_refine_regenerate_triggers_new_seed(tmp_path):
    """`regenerate` in the repair session re-rolls the mesh with a new seed."""
    fail_critique = {
        "score": 20,
        "passed": False,
        "discrepancies": [
            {
                "feature": "overall",
                "issue": "wrong object entirely",
                "severity": "major",
                "source": "spec",
            }
        ],
        "matches": [],
        "confidence": 0.9,
    }
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            fail_critique,  # baseline critique fails → repair session
            _tc("regenerate", {"subject_hint": "flatter knob", "note": "wrong shape"}),
            # iteration 2: orientation is cached, baseline critique passes
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
    # accepted as-is → the repair session recorded no geometry ops
    # (deterministic spec ops in params are a separate, legit source)
    assert not (outcome.mesh.get("ops") or [])


async def test_mesh_refine_repair_round_after_failed_critique(tmp_path):
    """A failing critique seeds an issues-driven repair session; its ops fix
    the candidate in place instead of re-rolling the seed."""
    fail_critique = {
        "score": 40,
        "passed": False,
        "discrepancies": [
            {
                "feature": "base",
                "issue": "wobbly underside",
                "severity": "major",
                "source": "spec",
            }
        ],
        "matches": [],
        "confidence": 0.8,
    }
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            fail_critique,  # baseline critique fails
            _tc("mesh_scale", {"factor": 1.0}),  # repair session: inert op
            _tc("done", {"reasoning": "fixed in place"}),
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
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1, agent_parametrize=False),
    )
    outcome = await run_generation(ctx)
    assert outcome.status == "succeeded"
    assert len(mesh.calls) == 1  # fixed in place — no re-generation
    assert [o["op"] for o in outcome.params.get("ops", [])] == ["scale"]


async def test_mesh_refine_regression_reverts_ops(tmp_path):
    """A repair round that deterministically regresses (worse rank) is rolled
    back — the wrecked op list must not become the session state."""
    fail_critique = {
        "score": 40,
        "passed": False,
        "discrepancies": [
            {
                "feature": "base",
                "issue": "rough underside",
                "severity": "major",
                "source": "spec",
            }
        ],
        "matches": [],
        "confidence": 0.8,
    }
    worse_critique = {
        "score": 10,
        "passed": False,
        "discrepancies": [
            {
                "feature": "shape",
                "issue": "flattened into a sliver",
                "severity": "major",
                "source": "spec",
            },
            {
                "feature": "dims",
                "issue": "height destroyed",
                "severity": "major",
                "source": "spec",
            },
        ],
        "matches": [],
        "confidence": 0.9,
    }
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            fail_critique,
            _tc("mesh_flatten_bottom", {"height_mm": 20.0}),  # wrecks the part
            _tc("done", {"reasoning": "flattened"}),
            worse_critique,  # re-eval: clearly worse → ops revert
        ]
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img(seed=1)],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1),
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
    )
    outcome = await run_generation(ctx)
    # the regressed ops were reverted — params carry the pre-session state
    # (deterministic spec ops like silhouette_carve may legitimately remain)
    ops = outcome.params.get("ops") or []
    assert "flatten_bottom" not in [o["op"] for o in ops]


async def test_mesh_repair_skipped_after_two_regressions(tmp_path):
    """Two consecutive repair regressions mean ops can't fix this shape —
    later iterations must not burn a session; a fresh seed is cheaper."""
    fail_critique = {
        "score": 40,
        "passed": False,
        "discrepancies": [
            {
                "feature": "base",
                "issue": "rough underside",
                "severity": "major",
                "source": "spec",
            }
        ],
        "matches": [],
        "confidence": 0.8,
    }
    worse_critique = {
        "score": 10,
        "passed": False,
        "discrepancies": [
            {
                "feature": "shape",
                "issue": "flattened into a sliver",
                "severity": "major",
                "source": "spec",
            },
            {
                "feature": "dims",
                "issue": "height destroyed",
                "severity": "major",
                "source": "spec",
            },
        ],
        "matches": [],
        "confidence": 0.9,
    }
    wreck = [
        _tc("mesh_flatten_bottom", {"height_mm": 20.0}),
        _tc("done", {"reasoning": "flattened"}),
    ]
    llm = FakeLLM(
        [
            MESH_SPEC,
            MESH_PLAN,
            MESH_ORIENT,
            fail_critique,
            *wreck,
            worse_critique,  # it1 repair regresses
            fail_critique,
            *wreck,
            worse_critique,  # it2 repair regresses → it3 skips the session
            fail_critique,  # it3 baseline critique; no session scripted
        ]
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        llm=llm,
        gen_mode="mesh",
        assets=[_img(seed=1)],
        mesh=mesh,
        mesh_available=True,
        prompt="organic knob 40mm",
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1),
        settings=ProjectSettings(clarification_mode="assume", max_iterations=3),
    )
    msgs: list[str] = []
    orig_emit = ctx.emit

    async def rec(step: str, it: int, msg: str, **f) -> None:
        msgs.append(msg)
        await orig_emit(step, it, msg, **f)

    ctx.emit = rec
    await run_generation(ctx)
    assert any("last repair sessions all regressed" in m for m in msgs)


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
    # no tool sessions → no recorded session ops; deterministic spec ops
    # (silhouette carve/drill) may still appear in params
    assert not (outcome.mesh.get("ops") or [])
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
    fail_crit = {
        "score": 20,
        "passed": False,
        # spec-sourced so fact calibration keeps the major — "image"/"other"
        # majors downgrade to minor when every measured fact passes
        "discrepancies": [
            {
                "what": "wrong shape vs the stated part",
                "where": "overall",
                "severity": "major",
                "source": "spec",
                "fix_hint": "reshape the body",
            }
        ],
        "matches": [],
        "confidence": 0.9,
    }
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            spec,
            MESH_PLAN,
            MESH_ORIENT,
            fail_crit,  # iter1 baseline critique → repair session → regenerate
            fail_crit,  # iter2 baseline critique → repair session → regenerate
            fail_crit,  # iter3 baseline critique → repair session → regenerate
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
        assets=[_img(seed=1)],
        mesh=FakeMeshClient(),
        mesh_available=True,
        prompt="40x40x30 block",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=3),
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


def test_mesh_source_images_blank_intake_roles_use_measurements():
    """Local models emit image_roles with blank asset_id/role. When intake
    produced no usable roles, the deterministic measure step's plan-view
    classification must drive the generator mapping (regression: blank roles
    collapsed to front+other x3 -> a standing 'banana' mesh for a flat scale)."""
    from cadx.agent.orchestrator import _mesh_source_images

    top = _img()
    side = _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "inferred"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[{"asset_id": "", "role": ""}, {"asset_id": "", "role": ""}],
    )
    ctx = _ctx(assets=[top, side])
    ctx.measurements = {top.asset_id: {"role": "top"}}
    ordered, roles = _mesh_source_images(ctx, 1, spec)
    assert ordered[0].asset_id == top.asset_id
    assert roles == ["front", "other"]


def test_mesh_source_images_measured_plan_overrides_intake_side():
    """For flat parts a measured 'top' beats an intake that mislabeled the
    face-up photo as a side elevation."""
    from cadx.agent.orchestrator import _mesh_source_images

    face = _img()
    side = _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[
            {"asset_id": face.asset_id, "role": "left"},
            {"asset_id": side.asset_id, "role": "right"},
        ],
    )
    ctx = _ctx(assets=[face, side])
    ctx.measurements = {face.asset_id: {"role": "top"}}
    ordered, roles = _mesh_source_images(ctx, 1, spec)
    assert ordered[0].asset_id == face.asset_id
    assert roles == ["front", "right"]


def test_intake_repairs_blank_image_roles_positionally():
    """_repair_image_roles binds all-blank asset_ids by image order."""
    from cadx.agent.steps.intake import _repair_image_roles

    a, b = _img(), _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[{"asset_id": "", "role": "top"}, {"asset_id": "", "role": "left"}],
    )
    ctx = _ctx(assets=[a, b])
    _repair_image_roles(spec, ctx)
    assert [r.asset_id for r in spec.image_roles] == [a.asset_id, b.asset_id]
    # partial/blank roles are left alone once ids are bound
    spec2 = PartSpec(
        name="x",
        length_x={"value": 10.0, "source": "stated"},
        width_y={"value": 10.0, "source": "stated"},
        height_z={"value": 5.0, "source": "stated"},
        image_roles=[{"asset_id": a.asset_id, "role": "top"}, {"asset_id": "", "role": "left"}],
    )
    _repair_image_roles(spec2, ctx)
    assert spec2.image_roles[1].asset_id == ""


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


# --------------------------------------------------------------------------- #
# M2: silhouette back view + heightfield candidate + multi-adapter screening


def test_mesh_source_images_adds_synth_back_view_for_flat_parts():
    """Flat part with a measured outline and no bottom view gets a synthetic
    filled-silhouette 'back' so the generator anchors the underside."""
    from cadx.agent.orchestrator import _mesh_source_images

    top = _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[{"asset_id": top.asset_id, "role": "top"}],
    )
    ctx = _ctx(assets=[top])
    ctx.measurements = {
        top.asset_id: {
            "role": "top",
            "confidence": 0.9,
            "authoritative": True,
            "outline_mm": [[-60, -17], [60, -17], [60, 17], [-60, 17], [-60, -17]],
        }
    }
    ordered, roles = _mesh_source_images(ctx, 1, spec)
    assert roles[0] == "front"
    assert roles[-1] == "back"
    assert ordered[-1].asset_id == "_synth_back"
    assert ordered[-1].path.exists()


def test_mesh_source_images_no_synth_back_when_outline_missing():
    from cadx.agent.orchestrator import _mesh_source_images

    top = _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[{"asset_id": top.asset_id, "role": "top"}],
    )
    ctx = _ctx(assets=[top])
    ctx.measurements = {top.asset_id: {"role": "top", "confidence": 0.9}}
    _ordered, roles = _mesh_source_images(ctx, 1, spec)
    assert roles == ["front"]


async def test_candidate_adapters_prefers_configured_csv():
    from cadx.agent.orchestrator import _candidate_adapters

    ctx = _ctx(
        runtime_settings=RuntimeSettings(mesh_candidate_adapters="triposg, hunyuan3d, trellis2")
    )
    assert await _candidate_adapters(ctx, 2) == ["triposg", "hunyuan3d"]


async def test_candidate_adapters_auto_spreads_over_health():
    from cadx.agent.orchestrator import _candidate_adapters

    class _Multi(FakeMeshClient):
        async def health(self):
            return MeshHealth(
                ok=True,
                device="mps",
                models=[
                    MeshModelInfo(id="hunyuan3d", device="mps", supports_multiview=True),
                    MeshModelInfo(id="triposg", device="cpu", supports_multiview=True),
                    MeshModelInfo(id="dead", available=False),
                ],
            )

    ctx = _ctx(mesh=_Multi())
    got = await _candidate_adapters(ctx, 4)
    assert got == ["hunyuan3d", "triposg"]  # device-native first, dead skipped


async def test_candidate_adapters_skips_image_adapters():
    """Image-generation adapters (flux/sdxl) must never become mesh candidates —
    they report available + max_images=0 and would poison candidate screening."""
    from cadx.agent.orchestrator import _candidate_adapters

    class _Mixed(FakeMeshClient):
        async def health(self):
            return MeshHealth(
                ok=True,
                device="mlx",
                models=[
                    MeshModelInfo(id="hunyuan3d-2.1-mlx", device="mlx", supports_multiview=False),
                    MeshModelInfo(id="triposg", device="mps", supports_multiview=False),
                    MeshModelInfo(id="flux-schnell-mlx", device="mlx", kind="image", max_images=0),
                    MeshModelInfo(id="sdxl", device="mps", kind="image", max_images=0),
                ],
            )

    ctx = _ctx(mesh=_Mixed())
    got = await _candidate_adapters(ctx, 4)
    assert got == ["hunyuan3d-2.1-mlx", "triposg"]  # no image adapters


def test_pick_adapter_skips_image_adapters():
    from cadx.agent.mesh_client import MeshModelInfo
    from cadx.agent.model_pick import pick_adapter

    models = [
        MeshModelInfo(id="flux-schnell-mlx", device="mlx", kind="image", available=True),
        MeshModelInfo(id="triposg", device="mps", available=True),
    ]
    pick = pick_adapter(models, "flat")
    assert pick is not None and pick.id == "triposg"


def test_pick_adapter_prefers_mlx_over_torch_mps_multiview():
    """B.1: on an MLX host the only multiview adapters are torch-mps builds
    (stall-prone under memory pressure) — auto-pick must choose a native
    single-image adapter instead, capping the images it can use."""
    from cadx.agent.mesh_client import MeshModelInfo
    from cadx.agent.model_pick import pick_adapter

    models = [
        MeshModelInfo(
            id="hunyuan3d-2mv",
            device="mps",
            backend="torch-mps",
            supports_multiview=True,
            max_images=4,
            available=True,
        ),
        MeshModelInfo(
            id="hunyuan3d-2.1-mlx",
            device="mlx",
            backend="mlx",
            supports_multiview=False,
            max_images=1,
            available=True,
        ),
    ]
    pick = pick_adapter(models, "organic", need_multiview=True, device="mlx")
    assert pick is not None and pick.id == "hunyuan3d-2.1-mlx"
    # but a genuinely multiview-capable native adapter still wins
    models.append(
        MeshModelInfo(
            id="triposg-mv",
            device="mlx",
            backend="mlx",
            supports_multiview=True,
            max_images=4,
            available=True,
        )
    )
    pick = pick_adapter(models, "organic", need_multiview=True, device="mlx")
    assert pick is not None and pick.id == "triposg-mv"


async def test_generate_mesh_uses_heightfield_for_flat_part():
    """With a measured outline + flat spec the deterministic heightfield
    candidate wins even without candidate screening — no generator call."""
    import time

    from cadx.agent.orchestrator import _generate_mesh

    top = _img()
    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
        image_roles=[{"asset_id": top.asset_id, "role": "top"}],
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        assets=[top],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=1, mesh_prep_images=False),
    )
    ctx.measurements = {
        top.asset_id: {
            "role": "top",
            "confidence": 0.9,
            "authoritative": True,
            "outline_mm": [[-60, -17], [60, -17], [60, 17], [-60, 17], [-60, -17]],
        }
    }
    rel = await _generate_mesh(ctx, 1, spec, None, t0=time.monotonic())
    assert rel and rel.endswith("mesh_raw_hf.glb")
    assert mesh.calls == []  # generator never ran
    assert ctx.mesh_info["generation"]["heightfield"] is True


async def test_generate_mesh_hopeless_gate_skips_full_generation(monkeypatch):
    """All cheap candidates below the viability threshold → no full-res roll.

    Regression: _generate_mesh must return None (callers route to the CAD
    fallback) instead of re-rolling the same hopeless inputs at full cost.
    """
    import time

    import cadx.agent.orchestrator as orch
    from cadx.agent.orchestrator import _generate_mesh

    spec = PartSpec(
        name="scale",
        length_x={"value": 119.0, "source": "stated"},
        width_y={"value": 34.0, "source": "stated"},
        height_z={"value": 8.0, "source": "stated"},
    )
    mesh = FakeMeshClient()
    ctx = _ctx(
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(mesh_candidates_per_iter=2, mesh_prep_images=False),
    )
    # the fake blob legitimately scores above the viability bar in its best
    # orientation — force hopeless scores so the test exercises the gate
    # itself instead of depending on fragile geometry scoring
    monkeypatch.setattr(orch, "_score_candidate_mesh", lambda *a, **k: 0.1)
    # no measurements → no outline → no heightfield candidate
    rel = await _generate_mesh(ctx, 1, spec, None, t0=time.monotonic())
    assert rel is None
    assert ctx.mesh_info["_mesh_hopeless"] is True
    assert len(mesh.calls) == 2  # only the two cheap candidates ran


def test_mesh_source_images_dedupes_near_identical(tmp_path):
    """Two uploads of the same photo (near-identical aHash) must not both
    reach the generator — it would waste a view slot on the same pose."""
    from PIL import Image

    from cadx.agent.orchestrator import _mesh_source_images

    a = _img()
    # byte-different but perceptually identical copy of a's image
    dup_path = tmp_path / "dup.png"
    img = Image.open(a.path).convert("RGB")
    img.save(dup_path, quality=95)
    b = InputAsset(asset_id=str(uuid.uuid4()), kind="image", path=dup_path, filename="dup.png")
    c = _img()
    spec = PartSpec(
        name="knob",
        length_x={"value": 40.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 25.0, "source": "stated"},
        image_roles=[
            {"asset_id": a.asset_id, "role": "front"},
            {"asset_id": b.asset_id, "role": "back"},
            {"asset_id": c.asset_id, "role": "left"},
        ],
    )
    ctx = _ctx(assets=[a, b, c])
    ordered, _roles = _mesh_source_images(ctx, 1, spec)
    ids = [o.asset_id for o in ordered]
    assert b.asset_id not in ids
    assert set(ids) == {a.asset_id, c.asset_id}


def test_hybrid_missing_mesh_ranks_lower():
    """T4.5: an improve result that dropped load_mesh_solid() must rank below
    any candidate that kept the organic base body — regardless of score."""
    from cadx.agent.orchestrator import Candidate
    from cadx.llm.schemas import Critique

    good = Candidate(1, None, {}, Critique(score=70, passed=True), [])
    bad = Candidate(2, None, {}, Critique(score=90, passed=True), [], hybrid_missing=True)
    assert good.rank > bad.rank


def test_wrecked_hybrid_candidate_never_ranks_best():
    """Observed live: score sequence [80,0,80,80,0] exported the iteration-2
    candidate — a faceted score-0 wreck that happened to keep load_mesh_solid()
    while every healthier iteration had dropped it. Contract fidelity must only
    decide BETWEEN viable candidates; a critique-floor wreck with majors can
    never claim best."""
    from cadx.agent.orchestrator import Candidate
    from cadx.llm.schemas import Critique, Discrepancy

    def _majors(n: int) -> list[Discrepancy]:
        return [
            Discrepancy(
                what=f"defect {i}",
                where="arm",
                severity="major",
                fix_hint="fix it",
            )
            for i in range(n)
        ]

    viable_parametric = Candidate(
        1,
        None,
        {},
        Critique(score=80, passed=True, discrepancies=[]),
        [],
        hybrid_missing=True,
    )
    wrecked_mesh = Candidate(
        2,
        None,
        {},
        Critique(score=0, passed=False, discrepancies=_majors(5)),
        [],
        wrecked=True,
    )
    assert viable_parametric.rank > wrecked_mesh.rank
    # viability gate sits ABOVE the hybrid contract: a wrecked mesh-keeper
    # also loses against an equally wrecked parametric part on the fine keys,
    # and a viable mesh-keeper still outranks a viable parametric one
    viable_mesh = Candidate(4, None, {}, Critique(score=70, passed=True, discrepancies=[]), [])
    assert viable_mesh.rank > viable_parametric.rank


def test_feature_regression_ranks_lower():
    """Measured feature count is the LAST rank tiebreak now: it still splits
    otherwise-identical candidates, but it no longer outweighs dq/score —
    extra holes drilled through a gusset used to be rewarded (bracket run)."""
    from cadx.agent.orchestrator import Candidate
    from cadx.llm.schemas import Critique

    with_feats = Candidate(
        1,
        None,
        {"features": {"total_holes": 2, "rounds": [2.0, 2.0]}},
        Critique(score=40, passed=False),
        [],
    )
    with_feats.dq = 50
    dropped = Candidate(
        2,
        None,
        {"features": {"total_holes": 0}},
        Critique(score=40, passed=False),
        [],
    )
    dropped.dq = 50
    assert with_feats.measured_feature_count == 4
    assert dropped.measured_feature_count == 0
    assert with_feats.rank > dropped.rank
    # dq dominates the raw feature count: a measured-clean candidate beats
    # one that merely accumulated (possibly misplaced) features
    better = Candidate(
        3,
        None,
        {"features": {"total_holes": 0}},
        Critique(score=40, passed=False),
        [],
    )
    better.dq = 80
    assert better.rank > with_feats.rank
    # a fractured multi-solid part is a hard gate — ranked below any
    # single-solid candidate regardless of score
    fractured = Candidate(
        4,
        None,
        {"features": {"total_holes": 4}, "analysis": {"solid_count": 3}},
        Critique(score=95, passed=True),
        [],
    )
    fractured.dq = 30
    assert with_feats.rank > fractured.rank


async def test_cad_fallback_strips_mesh_plan(monkeypatch, tmp_path):
    """F4: a mesh->cad fallback must not hand the mesh plan to codegen —
    steps mentioning load_mesh_solid()/image-to-3d are stripped, mesh fields
    cleared, approach rewritten to cad."""
    import json

    from cadx.agent.steps import mesh_refine as mesh_refine_step

    async def always_regen(*a, **kw):
        return mesh_refine_step.RefineResult(regenerate=True, reasoning="flat slab")

    monkeypatch.setattr(mesh_refine_step, "run", always_regen)

    spec = dict(MESH_SPEC)
    spec["width_y"] = {"value": 40.0, "unit": "mm", "source": "stated"}
    spec["height_z"] = {"value": 30.0, "unit": "mm", "source": "stated"}
    mesh_plan = {
        **MESH_PLAN,
        "steps": [
            "1. generate base mesh from the reference image (image-to-3d)",
            "2. call load_mesh_solid() and cut the bore",
        ],
        "mesh_up_hint": "dome up",
    }
    fail_crit = {
        "score": 20,
        "passed": False,
        "discrepancies": [
            {
                "what": "wrong shape vs the stated part",
                "where": "overall",
                "severity": "major",
                "source": "spec",
                "fix_hint": "reshape the body",
            }
        ],
        "matches": [],
        "confidence": 0.9,
    }
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            spec,
            mesh_plan,
            MESH_ORIENT,
            fail_crit,
            fail_crit,
            fail_crit,
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
        assets=[_img(seed=1)],
        mesh=FakeMeshClient(),
        mesh_available=True,
        prompt="40x40x30 block",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=3),
    )
    outcome = await run_generation(ctx)
    assert outcome.resolved_mode == "cad"
    assert ctx.plan is not None and ctx.plan.approach == "cad"
    assert ctx.plan.mesh_subject == "" and ctx.plan.mesh_up_hint == ""
    # the codegen prompt embeds the plan JSON — the plan must carry no
    # mesh-only instructions (code_contract.md still documents the helper
    # generically; the plan must not instruct its use)
    plan_calls = [c for c in llm.calls if "MODELING PLAN" in json.dumps(c["messages"])]
    assert plan_calls
    for c in plan_calls:
        blob = json.dumps(c["messages"])
        plan_json = blob.split("MODELING PLAN (JSON):", 1)[1]
        assert "load_mesh_solid" not in plan_json
        assert "image-to-3d" not in plan_json
    # the persisted plan is the rewritten cad one
    assert outcome.plan and outcome.plan.get("approach") == "cad"


def test_cad_fallback_plan_unit():
    """Deterministic plan rewrite: mesh steps dropped, hybrid features kept."""
    from cadx.agent.steps.plan import cad_fallback_plan

    plan = ModelingPlan(
        approach="hybrid",  # type: ignore[arg-type]
        approach_reason="organic base + precise holes",
        strategy=["hybrid"],  # type: ignore[list-item]
        mesh_subject="domed knob",
        mesh_source_images=["a1"],
        mesh_up_hint="dome up",
        hybrid_cad_features=["cut M4 through-hole"],
        steps=[
            "1. generate organic mesh via image-to-3d",
            "2. load_mesh_solid() then cut holes",
            "3. add the top chamfer",
        ],
    )
    fb = cad_fallback_plan(plan)
    assert fb is not None
    assert fb.approach == "cad" and fb.strategy == ["cad"]
    assert "cad fallback" in fb.approach_reason
    assert fb.steps[0].startswith("0. CAD fallback")
    assert any("top chamfer" in s for s in fb.steps)
    assert any("M4 through-hole" in s for s in fb.steps)
    assert not any("mesh" in s.lower() for s in fb.steps[1:])
    assert fb.mesh_subject == "" and fb.mesh_source_images == [] and fb.mesh_up_hint == ""
    # original untouched
    assert plan.approach == "hybrid" and plan.mesh_subject == "domed knob"
    assert cad_fallback_plan(None) is None
