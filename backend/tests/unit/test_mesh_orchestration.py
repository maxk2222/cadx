"""Workstream-E mesh orchestration: synthetic side views, Omni bbox control,
deterministic silhouette ops, the pre-critique rejection gate and the
2-iteration stagnation patience."""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import pytest

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.mesh_client import FakeMeshClient, MeshHealth, MeshModelInfo
from cadx.agent.orchestrator import RunOutcome
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import ModelingPlan, PartSpec
from cadx.runtime_config import RuntimeSettings


def _ctx(**kw) -> AgentCtx:
    ctx_msgs: list[tuple[str, str]] = []

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        ctx_msgs.append((step, msg))

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=1))
    kw.setdefault("mode", "generate")
    kw.setdefault("prompt", "x")
    if kw["settings"].mesh_seed is None:
        kw["settings"].mesh_seed = 4242
    rs = kw.get("runtime_settings") or RuntimeSettings()
    kw["runtime_settings"] = rs.model_copy(update={"agent_max_wall_s": 86_400})
    ctx = AgentCtx(
        llm=kw.pop("llm", None) or FakeLLM([]),
        executor=kw.pop("executor", None) or InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        **kw,
    )
    ctx._test_msgs = ctx_msgs  # type: ignore[attr-defined]
    return ctx


def _img(seed: int = 7) -> InputAsset:
    """Opaque test image (no alpha mask, rembg never runs in unit tests)."""
    import random

    from PIL import Image, ImageDraw

    aid = str(uuid.uuid4())
    p = Path(tempfile.mkdtemp(prefix="cadx-img-")) / f"{aid[:8]}.png"
    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64))
    dr = ImageDraw.Draw(img)
    for gy in range(0, 64, 8):
        for gx in range(0, 64, 8):
            v = rng.choice((30, 225))
            dr.rectangle([gx, gy, gx + 8, gy + 8], fill=(v, v, v))
    img.save(p)
    return InputAsset(asset_id=aid, kind="image", path=p, filename=p.name)


def _spec(dims: tuple[float, float, float], **kw) -> PartSpec:
    return PartSpec(
        name=kw.pop("name", "part"),
        summary=kw.pop("summary", ""),
        length_x={"value": dims[0], "unit": "mm", "source": "stated"},
        width_y={"value": dims[1], "unit": "mm", "source": "stated"},
        height_z={"value": dims[2], "unit": "mm", "source": "stated"},
        features=kw.pop("features", []),
        **kw,
    )


def _plan() -> ModelingPlan:
    return ModelingPlan(approach="mesh", approach_reason="r", mesh_subject="knob")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Omni bbox control


class _OmniMesh(FakeMeshClient):
    async def health(self) -> MeshHealth:
        return MeshHealth(
            ok=True,
            device="fake",
            models=[
                MeshModelInfo(
                    id="hunyuan3d-omni",
                    device="fake",
                    max_images=4,
                    supports_multiview=True,
                )
            ],
        )


async def test_omni_bbox_control_passed_to_client():
    from cadx.agent.orchestrator import _generate_mesh

    mesh = _OmniMesh()
    ctx = _ctx(
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(
            mesh_cache_enabled=False, mesh_prep_images=False, agent_max_wall_s=86_400
        ),
    )
    spec = _spec((40.0, 30.0, 20.0))
    rel = await _generate_mesh(ctx, 1, spec, _plan(), seed=7, model="hunyuan3d-omni")
    assert rel
    assert mesh.calls, "generator was not invoked"
    call = mesh.calls[-1]
    assert call["control_type"] == "bbox"
    assert call["control_input"] == [40.0, 30.0, 20.0]


async def test_bbox_control_not_sent_to_other_adapters():
    from cadx.agent.orchestrator import _generate_mesh

    mesh = FakeMeshClient()
    ctx = _ctx(
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(
            mesh_cache_enabled=False, mesh_prep_images=False, agent_max_wall_s=86_400
        ),
    )
    spec = _spec((40.0, 30.0, 20.0))
    rel = await _generate_mesh(ctx, 1, spec, _plan(), seed=7)
    assert rel
    assert mesh.calls[-1]["control_type"] is None
    assert mesh.calls[-1]["control_input"] is None


async def test_omni_bbox_control_in_batch():
    from cadx.agent.orchestrator import _generate_mesh_batch

    mesh = _OmniMesh()
    ctx = _ctx(
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(
            mesh_cache_enabled=False, mesh_prep_images=False, agent_max_wall_s=86_400
        ),
    )
    spec = _spec((40.0, 30.0, 20.0))
    rels = await _generate_mesh_batch(
        ctx,
        1,
        spec,
        _plan(),
        seeds=[11, 22],
        steps=5,
        resolution=256,
        model="hunyuan3d-omni",
        tag_prefix="_cand",
    )
    assert any(rels)
    assert mesh.calls
    assert all(c["control_type"] == "bbox" for c in mesh.calls)
    assert all(c["control_input"] == [40.0, 30.0, 20.0] for c in mesh.calls)


# --------------------------------------------------------------------------- #
# synthetic side views


async def test_synth_views_assigned_left_right_roles(monkeypatch, tmp_path):
    from PIL import Image

    from cadx.agent import orchestrator as orch

    async def fake_gen(ctx, prompt, *, count=1, **kw):
        out = []
        for i in range(count):
            p = tmp_path / f"synth{i}.png"
            Image.new("RGB", (64, 64), (200, 200, 200)).save(p)
            a = InputAsset(asset_id=f"synth{i}", kind="generated_image", path=p)
            ctx.assets.append(a)
            out.append(a)
        return out

    monkeypatch.setattr("cadx.agent.concept_image.generate_concept_images", fake_gen)

    ctx = _ctx(assets=[_img()], image=object(), image_available=True)
    spec = _spec((40.0, 30.0, 35.0))  # non-flat
    await orch._ensure_synth_views(ctx, 1, spec)

    images, roles = orch._mesh_source_images(ctx, 1, spec)
    assert len(images) == 3
    assert roles == ["front", "left", "right"]
    synth = [a for a in ctx.assets if a.role_hint == "synth_view"]
    assert len(synth) == 2
    assert all(a.caption.startswith("synth:") for a in synth)
    # runs once per run — a second call must not generate again
    await orch._ensure_synth_views(ctx, 1, spec)
    assert sum(1 for a in ctx.assets if a.role_hint == "synth_view") == 2


async def test_synth_views_skipped_when_flat_or_multi(monkeypatch, tmp_path):
    from cadx.agent import orchestrator as orch

    calls = []

    async def fake_gen(ctx, prompt, **kw):
        calls.append(prompt)
        return []

    monkeypatch.setattr("cadx.agent.concept_image.generate_concept_images", fake_gen)

    # flat part: the measured-outline 'back' synth covers that case instead
    ctx = _ctx(assets=[_img()], image=object(), image_available=True)
    await orch._ensure_synth_views(ctx, 1, _spec((100.0, 80.0, 5.0)))
    assert calls == []

    # more than one input image — no synthesis
    ctx = _ctx(assets=[_img(1), _img(2)], image=object(), image_available=True)
    await orch._ensure_synth_views(ctx, 1, _spec((40.0, 30.0, 35.0)))
    assert calls == []

    # generation failure is tolerated (flagged done, no raise)
    async def boom(ctx, prompt, **kw):
        raise RuntimeError("image backend down")

    monkeypatch.setattr("cadx.agent.concept_image.generate_concept_images", boom)
    ctx = _ctx(assets=[_img()], image=object(), image_available=True)
    await orch._ensure_synth_views(ctx, 1, _spec((40.0, 30.0, 35.0)))
    assert ctx.mesh_info["_synth_views_done"] is True
    assert not ctx.mesh_info.get("_synth_view_roles")


async def test_synth_views_never_used_as_reference_mask(monkeypatch, tmp_path):
    from PIL import Image

    from cadx.agent import orchestrator as orch

    async def fake_gen(ctx, prompt, *, count=1, **kw):
        out = []
        for i in range(count):
            p = tmp_path / f"synth{i}.png"
            Image.new("RGB", (64, 64), (200, 200, 200)).save(p)
            a = InputAsset(asset_id=f"synth{i}", kind="generated_image", path=p)
            ctx.assets.append(a)
            out.append(a)
        return out

    monkeypatch.setattr("cadx.agent.concept_image.generate_concept_images", fake_gen)
    seen: list[str] = []

    def fake_mask(p):
        seen.append(Path(p).name)

    monkeypatch.setattr("cadx.cad.silhouette.reference_mask", fake_mask)

    hero = _img()
    ctx = _ctx(assets=[hero], image=object(), image_available=True)
    spec = _spec((40.0, 30.0, 35.0))
    await orch._ensure_synth_views(ctx, 1, spec)
    # a measured-outline back render must also never feed the reference mask
    ctx.assets.append(
        InputAsset(
            asset_id="_synth_back",
            kind="generated_image",
            path=tmp_path / "back.png",
            caption="synthetic bottom silhouette",
        )
    )
    mask, rel = orch._ref_side_mask(ctx, 1, spec)
    assert mask is None and rel is None
    assert seen == [hero.path.name]


# --------------------------------------------------------------------------- #
# deterministic silhouette ops


def _circle_mask(size: int = 64):
    import numpy as np

    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2.0
    return (xx - c) ** 2 + (yy - c) ** 2 < (size * 0.4) ** 2


def test_spec_silhouette_ops_revolve_emits_radial_profile():
    from cadx.agent.orchestrator import _spec_silhouette_ops

    pytest.importorskip("skimage")
    ctx = _ctx()
    ctx.mesh_info["_ref_mask"] = _circle_mask()
    spec = _spec((40.0, 40.0, 30.0), name="knob", summary="round cylindrical knob")
    res = {
        "analysis": {"bbox_size": [40.0, 40.0, 30.0]},
        "mesh": {"silhouette_view": "+y", "silhouette_iou": 0.5, "features": {}},
    }
    ops = _spec_silhouette_ops(spec, ctx, res)
    assert len(ops) == 1
    assert ops[0].op == "radial_profile_auto"
    assert ops[0].args["axis"] == "z"
    assert 2 <= len(ops[0].args["stations"]) <= 32


def _wide_mask(size: int = 64, aspect: float = 5.0):
    """Wide subject mask — aspect consistent with a 100x20 side view."""
    import numpy as np

    m = np.zeros((size, size), dtype=bool)
    h = int(size * 0.7 / aspect)
    w = int(size * 0.7)
    y0, x0 = (size - h) // 2, (size - w) // 2
    m[y0 : y0 + h, x0 : x0 + w] = True
    return m


def test_spec_silhouette_ops_generic_part_carves():
    from cadx.agent.orchestrator import _spec_silhouette_ops

    pytest.importorskip("skimage")
    ctx = _ctx()
    ctx.mesh_info["_ref_mask"] = _wide_mask(aspect=5.0)
    spec = _spec((100.0, 40.0, 20.0), name="housing", summary="rectangular housing")
    res = {
        "analysis": {"bbox_size": [100.0, 40.0, 20.0]},
        "mesh": {"silhouette_view": "+y", "silhouette_iou": 0.5, "features": {}},
    }
    ops = _spec_silhouette_ops(spec, ctx, res)
    assert len(ops) == 1
    assert ops[0].op == "silhouette_carve"
    assert ops[0].args["view"] == "+y"
    assert len(ops[0].args["outline"]) >= 3


def test_spec_silhouette_ops_rejects_aspect_mismatched_mask():
    """A mask whose subject aspect is wildly off the measured view-plane
    aspect is junk/mismatched evidence — emit nothing rather than trim the
    mesh into a wrong-aspect sliver."""
    from cadx.agent.orchestrator import _spec_silhouette_ops

    pytest.importorskip("skimage")
    ctx = _ctx()
    ctx.mesh_info["_ref_mask"] = _circle_mask()  # 1:1 subject vs 5:1 view
    spec = _spec((100.0, 40.0, 20.0), summary="rectangular housing")
    res = {
        "analysis": {"bbox_size": [100.0, 40.0, 20.0]},
        "mesh": {"silhouette_view": "+y", "silhouette_iou": 0.5, "features": {}},
    }
    assert _spec_silhouette_ops(spec, ctx, res) == []


def test_spec_silhouette_ops_conservative_without_evidence():
    from cadx.agent.orchestrator import _spec_silhouette_ops

    ctx = _ctx()
    spec = _spec((40.0, 40.0, 30.0), summary="round knob")
    res = {
        "analysis": {"bbox_size": [40.0, 40.0, 30.0]},
        "mesh": {"silhouette_view": "+y", "silhouette_iou": 0.5, "features": {}},
    }
    # no reference mask extracted -> no op
    assert _spec_silhouette_ops(spec, ctx, res) == []
    # no matched view -> no op
    ctx.mesh_info["_ref_mask"] = _circle_mask()
    res2 = {
        "analysis": {"bbox_size": [40.0, 40.0, 30.0]},
        "mesh": {"silhouette_view": None, "silhouette_iou": 0.0, "features": {}},
    }
    assert _spec_silhouette_ops(spec, ctx, res2) == []
    # a non-round spec without measured roundness -> no revolve axis
    spec_sq = _spec((40.0, 40.0, 30.0), summary="square box")
    assert _spec_silhouette_ops(spec_sq, ctx, res)[0].op == "silhouette_carve"


# --------------------------------------------------------------------------- #
# pre-critique rejection gate


class _BadMeshExec(InProcessExecutor):
    """process_mesh always returns the same clearly-bad result: the shape
    misses the reference silhouette AND violates stated dims."""

    def __init__(self) -> None:
        self.calls = 0

    async def process_mesh(self, payload: dict) -> dict:
        self.calls += 1
        return {
            "ok": True,
            "analysis": {"bbox_size": [50.0, 20.0, 18.0]},
            "mesh": {
                "silhouette_iou": 0.12,
                "silhouette_view": "+y",
                "features": {},
                "post": payload.get("post") or {},
                "dim_issues": [],
            },
        }


async def test_pregate_rejects_silhouette_dim_mismatch_without_critique():
    from cadx.agent.orchestrator import _run_mesh

    spec = _spec((100.0, 40.0, 40.0), name="block")
    plan = _plan()
    mesh = FakeMeshClient()
    executor = _BadMeshExec()
    llm = FakeLLM([{"up": "+z", "front": "+y", "confidence": 0.9, "reason": "x"}])
    ctx = _ctx(
        llm=llm,
        executor=executor,
        assets=[_img()],
        mesh=mesh,
        mesh_available=True,
        runtime_settings=RuntimeSettings(
            mesh_cache_enabled=False,
            mesh_prep_images=False,
            mesh_candidates_per_iter=1,
            mesh_heightfield_flat=False,
            agent_mesh_hard_fail_limit=2,
            agent_max_wall_s=86_400,
        ),
    )
    ctx.spec = spec
    ctx.plan = plan
    outcome = RunOutcome()

    async def no() -> bool:
        return False

    outcome = await _run_mesh(
        ctx,
        outcome,
        spec,
        plan,
        time.monotonic(),
        no,
        no,
        max_iterations=5,
        pass_score=70,
        tol_pct=5.0,
        tol_mm=0.5,
    )
    msgs = ctx._test_msgs  # type: ignore[attr-defined]
    rejects = [m for s, m in msgs if "rejected by pre-gate" in m]
    assert len(rejects) == 2
    assert any("silhouette IoU" in m for m in rejects)
    # the critique step never ran — the candidate was rejected deterministically
    assert not any(s == "critique" for s, _ in msgs)
    assert outcome.fallback_to_cad is True  # dimensioned part
    assert outcome.error


def test_stall_patience_default_is_two():
    """Workstream-E: mesh loop gives up after 2 improvement-free iterations
    (v7 observation: nothing improved after iteration 3 anyway)."""
    assert RuntimeSettings().agent_stall_patience == 2
