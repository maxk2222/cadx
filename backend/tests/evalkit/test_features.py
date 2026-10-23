"""Lazy feature backfill: StageOutcome.features from STEP/mesh artifacts."""

from __future__ import annotations

import pytest

from cadx.evalkit.features import ensure_features, needs_features
from cadx.evalkit.models import StageOutcome

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("numpy")


def test_needs_features_gate_types():
    assert needs_features([{"type": "holes", "count": 2}])
    assert needs_features([{"type": "fillet"}])
    assert needs_features([{"type": "features", "entries": []}])
    assert not needs_features([{"type": "status"}, {"type": "bbox"}])
    assert not needs_features(None)


def test_ensure_features_noop_when_set():
    o = StageOutcome(status="succeeded", features={"hole_groups": []})
    import asyncio

    asyncio.run(ensure_features(o))
    assert o.features == {"hole_groups": []}
    assert o.features_available is True


@pytest.mark.asyncio
async def test_ensure_features_mesh_fallback(tmp_path):
    """STL-only artifact -> mesh tunnel detection fills features."""
    # plate with a clear Z-axis bore
    plate = trimesh.creation.box(extents=(40, 40, 8))
    hole = trimesh.creation.cylinder(radius=4, height=20, sections=48)
    mesh = plate.difference(hole)
    p = tmp_path / "plate.stl"
    mesh.export(p)
    o = StageOutcome(status="succeeded", artifact_files=[str(p)])
    await ensure_features(o)
    assert o.features, "mesh fallback produced no features"
    groups = o.features.get("hole_groups") or []
    assert any(abs(float(g.get("diameter") or 0) - 8.0) < 1.5 for g in groups), groups


@pytest.mark.asyncio
async def test_ensure_features_step_path(tmp_path, monkeypatch):
    """STEP artifact -> extract_features (BRep) path, resolved via storage."""
    build123d = pytest.importorskip("build123d")
    from cadx.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        shape = build123d.Box(40, 40, 8) - build123d.Cylinder(radius=4, height=20)
        exports = tmp_path / "exports"
        exports.mkdir()
        step = exports / "model.step"
        build123d.export_step(shape, str(step))
        o = StageOutcome(status="succeeded", artifacts={"step|": "exports/model.step"})
        await ensure_features(o)
        assert o.features, "STEP measurement produced no features"
        assert o.features.get("approximate") is False
        groups = o.features.get("hole_groups") or []
        assert any(abs(float(g.get("diameter") or 0) - 8.0) < 1.0 for g in groups), groups
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_ensure_features_marks_missing_artifact_unavailable():
    o = StageOutcome(status="timeout", version_id="v-timeout")
    await ensure_features(o)
    assert o.features is None
    assert o.features_available is False
    assert "no STEP/STL artifact" in (o.features_error or "")


@pytest.mark.asyncio
async def test_ensure_features_marks_extraction_failure_unavailable(tmp_path, monkeypatch):
    from cadx.evalkit import features as feature_module

    artifact = tmp_path / "broken.stl"
    artifact.write_text("not a mesh")
    monkeypatch.setattr(
        feature_module,
        "_measure_mesh",
        lambda _outcome: (_ for _ in ()).throw(RuntimeError("bad mesh")),
    )
    o = StageOutcome(status="succeeded", artifact_files=[str(artifact)])
    await ensure_features(o)
    assert o.features is None
    assert o.features_available is False
    assert "RuntimeError: bad mesh" in (o.features_error or "")
