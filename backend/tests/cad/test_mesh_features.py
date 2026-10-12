"""Tests for cadx.cad.mesh_features — deterministic mesh feature detection."""

from __future__ import annotations

import pytest
import trimesh

from cadx.cad.mesh_features import detect_mesh_features


def _handle_like() -> trimesh.Trimesh:
    """100x30x8 plate, two Ø5 through-holes at x=+-30, one dimple at origin."""
    out = trimesh.creation.box(extents=[100, 30, 8])
    out.apply_translation([0, 0, 4])
    for x in (-30.0, 30.0):
        cyl = trimesh.creation.cylinder(radius=2.5, height=30)
        cyl.apply_translation([x, 0, 4])
        out = out.difference(cyl, engine="manifold")
    ball = trimesh.creation.icosphere(subdivisions=3, radius=4.0)
    ball.apply_translation([0, 0, 8])
    return out.difference(ball, engine="manifold")


def test_detect_mesh_features():
    mesh = _handle_like()
    assert mesh.is_watertight
    feats = detect_mesh_features(mesh)
    tunnels = [h for h in feats.holes if h.kind == "tunnel"]
    dimples = [h for h in feats.holes if h.kind == "dimple"]
    assert len(tunnels) == 2
    assert [h.x for h in tunnels] == pytest.approx([-30, 30], abs=0.6)
    for h in tunnels:
        assert h.dia == pytest.approx(5.0, abs=1.0)
        assert abs(h.y) < 1.5
    assert len(dimples) >= 1
    d0 = dimples[0]
    assert d0.x == pytest.approx(0, abs=0.8)
    assert d0.y == pytest.approx(0, abs=0.8)
    xs = [p[0] for p in feats.outline_mm]
    ys = [p[1] for p in feats.outline_mm]
    assert max(xs) - min(xs) == pytest.approx(100, abs=1.0)
    assert max(ys) - min(ys) == pytest.approx(30, abs=1.0)


def test_detect_mesh_features_solid_block():
    mesh = trimesh.creation.box(extents=[20, 20, 20])
    feats = detect_mesh_features(mesh)
    assert feats.holes == []
    assert len(feats.outline_mm) >= 4
