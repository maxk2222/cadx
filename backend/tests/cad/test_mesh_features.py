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


def _bored_body(axis: str) -> trimesh.Trimesh:
    """Cylinder grip with a longitudinal Ø10 through-bore along `axis`."""
    import numpy as np

    body = trimesh.creation.cylinder(radius=12, height=80, sections=48)
    bore = trimesh.creation.cylinder(radius=5, height=120, sections=32)
    if axis == "x":
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        body.apply_transform(rot)
        bore.apply_transform(rot)
    elif axis == "y":
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        body.apply_transform(rot)
        bore.apply_transform(rot)
    mesh = body.difference(bore, engine="manifold")
    assert mesh.is_watertight
    return mesh


def test_tunnel_dia_survives_grazing_backdrop():
    """A bore measured through its host wall must read full size even when
    unrelated geometry grazes its full-part sightline — a boss rim behind
    the flange bore shrank a Ø6.4 bore to Ø5.6 (robot-arm-mount eval)."""
    import numpy as np

    # 8mm flange plate in the YZ plane at x=42..50, Ø6.5 bore along X at (y=0, z=0)
    flange = trimesh.creation.box(extents=[8, 40, 30])
    flange.apply_translation([46, 0, 0])
    bore = trimesh.creation.cylinder(radius=3.25, height=30, sections=48)
    bore.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    bore.apply_translation([46, 0, 0])
    part = flange.difference(bore, engine="manifold")
    # unrelated boss behind the flange whose rim grazes the bore's lower
    # sightline — the old whole-part no-hit test read the bore ~14% small
    backdrop = trimesh.creation.cylinder(radius=3.0, height=10, sections=32)
    backdrop.apply_translation([0, 0, -4.9])  # rim just inside the bore's shadow edge
    mesh = trimesh.util.concatenate([part, backdrop])
    feats = detect_mesh_features(mesh)
    tunnels = [h for h in feats.holes if h.kind == "tunnel" and h.axis == "x"]
    assert tunnels, "bore not detected"
    bore_h = min(tunnels, key=lambda h: abs(h.dia - 6.5))
    assert bore_h.dia == pytest.approx(6.5, abs=0.7)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_detect_tunnel_each_axis(axis):
    """T8.1: through-bores must be found on every principal axis — a Z-only
    pass made the refine agent undo a correct X-axis bore (no evidence)."""
    feats = detect_mesh_features(_bored_body(axis))
    tunnels = [h for h in feats.holes if h.kind == "tunnel"]
    matching = [h for h in tunnels if h.axis == axis]
    assert len(matching) >= 1, f"no tunnel detected along {axis}"
    assert matching[0].dia == pytest.approx(10.0, abs=2.0)
