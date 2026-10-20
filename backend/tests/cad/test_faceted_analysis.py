"""Faceted mesh solids must not trigger unbounded OCC feature recognition.

Observed: a 40k-face hybrid solid hung the executor's analyze step for 448s
(pipeline timeout) while BRepCheck + per-face BRepAdaptor scans found nothing
a sewn mesh couldn't already report. These tests pin the fast path.
"""

import trimesh

from cadx.cad.analysis import _FACET_FACE_CAP, analyze
from cadx.cad.features import extract_features
from cadx.cad.mesh_ops import mesh_to_solid


def _faceted_solid():
    # 5120 triangular faces — well past _FACET_FACE_CAP
    return mesh_to_solid(trimesh.creation.icosphere(subdivisions=4, radius=20))


def test_faceted_analysis_skips_heavy_scans():
    solid = _faceted_solid()
    assert len(solid.faces()) > _FACET_FACE_CAP

    a = analyze(solid)
    assert a.is_valid  # sewn single solid, no BRepCheck grind
    assert a.solid_count == 1
    assert a.cylindrical_faces == []
    assert any("faceted" in w for w in a.warnings)
    # mass properties still reported
    assert a.volume > 0
    assert a.surface_area > 0


def test_faceted_feature_extraction_degrades_to_approximate():
    feats = extract_features(_faceted_solid())
    # all faces are PLANE — the analytic scan must not run; the result is
    # flagged approximate so callers treat it as mesh-derived
    assert feats.approximate
    assert feats.total_holes == 0
    assert feats.hole_groups == []


def test_attached_source_mesh_supplies_mass_props():
    solid = _faceted_solid()
    m = trimesh.creation.icosphere(subdivisions=4, radius=20)
    solid._cadx_mesh = m  # noqa: SLF001 — same attachment the runner performs
    a = analyze(solid)
    assert a.volume == round(float(m.volume), 3)
    assert a.surface_area == round(float(m.area), 3)
