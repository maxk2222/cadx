"""mesh_boolean — manifold-backed booleans, sewing, unify, anchor re-cut."""

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, Pos

from cadx.cad import mesh_boolean as mb
from cadx.cad.mesh_ops import mesh_to_solid


@pytest.fixture
def sphere_solid():
    """Faceted sphere solid (r=20 resting on XY) with its source mesh attached."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=20)
    m.apply_translation([0, 0, 20])
    return mb.attach_mesh(mesh_to_solid(m), m)


def test_sew_and_unify_merges_coplanar_facets():
    m = trimesh.creation.box(extents=[30, 20, 10])
    m.apply_translation([0, 0, 5])
    solid = mb.sew_mesh(m)
    assert len(solid.solids()) == 1
    uni = mb.unify_same_domain(solid)
    # 12 triangles -> 6 planar faces, volume preserved
    assert len(uni.faces()) == 6
    assert abs(uni.volume - solid.volume) / solid.volume < 0.01


def test_unify_checked_rejects_drift():
    b = Box(10, 10, 10)
    assert mb.unify_checked(b).volume == pytest.approx(b.volume)


def test_boolean_cut_uses_attached_mesh(sphere_solid):
    tool = Pos(0, 0, -10) * Cylinder(radius=4, height=80)
    out = mb.boolean(sphere_solid, tool, "cut")
    assert len(out.solids()) == 1
    removed = sphere_solid.volume - out.volume
    assert 1400 < removed < 2500  # Ø8 bore through ~r20 sphere
    # the manifold mesh must ride along for the next op in the chain
    assert getattr(out, mb.MESH_ATTR) is not None


def test_boolean_chain_reuses_mesh(sphere_solid):
    """Second op in a chain must not re-tessellate the sewn solid."""
    r1 = mb.boolean(sphere_solid, Pos(0, 0, -10) * Cylinder(radius=4, height=80), "cut")
    r2 = mb.boolean(r1, Pos(10, 0, 20) * Box(8, 8, 30), "cut")
    assert len(r2.solids()) == 1
    assert r2.volume < r1.volume


def test_boolean_fuse_and_intersect(sphere_solid):
    # box straddles z=0 — its lower half adds ~200 mm³ under the sphere base
    fused = mb.boolean(sphere_solid, Pos(0, 0, 0) * Box(10, 10, 4), "fuse")
    assert fused.volume > sphere_solid.volume
    inter = mb.boolean(sphere_solid, Pos(0, 0, 20) * Box(60, 60, 8), "intersect")
    assert len(inter.solids()) == 1
    assert inter.volume < sphere_solid.volume


def test_boolean_refuses_oversize_unbacked_body(sphere_solid):
    """A huge sewn shape without an attached mesh must raise (the cadlib
    caller then falls back to OCC) instead of spending minutes tessellating."""
    import cadx.cad.mesh_boolean as m

    monkey = m.MAX_FACES_FOR_TESS
    m.MAX_FACES_FOR_TESS = 10
    try:
        bare = mesh_to_solid(trimesh.creation.icosphere(subdivisions=3, radius=20))
        with pytest.raises(ValueError, match="tessellation"):
            mb.boolean(bare, Box(1, 1, 1), "cut")
    finally:
        m.MAX_FACES_FOR_TESS = monkey


def test_tunnel_open_probe():
    b = Box(20, 20, 10)
    assert not mb.tunnel_open(b, 0, 0, "+z")
    holed = b - Pos(0, 0, -1) * Cylinder(radius=2.5, height=12)
    assert mb.tunnel_open(holed, 0, 0, "+z")
    # off-axis from the hole the material still blocks
    assert not mb.tunnel_open(holed, 6, 6, "+z")


def test_recut_missing_hole():
    b = Box(20, 20, 10)
    out, applied = mb.recut_missing_holes(
        b, [{"x": 0, "y": 0, "dia": 5.0, "axis": "+z", "kind": "hole"}]
    )
    assert applied
    assert mb.tunnel_open(out, 0, 0, "+z")
    assert out.volume == pytest.approx(b.volume - np.pi * 2.5**2 * 10, rel=0.02)


def test_recut_skips_open_and_nonauthoritative():
    b = Box(20, 20, 10) - Pos(0, 0, -1) * Cylinder(radius=2.5, height=12)
    # already open → untouched
    out, applied = mb.recut_missing_holes(
        b, [{"x": 0, "y": 0, "dia": 5.0, "axis": "+z", "kind": "hole"}]
    )
    assert applied == [] and out is b
    # image-approx anchors are hints, not drilling orders
    out2, applied2 = mb.recut_missing_holes(
        b, [{"x": 6, "y": 6, "dia": 5.0, "axis": "+z", "source": "image-approx"}]
    )
    assert applied2 == [] and out2 is b
    # dimple anchors never force a bore
    out3, applied3 = mb.recut_missing_holes(
        b, [{"x": 6, "y": 6, "dia": 5.0, "axis": "+z", "kind": "dimple"}]
    )
    assert applied3 == [] and out3 is b


def test_recut_spec_anchor():
    """Spec-sourced anchors carry declared intent: a spec bore missing from
    the geometry must be cut even though no image/mesh evidence exists."""
    b = Box(20, 20, 10)
    out, applied = mb.recut_missing_holes(
        b, [{"x": 3, "y": -4, "dia": 5.0, "axis": "+z", "kind": "hole", "source": "spec"}]
    )
    assert applied
    assert mb.tunnel_open(out, 3, -4, "+z")
    # spec-enriched measured anchors ("mesh+spec" etc.) are authoritative too
    out2, applied2 = mb.recut_missing_holes(
        b, [{"x": 3, "y": -4, "dia": 5.0, "axis": "+z", "source": "image-approx+mesh+spec"}]
    )
    assert applied2
    assert mb.tunnel_open(out2, 3, -4, "+z")


def test_recut_blind_depth():
    b = Box(20, 20, 10)
    out, applied = mb.recut_missing_holes(
        b,
        [{"x": 5, "y": 5, "dia": 4.0, "axis": "+z", "kind": "hole", "depth_mm": 4.0}],
    )
    assert applied
    assert not mb.tunnel_open(out, 5, 5, "+z")
    removed = b.volume - out.volume
    assert removed == pytest.approx(np.pi * 4.0 * 4.0, rel=0.15)  # r²·π·4mm


def test_recut_blind_depth_centred_on_axial():
    """Anchor pos[axial] centres a bounded re-cut on the host plate — the
    envelope's axis-sign face would miss a plate sitting at the other end
    (bracket leg at x=-60..-55 vs bbox x max=+60)."""
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    part = leg + arm
    out, applied = mb.recut_missing_holes(
        part,
        [
            {
                "x": 0.0,
                "y": 0.0,
                "dia": 5.5,
                "axis": "+x",
                "kind": "hole",
                "source": "spec",
                "depth_mm": 5.0,
                "pos": [-57.5, 0.0, 15.0],
            }
        ],
    )
    assert applied
    # bore landed inside the leg (material removed there), not at +x end
    removed = part.volume - out.volume
    assert removed == pytest.approx(np.pi * 2.75**2 * 5.0, rel=0.15)
    # the ray along +x at (y=0, z=15) is now open — the leg got the bore
    assert mb.tunnel_open(out, 0.0, 15.0, "+x")
    assert not mb.tunnel_open(part, 0.0, 15.0, "+x")  # sanity: was blocked before


def test_recut_bounded_does_not_pierce_strut_on_same_ray():
    """Live hybrid failure: the leg bore at z=70 shares its +x ray with the
    diagonal strut (material at x≈5..10 there). A bounded anchor
    (depth_mm = host plate span) must drill the leg only — the unbounded
    cutter pierced the strut and produced a phantom bore."""
    leg = Pos(-57.5, 0, 50) * Box(5, 30, 100)
    arm = Pos(0, 0, 97.5) * Box(120, 30, 5)
    from cadx.cad import cadlib

    part = leg + arm + cadlib.strut((-55, 0, 35), (15, 0, 95), 10, 6)
    strut_probe = (-14.0, 0.0, 70.0)
    assert part.is_inside(strut_probe)  # strut material on the bore ray
    out, applied = mb.recut_missing_holes(
        part,
        [
            {
                "x": -57.5,
                "y": 0.0,
                "dia": 5.5,
                "axis": "+x",
                "kind": "hole",
                "source": "spec",
                "depth_mm": 5.0,
                "pos": [-57.5, 0.0, 70.0],
            }
        ],
    )
    assert applied
    assert not out.is_inside((-57.5, 0.0, 70.0))  # leg bored
    assert out.is_inside(strut_probe)  # strut intact — the bug pierced it


def test_cadlib_backend_injection(sphere_solid):
    """cadlib.mesh_cut must route to the injected backend when OCC output is
    unusable — and keep working identically when OCC succeeds."""
    from cadx.cad import cadlib

    calls = []

    def recording_backend(a, b, kind):
        calls.append(kind)
        return mb.boolean(a, b, kind)

    cadlib.ops.set_boolean_backend(recording_backend)
    try:
        out = cadlib.mesh_cut(sphere_solid, Pos(0, 0, -10) * Cylinder(radius=4, height=80))
        assert len(out.solids()) == 1
        # OCC succeeded — backend may or may not have been consulted. The
        # fuse box straddles the sphere base so it adds real material (a
        # tangent tool is a no-op and raises under the new guard).
        out2 = cadlib.mesh_fuse(out, Pos(0, 0, 0) * Box(10, 10, 4))
        assert len(out2.solids()) == 1
        assert out2.volume > out.volume
    finally:
        cadlib.ops.set_boolean_backend(None)


def test_cadlib_mesh_cut_rescue_via_backend(sphere_solid, monkeypatch):
    """When the OCC boolean raises, the injected mesh backend must still
    deliver the cut — that is the entire point of the rescue lane."""
    from build123d import Shape

    from cadx.cad import cadlib

    calls = []

    def backend(a, b, kind):
        calls.append(kind)
        return mb.boolean(a, b, kind)

    def boom(self, other):
        raise RuntimeError("simulated OCC failure")

    cadlib.ops.set_boolean_backend(backend)
    monkeypatch.setattr(Shape, "__sub__", boom)
    try:
        out = cadlib.mesh_cut(sphere_solid, Pos(0, 0, -10) * Cylinder(radius=4, height=80))
        assert calls == ["cut"]
        assert len(out.solids()) == 1
        assert out.volume < sphere_solid.volume
    finally:
        cadlib.ops.set_boolean_backend(None)


def test_tunnel_open_negative_axis():
    """-z probes must start BELOW the part and travel up — the old ray
    always started above and reported every -z bore as open."""
    b = Pos(0, 0, 10) * Box(20, 20, 10)
    assert not mb.tunnel_open(b, 0, 0, "-z")
    holed = b - Pos(0, 0, -1) * Cylinder(radius=2.5, height=30)
    assert mb.tunnel_open(holed, 0, 0, "-z")


def test_tunnel_open_elevated_part():
    """A part floating above z=0 (or offset in x) must not read as open —
    the old probe projected hit points onto the direction from the world
    origin, so every hit beyond the span was filtered away (observed: all
    spec anchors skipped on a 120x40x30 mesh handle)."""
    b = Pos(0, 0, 15) * Box(20, 20, 10)  # z in [10, 20]
    assert not mb.tunnel_open(b, 0, 0, "+z")
    off = Pos(40, 0, 5) * Box(10, 10, 10)  # x in [35, 45]
    assert not mb.tunnel_open(off, 40, 0, "+z")
    assert mb.tunnel_open(off, 0, 0, "+z")  # ray at x=0 misses the part
