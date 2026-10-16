"""Tests for cad/mesh_repair_pml — optional PyMeshLab repair engine.

Every pymeshlab code path skips via pytest.importorskip so CI without the
'repair' extra stays green; the fallback tests mock `pm_available` and need
no pymeshlab at all.
"""

import sys

import numpy as np
import pytest
import trimesh

from cadx.cad import mesh_ops
from cadx.cad import mesh_repair_pml as pml_mod


def _sphere_open() -> trimesh.Trimesh:
    """Icosphere with a hole cut out of the top — open but otherwise clean."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    mask = np.asarray(m.triangles_center)[:, 2] < 8.5
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    assert not m.is_watertight
    return m


def _block_clean_stage(m: trimesh.Trimesh, monkeypatch) -> None:
    """Force the mesh through the deep rungs of _ensure_watertight:
    fill_holes raises and manifold3d is 'not installed'."""
    monkeypatch.setattr(
        trimesh.repair, "fill_holes", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setitem(sys.modules, "manifold3d", None)


# --------------------------------------------------------------------------- #
# availability probe (no pymeshlab required)


def test_pm_available_returns_bool():
    assert isinstance(pml_mod.pm_available(), bool)


def test_pm_available_matches_import():
    try:
        import pymeshlab  # noqa: F401

        expected = True
    except ImportError:
        expected = False
    assert pml_mod.pm_available() is expected


def test_functions_raise_without_pymeshlab(monkeypatch):
    """Every public function must fail loudly when pymeshlab is missing."""
    monkeypatch.setattr(pml_mod, "_pml", None)
    monkeypatch.setattr(pml_mod, "_pml_tried", True)
    box = trimesh.creation.box(extents=[10, 10, 10])
    with pytest.raises(RuntimeError):
        pml_mod.alpha_wrap(box)
    with pytest.raises(RuntimeError):
        pml_mod.isotropic_remesh(box, 1.0)
    with pytest.raises(RuntimeError):
        pml_mod.quadric_decimate(box, 50)
    with pytest.raises(RuntimeError):
        pml_mod.taubin_smooth(box, 3)
    with pytest.raises(RuntimeError):
        pml_mod.close_holes(box)
    with pytest.raises(RuntimeError):
        pml_mod.remove_non_manifold(box)


# --------------------------------------------------------------------------- #
# pymeshlab engine functions


def test_alpha_wrap_repairs_open_mesh():
    pytest.importorskip("pymeshlab")
    wrapped = pml_mod.alpha_wrap(_sphere_open())
    assert wrapped.is_watertight
    assert wrapped.is_winding_consistent
    assert len(wrapped.faces) > 0


def test_isotropic_remesh_watertight():
    pytest.importorskip("pymeshlab")
    box = trimesh.creation.box(extents=[20, 20, 20])
    out = pml_mod.isotropic_remesh(box, 1.0)
    assert out.is_watertight
    # mean edge length lands near the target (uniform remesh)
    mean_edge = float(np.asarray(out.edges_unique_length).mean())
    assert mean_edge == pytest.approx(1.0, rel=0.5)


def test_quadric_decimate_halves_faces():
    pytest.importorskip("pymeshlab")
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    target = len(sphere.faces) // 2
    out = pml_mod.quadric_decimate(sphere, target)
    assert len(out.faces) < len(sphere.faces)
    assert len(out.faces) == pytest.approx(target, rel=0.1)
    assert out.is_watertight


def test_taubin_smooth_preserves_volume():
    pytest.importorskip("pymeshlab")
    # noisy sphere: jitter vertices along normals so smoothing has work to do
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    rng = np.random.default_rng(0)
    v = np.asarray(sphere.vertices).copy()
    v += np.asarray(sphere.vertex_normals) * rng.uniform(-0.1, 0.1, len(v))[:, None]
    noisy = trimesh.Trimesh(vertices=v, faces=np.asarray(sphere.faces), process=False)
    out = pml_mod.taubin_smooth(noisy, 10)
    assert len(out.faces) == len(noisy.faces)
    rough_before = mesh_ops.surface_roughness_deg(noisy)
    rough_after = mesh_ops.surface_roughness_deg(out)
    assert rough_after < rough_before


def test_close_holes_makes_watertight():
    pytest.importorskip("pymeshlab")
    closed = pml_mod.close_holes(_sphere_open(), max_hole_edges=500)
    assert closed.is_watertight


def test_remove_non_manifold():
    pytest.importorskip("pymeshlab")
    # flap: one extra triangle sharing an edge with the box top -> that edge
    # is incident to 3 faces (non-manifold)
    box = trimesh.creation.box(extents=[10, 10, 10])
    top_edge = np.asarray(box.vertices)[[0, 1]]
    apex = np.array([[top_edge[:, 0].mean(), top_edge[:, 1].mean(), 10.0]])
    verts = np.vstack([np.asarray(box.vertices), apex])
    faces = np.vstack([np.asarray(box.faces), [[0, 1, len(box.vertices)]]])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    def non_manifold_edges(mm: trimesh.Trimesh) -> int:
        _, counts = np.unique(np.sort(np.asarray(mm.edges), axis=1), axis=0, return_counts=True)
        return int((counts > 2).sum())

    assert non_manifold_edges(m) > 0
    out = pml_mod.remove_non_manifold(m)
    assert non_manifold_edges(out) == 0


# --------------------------------------------------------------------------- #
# _ensure_watertight ladder integration


def test_ensure_watertight_uses_alpha_wrap(monkeypatch):
    pytest.importorskip("pymeshlab")
    m = _sphere_open()
    _block_clean_stage(m, monkeypatch)

    calls = []
    real_wrap = pml_mod.alpha_wrap

    def spy(mesh, *a, **k):
        calls.append(len(mesh.faces))
        return real_wrap(mesh, *a, **k)

    monkeypatch.setattr(pml_mod, "alpha_wrap", spy)
    out = mesh_ops._ensure_watertight(m)
    assert calls, "alpha_wrap rung was not reached"
    assert out.is_watertight


def test_ensure_watertight_falls_back_without_pml(monkeypatch):
    """pm_available=False -> the old trimesh ladder, bit-identical."""
    m = _sphere_open()
    _block_clean_stage(m, monkeypatch)
    monkeypatch.setattr(pml_mod, "pm_available", lambda: False)

    wrap_called = []

    def forbidden(*a, **k):
        wrap_called.append(True)
        raise AssertionError("alpha_wrap must not run when pm_available() is False")

    monkeypatch.setattr(pml_mod, "alpha_wrap", forbidden)
    out = mesh_ops._ensure_watertight(m)
    assert not wrap_called
    assert out.is_watertight


# --------------------------------------------------------------------------- #
# ops integration


def test_op_remesh_uses_pml_when_available():
    pytest.importorskip("pymeshlab")
    if not pml_mod.pm_available():
        pytest.skip("pymeshlab importable but engine probe failed")
    box = trimesh.creation.box(extents=[20, 20, 20])
    out, report = mesh_ops.apply_ops(box, [{"op": "remesh", "args": {"voxel_mm": 1.0}}])
    assert out.is_watertight
    assert report[0]["op"] == "remesh"


def test_op_remesh_voxel_fallback(monkeypatch):
    """Without pymeshlab the remesh op keeps the voxel path."""
    monkeypatch.setattr(pml_mod, "pm_available", lambda: False)
    box = trimesh.creation.box(extents=[20, 20, 20])
    out, _ = mesh_ops.apply_ops(box, [{"op": "remesh", "args": {"voxel_mm": 1.0}}])
    assert out.is_watertight


def test_op_decimate_reduces_faces():
    pytest.importorskip("pymeshlab")
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    target = len(sphere.faces) // 4
    out, _ = mesh_ops.apply_ops(sphere, [{"op": "decimate", "args": {"faces": target}}])
    assert len(out.faces) < len(sphere.faces)


def test_op_smooth_taubin():
    pytest.importorskip("pymeshlab")
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    out, _ = mesh_ops.apply_ops(
        sphere, [{"op": "smooth", "args": {"scheme": "taubin", "iterations": 5}}]
    )
    assert len(out.faces) == len(sphere.faces)
    assert out.is_watertight
