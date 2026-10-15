"""Pseudo-topology organic restyle (`cad/fem/organic.py`).

Deterministic procedural density field extracted through the shared TopOpt
path — watertight, connectivity-pruned, with pinned floor/keep/anchor zones.
Aesthetic only — the report must say so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from cadx.cad.fem.organic import organic_transform
from cadx.cad.mesh_ops import OPS_REGISTRY, MeshOpError, apply_ops
from cadx.llm.schemas import OrganicParams, RegionSpec


def _box(x: float = 60.0, y: float = 30.0, z: float = 15.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _p(**kw) -> OrganicParams:
    # coarse grid keeps the tests fast
    kw.setdefault("voxel_mm", 2.0)
    kw.setdefault("period_mm", 10.0)
    return OrganicParams(**kw)


# --------------------------------------------------------------------------- #
# params validation                                                           #
# --------------------------------------------------------------------------- #


def test_params_defaults():
    p = OrganicParams()
    assert p.style == "gyroid" and p.volfrac == 0.35 and p.auto_keep
    # anchors is runtime-injected, never serialized into stored op args
    assert "anchors" not in p.model_dump(mode="json")


def test_params_validation():
    with pytest.raises(ValidationError):
        OrganicParams(style="lattice")  # unknown style
    with pytest.raises(ValidationError):
        OrganicParams(volfrac=1.5)
    with pytest.raises(ValidationError):
        OrganicParams(seed=-1)


# --------------------------------------------------------------------------- #
# transform                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("style", ["gyroid", "trabecular", "voronoi", "branches"])
def test_styles_produce_solid_mesh(style):
    res = organic_transform(_box(), _p(style=style, seed=7))
    m = res.mesh
    assert len(m.faces) > 500, f"{style}: degenerate output"
    assert m.is_watertight, f"{style}: not watertight"
    # stays inside the source envelope (one voxel tolerance)
    tol = res.domain.pitch * 1.5
    assert m.bounds[0][0] >= _box().bounds[0][0] - tol
    assert m.bounds[1][0] <= _box().bounds[1][0] + tol
    # the report marks the result as aesthetic-only
    assert any("not structurally" in w.lower() for w in res.report["warnings"])
    assert res.report["style"] == style


def test_deterministic_seed():
    a = organic_transform(_box(), _p(style="voronoi", seed=42))
    b = organic_transform(_box(), _p(style="voronoi", seed=42))
    assert np.allclose(a.density, b.density)
    assert len(a.mesh.faces) == len(b.mesh.faces)
    assert np.allclose(a.mesh.vertices, b.mesh.vertices)


def test_seed_changes_output():
    a = organic_transform(_box(), _p(style="trabecular", seed=1))
    b = organic_transform(_box(), _p(style="trabecular", seed=2))
    assert not np.allclose(a.density, b.density)


def test_volfrac_changes_mass():
    lo = organic_transform(_box(), _p(style="trabecular", seed=3, volfrac=0.15))
    hi = organic_transform(_box(), _p(style="trabecular", seed=3, volfrac=0.6))
    assert int(hi.mask.sum()) > int(lo.mask.sum())


def test_floor_band_pinned_solid():
    """Every passive cell (floor band / keeps) is present in the final mask."""
    res = organic_transform(_box(), _p(floor_mm=4.0))
    assert res.passive.any(), "floor band produced no pinned cells"
    assert res.mask.ravel()[res.passive].all()
    # and the floor actually exists in the output mesh
    assert abs(res.mesh.bounds[0][2]) < res.domain.pitch


def test_keep_region_survives():
    """A painted/cylinder keep zone stays solid even deep in the part."""
    keep = RegionSpec(kind="cylinder", axis="z", center=[0.0, 0.0, 7.5], diameter_mm=12.0)
    res = organic_transform(_box(), _p(style="gyroid", volfrac=0.2, keep_regions=[keep]))
    assert res.passive.any()
    assert res.mask.ravel()[res.passive].all()


def test_auto_keep_anchor_ring():
    anchors = [
        {
            "index": 1,
            "kind": "hole",
            "axis": "+z",
            "pos": [10.0, 0.0, 7.5],
            "x": 10.0,
            "y": 0.0,
            "dia": 5.0,
            "depth_mm": None,
        }
    ]
    res = organic_transform(_box(), _p(auto_keep=True), anchors=anchors)
    assert res.passive.any()
    assert res.mask.ravel()[res.passive].all()
    # auto_keep=False must not pin the anchor
    res_off = organic_transform(_box(), _p(auto_keep=False, floor_mm=0.0), anchors=anchors)
    assert not res_off.passive.any()


def test_no_fallback_needed_on_extreme_density():
    """Absurd params still produce a mesh (fallback cascade inside)."""
    res = organic_transform(_box(), _p(style="gyroid", volfrac=0.05, period_mm=4.0))
    assert len(res.mesh.faces) > 0


# --------------------------------------------------------------------------- #
# mesh-op integration (the preview path)                                       #
# --------------------------------------------------------------------------- #


def test_organic_op_registered():
    assert "organic" in OPS_REGISTRY


def test_apply_ops_organic():
    mesh, report = apply_ops(
        _box(),
        [{"op": "organic", "args": {"style": "gyroid", "volfrac": 0.4, "voxel_mm": 2.5}}],
    )
    assert len(mesh.faces) > 500
    assert report[0]["op"] == "organic"
    assert report[0]["faces_out"] == len(mesh.faces)


def test_apply_ops_organic_anchors_injected():
    """auto_keep resolves anchors passed through apply_ops (preview path)."""
    anchors = {"holes": [{"index": 1, "kind": "hole", "x": 10.0, "y": 0.0, "dia": 5.0}]}
    mesh, report = apply_ops(
        _box(),
        [
            {
                "op": "organic",
                "args": {"style": "gyroid", "auto_keep": True, "voxel_mm": 2.5},
            }
        ],
        anchors=anchors,
    )
    assert len(mesh.faces) > 0
    assert report[0]["op"] == "organic"


def test_apply_ops_organic_bad_args():
    with pytest.raises(MeshOpError):
        apply_ops(_box(), [{"op": "organic", "args": {"style": "nope"}}])


# --------------------------------------------------------------------------- #
# executor pipeline (what `run_organic_job` runs)                              #
# --------------------------------------------------------------------------- #


def test_run_organic_pipeline(tmp_path: Path) -> None:
    """End-to-end: watertight STL + exports + resample-compatible density npz."""
    from cadx.cad.engine import run_organic_pipeline

    src = tmp_path / "b.stl"
    _box().export(str(src))
    res = run_organic_pipeline(
        mesh_path=src,
        out_dir=tmp_path / "o",
        organic={"style": "trabecular", "volfrac": 0.35, "voxel_mm": 3.0},
    )
    assert res.ok, res.error
    stl = tmp_path / "o" / "_run" / "model.stl"
    assert stl.exists()
    m = trimesh.load(str(stl), force="mesh", process=False)
    m.merge_vertices()
    assert m.is_watertight
    npz = tmp_path / "o" / "organic" / "density.npz"
    assert npz.exists()
    data = np.load(npz)
    assert {"density", "mask", "solid", "passive", "origin", "pitch", "dims"} <= set(data.files)
    org = res.mesh["organic"]
    assert org["style"] == "trabecular" and org["triangles"] > 500
    assert any("not structurally" in w.lower() for w in org["warnings"])
    # same export contract as the other pipelines
    assert res.exports.get("stl") and res.exports.get("glb")
