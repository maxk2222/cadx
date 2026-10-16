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
    # carved skin by default — the silhouette survives at ~65% coverage
    assert p.surface_open == 0.35 and p.shell_mm == 0.0
    # anchors is runtime-injected, never serialized into stored op args
    assert "anchors" not in p.model_dump(mode="json")


def test_params_validation():
    with pytest.raises(ValidationError):
        OrganicParams(style="lattice")  # unknown style
    with pytest.raises(ValidationError):
        OrganicParams(volfrac=1.5)
    with pytest.raises(ValidationError):
        OrganicParams(seed=-1)
    with pytest.raises(ValidationError):
        OrganicParams(surface_open=1.5)
    with pytest.raises(ValidationError):
        OrganicParams(surface_open=-0.1)


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
    # auto_keep=False + open skeleton (surface_open=1) must not pin anything —
    # with the default carved skin, skin cells land in passive legitimately
    res_off = organic_transform(
        _box(), _p(auto_keep=False, floor_mm=0.0, surface_open=1.0), anchors=anchors
    )
    assert not res_off.passive.any()


def test_no_fallback_needed_on_extreme_density():
    """Absurd params still produce a mesh (fallback cascade inside)."""
    res = organic_transform(_box(), _p(style="gyroid", volfrac=0.05, period_mm=4.0))
    assert len(res.mesh.faces) > 0


# --------------------------------------------------------------------------- #
# carved skin — recognizability                                                #
# --------------------------------------------------------------------------- #


def test_solid_skin_fully_preserved():
    """surface_open=0 pins the whole skin — every skin cell survives."""
    res = organic_transform(_box(), _p(style="gyroid", surface_open=0.0, volfrac=0.4))
    assert res.report["surface_coverage"] is not None
    assert res.report["surface_coverage"] >= 0.99


def test_carved_skin_ordering():
    """surface coverage drops as surface_open grows — monotonic recognizability."""
    cov = {}
    for open_ in (0.0, 0.35, 1.0):
        res = organic_transform(
            _box(), _p(style="trabecular", seed=5, surface_open=open_, volfrac=0.35)
        )
        cov[open_] = res.report["surface_coverage"]
    assert cov[0.0] >= 0.99
    assert cov[0.35] is not None and cov[0.35] < cov[0.0]
    assert cov[1.0] is not None and cov[1.0] < cov[0.35]
    # the open skeleton is the only regime allowed to lose most of the skin
    assert cov[1.0] < 0.4


def test_carved_skin_is_bounded_by_field():
    """Pinned skin cells are exactly the top (1-open) field fraction of skin."""
    res = organic_transform(_box(), _p(style="gyroid", surface_open=0.35, seed=3))
    cov = res.report["surface_coverage"]
    assert cov is not None and 0.5 < cov < 0.9


def test_void_carves_through_skin():
    """A painted void punches through the pinned skin — deliberate erase wins."""
    void = RegionSpec(kind="box", center=[0.0, 0.0, 14.0], diameter_mm=10.0)
    res = organic_transform(
        _box(), _p(style="gyroid", surface_open=0.0, void_regions=[void], volfrac=0.5)
    )
    # the void centre column must be empty in the final mask even though the
    # skin is fully pinned
    d = res.domain
    i = round((0.0 - d.origin[0]) / d.pitch - 0.5)
    j = round((0.0 - d.origin[1]) / d.pitch - 0.5)
    k = round((14.0 - d.origin[2]) / d.pitch - 0.5)
    assert not res.mask[max(i, 0), max(j, 0), max(k, 0)]
    assert not res.passive.ravel().reshape(d.dims)[max(i, 0), max(j, 0), max(k, 0)]


def test_void_survives_min_member_dilation():
    """Min-member thickening grows material — it must NOT heal a carved void:
    void_cells is re-applied after every growing pass in density_to_mask."""
    void = RegionSpec(kind="box", center=[0.0, 0.0, 14.0], diameter_mm=8.0)
    res = organic_transform(
        _box(),
        _p(
            style="gyroid",
            surface_open=0.0,
            min_member_mm=3.0,
            void_regions=[void],
            volfrac=0.6,
        ),
    )
    d = res.domain
    i = round((0.0 - d.origin[0]) / d.pitch - 0.5)
    j = round((0.0 - d.origin[1]) / d.pitch - 0.5)
    k = round((14.0 - d.origin[2]) / d.pitch - 0.5)
    assert not res.mask[max(i, 0), max(j, 0), max(k, 0)]


def test_explicit_threshold_reaches_extraction():
    """A user-set density_threshold must survive to extraction — only the
    fallback RETRIES may drop it (an over-high iso is a plausible cause of
    an empty first attempt)."""
    base = dict(style="trabecular", seed=3, surface_open=1.0, floor_mm=0.0, auto_keep=False)
    auto = organic_transform(_box(), _p(**base, volfrac=0.4))
    hi = organic_transform(
        _box(), _p(**base, volfrac=0.4, density_threshold=0.9)
    )
    assert hi.report["threshold"] == 0.9
    assert hi.report["solid_elements"] < auto.report["solid_elements"]


def test_thin_feature_survives_voxelization():
    """A wall thinner than one pitch still lands in the design domain —
    near-surface cells are unioned into domain.solid before the field runs."""
    base = trimesh.creation.box(extents=[40.0, 40.0, 8.0])
    base.apply_translation([0.0, 0.0, 4.0])
    # 0.5 mm fin — vanishes under cell-center containment at pitch 2.5
    fin = trimesh.creation.box(extents=[40.0, 0.5, 6.0])
    fin.apply_translation([0.0, 20.1, 3.0])
    m = trimesh.util.concatenate([base, fin])
    res = organic_transform(m, _p(style="trabecular", voxel_mm=2.5, surface_open=1.0))
    # the fin protrudes past the base surface (y=20) — cells whose centres sit
    # beyond the base only exist if the near-surface rescue unioned them in
    centers = res.domain.element_centers()
    beyond = (centers[:, 1] > 20.0).reshape(res.domain.dims)
    assert (res.domain.solid & beyond).any(), "thin fin was erased by voxelization"


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


def test_apply_ops_report_has_organic_metrics():
    """The op stash surfaces recognizability metrics into the preview stats."""
    _, report = apply_ops(
        _box(),
        [
            {
                "op": "organic",
                "args": {"style": "trabecular", "volfrac": 0.4, "seed": 7, "voxel_mm": 3.0},
            }
        ],
    )
    org = report[0]["organic"]
    assert org["volfrac_achieved"] > 0
    assert org["n_components"] >= 1
    assert org["shell_mm"] > 0
    assert org["surface_coverage"] is not None
    assert isinstance(org["warnings"], list)


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


def test_apply_ops_organic_keeps_disjoint_bodies():
    """The wreck-guard in apply_ops must NOT 'salvage' a legitimately
    multi-body organic result — disjoint lattices/islands are valid output
    (reported via n_components), while the commit pipeline exports them."""
    mesh, report = apply_ops(
        _box(),
        [
            {
                "op": "organic",
                "args": OrganicParams(
                    style="trabecular",
                    volfrac=0.05,
                    density_threshold=0.9,
                    surface_open=1.0,
                    floor_mm=0.0,
                    min_member_mm=0.0,
                    auto_keep=False,
                    keep_regions=[
                        RegionSpec(kind="face", axis="-x", margin_mm=2.0),
                        RegionSpec(kind="face", axis="+x", margin_mm=2.0),
                    ],
                    voxel_mm=3.0,
                    seed=1,
                ).model_dump(exclude_none=True),
            }
        ],
    )
    assert len(mesh.split(only_watertight=False)) == 2
    assert report[0]["organic"]["n_components"] == 2
    assert not report[0].get("salvaged")
    assert not report[0].get("reverted")


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


def test_organic_resample_preserves_carve(tmp_path: Path) -> None:
    """Saved organic voids must survive topopt-resample min-member regrowth.

    Reproduces the bug where density.npz did not persist carved cells: the
    resample's min-member dilation then regrew material into painted carve
    regions, silently undoing the user's brush work.
    """
    from cadx import storage
    from cadx.cad.engine import run_organic_pipeline, run_topopt_resample

    out_dir = storage.tmp_dir() / f"organic-resample-{id(tmp_path)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "b.stl"
    _box().export(str(src))
    # full-height cylinder carve through the middle of the 60x30x15 box
    carve = RegionSpec(kind="cylinder", axis="+z", center=[0.0, 0.0, 7.5], diameter_mm=8.0)
    res = run_organic_pipeline(
        mesh_path=src,
        out_dir=out_dir / "o",
        organic={
            "style": "gyroid",
            "volfrac": 0.5,
            "voxel_mm": 2.5,
            "surface_open": 1.0,
            "floor_mm": 0.0,
            "auto_keep": False,
            "min_member_mm": 0.0,
            "void_regions": [carve.model_dump()],
        },
    )
    assert res.ok, res.error
    npz = out_dir / "o" / "organic" / "density.npz"
    data = np.load(npz)
    assert "voids" in data.files and data["voids"].any()

    # resample with a large min-member — without the persisted void mask the
    # dilation pass would regrow material straight back into the carve
    res2 = run_topopt_resample(
        density_relpath=storage.relpath(npz),
        out_dir=out_dir / "r",
        load_cases=[],
        optimize={"min_member_mm": 10.0},
    )
    assert res2.ok, res2.error
    assert any("suppressed" in w for w in res2.topopt.get("warnings", []))
    m = trimesh.load(str(out_dir / "r" / "_run" / "model.stl"), force="mesh")
    m.merge_vertices()
    assert m.is_watertight
    assert not m.contains(np.array([[0.0, 0.0, 7.5]]))[0]
