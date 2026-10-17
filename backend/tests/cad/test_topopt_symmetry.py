"""Mirror-symmetry constraint for SIMP (`cad/fem/topopt.py`
TopOptParams.symmetry): after every design update the blueprint density is
averaged with its reflection in the chosen domain mid-plane
(x = 0.5*(x + flip(x)), passive cells re-pinned), so the extracted part
comes out symmetric about "xy"/"xz"/"yz"."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import TopOptParams, passive_solid_mask, simp
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _beam_domain():
    """20x8x8 voxel mini cantilever — symmetric about every mid-plane."""
    m = trimesh.creation.box(extents=[20.0, 8.0, 8.0])
    m.apply_translation([0.0, 0.0, 4.0])
    return voxelize(m, 1.0, 200_000)


def _setup(dom):
    """Cantilever: fixed -x face, downward -z tip load on the +x face.
    BCs are symmetric about xz (the y mid-plane)."""
    case = LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, -30.0],
            )
        ],
    )
    fixed, F, _ = resolve_case(dom, case, [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    return fixed, F, passive


def test_symmetry_xz_produces_mirrored_density():
    """symmetry="xz" must leave the result mirror-symmetric in y — the
    blueprint bitwise (0.5*(a+b) == 0.5*(b+a) in IEEE), the filtered
    physical density within tolerance."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=15, symmetry="xz"),
    )
    assert res.iterations > 0
    g = res.density.reshape(dom.dims)
    assert np.allclose(g, np.flip(g, axis=1))
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=1))


def test_symmetry_xy_produces_mirrored_blueprint():
    """symmetry="xy" forces a z-mirrored blueprint even though the tip
    load is NOT symmetric about the xy mid-plane — the constraint is on
    the density, not the physics."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10, symmetry="xy"),
    )
    assert res.iterations > 0
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=2))


def test_symmetry_off_runs_unchanged():
    """Default symmetry=None keeps the plain deterministic path — just
    check the run completes and returns a full density field."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=10),
    )
    assert res.iterations > 0
    assert res.density.shape == (dom.cell_count,)


def test_symmetry_unknown_plane_warns_and_ignores():
    """An out-of-enum plane name must not crash — warn and run plain."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=8, symmetry="diag"),
    )
    assert res.iterations > 0
    assert any("symmetry" in w for w in res.warnings)


def test_symmetry_xz_mma_path():
    """The MMA optimizer path symmetrises the unpacked blueprint every
    evaluation — the returned design is mirrored in y as well."""
    pytest.importorskip("nlopt")
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=8, solver="mma", symmetry="xz"),
    )
    assert res.iterations > 0
    assert res.blueprint is not None
    bg = res.blueprint.reshape(dom.dims)
    np.testing.assert_array_equal(bg, np.flip(bg, axis=1))
