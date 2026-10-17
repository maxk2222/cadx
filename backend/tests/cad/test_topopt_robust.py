"""Robust SIMP formulation (Sigmund 2009 / Wang et al. 2011): the blueprint
density is evaluated under eroded / intermediate / dilated Heaviside
projections (eta = 0.5+d, 0.5, 0.5-d), the worst case drives the update and
the volume constraint acts on the dilated realization — threshold-insensitive,
print-tolerant designs (`cad/fem/topopt.py`)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.regions import resolve_case
from cadx.cad.fem.topopt import (
    TopOptParams,
    _filter_matrix,
    passive_solid_mask,
    simp,
)
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _beam_domain():
    """20x8x8 voxel mini cantilever — small enough for ~3x FEM per iter."""
    m = trimesh.creation.box(extents=[20.0, 8.0, 8.0])
    m.apply_translation([0.0, 0.0, 4.0])
    return voxelize(m, 1.0, 200_000)


def _plate_domain():
    """30x30x6 plate — thick enough members for the robust formulation to
    actually enforce minimum length scale in both phases."""
    m = trimesh.creation.box(extents=[30.0, 30.0, 6.0])
    m.apply_translation([0.0, 0.0, 3.0])
    return voxelize(m, 1.5, 300_000)


def _tip_case(margin: float = 1.0):
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=margin),
                force_n=[0.0, 0.0, -30.0],
            )
        ],
    )


def _setup(dom, margin: float = 1.0):
    fixed, F, _ = resolve_case(dom, _tip_case(margin), [])
    passive = passive_solid_mask(dom, [], [fixed], [])
    return fixed, F, passive


def _mask_variation(res, dom, rmin_mm: float, d: float = 0.25) -> float:
    """Mask-volume difference between extractions at eta=0.5±d — the fraction
    of design-space cells whose cone-filtered blueprint density xt lies inside
    the eta interval (exactly the cells that flip between the dilated and the
    eroded realization). Computed on the blueprint, not the projected physical
    density, so it is independent of the final beta."""
    solid_idx = np.flatnonzero(dom.solid.ravel())
    rmin_cells = max(1.2, rmin_mm / dom.pitch)  # same rule as simp()
    H = _filter_matrix(dom, rmin_cells)
    Hs = np.asarray(H.sum(axis=1)).ravel()
    Hs[Hs <= 0] = 1.0
    xt = np.asarray(H @ res.blueprint[solid_idx]).ravel() / Hs
    return float(((xt > 0.5 - d) & (xt <= 0.5 + d)).sum()) / solid_idx.size


def test_robust_edi_converges_and_orders_designs():
    """robust=True converges, records obj_per_design in eroded/intermediate/
    dilated order, and compliance ranks eroded >= intermediate >= dilated."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=40, robust=True),
    )
    assert res.iterations > 0
    assert res.history, "no history recorded"
    assert res.blueprint is not None
    objs = np.asarray(res.history[-1]["obj_per_design"], dtype=float)
    assert objs.shape == (3,)
    # less material -> more compliance: eroded >= intermediate >= dilated
    assert objs[0] >= objs[1] >= objs[2]
    assert res.history[-1]["compliance_mj"] == pytest.approx(float(objs[0]))
    # the enforced volume is the dilated realization's
    assert res.history[-1]["volfrac"] == pytest.approx(0.4, abs=0.02)


def test_robust_is_threshold_insensitive():
    """Core benefit: extracting the robust design at thresholds eta=0.5±d
    must vary the solid mask much less than for a design optimized at
    eta=0.5 only (same setup, heaviside filter in both)."""
    dom = _plate_domain()
    fixed, F, passive = _setup(dom, margin=1.5)
    base = dict(
        volfrac=0.5,
        rmin_mm=3.0,
        max_iter=55,
        tol=0.01,
        converge_patience=5,
        filter="heaviside",
        beta_max=8.0,
    )
    det = simp(dom, E_PLA, NU, [(fixed, F)], passive, TopOptParams(**base))
    rob = simp(dom, E_PLA, NU, [(fixed, F)], passive, TopOptParams(robust=True, **base))
    var_det = _mask_variation(det, dom, rmin_mm=3.0)
    var_rob = _mask_variation(rob, dom, rmin_mm=3.0)
    assert var_rob < 0.5 * var_det


def test_robust_forces_heaviside_filter():
    """robust without filter='heaviside' must warn and force the projection
    instead of silently running a cone-filter robust design."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=15, filter="cone", robust=True),
    )
    assert res.iterations > 0
    assert any("heaviside" in w for w in res.warnings)
    # the projection actually ran — beta is recorded per iteration
    assert "beta" in res.history[-1]


def test_robust_e_evaluates_eroded_only():
    """robust_designs='e' (Fernández) evaluates only the eroded realization —
    one entry in obj_per_design instead of three."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    res = simp(
        dom,
        E_PLA,
        NU,
        [(fixed, F)],
        passive,
        TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=15, robust=True, robust_designs="e"),
    )
    assert res.iterations > 0
    objs = res.history[-1]["obj_per_design"]
    assert len(objs) == 1
    assert res.history[-1]["compliance_mj"] == pytest.approx(objs[0])


def test_robust_off_is_byte_identical():
    """Regression: robust=False must be the unchanged deterministic path —
    SIMP is seed-free, so two runs on the same setup are byte-identical and
    carry no obj_per_design entries."""
    dom = _beam_domain()
    fixed, F, passive = _setup(dom)
    p = TopOptParams(volfrac=0.4, rmin_mm=2.0, max_iter=12)
    r1 = simp(dom, E_PLA, NU, [(fixed, F)], passive, p)
    r2 = simp(dom, E_PLA, NU, [(fixed, F)], passive, p)
    np.testing.assert_array_equal(r1.density, r2.density)
    assert all("obj_per_design" not in rec for rec in r1.history)
