"""Shell+Infill structure method (`cad/fem/infill.py`).

FDM-native alternative to SIMP: closed outer skin + interior density-graded
lattice driven by ONE FEM solve per load case on the full solid domain. The
(rho, mask) output must drop straight into the shared `density_to_mesh`
extraction and `run_fem_domain` verification paths unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh
from scipy import ndimage

from cadx.cad.fem.infill import (
    InfillParams,
    _skin_mask,
    element_principal_dirs,
    element_strain_energy,
    gyroid_occupancy,
    infill_field,
    strut_occupancy,
    target_density,
)
from cadx.cad.fem.regions import resolve_case_dofs
from cadx.cad.fem.solver import assemble, reduce_case, solve_reduced
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

E_PLA = 3500.0
NU = 0.36


def _box_mesh(x: float = 60.0, y: float = 24.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])  # contract: rests on XY, centred
    return m


def _cantilever_case(force_n: float = -100.0) -> LoadCase:
    return LoadCase(
        name="tip",
        fixed=[RegionSpec(kind="face", axis="-x")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+x", margin_mm=1.0),
                force_n=[0.0, 0.0, force_n],
            )
        ],
    )


def _solve_cantilever(dom):
    """One solved case, mirroring analyze._solve_cases' call path:
    resolve_case_dofs -> assemble -> reduce_case -> solve_reduced."""
    case = _cantilever_case()
    fixed3, F, _w = resolve_case_dofs(dom, case)
    elem_E = np.where(dom.solid.ravel(), E_PLA, 0.0)
    K, edof, active = assemble(dom, elem_E, NU)
    ctx = reduce_case(dom, K, edof, fixed3)
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU, E_ref=E_PLA)
    assert sol.converged
    return case, sol, edof, active, elem_E


@pytest.fixture(scope="module")
def solved_cantilever():
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    case, sol, edof, active, elem_E = _solve_cantilever(dom)
    return dom, case, sol, edof, active, elem_E


def _air_leak_cells(mask: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Interior voids reachable by outside air — empty iff the skin is closed.

    ~mask is padded with a layer of open air so the grid exterior connects;
    any (solid & ~mask) cell sharing the border component's label leaks."""
    padded = np.pad(~mask, 1, constant_values=True)
    labels, _n = ndimage.label(padded, structure=ndimage.generate_binary_structure(3, 1))
    outside = labels[0, 0, 0]
    inner = labels[1:-1, 1:-1, 1:-1]
    return np.asarray(solid & ~mask & (inner == outside))


# --------------------------------------------------------------------------- #
# per-element fields                                                           #
# --------------------------------------------------------------------------- #


def test_element_strain_energy_matches_compliance(solved_cantilever) -> None:
    """Σ_e u_eᵀ(E_e·k_e)u_e over the assembled cells must equal uᵀKu = uᵀF,
    i.e. the solution's compliance — an end-to-end check of the edof gather
    and element stiffness path."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    assert w.shape == (dom.cell_count,)
    assert np.all(np.isfinite(w)) and (w >= 0.0).all()
    assert w.sum() == pytest.approx(sol.compliance_j, rel=1e-4)
    # cells outside the assembled (solid) set carry no energy
    assert not w[~dom.solid.ravel()].any()


def test_element_strain_energy_bending_profile(solved_cantilever) -> None:
    """Cantilever: bending moment grows toward the fixed end — energy must be
    concentrated on the root third, not the tip third (the -x tip's load
    singularity is local, not bulk)."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w3 = element_strain_energy(dom, sol.u, edof, active, elem_E, NU).reshape(dom.dims)
    t = dom.dims[0] // 3
    root, mid, tip = w3[:t].sum(), w3[t : 2 * t].sum(), w3[2 * t :].sum()
    assert root > mid > tip
    assert root > 4 * tip


def test_element_principal_dirs_bending_axis(solved_cantilever) -> None:
    """Extreme-fibre cells (top/bottom skin at mid-span) see a sigma_xx-dominated
    stress state — the principal direction must snap to ±x."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    dirs = element_principal_dirs(dom, sol.u, edof, active, elem_E, NU)
    assert dirs.shape == (dom.cell_count, 3)
    # every active cell returns a unit vector (stressed eigvec or +z default)
    norms = np.linalg.norm(dirs[dom.solid.ravel()], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    # inactive cells are zero
    assert not dirs[~dom.solid.ravel()].any()

    d3 = dirs.reshape((*dom.dims, 3))
    centers = dom.element_centers().reshape((*dom.dims, 3))
    mid_x = (centers[..., 0] > -15.0) & (centers[..., 0] < 15.0)
    extreme = (centers[..., 2] > 10.0) | (centers[..., 2] < 2.0)
    sel = mid_x & extreme & dom.solid
    x_comp = np.abs(d3[..., 0][sel])
    assert sel.sum() > 20
    assert (x_comp > 0.9).mean() > 0.9


def test_element_fields_zero_on_empty_active(solved_cantilever) -> None:
    dom, _case, sol, _edof, _active, elem_E = solved_cantilever
    empty = np.zeros(0, dtype=np.int64)
    edof0 = np.zeros((0, 24), dtype=np.int64)
    w = element_strain_energy(dom, sol.u, edof0, empty, elem_E, NU)
    assert not w.any()
    dirs = element_principal_dirs(dom, sol.u, edof0, empty, elem_E, NU)
    assert not dirs.any()


# --------------------------------------------------------------------------- #
# target density                                                               #
# --------------------------------------------------------------------------- #


def test_target_density_monotone_in_energy() -> None:
    """Higher relative energy -> higher target density (monotone grading)."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    centers = dom.element_centers()
    e = np.clip(centers[:, 0] - float(dom.origin[0]), 0.0, None)  # ramp along x
    d = target_density(dom, [e], dmin=0.1, dmax=0.8, alpha=1.0)
    left = dom.solid.ravel() & (centers[:, 0] < 0.0)
    right = dom.solid.ravel() & (centers[:, 0] > 10.0)
    assert d[right].mean() > d[left].mean()


def test_target_density_clips_and_pins() -> None:
    """d ∈ [dmin, dmax] on gradable cells; shell/passive → 1.0, void → 0.0."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)  # dims (20, 8, 4)
    shell = np.zeros(dom.dims, dtype=bool)
    shell[0, :, :] = True  # -x face layer acts as skin
    passive = np.zeros(dom.dims, dtype=bool)
    passive[10, 6, 3] = True  # one interior pin
    void = np.zeros(dom.dims, dtype=bool)
    void[-1, :, :] = True  # carved +x slab

    # ramp energy along x with an explicit cold interior cell (e=0 → dmin)
    # and a hot outlier (w_norm clips at 1 → dmax)
    centers = dom.element_centers()
    e = np.clip(centers[:, 0] - float(dom.origin[0]), 0.0, None)
    e[~dom.solid.ravel()] = 0.0
    cold = np.zeros(dom.dims, dtype=bool)
    cold[5, 4, 2] = True
    e[cold.ravel()] = 0.0
    hot = np.zeros(dom.dims, dtype=bool)
    hot[-2, 4, 2] = True
    e[hot.ravel()] = 1e9

    d = target_density(
        dom,
        [e],
        shell_mask=shell,
        passive=passive.ravel(),
        void=void.ravel(),
        dmin=0.1,
        dmax=0.8,
    )
    gradable = dom.solid.ravel() & ~shell.ravel() & ~passive.ravel() & ~void.ravel()
    assert d[gradable].min() >= 0.1 - 1e-9 and d[gradable].max() <= 0.8 + 1e-9
    assert d[cold.ravel()] == pytest.approx(0.1)  # energy floor → density floor
    assert d[hot.ravel()] == pytest.approx(0.8)  # outlier is clipped at dmax
    assert (d[shell.ravel()] == 1.0).all()
    assert d[passive.ravel()] == 1.0
    assert not d[void.ravel()].any()
    assert not d[~dom.solid.ravel()].any()


def test_target_density_weighted_max_across_cases() -> None:
    """Two cases hot on opposite ends: the max combine keeps BOTH ends dense;
    a small weight shrinks a case's contribution."""
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    centers = dom.element_centers()
    left_e = np.clip(5.0 - centers[:, 0], 0.0, None)  # hot at -x
    right_e = np.clip(centers[:, 0] + 25.0, 0.0, None)  # hot at +x
    left_e[~dom.solid.ravel()] = 0.0
    right_e[~dom.solid.ravel()] = 0.0

    d_both = target_density(dom, [left_e, right_e], dmin=0.1, dmax=0.8, alpha=1.0)
    d_left = target_density(dom, [left_e], dmin=0.1, dmax=0.8, alpha=1.0)
    right = dom.solid.ravel() & (centers[:, 0] > 15.0)
    assert d_both[right].mean() > d_left[right].mean()

    d_weak = target_density(
        dom, [left_e, right_e], weights=[1.0, 0.2], dmin=0.1, dmax=0.8, alpha=1.0
    )
    assert d_weak[right].mean() < d_both[right].mean()

    with pytest.raises(ValueError, match="weights"):
        target_density(dom, [left_e, right_e], weights=[1.0])


def test_target_density_no_energy_gives_floor() -> None:
    dom = voxelize(_box_mesh(), 3.0, 150_000)
    d = target_density(dom, [], dmin=0.12, dmax=0.8)
    np.testing.assert_allclose(d[dom.solid.ravel()], 0.12)
    # an all-zero case also yields the floor, not a crash on p95=0
    d = target_density(dom, [np.zeros(dom.cell_count)], dmin=0.12, dmax=0.8)
    np.testing.assert_allclose(d[dom.solid.ravel()], 0.12)


# --------------------------------------------------------------------------- #
# lattice occupancy                                                            #
# --------------------------------------------------------------------------- #


def test_gyroid_occupancy_uniform_density() -> None:
    """Uniform d → global occupancy fraction ≈ d (±10% relative)."""
    dom = voxelize(_box_mesh(48.0, 48.0, 24.0), 2.0, 150_000)
    d = np.where(dom.solid.ravel(), 0.3, 0.0)
    occ = gyroid_occupancy(dom, d, 8.0, margin_cells=0)
    frac = float(occ[dom.solid].mean())
    assert math.isclose(frac, 0.3, rel_tol=0.1), frac


def test_gyroid_occupancy_periodic() -> None:
    """The field is periodic with cell_mm per axis — shifting by one period
    reproduces the occupancy. Exact equality is impossible at threshold
    ties (the commensurate grid puts whole plateaus of cells at |g-t| < fp
    noise), so mismatches are only allowed on cells sitting at the iso
    boundary."""
    dom = voxelize(_box_mesh(48.0, 48.0, 24.0), 2.0, 150_000)
    d_val = 0.4
    d = np.where(dom.solid.ravel(), d_val, 0.0)
    cell_mm = 8.0
    period_cells = 4  # 8.0 mm / 2.0 mm pitch
    occ = gyroid_occupancy(dom, d, cell_mm, margin_cells=0)

    # recompute the spec'd field + the empirical-CDF threshold to identify ties
    c = dom.element_centers()
    k = 2.0 * math.pi / cell_mm
    g = (
        np.sin(c[:, 0] * k) * np.cos(c[:, 1] * k)
        + np.sin(c[:, 1] * k) * np.cos(c[:, 2] * k)
        + np.sin(c[:, 2] * k) * np.cos(c[:, 0] * k)
    ).reshape(dom.dims)
    g_sorted = np.sort(g[dom.solid])
    t = g_sorted[round((1.0 - d_val) * (g_sorted.size - 1))]

    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(None, -period_cells)
        hi[ax] = slice(period_cells, None)
        mism = occ[tuple(lo)] != occ[tuple(hi)]
        # every mismatch is a |g - t| < 1e-9 knife-edge cell, nothing else
        g_mism = np.abs(g[tuple(lo)][mism] - t)
        assert g_mism.size and g_mism.max() < 1e-9


def test_gyroid_occupancy_tracks_graded_density() -> None:
    """Spatially-varying d: the high-density half must be locally denser —
    bin the domain in halves and compare occupancy fractions."""
    dom = voxelize(_box_mesh(48.0, 48.0, 24.0), 2.0, 150_000)
    centers = dom.element_centers()
    d = np.where(centers[:, 0] < 0.0, 0.55, 0.15)
    d[~dom.solid.ravel()] = 0.0
    occ = gyroid_occupancy(dom, d, 8.0, margin_cells=0)
    left = dom.solid & (centers[:, 0].reshape(dom.dims) < 0.0)
    right = dom.solid & (centers[:, 0].reshape(dom.dims) > 0.0)
    fl = float(occ[left].mean())
    fr = float(occ[right].mean())
    assert fl == pytest.approx(0.55, abs=0.08)
    assert fr == pytest.approx(0.15, abs=0.08)
    assert fl > 2.0 * fr


def test_occupancy_bonds_full_density_cells() -> None:
    """d >= ~0.995 → unconditionally occupied; d == 0 → never occupied."""
    dom = voxelize(_box_mesh(24.0, 24.0, 24.0), 2.0, 150_000)
    d = np.full(dom.cell_count, 0.3)
    hot = np.zeros(dom.dims, dtype=bool)
    hot[2:4, 2:4, 2:4] = True
    d[hot.ravel()] = 1.0
    empty = np.zeros(dom.dims, dtype=bool)
    empty[-2:, -2:, -2:] = True
    d[empty.ravel()] = 0.0
    for fn in (gyroid_occupancy, strut_occupancy):
        if fn is strut_occupancy:
            occ = fn(dom, d, np.tile([1.0, 0.0, 0.0], (dom.cell_count, 1)), 8.0)
        else:
            occ = fn(dom, d, 8.0)
        assert occ[hot].all()
        assert not occ[empty].any()


def test_strut_occupancy_dominant_axis() -> None:
    """x-directed stress → mostly x-struts: occupancy ≈ d (square-rod
    quantization on the coarse grid) and the rods run the full beam length
    along x while no full-length column exists along z."""
    # 12 cells per lattice period keeps the (k/n)² occupancy quantization fine
    dom = voxelize(_box_mesh(48.0, 48.0, 24.0), 1.0, 150_000)
    n = dom.cell_count
    d = np.where(dom.solid.ravel(), 0.4, 0.0)
    dirs = np.tile([1.0, 0.0, 0.0], (n, 1))
    occ = strut_occupancy(dom, d, dirs, 12.0, margin_cells=0)
    frac = float(occ[dom.solid].mean())
    assert math.isclose(frac, 0.4, abs_tol=0.08)
    # x-struts are continuous rods along x: some (y,z) columns are occupied
    # for the full beam length, while no column along z is fully occupied
    full_x = int((occ.sum(axis=0) == dom.dims[0]).sum())
    full_z = int((occ.sum(axis=2) == dom.dims[2]).sum())
    assert full_x > 0
    assert full_z == 0


# --------------------------------------------------------------------------- #
# infill_field end-to-end                                                      #
# --------------------------------------------------------------------------- #


def test_infill_field_closed_shell(solved_cantilever) -> None:
    """The skin must be sealed: no interior void cell may be connected to
    outside air (6-connectivity through the padded domain)."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    rho, mask, meta = infill_field(
        dom, InfillParams(shell_mm=2.0, infill_cell_mm=8.0), [w], None, None, None
    )
    assert rho.dtype == np.float32 and rho.shape == (dom.cell_count,)
    np.testing.assert_array_equal(rho.reshape(dom.dims) >= 0.5, mask)
    # lattice actually carved the interior but the shell holds
    assert mask.sum() < dom.solid_count
    assert not _air_leak_cells(mask, dom.solid).any()
    # meta contract
    for key in (
        "shell_mm",
        "cell_mm",
        "d_mean",
        "d_p50",
        "d_p95",
        "volfrac_estimate",
        "style",
        "scope",
    ):
        assert key in meta
    assert meta["style"] == "gyroid_graded" and meta["scope"] == "full"
    assert 0.0 < meta["volfrac_estimate"] < 1.0


@pytest.mark.parametrize("style", ["gyroid_graded", "struts_aligned"])
def test_infill_field_styles_closed(solved_cantilever, style) -> None:
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    dirs = element_principal_dirs(dom, sol.u, edof, active, elem_E, NU)
    _rho, mask, meta = infill_field(
        dom,
        InfillParams(shell_mm=2.0, infill_cell_mm=8.0, style=style),
        [w],
        dirs,
        None,
        None,
    )
    assert meta["style"] == style
    assert mask.any() and mask.sum() < dom.solid_count
    assert not _air_leak_cells(mask, dom.solid).any()


def test_infill_field_struts_without_dirs_falls_back(solved_cantilever) -> None:
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    _rho, mask, meta = infill_field(
        dom, InfillParams(style="struts_aligned"), [w], None, None, None
    )
    assert meta["style"] == "gyroid_graded"
    assert meta["style_requested"] == "struts_aligned"
    assert mask.any()


def test_infill_field_passive_and_void(solved_cantilever) -> None:
    """Passive cells are pinned solid, void cells stay empty — including a
    void punched through the skin (declared carve beats the closed shell)."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)

    passive = np.zeros(dom.dims, dtype=bool)
    passive[8:10, 4:8, 2:4] = True  # interior pillar
    void = np.zeros(dom.dims, dtype=bool)
    void[12:16, 4:8, :] = True  # carve through top and bottom skin
    _rho, mask, _meta = infill_field(dom, InfillParams(), [w], None, passive.ravel(), void.ravel())
    assert mask[passive].all()
    assert not mask[void].any()


def test_infill_field_guided_scope(solved_cantilever) -> None:
    """scope='guided': the lattice exists only inside the guide mask; the
    rest of the interior stays a hollow pocket under the closed skin."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    centers = dom.element_centers().reshape((*dom.dims, 3))
    guide = dom.solid & (centers[..., 0] < 0.0)

    _rho, mask, meta = infill_field(
        dom, InfillParams(scope="guided"), [w], None, None, None, guide_mask=guide
    )
    # the interior the module saw: solid minus the resolved skin
    skin = _skin_mask(dom, 2.0)
    interior = dom.solid & ~skin
    guided = interior & guide
    unguided = interior & ~guide
    assert (mask & guided).sum() > 0
    # occupancy only inside the guide — unguided interior is empty
    assert not (mask & unguided).any()
    # shell still closed — a hollow pocket is fine, a leak is not
    assert not _air_leak_cells(mask, dom.solid).any()
    assert meta["scope"] == "guided"

    with pytest.raises(ValueError, match="guide_mask"):
        infill_field(dom, InfillParams(scope="guided"), [w], None, None, None)


def test_infill_field_deterministic(solved_cantilever) -> None:
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    p = InfillParams(seed=7)
    rho1, mask1, _ = infill_field(dom, p, [w], None, None, None)
    rho2, mask2, _ = infill_field(dom, p, [w], None, None, None)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(rho1, rho2)


def test_infill_field_bad_params(solved_cantilever) -> None:
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    with pytest.raises(ValueError, match="style"):
        infill_field(dom, InfillParams(style="nope"), [w], None, None, None)
    with pytest.raises(ValueError, match="scope"):
        infill_field(dom, InfillParams(scope="nope"), [w], None, None, None)


def test_seed_shifts_lattice(solved_cantilever) -> None:
    """Different seeds translate the lattice phase — masks differ but the
    occupancy statistics and the closed shell stay the same."""
    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    _r1, m1, _meta1 = infill_field(dom, InfillParams(seed=1), [w], None, None, None)
    _r2, m2, _meta2 = infill_field(dom, InfillParams(seed=2), [w], None, None, None)
    assert not (m1 == m2).all()
    assert not _air_leak_cells(m2, dom.solid).any()


# --------------------------------------------------------------------------- #
# density_to_mesh / run_fem_domain integration                                 #
# --------------------------------------------------------------------------- #


def test_infill_density_to_mesh_watertight(solved_cantilever) -> None:
    """The (rho, mask) pair feeds the unchanged extraction path: threshold
    0.5, watertight mesh, bounds inside the design domain."""
    pytest.importorskip("skimage")
    from cadx.cad.fem.extract import density_to_mesh

    dom, _case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    rho, mask, _meta = infill_field(
        dom, InfillParams(shell_mm=2.0, infill_cell_mm=8.0), [w], None, None, None
    )
    mesh, final_mask, _warnings, n_components, _info = density_to_mesh(
        dom,
        rho,
        threshold=0.5,
        void_cells=(~dom.solid).reshape(dom.dims),
        min_member_mm=0.0,
        smooth_iterations=0,
    )
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert n_components >= 1
    # the extracted mask reproduces the requested structure (connectivity
    # pruning may drop small lattice islands — the shell+core survive)
    assert (final_mask & mask).sum() >= 0.9 * int(mask.sum())
    lo = dom.origin
    hi = dom.origin + dom.pitch * np.asarray(dom.dims)
    assert (mesh.bounds[0] >= lo - 1e-6).all()
    assert (mesh.bounds[1] <= hi + 1e-6).all()


def test_infill_mask_verifies_under_fem(solved_cantilever) -> None:
    """run_fem_domain on the lattice mask — the same call the TopOpt
    post-verification uses — must solve and report a safety factor."""
    from cadx.cad.fem.analyze import run_fem_domain

    dom, case, sol, edof, active, elem_E = solved_cantilever
    w = element_strain_energy(dom, sol.u, edof, active, elem_E, NU)
    _rho, mask, _meta = infill_field(
        dom, InfillParams(shell_mm=2.0, infill_cell_mm=8.0), [w], None, None, None
    )
    rep = run_fem_domain(dom, mask, [case], "pla")
    c = rep.cases[0]
    assert c.safety_factor is not None
    assert c.max_disp_mm > 0
