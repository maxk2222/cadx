"""Shell+Infill structure method — an FDM-native alternative to SIMP.

Where SIMP carves the part into hollow pockets and organic members, this
method keeps a closed outer skin (the slicer's "wall") and fills the
interior with a density-graded lattice — the shell+infill strategy every
FDM slicer implements, but graded by the part's own stress field. ONE
ordinary FEM solve per load case (computed by the caller on the FULL solid
domain — `analyze.run_fem_domain` / `solver.solve_reduced` over
`domain.solid`) supplies the element strain energies that drive the local
lattice density.

Pipeline:

1. `element_strain_energy`  — per-cell energy u_eᵀ(E_e·k_e)u_e from the
   solved displacement field (the edof/active tables of `solver.assemble`).
2. `element_principal_dirs` — per-cell direction of the largest-|sigma|
   principal stress, recomputed from u via the solver's own _B0/_D
   matrices (only needed for the "struts_aligned" style).
3. `target_density`         — p95-normalized energy, weighted max across
   load cases, shaped into [dmin, dmax] by the grading exponent alpha.
4. `gyroid_occupancy` / `strut_occupancy` — binary lattice masks whose
   local occupancy fraction approximates the target density.
5. `infill_field`           — orchestration: closed skin + graded lattice +
   passive/void composition; returns (rho, mask, meta) that drop straight
   into `density_to_mesh(rho, threshold=0.5)` and `run_fem_domain(mask)`
   unchanged.

Deterministic: `seed` only shifts the lattice phase — no sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from cadx.cad.fem.solver import _B0, _D, _ke_unit
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

INFILL_STYLES = ("gyroid_graded", "struts_aligned")
INFILL_SCOPES = ("full", "guided")

# d_target >= this reads as "print solid" — unconditionally occupied cells
# that bond the lattice to the skin/passive zones (the 0.995 edge leaves a
# gap to dmax so graded cells keep lattice character even at the cap)
_FULL_DENSITY = 0.995


@dataclass
class InfillParams:
    """Parameters of the Shell+Infill structure method.

    shell_mm:       closed outer skin thickness; 0 = auto (~2 voxel pitch)
    infill_cell_mm: lattice period (gyroid wavelength / strut pitch)
    dmin/dmax:      interior local density floor / cap — the shell and
                    passive regions sit above the cap and print solid
    alpha:          energy→density grading exponent (<1 flattens the
                    grading, >1 concentrates density on hot cells)
    style:          "gyroid_graded" (isotropic TPMS sheet) or
                    "struts_aligned" (axis-strut lattice weighted by the
                    principal stress direction — best-effort, see
                    `strut_occupancy` for limits)
    scope:          "full" fills the whole interior; "guided" restricts the
                    lattice to a caller-supplied guide mask (the rest of the
                    interior stays a hollow pocket enclosed by the skin)
    seed:           deterministic lattice phase shift
    """

    shell_mm: float = 2.0
    infill_cell_mm: float = 8.0
    dmin: float = 0.08
    dmax: float = 0.90
    alpha: float = 0.7
    style: str = "gyroid_graded"
    scope: str = "full"
    seed: int = 0


def _flat(mask: np.ndarray | None, domain: VoxelDomain) -> np.ndarray:
    """(ncells,) bool — accepts flat or (nx,ny,nz) masks, None → empty."""
    if mask is None:
        return np.zeros(domain.cell_count, dtype=bool)
    return np.asarray(mask, dtype=bool).ravel()


def _skin_mask(domain: VoxelDomain, shell_mm: float) -> np.ndarray:
    """(nx,ny,nz) bool — solid cells whose centre sits within `shell_mm` of
    the part surface. Same padded-EDT technique as organic._skin_mask: pad
    one void layer so the grid boundary counts as surface;
    depth = (EDT - 0.5) * pitch measures from the cell centre to the
    nearest void/exterior face."""
    if shell_mm <= 0:
        return np.zeros(domain.dims, dtype=bool)
    padded = np.pad(domain.solid, 1, constant_values=False)
    depth = (ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1] - 0.5) * domain.pitch
    return np.asarray(domain.solid & (depth <= shell_mm))


def _seed_phase(seed: int, cell_mm: float) -> np.ndarray:
    """Deterministic lattice phase offset (mm) from the seed — the whole
    pattern translates rigidly, so different seeds give different wall
    registrations without any structural change."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    return rng.uniform(0.0, max(float(cell_mm), 1e-6), size=3)


# --------------------------------------------------------------------------- #
# per-element fields from a solved load case                                   #
# --------------------------------------------------------------------------- #


def element_strain_energy(
    domain: VoxelDomain,
    u: np.ndarray,
    edof: np.ndarray,
    active: np.ndarray,
    elem_E: np.ndarray,
    nu: float,
) -> np.ndarray:
    """(ncells,) element strain energy w_e = u_eᵀ(E_e·k_e)u_e.

    `u` is one case's global displacement (FemSolution.u, flat or (nnodes,3));
    `edof`/`active` the (nelem,24) dof table + flat cell ids returned by
    `solver.assemble` for the solved cell set; `elem_E` the (ncells,) modulus
    field the solve used. Cells outside `active` get 0 — same convention as
    FemSolution.elem_vm. Summed over the domain, w_e equals uᵀKu (= uᵀF, the
    compliance). On the uniform grid w_e ∝ strain-energy density (divide by
    pitch³ for J/mm³); the grading only uses relative values.
    """
    out = np.zeros(domain.cell_count)
    active = np.asarray(active, dtype=np.int64)
    if not active.size:
        return out
    ke = _ke_unit(float(nu), domain.pitch)
    E = np.asarray(elem_E, dtype=float).ravel()[active]
    uf = np.asarray(u, dtype=float).reshape(-1)
    w = np.empty(active.size)
    for s in range(0, active.size, 200_000):  # chunked: 24 floats/cell adds up
        ue = uf[edof[s : s + 200_000]]  # (m, 24)
        w[s : s + 200_000] = np.einsum("ij,ij->i", ue, ue @ ke.T)
    out[active] = w * E
    return out


def element_principal_dirs(
    domain: VoxelDomain,
    u: np.ndarray,
    edof: np.ndarray,
    active: np.ndarray,
    elem_E: np.ndarray,
    nu: float,
) -> np.ndarray:
    """(ncells,3) unit direction of the largest-|sigma| principal stress.

    Recomputes the cell-centre stress exactly like `solve_reduced` post
    passes do: eps = u_e·_B0(pitch)ᵀ, sig = eps·Dᵀ with D = _D(nu) scaled by
    the element's own modulus (a positive per-element scale does not change
    eigenvectors — it only keeps the tensor honest). The Voigt 6-vector is
    [sx, sy, sz, txy, tyz, tzx] (row order of `solver._B_matrix`), mapped
    onto the symmetric 3x3 tensor and eigh'ed; the eigenvector of the
    largest-|λ| eigenvalue is returned.

    Signs are canonicalized (largest-|component| positive) so the result is
    reproducible across BLAS variants — a strut direction is unsigned
    anyway. Cells with vanishing stress get +z (build-axis default);
    inactive cells get zeros.
    """
    out = np.zeros((domain.cell_count, 3))
    active = np.asarray(active, dtype=np.int64)
    if not active.size:
        return out
    B0 = _B0(domain.pitch)
    D = _D(float(nu))
    E = np.asarray(elem_E, dtype=float).ravel()[active]
    uf = np.asarray(u, dtype=float).reshape(-1)
    dirs = np.zeros((active.size, 3))
    for s in range(0, active.size, 100_000):
        ue = uf[edof[s : s + 100_000]]  # (m, 24)
        eps = ue @ B0.T  # (m, 6)
        sig = (eps @ D.T) * E[s : s + 100_000, None]  # (m, 6) Voigt
        m = ue.shape[0]
        S = np.zeros((m, 3, 3))
        S[:, 0, 0] = sig[:, 0]
        S[:, 1, 1] = sig[:, 1]
        S[:, 2, 2] = sig[:, 2]
        S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]  # txy
        S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]  # tyz
        S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]  # tzx
        evals, evecs = np.linalg.eigh(S)  # ascending; columns are eigenvectors
        sl = np.arange(s, min(s + 100_000, active.size))
        k = np.argmax(np.abs(evals), axis=1)
        v = evecs[np.arange(m), :, k]  # (m, 3)
        # canonical sign: the largest-|component| entry is positive
        lead = np.argmax(np.abs(v), axis=1)
        sgn = np.sign(v[np.arange(m), lead])
        sgn[sgn == 0] = 1.0
        v = v * sgn[:, None]
        # vanishing stress → degenerate eigenvector → build-axis default
        v[np.abs(evals[np.arange(m), k]) <= 1e-12] = (0.0, 0.0, 1.0)
        dirs[sl] = v
    out[active] = dirs
    return out


# --------------------------------------------------------------------------- #
# energy -> target density                                                     #
# --------------------------------------------------------------------------- #


def target_density(
    domain: VoxelDomain,
    energy_cases: list[np.ndarray] | np.ndarray,
    weights: np.ndarray | list[float] | None = None,
    shell_mask: np.ndarray | None = None,
    passive: np.ndarray | None = None,
    void: np.ndarray | None = None,
    *,
    dmin: float = 0.08,
    dmax: float = 0.9,
    alpha: float = 0.7,
) -> np.ndarray:
    """(ncells,) interior target density in [dmin, dmax].

    Each case's energy is normalized by its p95 over the gradable cells —
    `solid` minus the supplied shell/passive/void masks, i.e. the "p95 over
    solid" of the cells the grading actually applies to (shell hotspots at
    fix/load boundaries would otherwise compress the interior grading band).
    With no masks supplied the whole solid is the reference, matching the
    documented p95-over-solid contract literally.

    Cases combine by weighted max (weights are relative — normalized to a
    max of 1 so the dominant case keeps full scale; default: equal), then
    d = clip(dmin + (dmax-dmin)·w^alpha, dmin, dmax).

    Composition contract (documented, applied here so the field is
    self-contained): shell_mask and passive cells return 1.0 — they print
    solid and gyroid_occupancy treats d >= ~0.995 as unconditionally
    occupied — while void and non-solid cells return 0.0. `infill_field`
    recomposes the mask explicitly and does not rely on these values.
    """
    solid = domain.solid.ravel()
    shell = _flat(shell_mask, domain)
    pas = _flat(passive, domain) & solid
    vod = _flat(void, domain)
    pinned = shell | pas
    gradable = solid & ~pinned & ~vod

    d = np.zeros(domain.cell_count)
    d[pinned & ~vod] = 1.0  # prints solid regardless; a void beats a pin
    ref = np.flatnonzero(gradable)
    if not ref.size:
        ref = np.flatnonzero(solid)

    if isinstance(energy_cases, np.ndarray):
        cases = [row for row in np.atleast_2d(np.asarray(energy_cases, dtype=float))]
    else:
        cases = [np.asarray(e, dtype=float).ravel() for e in energy_cases]
    if cases and ref.size:
        w = np.ones(len(cases)) if weights is None else np.asarray(list(weights), dtype=float)
        if w.size != len(cases):
            raise ValueError("weights must match the number of energy cases")
        if w.size and float(np.max(np.abs(w))) > 0:
            w = w / float(np.max(np.abs(w)))
        wn = np.zeros(ref.size)
        for wi, e in zip(w, cases, strict=True):
            p95 = float(np.quantile(e[ref], 0.95)) if ref.size else 0.0
            if p95 <= 0:
                continue  # zero/degenerate case contributes nothing
            wn = np.maximum(wn, wi * np.clip(e[ref] / p95, 0.0, 1.0))
        d[ref] = np.clip(dmin + (dmax - dmin) * np.power(wn, alpha), dmin, dmax)
    elif ref.size:
        d[ref] = dmin  # no energy information — uniform floor-density infill
    return d


# --------------------------------------------------------------------------- #
# target density -> binary lattice occupancy                                   #
# --------------------------------------------------------------------------- #


def _bonding_band(full: np.ndarray, dims: tuple[int, int, int], cells: int) -> np.ndarray:
    """Flat bool — the `cells`-wide dilation ring of unconditionally-occupied
    regions: the solid band where the lattice lands flush on the skin and on
    passive zones instead of touching at isolated threshold-edge points."""
    if cells <= 0 or not full.any():
        return np.zeros(len(full), dtype=bool)
    band = ndimage.binary_dilation(full.reshape(dims), iterations=int(cells))
    return np.asarray(band.ravel() & ~full)


def gyroid_occupancy(
    domain: VoxelDomain,
    d_target: np.ndarray,
    cell_mm: float,
    *,
    margin_cells: int = 1,
    phase_mm: np.ndarray | None = None,
) -> np.ndarray:
    """(nx,ny,nz) bool — TPMS gyroid sheet occupancy.

    g = sin(2πx/p)cos(2πy/p) + sin(2πy/p)cos(2πz/p) + sin(2πz/p)cos(2πx/p)
    evaluated at cell centres (world mm, period p = cell_mm, plus the
    deterministic `phase_mm` shift). Occupancy where g > t(x) with the
    per-cell threshold t taken from the GLOBAL empirical CDF of g over the
    gradable cells (0 < d < ~0.995): t(x) = quantile(g, 1 - d(x)), so local
    occupancy ≈ d for slowly varying d.

    Cells with d >= ~0.995 are occupied unconditionally — that is what bonds
    the lattice to the skin/passive zones (they return 1.0 from
    `target_density`). Cells with d <= 0 (void, outside the part) are never
    occupied. `margin_cells` adds a solid bonding ring around the
    unconditionally-occupied set — it effectively thickens the skin by
    margin_cells·pitch on the interior face only.
    """
    centers = domain.element_centers()
    if phase_mm is not None:
        centers = centers + np.asarray(phase_mm, dtype=float)[None, :]
    k = 2.0 * math.pi / max(float(cell_mm), 1e-6)
    x, y, z = centers[:, 0] * k, centers[:, 1] * k, centers[:, 2] * k
    g = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)

    d = np.asarray(d_target, dtype=float).ravel()
    full = d >= _FULL_DENSITY
    graded = (d > 0.0) & ~full
    occ = np.zeros(domain.cell_count, dtype=bool)
    ref = g[graded]
    if ref.size:
        order = np.sort(ref)
        q = np.clip(1.0 - d[graded], 0.0, 1.0)
        idx = np.clip(np.round(q * (ref.size - 1)).astype(np.int64), 0, ref.size - 1)
        occ[graded] = ref > order[idx]
    occ |= full
    # guaranteed bond: solid ring where the lattice meets skin/passives
    occ |= _bonding_band(full, domain.dims, margin_cells) & (d > 0.0)
    return np.asarray(occ.reshape(domain.dims))


def _line_dist(x: np.ndarray, period: float) -> np.ndarray:
    """Distance to the nearest multiple of `period`, normalized to [0,1] —
    0 on a grid line, 1 at the mid-plane between lines."""
    frac = np.mod(x / period, 1.0)
    return 2.0 * np.minimum(frac, 1.0 - frac)


def strut_occupancy(
    domain: VoxelDomain,
    d_target: np.ndarray,
    dirs: np.ndarray,
    cell_mm: float,
    *,
    margin_cells: int = 1,
    phase_mm: np.ndarray | None = None,
) -> np.ndarray:
    """(nx,ny,nz) bool — direction-weighted strut lattice (stage-2).

    Three orthogonal strut families — square rods along x, y, z on a
    cell_mm grid — are superposed by max. Per cell, family a's rod
    half-width is w_a = |dir_a|·sqrt(d) (in units of half the period), so a
    purely x-directed stress state grows only x-struts at occupancy ≈ d and
    a general direction gets the thickest member along its dominant axis.
    `dirs` is the (ncells,3) principal-direction field from
    `element_principal_dirs`; zero-length rows fall back to an isotropic
    (1,1,1)/√3 blend.

    Documented limits (best-effort stage): principal directions snap to
    grid axes — a 45° stress state builds stair-stepped x/y struts rather
    than true diagonal members; rod intersections overfill locally (the
    Σw² superposition beats the d target by up to ~3x at nodes); lattice
    connectivity comes from family overlap plus the same bonding band the
    gyroid uses, not from explicit node placement.
    """
    centers = domain.element_centers()
    if phase_mm is not None:
        centers = centers + np.asarray(phase_mm, dtype=float)[None, :]
    d = np.asarray(d_target, dtype=float).ravel()
    v = np.asarray(dirs, dtype=float).reshape(-1, 3)
    vn = np.linalg.norm(v, axis=1)
    iso = np.full(3, 1.0 / math.sqrt(3.0))
    uvec = np.where(vn[:, None] > 1e-9, v / np.maximum(vn[:, None], 1e-30), iso)
    width = np.abs(uvec) * np.sqrt(np.clip(d, 0.0, 1.0))[:, None]  # (n,3)

    p = max(float(cell_mm), 1e-6)
    occ = np.zeros(domain.cell_count, dtype=bool)
    for ax in range(3):
        o0, o1 = tuple(i for i in range(3) if i != ax)
        saw = np.maximum(
            _line_dist(centers[:, o0], p), _line_dist(centers[:, o1], p)
        )  # 0 on a strut line along `ax`
        occ |= saw <= np.clip(width[:, ax], 0.0, 1.0)
    full = d >= _FULL_DENSITY
    occ = (occ | full) & (d > 0.0)
    occ |= _bonding_band(full, domain.dims, margin_cells) & (d > 0.0)
    return np.asarray(occ.reshape(domain.dims))


# --------------------------------------------------------------------------- #
# orchestration                                                                #
# --------------------------------------------------------------------------- #


def _resolved_shell_mm(domain: VoxelDomain, p: InfillParams) -> float:
    """Skin depth — explicit `shell_mm`, or ~2 pitch when auto (0): one
    closed wall layer even on the coarsest grid."""
    if p.shell_mm > 0:
        return float(p.shell_mm)
    return float(2.0 * domain.pitch)


def infill_field(
    domain: VoxelDomain,
    p: InfillParams,
    energies: list[np.ndarray] | np.ndarray,
    dirs: np.ndarray | None,
    passive: np.ndarray | None,
    void: np.ndarray | None,
    *,
    guide_mask: np.ndarray | None = None,
    weights: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the shell+infill density field and occupancy mask.

    interior = solid - skin - passive - void (and ∩ guide_mask when
    scope="guided"); the skin is the padded-EDT band `_skin_mask` at the
    resolved shell_mm, `energies` is one (ncells,) array per load case from
    `element_strain_energy`, `dirs` the matching principal-direction field
    (required for style="struts_aligned" — without it the style falls back
    to "gyroid_graded" and the fallback is reported in meta).

    mask = (skin | occupancy | passive) & ~void & solid — void wins over
    every pin (a declared carve opens the shell deliberately). Cells of the
    interior left outside a guided mask stay empty: an enclosed hollow
    pocket under the closed skin, printed hollow like a 0%-infill region —
    pin it via `passive` if it should stay solid.

    Returns (rho (ncells,) float32, mask (nx,ny,nz) bool, meta). rho is the
    binary field — 1.0 inside mask, 0 outside — so `density_to_mesh(rho,
    threshold=0.5, passive_solid=..., void_cells=void|~solid)` reproduces
    exactly this structure and `run_fem_domain(domain, mask, ...)` verifies
    the same cells the export mesh was cut from.
    """
    if p.style not in INFILL_STYLES:
        raise ValueError(f"unknown infill style {p.style!r} — expected one of {INFILL_STYLES}")
    if p.scope not in INFILL_SCOPES:
        raise ValueError(f"unknown infill scope {p.scope!r} — expected one of {INFILL_SCOPES}")

    solid3 = domain.solid
    solid = solid3.ravel()
    pas = _flat(passive, domain) & solid
    vod = _flat(void, domain)
    pas3, vod3 = pas.reshape(domain.dims), vod.reshape(domain.dims)

    shell_mm = _resolved_shell_mm(domain, p)
    skin = _skin_mask(domain, shell_mm)
    core = solid3 & ~skin & ~pas3 & ~vod3
    interior = core.copy()
    if p.scope == "guided":
        if guide_mask is None:
            raise ValueError('scope="guided" requires a guide_mask')
        interior &= np.asarray(guide_mask, dtype=bool).reshape(domain.dims)

    d = target_density(
        domain,
        energies,
        weights,
        shell_mask=skin.ravel(),
        passive=pas,
        void=vod,
        dmin=p.dmin,
        dmax=p.dmax,
        alpha=p.alpha,
    )

    phase = _seed_phase(p.seed, p.infill_cell_mm)
    style = p.style
    if style == "struts_aligned" and dirs is None:
        log.warning("infill.struts_no_dirs", fallback="gyroid_graded")
        style = "gyroid_graded"
    if style == "struts_aligned":
        occ = strut_occupancy(
            domain, d, np.asarray(dirs, dtype=float), p.infill_cell_mm, phase_mm=phase
        )
    else:
        occ = gyroid_occupancy(domain, d, p.infill_cell_mm, phase_mm=phase)
    occ3 = np.asarray(occ, dtype=bool) & interior  # occupancy is interior-only

    # drop lattice islands not bonded to the skin/passive skeleton — a
    # floating gyroid fragment inside the closed shell cannot be printed
    # (nothing holds it) and would fail export verification as a second
    # body. The bonding band widens the weld region; whatever still does
    # not reach the skeleton goes.
    dropped_islands = 0
    if occ3.any():
        anchor = skin | pas3
        lab, _n = ndimage.label(occ3 | anchor)
        bonded = np.unique(lab[anchor])
        bonded = bonded[bonded > 0]
        keep = np.isin(lab, bonded)
        dropped_islands = int(occ3.sum() - (occ3 & keep).sum())
        occ3 &= keep

    mask = np.asarray((skin | occ3 | pas3) & ~vod3 & solid3)
    rho = mask.ravel().astype(np.float32)

    di = d[interior.ravel()]
    meta: dict[str, Any] = {
        "style": style,
        "style_requested": p.style,
        "scope": p.scope,
        "shell_mm": round(shell_mm, 4),
        "cell_mm": float(p.infill_cell_mm),
        "skin_cells": int(skin.sum()),
        "interior_cells": int(interior.sum()),
        "core_cells": int(core.sum()),
        "lattice_cells": int(occ3.sum()),
        "degenerated_to_solid": bool(
            int(core.sum()) == 0
            or (int(interior.sum()) > 0 and int(occ3.sum()) >= int(interior.sum()))
        ),
        "d_mean": round(float(di.mean()), 4) if di.size else None,
        "d_p50": round(float(np.quantile(di, 0.5)), 4) if di.size else None,
        "d_p95": round(float(np.quantile(di, 0.95)), 4) if di.size else None,
        "infill_occupancy": round(float(occ3.sum() / max(int(interior.sum()), 1)), 4),
        "dropped_lattice_cells": dropped_islands,
        "volfrac_estimate": round(float(mask.sum() / max(domain.solid_count, 1)), 4),
    }
    log.info(
        "infill.field",
        style=style,
        scope=p.scope,
        skin_cells=int(skin.sum()),
        interior_cells=int(interior.sum()),
        volfrac=meta["volfrac_estimate"],
    )
    return rho, mask, meta
