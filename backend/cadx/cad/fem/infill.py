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

from cadx.cad.fem.solver import _ke_from_D, _ke_unit, recover_elem_stress
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
    octet:          struts_aligned only — add the 4 body-diagonal families
                    (octet lattice) so 45°-dominated stress states grow real
                    diagonal members instead of stair-stepped axis pairs
    """

    shell_mm: float = 2.0
    infill_cell_mm: float = 8.0
    dmin: float = 0.08
    dmax: float = 0.90
    alpha: float = 0.7
    style: str = "gyroid_graded"
    scope: str = "full"
    seed: int = 0
    octet: bool = True


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
    *,
    aniso: tuple[float, float] | None = None,
    elem_class: np.ndarray | None = None,
    class_D: np.ndarray | None = None,
) -> np.ndarray:
    """(ncells,) element strain energy w_e = u_eᵀ(E_e·k_e)u_e.

    `u` is one case's global displacement (FemSolution.u, flat or (nnodes,3));
    `edof`/`active` the (nelem,24) dof table + flat cell ids returned by
    `solver.assemble` for the solved cell set; `elem_E` the (ncells,) modulus
    field the solve used. Cells outside `active` get 0 — same convention as
    FemSolution.elem_vm. Summed over the domain, w_e equals uᵀKu (= uᵀF, the
    compliance). On the uniform grid w_e ∝ strain-energy density (divide by
    pitch³ for J/mm³); the grading only uses relative values.

    `aniso`/`elem_class`/`class_D` must match the values `assemble` used —
    the energy is only consistent when k_e is the SAME element stiffness the
    solve ran with (`class_D` wins over `aniso`, like everywhere else).
    """
    out = np.zeros(domain.cell_count)
    active = np.asarray(active, dtype=np.int64)
    if not active.size:
        return out
    E = np.asarray(elem_E, dtype=float).ravel()[active]
    uf = np.asarray(u, dtype=float).reshape(-1)
    w = np.empty(active.size)
    if class_D is not None:
        ke_table = np.stack(
            [_ke_from_D(np.asarray(Dk, dtype=float), domain.pitch) for Dk in class_D]
        )
        cls = np.asarray(elem_class, dtype=np.int64).ravel()[active]
        for s in range(0, active.size, 200_000):  # chunked: 24 floats/cell adds up
            e = min(s + 200_000, active.size)
            ue = uf[edof[s:e]]  # (m, 24)
            for c in range(ke_table.shape[0]):
                sel = cls[s:e] == c
                if sel.any():
                    ue_c = ue[sel]
                    w[s:e][sel] = np.einsum("ij,ij->i", ue_c, ue_c @ ke_table[c].T)
    else:
        ke = (
            _ke_unit(float(nu), domain.pitch)
            if aniso is None
            else _ke_unit(float(nu), domain.pitch, float(aniso[0]), int(aniso[1]))
        )
        for s in range(0, active.size, 200_000):
            ue = uf[edof[s : s + 200_000]]  # (m, 24)
            w[s : s + 200_000] = np.einsum("ij,ij->i", ue, ue @ ke.T)
    out[active] = w * E
    return out


def _canonical_frame(evals: np.ndarray, evecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic eigensystem: per-eigenvector sign canonicalized so the
    largest-|component| entry is positive (strut directions are unsigned
    anyway), and degenerate tensors (max |λ| <= 1e-12) collapsed to zero
    evals + identity evecs so downstream code has one convention."""
    lead = np.argmax(np.abs(evecs), axis=1)  # (m,3): argmax component per column
    comps = np.take_along_axis(evecs, lead[:, None, :], axis=1)[:, 0, :]
    sgn = np.sign(comps)
    sgn[sgn == 0] = 1.0
    evecs = evecs * sgn[:, None, :]
    deg = np.max(np.abs(evals), axis=1) <= 1e-12
    if deg.any():
        evals = np.where(deg[:, None], 0.0, evals)
        evecs = np.where(deg[:, None, None], np.eye(3)[None], evecs)
    return evals, evecs


def element_principal_frames(
    domain: VoxelDomain,
    u: np.ndarray,
    edof: np.ndarray,
    active: np.ndarray,
    elem_E: np.ndarray,
    nu: float,
    *,
    aniso: tuple[float, float] | None = None,
    elem_class: np.ndarray | None = None,
    class_D: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(evals (ncells,3) ascending, evecs (ncells,3,3) columns) — the full
    principal-stress frame of each cell's centre stress tensor.

    Shares the `recover_elem_stress` path so aniso/per-class constitutive
    tensors produce the same stress the solve reported (a shared-D recovery
    would hand material zones the wrong frame). Eigenvalues carry the TRUE
    stress magnitude (unit-E stress scaled by E_e — eigenvectors are
    scale-invariant, but `combine_stress_frames` weights families by |λ|).
    Signs canonicalized via `_canonical_frame`; degenerate cells get zero
    evals + identity evecs, inactive cells stay all-zero.
    """
    evals_out = np.zeros((domain.cell_count, 3))
    evecs_out = np.zeros((domain.cell_count, 3, 3))
    active = np.asarray(active, dtype=np.int64)
    if not active.size:
        return evals_out, evecs_out
    _vm, sig = recover_elem_stress(
        domain,
        u,
        edof,
        active,
        elem_E,
        nu,
        aniso=aniso,
        # class tensors already carry E_e — force unit E on the shared-D
        # path so the per-element modulus can be applied honestly
        E_ref=None if class_D is not None else 1.0,
        elem_class=elem_class,
        class_D=class_D,
    )
    if class_D is None:
        sig = sig * np.asarray(elem_E, dtype=float).ravel()[active][:, None]
    for s in range(0, active.size, 100_000):
        sig_c = sig[s : s + 100_000]  # (m, 6) Voigt [sx,sy,sz,txy,tyz,tzx]
        m = sig_c.shape[0]
        S = np.zeros((m, 3, 3))
        S[:, 0, 0] = sig_c[:, 0]
        S[:, 1, 1] = sig_c[:, 1]
        S[:, 2, 2] = sig_c[:, 2]
        S[:, 0, 1] = S[:, 1, 0] = sig_c[:, 3]  # txy
        S[:, 1, 2] = S[:, 2, 1] = sig_c[:, 4]  # tyz
        S[:, 0, 2] = S[:, 2, 0] = sig_c[:, 5]  # tzx
        evals, evecs = np.linalg.eigh(S)  # ascending; columns are eigenvectors
        evals, evecs = _canonical_frame(evals, evecs)
        evals_out[active[s : s + m]] = evals
        evecs_out[active[s : s + m]] = evecs
    return evals_out, evecs_out


def combine_stress_frames(
    frames_list: list[tuple[np.ndarray, np.ndarray]],
    weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-case principal frames into one stress-weighted frame.

    T_c = Σ_j |λ_j|·(v_j ⊗ v_j) is a case's |stress| tensor; T = Σ_c w_c·T_c
    accumulates the weighted tensor once — a dominant case wins the
    dominant axis and orthogonal cases produce a genuinely bi-axial frame
    (unlike the old mean-of-directions, which snaps to whichever axis the
    average vector favours). Returns (evals (n,3) ascending, evecs
    (n,3,3) columns) in `_canonical_frame` convention."""
    T = None
    for (evals, evecs), w in zip(frames_list, weights, strict=True):
        if not w:
            continue
        lam = np.abs(np.asarray(evals, dtype=float))
        V = np.asarray(evecs, dtype=float)
        T_c = np.einsum("nj,naj,nbj->nab", lam, V, V)
        T = (float(w) * T_c) if T is None else T + float(w) * T_c
    if T is None:
        n = int(np.asarray(frames_list[0][0]).shape[0]) if frames_list else 0
        return np.zeros((n, 3)), np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    evals, evecs = np.linalg.eigh(T)
    return _canonical_frame(evals, evecs)


def element_principal_dirs(
    domain: VoxelDomain,
    u: np.ndarray,
    edof: np.ndarray,
    active: np.ndarray,
    elem_E: np.ndarray,
    nu: float,
    *,
    aniso: tuple[float, float] | None = None,
    elem_class: np.ndarray | None = None,
    class_D: np.ndarray | None = None,
) -> np.ndarray:
    """(ncells,3) unit direction of the largest-|sigma| principal stress.

    Thin wrapper over `element_principal_frames` (same `recover_elem_stress`
    stress path — aniso- and class-D-aware): picks the eigenvector of the
    largest-|λ| eigenvalue per cell. Signs are canonicalized (largest-
    |component| positive) so the result is reproducible across BLAS
    variants — a strut direction is unsigned anyway. Cells with vanishing
    stress get +z (build-axis default); inactive cells get zeros.
    """
    out = np.zeros((domain.cell_count, 3))
    active = np.asarray(active, dtype=np.int64)
    if not active.size:
        return out
    evals, evecs = element_principal_frames(
        domain,
        u,
        edof,
        active,
        elem_E,
        nu,
        aniso=aniso,
        elem_class=elem_class,
        class_D=class_D,
    )
    ev = evals[active]
    k = np.argmax(np.abs(ev), axis=1)
    v = evecs[active][np.arange(active.size), :, k]  # (nact, 3)
    # vanishing stress → degenerate eigenvector → build-axis default
    v[np.abs(ev[np.arange(active.size), k]) <= 1e-12] = (0.0, 0.0, 1.0)
    out[active] = v
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


_AXIS_FAMILIES = np.eye(3)
_OCTET_FAMILIES = np.array(
    [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]
) / math.sqrt(3.0)


def _strut_families(octet: bool) -> np.ndarray:
    """(nfam,3) strut family directions — the 3 axis families first, then
    the 4 body-diagonal families when `octet` (the octet-truss lattice:
    45°-dominated stress states get real diagonal members instead of
    stair-stepped axis pairs)."""
    return np.concatenate([_AXIS_FAMILIES, _OCTET_FAMILIES], axis=0) if octet else _AXIS_FAMILIES


def _perp_lattice_basis(f: np.ndarray) -> np.ndarray:
    """(3,2) basis of the strut-line lattice proj_{f⊥}(Z³).

    Family-f struts run along f through the integer grid — two grid points
    share one strut line iff they differ along f, so the in-plane strut
    positions are the projection of Z³ onto the plane ⊥f. Only commensurate
    integer directions are allowed (the projected point set is a lattice
    only then) — asserted. For (±1,±1,±1) body diagonals the two shortest
    projected axis vectors span the triangular lattice."""
    fn = np.asarray(f, dtype=float)
    fn = fn / np.linalg.norm(fn)
    scaled = fn / np.min(np.abs(fn[np.abs(fn) > 1e-12]))
    assert np.allclose(scaled, np.rint(scaled), atol=1e-9), (
        f"diagonal strut family must be a commensurate integer direction, got {f}"
    )
    cands: list[tuple[float, np.ndarray]] = []
    for n in np.indices((3, 3, 3)).reshape(3, -1).T - 1:
        v = n.astype(float) - (n.astype(float) @ fn) * fn
        ln = float(v @ v)
        if ln > 1e-9:
            cands.append((ln, v))
    cands.sort(key=lambda t: t[0])
    b1 = cands[0][1]
    b2 = cands[-1][1]
    for _ln, v in cands[1:]:
        if np.linalg.norm(np.cross(b1, v)) > 1e-9:
            b2 = v
            break
    return np.stack([b1, b2], axis=1)


def _strut_family_widths(
    n: int,
    d: np.ndarray,
    dirs: np.ndarray | None,
    frames: tuple[np.ndarray, np.ndarray] | None,
    fams: np.ndarray,
) -> np.ndarray:
    """(n, nfam) per-family rod half-widths in units of half the period.

    - `frames` = (evals, evecs): tensor-projected weight
      ``w_f² = clip(d,0,1) · Σ_i|λ_i|(f·v_i)² / max(Σ_i|λ_i|, 1e-12)`` —
      the family aligned with the dominant principal direction carries most
      of the local density, and a bi-axial stress state thickens two
      families honestly. Degenerate cells (Σ|λ| <= 1e-12) split evenly:
      ``w_f = sqrt(d / nfam)``.
    - `dirs` (legacy uniaxial): ``w_f = |f·d_unit|·sqrt(d)`` — the special
      case of the frame formula for a single eigenpair; zero-length rows
      fall back to the isotropic (1,1,1)/√3 direction.
    - neither: the isotropic blend ``w_f = sqrt(d / nfam)``.
    """
    nfam = fams.shape[0]
    dc = np.clip(d, 0.0, 1.0)
    w = np.empty((n, nfam))
    if frames is not None:
        evals = np.asarray(frames[0], dtype=float).reshape(n, 3)
        evecs = np.asarray(frames[1], dtype=float).reshape(n, 3, 3)
        lam = np.abs(evals)
        tsum = lam.sum(axis=1)
        deg = tsum <= 1e-12
        for fi in range(nfam):
            pr = evecs @ fams[fi]  # (n,3): (f·v_i) per eigenpair
            w2 = dc * (lam * pr * pr).sum(axis=1) / np.maximum(tsum, 1e-12)
            w[:, fi] = np.sqrt(np.clip(w2, 0.0, None))
        w[deg] = np.sqrt(dc / nfam)[deg][:, None]
    elif dirs is not None:
        v = np.asarray(dirs, dtype=float).reshape(n, 3)
        vn = np.linalg.norm(v, axis=1)
        iso = np.full(3, 1.0 / math.sqrt(3.0))
        uvec = np.where(vn[:, None] > 1e-9, v / np.maximum(vn[:, None], 1e-30), iso)
        for fi in range(nfam):
            w[:, fi] = np.abs(uvec @ fams[fi]) * np.sqrt(dc)
    else:
        w[:] = np.sqrt(dc / nfam)[:, None]
    return np.clip(w, 0.0, 1.0)


def _strut_family_dists(centers: np.ndarray, p: float, fams: np.ndarray) -> np.ndarray:
    """(n, nfam) normalized distance to the nearest strut line per family —
    0 on a strut axis, 1 at the in-plane mid-point between struts. Axis
    families use the square-rod Chebyshev distance on the p-grid; diagonal
    families the Euclidean distance to the projected integer lattice
    proj_{f⊥}(Z³) (triangular for body diagonals), expressed in the same
    half-period units."""
    n = centers.shape[0]
    dists = np.empty((n, fams.shape[0]))
    for fi in range(fams.shape[0]):
        f = fams[fi]
        if fi < 3:
            o0, o1 = tuple(i for i in range(3) if i != fi)
            dists[:, fi] = np.maximum(_line_dist(centers[:, o0], p), _line_dist(centers[:, o1], p))
        else:
            xu = centers / p  # period units
            xp = xu - (xu @ f)[:, None] * f[None, :]
            B = _perp_lattice_basis(f)  # (3,2)
            c = np.linalg.lstsq(B, xp.T, rcond=None)[0].T  # (n,2)
            c0 = np.floor(c)
            dd = np.full(n, np.inf)
            # the 4 parallelogram corners around the lattice coords — for
            # the triangular lattice the nearest site is always among them
            for d1 in (0.0, 1.0):
                for d2 in (0.0, 1.0):
                    cand = c0 + np.array([d1, d2])
                    dd = np.minimum(dd, np.linalg.norm(xp - cand @ B.T, axis=1))
            dists[:, fi] = 2.0 * dd  # dist/(p/2) expressed in period units
    return dists


def strut_occupancy(
    domain: VoxelDomain,
    d_target: np.ndarray,
    dirs: np.ndarray | None = None,
    cell_mm: float = 8.0,
    *,
    frames: tuple[np.ndarray, np.ndarray] | None = None,
    margin_cells: int = 1,
    phase_mm: np.ndarray | None = None,
    octet: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """(nx,ny,nz) bool — direction/frame-weighted strut lattice.

    Strut families — the 3 axis rods plus the 4 body-diagonal rods when
    `octet` — are superposed by max; per-cell rod half-widths come from
    `_strut_family_widths` (tensor-projected frame weights, uniaxial `dirs`
    legacy, or the isotropic blend).

    `normalize` rescales ALL widths by one bisected global factor so the
    measured interior occupancy meets the target mean — the Σw² union
    superposition overfills intersections ~3x otherwise. Applied before the
    bonding band; `margin_cells`/`phase_mm`/`full`-pin/`d>0` semantics are
    identical to the axis-only version.
    """
    centers = domain.element_centers()
    if phase_mm is not None:
        centers = centers + np.asarray(phase_mm, dtype=float)[None, :]
    d = np.asarray(d_target, dtype=float).ravel()
    n = domain.cell_count
    fams = _strut_families(octet)
    p = max(float(cell_mm), 1e-6)

    w = _strut_family_widths(n, d, dirs, frames, fams)
    dists = _strut_family_dists(centers, p, fams)

    # ---- optional global width normalization ----------------------------
    # the union superposition overshoots Σw²; a single bisected multiplier
    # pulls the measured interior fraction back onto the target mean
    scale = 1.0
    graded = (d > 0.0) & (d < _FULL_DENSITY)
    gidx = np.flatnonzero(graded)
    if normalize and gidx.size:
        target = float(d[gidx].mean())
        if target > 0.0:
            wd = w[gidx]
            dd = dists[gidx]

            def _frac(s: float) -> float:
                return float((dd <= np.clip(s * wd, 0.0, 1.0)).any(axis=1).mean())

            lo_s, hi_s = 0.0, 1.0
            while _frac(hi_s) < target and hi_s < 16.0:
                hi_s *= 2.0
            for _ in range(5):
                mid = 0.5 * (lo_s + hi_s)
                if _frac(mid) < target:
                    lo_s = mid
                else:
                    hi_s = mid
            # quantized occupancy: pick whichever bracket lands closer to
            # the target (a coarse grid can only step in whole cell rings)
            scale = lo_s if abs(_frac(lo_s) - target) <= abs(_frac(hi_s) - target) else hi_s
    occ = (dists <= np.clip(scale * w, 0.0, 1.0)).any(axis=1)

    full = d >= _FULL_DENSITY
    occ = (occ | full) & (d > 0.0)
    occ |= _bonding_band(full, domain.dims, margin_cells) & (d > 0.0)
    return np.asarray(occ.reshape(domain.dims))


def strut_alignment_deg(
    domain: VoxelDomain,
    occ3: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    *,
    octet: bool = True,
) -> dict[str, float | None]:
    """Alignment error of the built lattice vs the target stress frame.

    For each occupied, non-degenerate cell the achieved dominant strut is
    the family direction closest to the target dominant principal axis v1;
    the error is the unsigned angle between them (deg). Returns
    {"mean", "p95"} over the measured cells — None entries when no cell
    qualifies (no occupied lattice or all frames degenerate)."""
    occ = np.asarray(occ3, dtype=bool).ravel()
    ev = np.asarray(evals, dtype=float).reshape(-1, 3)
    V = np.asarray(evecs, dtype=float).reshape(-1, 3, 3)
    good = occ & (np.max(np.abs(ev), axis=1) > 1e-12)
    if not good.any():
        return {"mean": None, "p95": None}
    v1 = V[good, :, 2]  # dominant eigenvector (ascending eval order)
    fams = _strut_families(octet)
    cos = np.clip(np.abs(v1 @ fams.T).max(axis=1), -1.0, 1.0)
    err = np.degrees(np.arccos(cos))
    return {"mean": float(err.mean()), "p95": float(np.quantile(err, 0.95))}


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
    density_offset: float = 0.0,
    frames: tuple[np.ndarray, np.ndarray] | None = None,
    fields_out: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the shell+infill density field and occupancy mask.

    interior = solid - skin - passive - void (and ∩ guide_mask when
    scope="guided"); the skin is the padded-EDT band `_skin_mask` at the
    resolved shell_mm, `energies` is one (ncells,) array per load case from
    `element_strain_energy`, `dirs` the matching principal-direction field.

    For style="struts_aligned" either `frames` — the combined
    (evals, evecs) field from `element_principal_frames` /
    `combine_stress_frames` — or `dirs` is required; with neither the style
    falls back to "gyroid_graded" (reported in meta). `frames` takes the
    tensor-projected family path in `strut_occupancy`; `dirs` stays the
    meta/fallback driver.

    `fields_out` (optional dict) receives the internal fields the
    homogenized-verification path needs: ``d`` (post-offset target
    density), ``interior`` (flat bool, post guide-intersection) and
    ``occ3`` (flat bool, final lattice occupancy).

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
    if density_offset:
        gradable = interior.ravel()
        d[gradable] = np.clip(d[gradable] + float(density_offset), p.dmin, p.dmax)

    phase = _seed_phase(p.seed, p.infill_cell_mm)
    style = p.style
    if style == "struts_aligned" and dirs is None and frames is None:
        log.warning("infill.struts_no_dirs", fallback="gyroid_graded")
        style = "gyroid_graded"
    if style == "struts_aligned":
        occ = strut_occupancy(
            domain,
            d,
            None if dirs is None else np.asarray(dirs, dtype=float),
            p.infill_cell_mm,
            frames=frames,
            phase_mm=phase,
            octet=p.octet,
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

    # achieved-vs-target alignment of the strut lattice — scored on the
    # same family set the occupancy was built from; dirs only get a
    # synthesized uniaxial frame when no tensor frame was fed in
    align_deg: float | None = None
    if style == "struts_aligned":
        if frames is not None:
            fa_e = np.asarray(frames[0], dtype=float)
            fa_v = np.asarray(frames[1], dtype=float)
        else:
            vv = np.asarray(dirs, dtype=float).reshape(-1, 3)
            vn = np.linalg.norm(vv, axis=1)
            fa_e = np.zeros((domain.cell_count, 3))
            fa_e[:, 2] = (vn > 1e-9).astype(float)
            fa_v = np.zeros((domain.cell_count, 3, 3))
            fa_v[:, :, 2] = np.where(vn[:, None] > 1e-9, vv / np.maximum(vn[:, None], 1e-30), 0.0)
        align_deg = strut_alignment_deg(domain, occ3, fa_e, fa_v, octet=p.octet)["mean"]

    if fields_out is not None:
        fields_out.update(
            d=d,
            interior=interior.ravel(),
            occ3=occ3.ravel(),
        )

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
        "families": (7 if p.octet else 3) if style == "struts_aligned" else None,
        "strut_align_deg": round(align_deg, 2) if align_deg is not None else None,
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
