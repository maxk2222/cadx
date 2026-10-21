"""Pseudo-topology-optimization: procedural organic restyling (no FEM/SIMP).

Generates a deterministic synthetic density field inside the part's voxel
design domain and reuses the shared TopOpt extraction path
(`density_to_mesh`) so the output is watertight, connectivity-pruned and
respects min-member constraints. Purely aesthetic — the result is NOT
structurally optimized or verified.

Field styles:
- gyroid:     TPMS sheet — smooth minimal-surface walls
- trabecular: bone-like spongy core (smoothed white noise, two octaves)
- voronoi:    organic cell walls from a seeded Voronoi diagram
- branches:   floor-rooted struts growing into a canopy (anisotropic
              noise + segment trunks toward keep regions / top surface)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

from cadx.cad.fem.extract import density_to_mesh
from cadx.cad.fem.regions import auto_keep_regions, empty_region_warnings, keep_element_mask
from cadx.cad.fem.voxelize import VoxelDomain, voxelize
from cadx.llm.schemas import OrganicParams
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)

ORGANIC_STYLES = ("gyroid", "trabecular", "voronoi", "branches")


@dataclass
class OrganicResult:
    """In-process result — `report` is the JSON-safe part; the numpy fields
    let the pipeline persist the density field for later resampling."""

    mesh: trimesh.Trimesh
    report: dict[str, Any]
    density: np.ndarray  # flat float32 field (cell order)
    mask: np.ndarray  # final solid mask (nx,ny,nz)
    passive: np.ndarray  # flat bool — pinned-solid cells
    void: np.ndarray  # flat bool — carved cells (persisted for resample)
    domain: VoxelDomain


# --------------------------------------------------------------------------- #
# field generators (all return a (nx,ny,nz) float field normalized to [0,1])    #
# --------------------------------------------------------------------------- #


def _normalize(rho: np.ndarray, solid: np.ndarray) -> np.ndarray:
    out = np.asarray(rho, dtype=np.float32)
    out[~solid] = 0.0
    peak = float(out[solid].max()) if solid.any() else 0.0
    if peak > 1e-9:
        out = out / peak
    out[~solid] = 0.0
    return np.asarray(out, dtype=np.float32)


def _field_gyroid(domain: VoxelDomain, p: OrganicParams) -> np.ndarray:
    c = domain.element_centers()  # (ncells, 3) world-mm
    k = 2.0 * math.pi / p.period_mm
    x, y, z = c[:, 0] * k, c[:, 1] * k, c[:, 2] * k
    g = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    # sheet solid: density peaks where |g| -> 0 (the minimal surface)
    return 1.0 - np.abs(g.reshape(domain.dims)) / 3.0


def _field_trabecular(
    domain: VoxelDomain, p: OrganicParams, rng: np.random.Generator
) -> np.ndarray:
    sigma = max(p.period_mm / (2.0 * domain.pitch), 0.8)
    n1 = ndimage.gaussian_filter(rng.random(domain.dims), sigma)
    n2 = ndimage.gaussian_filter(rng.random(domain.dims), sigma * 0.5)
    return n1 + 0.35 * n2  # two octaves -> spongy bone look


def _field_voronoi(domain: VoxelDomain, p: OrganicParams, rng: np.random.Generator) -> np.ndarray:
    solid_idx = np.flatnonzero(domain.solid.ravel())
    vol = domain.solid_count * domain.pitch**3
    n_sites = int(np.clip(round(vol / max(p.period_mm**3, 1e-6)), 6, 1200))
    sites = rng.choice(solid_idx, size=min(n_sites, len(solid_idx)), replace=False)
    seed_mask = np.zeros(domain.cell_count, dtype=bool)
    seed_mask[sites] = True
    # nearest seed index per cell -> voronoi cell labels
    _, inds = ndimage.distance_transform_edt(~seed_mask.reshape(domain.dims), return_indices=True)
    flat_lbl = np.empty(domain.cell_count, dtype=np.int64)
    flat_lbl[sites] = np.arange(len(sites))
    lbl = flat_lbl[tuple(inds)]
    # walls: cells whose 6-neighbour carries a different label
    wall = np.zeros(domain.dims, dtype=bool)
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(None, -1)
        hi[ax] = slice(1, None)
        diff = lbl[tuple(lo)] != lbl[tuple(hi)]
        wall[tuple(lo)] |= diff
        wall[tuple(hi)] |= diff
    wall &= domain.solid
    dwall = ndimage.distance_transform_edt(~wall) * domain.pitch
    return 1.0 - np.clip(dwall / max(p.period_mm * 0.22, domain.pitch), 0.0, 1.0)


def _seg_field(centers: np.ndarray, a: np.ndarray, b: np.ndarray, r_mm: float) -> np.ndarray:
    """Per-cell density of a capsule segment a->b (radius r_mm), flat order."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        d = np.linalg.norm(centers - a, axis=1)
    else:
        t = np.clip(((centers - a) @ ab) / denom, 0.0, 1.0)
        d = np.linalg.norm(centers - (a[None, :] + t[:, None] * ab[None, :]), axis=1)
    return np.exp(-0.5 * (d / max(r_mm, 1e-3)) ** 2)


def _field_branches(
    domain: VoxelDomain,
    p: OrganicParams,
    rng: np.random.Generator,
    keep_targets: np.ndarray | None,
) -> np.ndarray:
    solid = domain.solid
    centers = domain.element_centers()
    lo, hi = domain.origin, domain.origin + domain.pitch * np.asarray(domain.dims, float)
    z0, z1 = float(lo[2]), float(hi[2])
    span = max(z1 - z0, 1e-6)

    # base: anisotropic noise stretched in Z -> vertical striations
    s = max(p.period_mm / (2.5 * domain.pitch), 0.8)
    rho = ndimage.gaussian_filter(rng.random(domain.dims), sigma=(s, s, s * 2.2)).ravel()

    # trunks: floor roots -> targets (keep-region centroids, else top surface)
    solid3 = solid
    floor_band = solid3 & (
        centers[:, 2].reshape(domain.dims) < z0 + max(p.floor_mm, domain.pitch) + domain.pitch
    )
    top_band = solid3 & (centers[:, 2].reshape(domain.dims) > z0 + span * 0.75)
    roots = np.flatnonzero(floor_band.ravel())
    tips = np.flatnonzero(top_band.ravel())
    n_trunks = int(np.clip(round(domain.solid_count * domain.pitch**3 / (p.period_mm**3)), 6, 60))
    targets: list[np.ndarray] = []
    if keep_targets is not None and len(keep_targets):
        targets.extend(np.asarray(t, dtype=float) for t in keep_targets)
    if len(tips):
        pick = rng.choice(tips, size=min(n_trunks, len(tips)), replace=False)
        targets.extend(centers[pick])
    trunk = np.zeros(domain.cell_count, dtype=np.float64)
    r_mm = max(p.period_mm * 0.22, domain.pitch * 1.2)
    for i, tip in enumerate(targets):
        if len(roots):
            root = centers[roots[rng.integers(0, len(roots))]]
        else:
            root = np.array([tip[0], tip[1], z0 + domain.pitch])
        # one jittered midpoint -> bent, organic struts
        mid = (root + tip) / 2 + rng.normal(0.0, p.period_mm * 0.35, size=3)
        trunk = np.maximum(trunk, _seg_field(centers, root, mid, r_mm))
        trunk = np.maximum(trunk, _seg_field(centers, mid, np.asarray(tip, float), r_mm * 0.8))
        if i >= 2 * n_trunks:
            break
    rho = np.maximum(rho / max(rho.max(), 1e-9), trunk).reshape(domain.dims)
    return np.asarray(rho, dtype=np.float32)


# --------------------------------------------------------------------------- #
# passive (pinned-solid) mask                                                  #
# --------------------------------------------------------------------------- #


def organic_passive_mask(
    domain: VoxelDomain, p: OrganicParams, anchors: list[dict] | None
) -> tuple[np.ndarray, list[str], list[Any]]:
    """Flat bool mask of cells forced solid + warnings + resolved keep list.

    Unions manual keep regions, auto-keep rings around detected anchors and
    the floor band (cell centers below min Z + floor_mm). The outer skin is
    NOT pinned here — `organic_density` carves it according to
    `p.surface_open`."""
    warnings: list[str] = []
    keep_list = list(p.keep_regions)
    if p.auto_keep and anchors:
        keep_list += auto_keep_regions(anchors, p.auto_keep_margin_mm)
    warnings.extend(empty_region_warnings(domain, keep_list, anchors))
    warnings.extend(empty_region_warnings(domain, list(p.void_regions), anchors))

    passive = np.zeros(domain.cell_count, dtype=bool)
    if keep_list:
        passive |= keep_element_mask(domain, keep_list, anchors)

    centers = domain.element_centers()
    if p.floor_mm > 0:
        floor = (centers[:, 2] < float(domain.origin[2]) + p.floor_mm).reshape(domain.dims)
        passive |= floor.ravel()

    passive &= domain.solid.ravel()
    return passive, warnings, keep_list


def _resolved_shell_mm(domain: VoxelDomain, p: OrganicParams) -> float:
    """Skin depth — explicit `shell_mm` or ~20% of the feature size, clamped
    to [1.6 pitch, 10 mm] so at least one skin layer always exists."""
    if p.shell_mm > 0:
        return float(p.shell_mm)
    return float(min(max(0.2 * p.period_mm, 1.6 * domain.pitch), 10.0))


def _skin_mask(domain: VoxelDomain, p: OrganicParams) -> np.ndarray:
    """(nx,ny,nz) bool — solid cells whose centre sits within the resolved
    skin depth of the part surface.

    The voxel grid is exact-bounds, so a box filling the whole domain has no
    background cells and a plain EDT degenerates — pad one void layer so the
    grid boundary counts as surface. depth = (EDT - 0.5) * pitch measures
    from the cell centre to the nearest void/exterior face."""
    shell_mm = _resolved_shell_mm(domain, p)
    if shell_mm <= 0:
        return np.zeros(domain.dims, dtype=bool)
    padded = np.pad(domain.solid, 1, constant_values=False)
    depth = (ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1] - 0.5) * domain.pitch
    return np.asarray(domain.solid & (depth <= shell_mm))


def organic_density(
    domain: VoxelDomain, p: OrganicParams, anchors: list[dict] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Build the pseudo-density field + passive mask for the domain.

    Returns (rho flat float32, passive flat bool, warnings, meta). rho is
    normalized to [0,1] on solid cells and 0 elsewhere; void regions are
    pinned empty AND removed from the passive set (a painted carve beats
    skin/floor/auto pins). meta carries `threshold` + `shell_mm`."""
    rng = np.random.default_rng(int(p.seed) & 0x7FFFFFFF)
    if p.style == "gyroid":
        rho = _field_gyroid(domain, p)
    elif p.style == "trabecular":
        rho = _field_trabecular(domain, p, rng)
    elif p.style == "voronoi":
        rho = _field_voronoi(domain, p, rng)
    else:
        # branch trunks grow toward keep regions when they exist
        keep_list = list(p.keep_regions)
        if p.auto_keep and anchors:
            keep_list += auto_keep_regions(anchors, p.auto_keep_margin_mm)
        targets = keep_element_mask(domain, keep_list, anchors)
        tgt_centers = domain.element_centers()[np.flatnonzero(targets)] if targets.any() else None
        rho = _field_branches(domain, p, rng, tgt_centers)
    rho = _normalize(rho, domain.solid)

    passive, warnings, _ = organic_passive_mask(domain, p, anchors)
    rho_flat = rho.ravel()

    # --- carved skin -----------------------------------------------------
    # Pin the top (1 - surface_open) fraction of the skin: windows open where
    # the field is weakest while the silhouette stays readable. surface_open
    # =0 keeps the whole skin (solid shell), =1 no protection (legacy look).
    skin3 = _skin_mask(domain, p) if p.surface_open < 1.0 else None
    if skin3 is not None and skin3.any():
        skin_free = np.flatnonzero(skin3.ravel() & ~passive)
        if len(skin_free):
            t_skin = float(np.quantile(rho_flat[skin_free], p.surface_open))
            passive[skin_free[rho_flat[skin_free] >= t_skin]] = True
        else:
            warnings.append("skin is fully occupied by keep regions — surface_open has no effect")

    # --- void regions ----------------------------------------------------
    # A painted carve is a deliberate "erase here" — it punches through
    # pinned skin, floor band and auto-keep rings alike. Void cells are
    # excluded from the free pool too: they can never be selected and would
    # otherwise inflate n_free, skewing the threshold low.
    void = np.zeros(domain.cell_count, dtype=bool)
    if p.void_regions:
        void = keep_element_mask(domain, list(p.void_regions), anchors)
        passive &= ~void
        rho.flat[void] = 0.0

    # threshold: explicit iso, else the quantile that realizes volfrac over
    # the free (non-passive, non-void) design space
    free = domain.solid.ravel() & ~passive & ~void
    if p.density_threshold is not None:
        threshold = float(p.density_threshold)
    else:
        target = round(p.volfrac * domain.solid_count)
        n_free = int(free.sum())
        need = target - int(passive.sum())
        if n_free <= 0 or need <= 0:
            threshold = 0.98  # only the pinned zones survive
        else:
            q = float(np.clip(1.0 - need / n_free, 0.02, 0.98))
            threshold = float(np.quantile(rho.ravel()[free], q)) if n_free else 0.5
            threshold = float(np.clip(threshold, 1e-3, 0.999))
    meta = {
        "threshold": threshold,
        "shell_mm": _resolved_shell_mm(domain, p),
        # extraction must re-carve after min-member dilation / regularization
        "void_cells": void,
    }
    return rho.ravel(), passive, warnings, meta


# --------------------------------------------------------------------------- #
# mesh transform                                                               #
# --------------------------------------------------------------------------- #


def organic_transform(
    mesh: trimesh.Trimesh,
    p: OrganicParams,
    *,
    anchors: list[dict] | None = None,
    voxel_mm: float | None = None,
    max_elements: int | None = None,
    runtime: RuntimeSettings | None = None,
) -> OrganicResult:
    """Restyle a mesh as a procedural organic structure.

    Returns the extracted (unsmoothed-beyond-density_to_mesh) mesh plus the
    full field state for persistence. Raises ValueError when every fallback
    produces nothing."""
    s = runtime or RuntimeSettings()
    domain = voxelize(
        mesh,
        p.voxel_mm or voxel_mm or s.fem_voxel_mm,
        p.max_elements or max_elements or s.fem_max_elements,
    )
    # thin walls/fins under ~1 pitch never contain a cell centre and would
    # vanish from the design domain entirely — union near-surface cells back
    # in (organic scope only; the shared voxelizer stays conservative)
    try:
        centers = domain.element_centers()
        dist = np.zeros(len(centers), dtype=float)
        for c0 in range(0, len(centers), 512):
            dist[c0 : c0 + 512] = np.asarray(mesh.nearest.on_surface(centers[c0 : c0 + 512])[1])
        near = (dist <= domain.pitch * 0.55).reshape(domain.dims)
        if near.any():
            domain.solid |= near
    except Exception:
        log.warning("organic.thin_rescue_failed", exc_info=True)
    all_anchors = anchors if anchors is not None else p.anchors

    warnings: list[str] = ["pseudo-organic restyling — visual structure, NOT structurally verified"]
    out_mesh: trimesh.Trimesh | None = None
    final_mask = np.zeros(domain.dims, dtype=bool)
    passive = np.zeros(domain.cell_count, dtype=bool)
    rho = np.zeros(domain.cell_count, dtype=np.float32)
    threshold = 0.5
    n_components = 0
    fallback = 0

    attempts = [(p.style, p.volfrac), (p.style, min(p.volfrac + 0.15, 0.9)), ("trabecular", 0.5)]
    for i, (style, vf) in enumerate(attempts):
        try:
            # fallback retries must drop an explicit user iso — an over-high
            # threshold is a plausible reason the first attempt emptied out
            pa = p.model_copy(
                update={
                    "style": style,
                    "volfrac": vf,
                    **({"density_threshold": None} if i else {}),
                }
            )
            rho, passive, w, meta = organic_density(domain, pa, all_anchors)
            if i == 0:
                warnings += w
            threshold = meta["threshold"]
            # user voids + everything outside the part — min-member dilation
            # and regularization must not grow material past the original
            # surface (shared _enforce_min_member has no solid clip; organic
            # cannot afford silhouette protrusions)
            stay_empty = meta["void_cells"].reshape(domain.dims) | ~domain.solid
            out_mesh, final_mask, ext_w, n_components, _extract_info = density_to_mesh(
                domain,
                rho,
                threshold=threshold,
                passive_solid=passive,
                void_cells=stay_empty,
                min_member_mm=p.min_member_mm or s.topopt_min_member_mm or 0.0,
                min_gap_mm=p.min_gap_mm or 0.0,
                smooth_iterations=(
                    p.smooth_iterations
                    if p.smooth_iterations is not None
                    else s.topopt_smooth_iterations
                ),
                # aesthetic fields may legitimately produce disjoint
                # trabecular bodies — bridging them would smear unintended
                # material into the lattice
                bridge_islands=False,
            )
            warnings += ext_w
            if len(out_mesh.faces) and final_mask.any():
                fallback = i
                break
            out_mesh = None
        except Exception as e:
            warnings.append(f"attempt {i + 1} ({style}, volfrac {vf:g}) failed: {e}")
            out_mesh = None

    if out_mesh is None:
        raise ValueError(
            "organic restyle produced no geometry — try a higher density or larger period"
        )

    if fallback:
        warnings.append(
            f"fallback used ({attempts[fallback][0]} @ volfrac {attempts[fallback][1]:g})"
        )

    # how much of the original skin survived the carve — the recognizability
    # metric; warns early when the silhouette is about to disintegrate.
    # Always measured, even at surface_open=1.0 (open skeleton): that regime
    # is exactly where the number matters most.
    skin = _skin_mask(domain, p)
    coverage: float | None = None
    if int(skin.sum()):
        coverage = float((final_mask & skin).sum() / skin.sum())
        if coverage < 0.4:
            warnings.append(
                f"only {coverage:.0%} of the original surface survived — "
                'raise "preserve surface" or density for a more readable shape'
            )

    solid_cells = int(final_mask.sum())
    report = {
        "style": p.style,
        "seed": p.seed,
        "period_mm": p.period_mm,
        "volfrac_target": p.volfrac,
        "volfrac_achieved": round(solid_cells / max(domain.solid_count, 1), 4),
        "threshold": round(threshold, 4),
        "pitch_mm": round(domain.pitch, 4),
        "dims": list(domain.dims),
        "solid_elements": solid_cells,
        "n_components": n_components,
        # body count of the actual export mesh — smoothing can merge or
        # split mask-space components; this is the number the printed part
        # is judged by
        "n_bodies_export": int(out_mesh.body_count),
        "floor_mm": p.floor_mm,
        "surface_open": p.surface_open,
        "shell_mm": round(_resolved_shell_mm(domain, p), 3),
        "skin_cells": int(skin.sum()),
        "surface_coverage": round(coverage, 3) if coverage is not None else None,
        "min_member_mm": p.min_member_mm or s.topopt_min_member_mm or 0.0,
        "auto_keep": p.auto_keep,
        "keep_regions": len(p.keep_regions),
        "void_regions": len(p.void_regions),
        "triangles": len(out_mesh.faces),
        "watertight": bool(out_mesh.is_watertight),
        "warnings": warnings,
        "disclaimer": "visual stylization only — not a structural optimization",
    }
    return OrganicResult(
        mesh=out_mesh,
        report=report,
        density=rho.astype(np.float32),
        mask=final_mask,
        passive=passive,
        void=meta["void_cells"],
        domain=domain,
    )
