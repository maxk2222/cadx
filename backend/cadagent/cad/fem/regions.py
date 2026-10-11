"""Map LLM-declared RegionSpecs onto node/element masks of the VoxelDomain.

Region kinds (see llm.schemas.RegionSpec):
- face:     outer element layer on one bbox side (axis = outward normal)
- cylinder: material within radius around an axis line (e.g. a bore/mount)
- box:      cube centred at `center` with edge `diameter_mm` (or 4*margin)
- all:      the whole solid

An empty selection is reported as a warning (caller decides to retry via the
LLM loads step or to continue without that region).
"""

from __future__ import annotations

import numpy as np

from cadagent.cad.fem.voxelize import VoxelDomain
from cadagent.llm.schemas import LoadCase, RegionSpec

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _axis_parts(axis: str) -> tuple[int, int]:
    """('x'|'y'|'z' index, sign +1/-1) from strings like '+z'/'-x'."""
    a = (axis or "+z").strip().lower()
    sign = -1 if a.startswith("-") else 1
    letter = a.lstrip("+-")[-1] if a.lstrip("+-") else "z"
    return _AXIS_IDX.get(letter, 2), sign


def _layers(domain: VoxelDomain, margin_mm: float) -> int:
    return max(1, round(max(margin_mm, 0.1) / domain.pitch))


def _solid_node_mask(domain: VoxelDomain) -> np.ndarray:
    en = domain.element_nodes()
    solid_nodes = np.unique(en[domain.solid.ravel()])
    mask = np.zeros(domain.node_count, dtype=bool)
    mask[solid_nodes] = True
    return mask


def _nodes_of_cells(domain: VoxelDomain, cell_mask: np.ndarray) -> np.ndarray:
    """Union of corner nodes of selected cells -> flat node mask."""
    en = domain.element_nodes()
    mask = np.zeros(domain.node_count, dtype=bool)
    mask[np.unique(en[cell_mask])] = True
    return mask


def _face_cells(domain: VoxelDomain, axis: str, margin_mm: float) -> np.ndarray:
    idx, sign = _axis_parts(axis)
    n = domain.dims[idx]
    lay = _layers(domain, margin_mm)
    mask = np.zeros(domain.dims, dtype=bool)
    if sign < 0:
        sl = [slice(None)] * 3
        sl[idx] = slice(0, lay)
        mask[tuple(sl)] = True
    else:
        sl = [slice(None)] * 3
        sl[idx] = slice(n - lay, n)
        mask[tuple(sl)] = True
    return np.asarray(mask & domain.solid)


def _cylinder_cells(
    domain: VoxelDomain, axis: str, center: list[float], radius: float
) -> np.ndarray:
    idx, _ = _axis_parts(axis)
    others = [i for i in range(3) if i != idx]
    centers = domain.element_centers().reshape((*domain.dims, 3))
    if len(center) >= 2:
        # use the two perpendicular components of the given center hint
        c = np.array([center[others[0]], center[others[1]]], dtype=float)
    else:
        lo, hi = domain.origin, domain.origin + domain.pitch * np.asarray(domain.dims)
        c = np.array([(lo[others[0]] + hi[others[0]]) / 2, (lo[others[1]] + hi[others[1]]) / 2])
    d2 = (centers[..., others[0]] - c[0]) ** 2 + (centers[..., others[1]] - c[1]) ** 2
    return np.asarray((d2 <= radius * radius) & domain.solid)


def _box_cells(domain: VoxelDomain, center: list[float], edge: float) -> np.ndarray:
    centers = domain.element_centers().reshape((*domain.dims, 3))
    if len(center) >= 3:
        c = np.asarray(center[:3], dtype=float)
    else:
        lo = domain.origin
        hi = domain.origin + domain.pitch * np.asarray(domain.dims)
        c = (lo + hi) / 2
    half = edge / 2
    inside = np.all(np.abs(centers - c[None, None, None, :]) <= half, axis=-1)
    return np.asarray(inside & domain.solid)


def region_cell_mask(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """Bool (nx,ny,nz) over cells: which solid cells belong to the region."""
    kind = region.kind
    if kind == "all":
        return domain.solid.copy()
    if kind == "face":
        return _face_cells(domain, region.axis or "+z", region.margin_mm)
    if kind == "cylinder":
        radius = (region.diameter_mm or 4 * region.margin_mm) / 2 + region.margin_mm
        return _cylinder_cells(domain, region.axis or "+z", region.center, radius)
    if kind == "box":
        edge = region.diameter_mm or 4 * region.margin_mm
        return _box_cells(domain, region.center, edge)
    return np.zeros(domain.dims, dtype=bool)


def region_node_mask(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """Bool (nnodes,) over all grid nodes belonging to the region."""
    cells = region_cell_mask(domain, region)
    if not cells.any():
        return np.zeros(domain.node_count, dtype=bool)
    return _nodes_of_cells(domain, cells.ravel())


def region_face_normal(region: RegionSpec) -> np.ndarray | None:
    """Outward unit normal for `face` regions (used for pressure loads)."""
    if region.kind != "face":
        return None
    idx, sign = _axis_parts(region.axis or "+z")
    v = np.zeros(3)
    v[idx] = float(sign)
    return v


def resolve_case(
    domain: VoxelDomain,
    case: LoadCase,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve a load case into (fixed_node_mask, F vector, warnings).

    F is a dense (3*nnodes,) force vector in Newton. Loads with zero matching
    nodes are skipped with a warning.
    """
    warnings: list[str] = []
    nn = domain.node_count
    fixed = np.zeros(nn, dtype=bool)
    F = np.zeros(3 * nn)

    for r in case.fixed:
        m = region_node_mask(domain, r)
        if not m.any():
            warnings.append(
                f"case '{case.name}': fixed region {r.kind} {r.axis or ''} matched no nodes"
            )
            continue
        fixed |= m

    for load in case.loads:
        m = region_node_mask(domain, load.region)
        if not m.any():
            warnings.append(
                f"case '{case.name}': load region {load.region.kind} "
                f"{load.region.axis or ''} matched no nodes"
            )
            continue
        nsel = int(m.sum())
        if load.force_n and len(load.force_n) == 3:
            f = np.asarray(load.force_n, dtype=float)
            idx = np.flatnonzero(m)
            for c in range(3):
                F[3 * idx + c] += f[c] / nsel
        elif load.pressure_mpa is not None:
            n = region_face_normal(load.region)
            if n is None:
                warnings.append(
                    f"case '{case.name}': pressure loads need a face region "
                    f"(got {load.region.kind})"
                )
                continue
            # approximate: each selected boundary node carries pitch^2 of area;
            # pressure acts AGAINST the outward normal (into the part)
            idx = np.flatnonzero(m)
            f = -n * float(load.pressure_mpa) * domain.pitch**2
            for c in range(3):
                F[3 * idx + c] += f[c]
        else:
            warnings.append(f"case '{case.name}': load has neither force_n nor pressure_mpa")

    if not fixed.any():
        warnings.append(f"case '{case.name}': no fixed region — case is unsolvable")
    if not np.any(F):
        warnings.append(f"case '{case.name}': all loads resolved to zero force")

    return fixed, F, warnings


def keep_element_mask(domain: VoxelDomain, regions: list[RegionSpec]) -> np.ndarray:
    """Union of cell masks for non-design (keep) regions, flat cell order."""
    mask = np.zeros(domain.cell_count, dtype=bool)
    for r in regions:
        mask |= region_cell_mask(domain, r).ravel()
    return mask
