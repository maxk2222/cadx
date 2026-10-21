"""Deterministic load-path density field — the geodesic corridor between
the loaded and the fixed cells of a resolved case.

For each resolved load case (the same ``(fixed_mask, F)`` tuples
``regions.resolve_case_dofs`` produces and engine hands to simp()) the
solid-cell graph (6-connectivity, uniform edge weight = pitch) is queried
TWICE by multi-source Dijkstra: ``dL`` = distance from the cells carrying
load, ``dF`` = distance from the cells carrying fixtures. The "pathiness"

    p(c) = dL(c) + dF(c)

reaches its minimum ``p_min`` exactly on the shortest load->fixture
corridors — cells off any shortest path score strictly higher — so a
Gaussian bump on the excess distance paints a smooth corridor field without
enumerating a single path:

    rho_case(c) = base + (1 - base) * exp(-(p(c) - p_min)^2 / corridor^2)

Cases are max-united weighted by their total applied force (the strongest
case reaches full corridor density); passive_solid cells pin to 1.0 and
passive_void / non-solid cells to 0.0. The result is a float32
``(ncells,)`` field usable as a SIMP warm start (``x0``) or a
preview-grade structure on its own.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)


def _solid_graph(domain: VoxelDomain) -> sparse.csr_matrix:
    """6-connected adjacency of the solid cells, edge weight = pitch.

    Built on the FULL flat cell index space — non-solid cells end up
    isolated (dijkstra distance inf, they never enter a corridor)."""
    n = domain.cell_count
    solid = domain.solid.ravel()
    idx = np.flatnonzero(solid)
    coords = np.unravel_index(idx, domain.dims)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    # C-order flat index: i*ny*nz + j*nz + k — axis strides (ny*nz, nz, 1)
    strides = (domain.dims[1] * domain.dims[2], domain.dims[2], 1)
    for ax in range(3):
        inward = coords[ax] < domain.dims[ax] - 1
        src = idx[inward]
        dst = src + strides[ax]
        both = solid[dst]
        a, b = src[both], dst[both]
        rows.append(np.concatenate([a, b]))
        cols.append(np.concatenate([b, a]))
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    return sparse.coo_matrix(
        (np.full(r.size, float(domain.pitch)), (r, c)), shape=(n, n)
    ).tocsr()


def _cells_at_nodes(domain: VoxelDomain, node_mask: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Solid cells (flat) touching at least one node of `node_mask` —
    the same element-nodes adjacency ``passive_solid_mask`` uses."""
    nodes = np.flatnonzero(node_mask)
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    en = domain.element_nodes()
    touch = np.isin(en, nodes).any(axis=1)
    return np.flatnonzero(touch & solid)


def _multi_source_dist(
    graph: sparse.csr_matrix, sources: np.ndarray
) -> np.ndarray:
    """(n,) shortest distance from the nearest source node — inf where
    unreachable (other solid components, non-solid cells)."""
    if sources.size == 0:
        return np.full(graph.shape[0], np.inf)
    d = csgraph.dijkstra(graph, directed=False, indices=sources, min_only=True)
    return np.asarray(d, dtype=float)


def _case_field(
    domain: VoxelDomain,
    graph: sparse.csr_matrix,
    solid: np.ndarray,
    fixed: np.ndarray,
    F: np.ndarray,
    corridor: float,
    base_rho: float,
) -> np.ndarray | None:
    """Corridor density for one resolved case — None when the case has no
    usable load/fixture cells (caller skips it)."""
    fixed_nodes = np.asarray(fixed, dtype=bool)
    if fixed_nodes.ndim == 2:
        fixed_nodes = fixed_nodes.any(axis=1)
    load_nodes = np.abs(np.asarray(F, dtype=float).reshape(-1, 3)).sum(axis=1) > 0
    load_cells = _cells_at_nodes(domain, load_nodes, solid)
    fixed_cells = _cells_at_nodes(domain, fixed_nodes, solid)
    if load_cells.size == 0 or fixed_cells.size == 0:
        return None
    d_l = _multi_source_dist(graph, load_cells)
    d_f = _multi_source_dist(graph, fixed_cells)
    p = d_l + d_f  # inf where either endpoint set is unreachable
    reach = np.isfinite(p) & solid
    if not reach.any():
        return None
    p_min = float(p[reach].min())
    rho = np.full(domain.cell_count, base_rho, dtype=float)
    excess = p[reach] - p_min
    rho[reach] = base_rho + (1.0 - base_rho) * np.exp(-((excess / corridor) ** 2))
    return rho


def loadpath_field(
    domain: VoxelDomain,
    cases_resolved: list[tuple[np.ndarray, np.ndarray]],
    passive_solid: np.ndarray | None = None,
    passive_void: np.ndarray | None = None,
    *,
    corridor_mm: float | None = None,
    base_rho: float = 0.15,
) -> np.ndarray:
    """Load-path density over the domain's cells, flat order, float32.

    `cases_resolved` are ``(fixed_mask, F)`` tuples — the output of
    ``regions.resolve_case_dofs``/``resolve_case``; `fixed_mask` accepts
    the (nnodes,) node mask or the (nnodes, 3) per-DOF mask.
    `corridor_mm` is the Gaussian half-width of the corridor around the
    shortest load->fixture paths (default 2.5 * pitch); `base_rho` is the
    background density far from any corridor."""
    solid = domain.solid.ravel()
    corridor = float(corridor_mm) if corridor_mm else 2.5 * float(domain.pitch)
    corridor = max(corridor, 1e-6)
    base_rho = float(np.clip(base_rho, 0.0, 1.0))
    rho = np.zeros(domain.cell_count, dtype=float)
    rho[solid] = base_rho

    graph = _solid_graph(domain)
    norms = np.array(
        [float(np.abs(np.asarray(F, dtype=float)).sum()) for _, F in cases_resolved]
    )
    w_max = float(norms.max()) if norms.size else 0.0
    if w_max <= 0.0:
        norms = np.ones(len(cases_resolved))
        w_max = 1.0
    for ci, (fixed, F) in enumerate(cases_resolved):
        rho_c = _case_field(domain, graph, solid, fixed, F, corridor, base_rho)
        if rho_c is None:
            log.warning("loadpath.case_skipped", case=ci)
            continue
        w = float(norms[ci]) / w_max
        rho = np.maximum(rho, rho_c * w)

    if passive_solid is not None:
        rho[np.asarray(passive_solid, dtype=bool).ravel()] = 1.0
    if passive_void is not None:
        rho[np.asarray(passive_void, dtype=bool).ravel()] = 0.0
    rho[~solid] = 0.0
    return np.asarray(rho, dtype=np.float32)
