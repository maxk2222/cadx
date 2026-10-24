"""Deterministic load-path certificate — a FEM-independent lower bound.

On the voxelized export solid the node-adjacent cell graph is turned
into a flow network: every solid cell gets a node capacity proportional
to its local inscribed cross section ``(2 · EDT_face · pitch)^2`` —
``EDT_face = EDT - 0.5`` measures cell-centre-to-void-face depth, the
same convention ``infill._skin_mask`` uses, so a surface cell carries
exactly one cell face of section (a narrow neck throttles all flow
through it). Adjacency matches the FEM's node-sharing coupling (26-
connected): face/edge/corner contacts carry the thinner endpoint's
capacity scaled by shared-node count (1.0 / 0.5 / 0.25) — diagonal
lattice struts stay reachable while a corner-only waist still reads as
a thin section.
``scipy.sparse.csgraph.maximum_flow`` from a super-source attached
to the case's loaded cells into a super-sink on its fixed cells then
yields the bottleneck cross section the load must pass.

``bottleneck_mm2 · yield_mpa`` is the axial force capacity of the
thinnest section — ``sf_floor = capacity / applied force`` bounds the
safety factor from the geometry side without any elasticity solve. A
case whose loaded cells cannot reach its fixtures gets flow 0 →
``load_path_ok = False``.

Documented limit: the axial bottleneck floor reads optimistic for
bending-dominated parts (a slender member in bending peaks above axial
stress); it is a lower bound on capacity, always paired with the FEM
numbers — never a standalone guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadx.cad.fem.voxelize import VoxelDomain
from cadx.logging import get_logger

log = get_logger(__name__)

# Flow capacities are integer-scaled mm² — `csgraph.maximum_flow` only
# accepts integral capacities. The target resolution is 1e-4 mm² — far
# below the certificate's physical accuracy — but the *total* scaled
# capacity must stay under int32 (~2.1e9): scipy's flow internals overflow
# past that and silently return 0 on a perfectly connected part, which
# reads as a broken load path. `_eff_scale` shrinks the unit when the
# domain is large.
_SCALE = 10_000.0
_INT32_SAFE_SUM = 1_000_000_000.0

# the flow network is rebuilt per case — refuse domains where the
# augmented graph would dwarf the FEM solve it shadows
_MAX_CERT_CELLS = 2_000_000


def _node_adjacent_graph(
    domain: VoxelDomain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """26-connected solid-cell arcs — the FEM couples cells through every
    shared node, so the flow network must too, or a diagonal lattice
    member (cells meeting only at edges/corners after rasterization)
    reads as a broken load path while the solve sees one body.

    Returns (rows, cols, shared_nodes) with BOTH arc directions listed
    once each; ``shared_nodes`` is 4/2/1 for face/edge/corner contacts so
    callers can scale contact capacity by the contact patch (shared/4).
    """
    solid = domain.solid.ravel()
    idx = np.flatnonzero(solid)
    coords = np.unravel_index(idx, domain.dims)
    strides = (domain.dims[1] * domain.dims[2], domain.dims[2], 1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    shared: list[np.ndarray] = []
    # the 13 unique forward directions (all (di,dj,dk) in {-1,0,1}^3 with
    # a positive first nonzero component) — each pair counted once, then
    # symmetrized below
    dirs = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if (di, dj, dk) != (0, 0, 0)
        and (di > 0 or (di == 0 and (dj > 0 or (dj == 0 and dk > 0))))
    ]
    for di, dj, dk in dirs:
        inside = (
            (coords[0] + di >= 0)
            & (coords[0] + di < domain.dims[0])
            & (coords[1] + dj >= 0)
            & (coords[1] + dj < domain.dims[1])
            & (coords[2] + dk >= 0)
            & (coords[2] + dk < domain.dims[2])
        )
        src = idx[inside]
        dst = src + di * strides[0] + dj * strides[1] + dk * strides[2]
        both = solid[dst]
        a, b = src[both], dst[both]
        n_shared = 2 ** (3 - (abs(di) + abs(dj) + abs(dk)))
        rows.append(np.concatenate([a, b]))
        cols.append(np.concatenate([b, a]))
        shared.append(np.full(2 * a.size, n_shared, dtype=np.int64))
    if not rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(shared)


def _unit_reachable(
    g_row: np.ndarray,
    g_col: np.ndarray,
    n: int,
    load_cells: np.ndarray,
    fix_cells: np.ndarray,
) -> bool:
    """Cheap undirected reachability between endpoint cells on the 26-conn
    cell graph — the sanity check behind the max-flow verdict: integer
    scaling can make the flow solver return 0 on a part that provably has
    a path, and only this check separates 'broken' from 'solver glitch'."""
    from scipy import sparse
    from scipy.sparse import csgraph

    g = sparse.csr_matrix(
        (np.ones(g_row.size, dtype=np.int8), (g_row, g_col)), shape=(n, n)
    )
    dist = csgraph.dijkstra(
        g, directed=False, indices=fix_cells, min_only=True
    )
    return bool(np.isfinite(dist[load_cells]).any())


def loadpath_certificate(
    domain: VoxelDomain,
    cases_resolved: list[tuple[np.ndarray, np.ndarray]],
    *,
    yield_mpa: float,
) -> dict[str, Any]:
    """Min-cut load-path certificate on ``domain``'s solid cells.

    `cases_resolved` are ``(fixed_mask, F)`` tuples as returned by
    ``regions.resolve_case_dofs`` — re-resolved on THIS domain, never
    recycled from another grid (node indices are grid-local).

    Returns::

        {
            "load_path_ok": bool,        # every case's loads reach fixtures
            "bottleneck_mm2": float,     # min over cases of min-cut area
            "sf_floor": float | None,    # bottleneck*yield / F (worst case)
            "per_case": [
                {"case": int, "bottleneck_mm2": float | None,
                 "sf_floor": float | None, "reachable": bool}
            ],
            "warnings": [str, ...],
        }
    """
    from scipy import ndimage, sparse
    from scipy.sparse import csgraph

    from cadx.cad.fem.loadpath import _cells_at_nodes

    out: dict[str, Any] = {
        "load_path_ok": True,
        "bottleneck_mm2": None,
        "sf_floor": None,
        "per_case": [],
        "warnings": [],
    }
    warnings: list[str] = out["warnings"]
    n = domain.cell_count
    solid = domain.solid.ravel()
    if n == 0 or not solid.any():
        out["load_path_ok"] = False
        warnings.append("no solid cells — the load path is unresolvable")
        return out
    if n > _MAX_CERT_CELLS:
        # skipped rather than failed — `load_path_ok` stays unknown (None)
        out["load_path_ok"] = None
        warnings.append(
            f"domain too large for the load-path certificate ({n} cells) — skipped"
        )
        return out
    if not cases_resolved:
        out["load_path_ok"] = None
        warnings.append("no resolved load cases — certificate skipped")
        return out

    # Local inscribed half-thickness per cell = distance from the cell
    # centre to the nearest void FACE — the same convention
    # `infill._skin_mask` uses ((edt - 0.5) * pitch). The mask is padded by
    # one void ring first so cells on the grid boundary still measure
    # their true half-thickness instead of the EDT's border behaviour.
    padded = np.pad(domain.solid, 1, mode="constant", constant_values=False)
    edt = ndimage.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
    half_mm = np.maximum(edt.ravel() - 0.5, 0.25) * float(domain.pitch)
    # the inscribed square section (2·half)² — exactly one cell face for
    # surface cells, so thin members bound the flow by their real section
    cap_mm2 = (2.0 * half_mm) ** 2
    # scipy's maximum_flow silently returns 0 once any capacity or the
    # capacity sum exceeds int32 — keep the whole scaled network below
    # ~1e9 so a fine grid can never overflow into a false "unreachable".
    eff_scale = min(_SCALE, _INT32_SAFE_SUM / max(float(cap_mm2.sum()), 1.0))
    cap_int = np.round(cap_mm2 * eff_scale).astype(np.int64)
    # node-split flow network: solid cell i becomes (2i in, 2i+1 out) with
    # the inscribed-section capacity on the in->out arc — that is what
    # makes the min cut read a physical cross-section. Without it, a
    # 26-connected contact graph would sum every parallel contact arc
    # into the cut and overstate the section of a solid bar. Contacts go
    # out->in, scaled by the contact patch (face 1.0 / edge 0.5 / corner
    # 0.25) — a corner-only waist is a thin section, not a missing one.
    g_row, g_col, g_shared = _node_adjacent_graph(domain)
    contact_cap = np.minimum(cap_int[g_row], cap_int[g_col]) * g_shared // 4
    solid_idx = np.flatnonzero(solid)
    src_node, sink_node = 2 * n, 2 * n + 1
    inf_cap = int(cap_int.sum()) + 1  # strictly above any finite cut

    for ci, (fixed, F) in enumerate(cases_resolved):
        rec: dict[str, Any] = {
            "case": ci,
            "bottleneck_mm2": None,
            "sf_floor": None,
            "reachable": False,
        }
        fixed_nodes = np.asarray(fixed, dtype=bool)
        if fixed_nodes.ndim == 2:
            fixed_nodes = fixed_nodes.any(axis=1)
        F3 = np.asarray(F, dtype=float).reshape(-1, 3)
        load_nodes = np.abs(F3).sum(axis=1) > 0
        rec["load_nodes"] = int(load_nodes.sum())
        rec["fix_nodes"] = int(fixed_nodes.sum())
        load_cells = _cells_at_nodes(domain, load_nodes, solid)
        fix_cells = _cells_at_nodes(domain, fixed_nodes, solid)
        rec["load_cells"] = int(load_cells.size)
        rec["fix_cells"] = int(fix_cells.size)
        if load_cells.size == 0 or fix_cells.size == 0:
            warnings.append(
                f"case {ci}: no load or fixture cells on this grid — load path broken"
            )
            log.warning(
                "loadpath_cert.endpoint_missing",
                case=ci,
                load_cells=int(load_cells.size),
                fix_cells=int(fix_cells.size),
            )
            out["per_case"].append(rec)
            continue
        if np.intersect1d(load_cells, fix_cells).size:
            # a cell carries both a load and a fixture — flow would short-
            # circuit through the super-source/sink arcs (capacity inf_cap)
            # and report a spurious bottleneck. The path exists trivially;
            # the min cut is just indeterminate for this case.
            rec["reachable"] = True
            warnings.append(
                f"case {ci}: load and fixture share cells — reachable, "
                "bottleneck indeterminate"
            )
            out["per_case"].append(rec)
            continue
        # terminals bypass their own node arc: the boundary cells are
        # where the load enters/leaves, not part of the measured section.
        # source -> load_out; fix_in -> sink.
        arc_r = np.concatenate(
            [
                np.full(load_cells.size, src_node),
                2 * fix_cells,
                2 * solid_idx,
                2 * g_row + 1,
            ]
        )
        arc_c = np.concatenate(
            [
                2 * load_cells + 1,
                np.full(fix_cells.size, sink_node),
                2 * solid_idx + 1,
                2 * g_col,
            ]
        )
        arc_v = np.concatenate(
            [
                np.full(load_cells.size + fix_cells.size, inf_cap, dtype=np.int64),
                cap_int[solid_idx],
                contact_cap,
            ]
        )
        flow_net = sparse.csr_matrix(
            (arc_v, (arc_r, arc_c)), shape=(2 * n + 2, 2 * n + 2), dtype=np.int64
        )
        try:
            res = csgraph.maximum_flow(flow_net, src_node, sink_node)
            flow_mm2 = float(res.flow_value) / eff_scale
        except Exception as e:
            warnings.append(f"case {ci}: max-flow solve failed: {e}")
            log.warning("loadpath_cert.flow_failed", case=ci, error=str(e))
            rec["reachable"] = None
            out["per_case"].append(rec)
            continue
        if flow_mm2 <= 1e-9:
            # connectivity cross-check: the int-scaled solver returning 0
            # is a numerical artifact on a genuinely connected part — only
            # a missing cell-graph path makes the verdict real
            reach = _unit_reachable(g_row, g_col, n, load_cells, fix_cells)
            if reach:
                warnings.append(
                    f"case {ci}: max-flow returned 0 though the cell graph is "
                    "connected — solver anomaly, certificate inconclusive"
                )
                log.warning("loadpath_cert.flow_anomaly", case=ci)
                rec["reachable"] = None
            else:
                warnings.append(
                    f"case {ci}: loads cannot reach the fixtures through the solid"
                )
                log.warning("loadpath_cert.unreachable", case=ci)
            out["per_case"].append(rec)
            continue
        rec["reachable"] = True
        rec["bottleneck_mm2"] = round(flow_mm2, 3)
        f_mag = float(np.abs(F3).sum())
        if yield_mpa > 0 and f_mag > 0:
            rec["sf_floor"] = round(flow_mm2 * float(yield_mpa) / f_mag, 4)
        out["per_case"].append(rec)

    # reachable=None marks solver-inconclusive cases: they must not pull
    # the verdict down to False the way a proven disconnect does — the
    # certificate is unknown for the run, never a broken-path verdict.
    verdicts = [r["reachable"] for r in out["per_case"]]
    if not verdicts:
        out["load_path_ok"] = None
    elif any(v is False for v in verdicts):
        out["load_path_ok"] = False
    elif all(verdicts):
        out["load_path_ok"] = True
    else:
        out["load_path_ok"] = None
    bottlenecks = [
        r["bottleneck_mm2"] for r in out["per_case"] if r["bottleneck_mm2"] is not None
    ]
    floors = [r["sf_floor"] for r in out["per_case"] if r["sf_floor"] is not None]
    out["bottleneck_mm2"] = min(bottlenecks) if bottlenecks else None
    out["sf_floor"] = min(floors) if floors else None
    return out
