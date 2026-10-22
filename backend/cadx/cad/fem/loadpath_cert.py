"""Deterministic load-path certificate — a FEM-independent lower bound.

On the voxelized export solid the 6-connected cell graph
(``loadpath._solid_graph``) is turned into a flow network: every solid
cell gets a node capacity proportional to its local inscribed cross
section ``(2 · EDT · pitch)^2`` (a narrow neck throttles all flow through
it). ``scipy.sparse.csgraph.maximum_flow`` from a super-source attached
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
    raise NotImplementedError
