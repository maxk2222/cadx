"""Auto-keep axial bounds — bearing-bracket phantom-column regression.

`keep_mode="auto"` synthesizes a `hole` keep region per anchor. Anchors
without ``depth_mm`` used to resolve to an UNBOUNDED axial keep cylinder —
a bottom blind tap then pinned a solid column all the way to the part's
top face, and the column's flat top rendered as a phantom bore aligned
with the thread axis (the "extra holes" on the bearing bracket). The keep
must bind to the measured void run: through bore -> its full pierced wall;
blind hole -> mouth to bottom; shallow attach dimple -> the interface pad
at the face.
"""

from __future__ import annotations

import numpy as np

from cadx.cad.fem.refine import _anchor_keep_prism
from cadx.cad.fem.regions import (
    _axial_void_run,
    interface_regions,
    keep_element_mask,
)
from cadx.cad.fem.voxelize import VoxelDomain


def _block(dims: tuple[int, int, int] = (20, 20, 30), pitch: float = 2.0) -> VoxelDomain:
    return VoxelDomain(origin=np.zeros(3), pitch=pitch, dims=dims, solid=np.ones(dims, bool))


def _void_column(
    solid: np.ndarray, ax: int, uv: tuple[float, float], r: float, lo: float, hi: float
) -> None:
    """Clear a void column along `ax` at in-plane `uv` (mm), axial [lo,hi] mm —
    grid pitch 2.0, origin 0 (cell centres at odd millimetres)."""
    ij = np.indices(solid.shape, dtype=float)
    others = [a for a in range(3) if a != ax]
    du = (ij[others[0]] + 0.5) * 2.0 - uv[0]
    dv = (ij[others[1]] + 0.5) * 2.0 - uv[1]
    axial = (ij[ax] + 0.5) * 2.0
    solid[(du * du + dv * dv <= r * r) & (axial >= lo) & (axial <= hi)] = False


def test_blind_tap_keep_binds_to_void_run_not_part_height() -> None:
    """A bottom-face tap anchor WITHOUT depth_mm must not pin a column to
    the top face — the flat column top is what renders as a phantom bore."""
    dom = _block()
    _void_column(dom.solid, 2, (10.0, 20.0), 3.4, 0.0, 10.0)  # 10 mm blind tap
    anchors = [{"index": 1, "kind": "tap", "axis": "-z", "pos": [10.0, 20.0, 0.0], "dia": 6.8}]
    keeps, _voids = interface_regions(anchors, margin_mm=3.0)
    keep = keep_element_mask(dom, keeps, anchors).reshape(dom.dims)
    assert keep[:, :, :6].any()  # ring pinned around the hole mouth
    assert not keep[:, :, 6:].any()  # nothing above the pad — no column to the top


def test_through_bore_keep_covers_full_wall_span() -> None:
    """A through bore still keeps its whole pierced wall — bounding must
    not under-protect the dominant case."""
    dom = _block()
    _void_column(dom.solid, 1, (20.0, 30.0), 7.5, -1.0, 41.0)  # y-bore Ø15 through
    anchors = [{"index": 1, "kind": "hole", "axis": "+y", "pos": [20.0, None, 30.0], "dia": 15.0}]
    keeps, _ = interface_regions(anchors, margin_mm=3.0)
    keep = keep_element_mask(dom, keeps, anchors).reshape(dom.dims)
    assert keep[:, 0:3, :].any() and keep[:, -3:, :].any()


def test_tap_keep_survives_bore_crossing() -> None:
    """The (0,-22) defect: the tap column is crossed by a transverse bore —
    two void runs exist on the same column. The tap's keep must bind to its
    own mouth run, not stretch to the bore void above."""
    dom = _block()
    _void_column(dom.solid, 2, (10.0, 20.0), 3.4, 0.0, 2.0)  # shallow tap dimple
    _void_column(dom.solid, 1, (10.0, 30.0), 7.5, -1.0, 41.0)  # y-bore crossing at z=30
    anchors = [{"index": 1, "kind": "tap", "axis": "-z", "pos": [10.0, 20.0, 0.0], "dia": 6.8}]
    keeps, _ = interface_regions(anchors, margin_mm=3.0)
    keep = keep_element_mask(dom, keeps, anchors).reshape(dom.dims)
    assert keep[:, :, :5].any()
    assert not keep[:, :, 5:].any()  # nothing from z>=11 — no band inside the bore


def test_axial_void_run_prefers_measured_centre_then_entry_face() -> None:
    dom = _block()
    _void_column(dom.solid, 2, (10.0, 20.0), 3.4, 0.0, 2.0)
    _void_column(dom.solid, 1, (10.0, 30.0), 7.5, -1.0, 41.0)
    # measured axc inside the crossing run -> that run wins
    run = _axial_void_run(dom, "-z", [10.0, 20.0, 0.0], 6.8, 30.0)
    assert run is not None and run[0] >= 20.0
    # measured axc at the mouth -> the dimple run
    run = _axial_void_run(dom, "-z", [10.0, 20.0, 0.0], 6.8, 0.5)
    assert run is not None and run[1] <= 4.0
    # no axc at all -> entry-face run for a "-z" (bottom-face) feature
    run = _axial_void_run(dom, "-z", [10.0, 20.0, 0.0], 6.8, None)
    assert run is not None and run[1] <= 4.0
    # fully solid column -> None
    dom2 = _block()
    assert _axial_void_run(dom2, "-z", [10.0, 20.0, 0.0], 6.8, 0.0) is None


def test_anchor_keep_prism_blind_without_depth_bounds_to_pad() -> None:
    bounds = np.array([[0.0, 0.0, 0.0], [40.0, 40.0, 60.0]])
    anchor = {
        "kind": "tap",
        "axis": "-z",
        "pos": [10.0, 20.0, 0.0],
        "dia": 6.8,
        "blind": True,
    }
    prism = _anchor_keep_prism(anchor, margin=3.0, span_axis=(0.0, 60.0), bounds=bounds)
    assert prism is not None
    # pad around the mouth plane, not the 0..60 part extent
    assert prism["span"][1] - prism["span"][0] <= 20.0
    assert prism["span"][0] <= 0.0 <= prism["span"][1]


def test_anchor_keep_prism_through_keeps_full_extent() -> None:
    bounds = np.array([[0.0, 0.0, 0.0], [40.0, 40.0, 60.0]])
    anchor = {
        "kind": "hole",
        "axis": "+z",
        "pos": [10.0, 20.0, 30.0],
        "dia": 15.0,
    }
    prism = _anchor_keep_prism(anchor, margin=3.0, span_axis=(0.0, 60.0), bounds=bounds)
    assert prism is not None
    # through anchor without depth: still restores the pierced wall span
    assert prism["span"][1] - prism["span"][0] >= 55.0
