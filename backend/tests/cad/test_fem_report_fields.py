"""M2 report evidence fields — supports records, reactions.fixed_centroid_mm,
work_disp_mm and feature_peaks on FemCaseResult.

`supports`/`applied` are the where-did-it-land records the agent and the
eval gates read: a support centroid in the wrong place is the most common
BC misconfiguration, and feature_peaks turns a bare global sigma max into
"92 MPa at bore #2".
"""

from __future__ import annotations

import numpy as np
import trimesh

from cadx.cad.fem.analyze import run_fem_domain
from cadx.cad.fem.voxelize import voxelize
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec


def _box_mesh(x: float = 60.0, y: float = 24.0, z: float = 12.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([0.0, 0.0, z / 2])
    return m


def _tension_case() -> LoadCase:
    return LoadCase(
        name="pull",
        fixed=[RegionSpec(kind="plane", axis="-x")],
        loads=[LoadSpec(region=RegionSpec(kind="plane", axis="+x"), force_n=[100.0, 0, 0])],
    )


def test_supports_and_fixed_centroid() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    rep = run_fem_domain(dom, dom.solid, [_tension_case()], "pla")
    case = rep.cases[0]
    assert len(case.supports) == 1
    rec = case.supports[0]
    assert rec["kind"] == "plane" and rec["axis"] == "-x"
    assert rec["fix_mode"] == "all"
    assert rec["dofs_pinned"] == [rec["nodes"]] * 3
    # the -x end plane: node centroid at (x_min, y_mid, z_mid)
    c = rec["centroid_mm"]
    assert c[0] == -30.0 and abs(c[1]) < 1e-6 and c[2] == 6.0
    # reactions carries the aggregate support centroid
    fc = case.reactions["fixed_centroid_mm"]
    assert fc[0] == -30.0 and fc[2] == 6.0
    assert case.reactions["resultant_n"][0] == 100.0


def test_work_disp_mm_matches_tension_theory() -> None:
    """work_disp = compliance/|F| — for a prismatic tension rod that is
    exactly FL/EA (the resultant-weighted mean displacement)."""
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    rep = run_fem_domain(dom, dom.solid, [_tension_case()], "pla")
    case = rep.cases[0]
    E, A, L, F = 3500.0, 24.0 * 12.0, 60.0, 100.0
    delta = F * L / (E * A)  # ~0.006 mm
    assert case.work_disp_mm is not None
    assert 0.5 * delta <= case.work_disp_mm <= 2.0 * delta
    # zero-resultant cases get None, not a crash
    empty = LoadCase(
        name="e",
        fixed=[RegionSpec(kind="plane", axis="-x")],
        loads=[LoadSpec(region=RegionSpec(kind="plane", axis="+x"), force_n=[0, 0, 0])],
    )
    rep2 = run_fem_domain(dom, dom.solid, [empty], "pla")
    assert rep2.cases[0].work_disp_mm is None


def test_feature_peaks_locate_anchor_stress() -> None:
    dom = voxelize(_box_mesh(), 2.0, 150_000)
    anchors = [
        {"index": 1, "kind": "hole", "pos": [10.0, 0.0, 6.0], "dia": 5.0},
        {"index": 2, "kind": "hole", "pos": [-29.0, 0.0, 6.0], "dia": 4.0},
    ]
    rep = run_fem_domain(dom, dom.solid, [_tension_case()], "pla", anchors=anchors)
    case = rep.cases[0]
    assert len(case.feature_peaks) == 2
    by_idx = {p["index"]: p for p in case.feature_peaks}
    assert by_idx[1]["pos"] == [10.0, 0.0, 6.0]
    assert by_idx[1]["vm_mpa"] > 0.0
    # the anchor near the clamped end sees the boundary-layer stress —
    # at least the uniform F/A level (0.35 MPa)
    assert by_idx[2]["vm_mpa"] >= 100.0 / (24.0 * 12.0) * 0.5
    # and the peak field is per-case, not report-global
    rep2 = run_fem_domain(dom, dom.solid, [_tension_case()], "pla")
    assert rep2.cases[0].feature_peaks == []
