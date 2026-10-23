"""Gate evaluation: outcome gates + trimesh mesh gates on StageOutcome."""

from __future__ import annotations

import math

import pytest

from cadx.evalkit.gates import run_gates
from cadx.evalkit.models import StageOutcome

trimesh = pytest.importorskip("trimesh")


def _out(**kw) -> StageOutcome:
    o = StageOutcome(status="succeeded", score=90)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _hole_groups(n: int, dia: float, axis: str, centers=None):
    if centers is None:
        centers = [[0, 0]] * n
    return [
        {
            "count": n,
            "diameter": dia,
            "axis": axis,
            "centers": centers,
            "depth_class": "through",
            "blocked": 0,
        }
    ]


def test_status_score_code_gates():
    gates = run_gates(
        [
            {"type": "status", "expect": ["succeeded", "needs_review"]},
            {"type": "score", "min": 80},
            {"type": "code"},
        ],
        _out(code="result = Box(1,1,1)"),
        1.0,
    )
    assert all(g.ok for g in gates)
    assert (
        run_gates([{"type": "status", "expect": "succeeded"}], _out(status="failed"), 1.0)[0].ok
        is False
    )
    assert run_gates([{"type": "score", "min": 95}], _out(), 1.0)[0].ok is False


def test_bbox_gate_match_and_within():
    out = _out(analysis={"bbox_size": [80, 60, 66]})
    assert run_gates(
        [{"type": "bbox", "dims": [80, 60, 66], "match": "match", "tol_mm": 1}], out, 0
    )[0].ok
    assert run_gates([{"type": "bbox", "dims": [100, 80, 70], "match": "within"}], out, 0)[0].ok
    assert not run_gates([{"type": "bbox", "dims": [50, 50, 50], "match": "within"}], out, 0)[0].ok
    assert not run_gates([{"type": "bbox", "dims": [80, 60, 60]}], out, 0)[0].ok
    # missing bbox fails, doesn't crash
    assert run_gates([{"type": "bbox", "dims": [1, 2, 3]}], _out(analysis={}), 0)[0].ok is False


def test_hole_gate_count_dia_axis_positions_pcd():
    feats = {
        "hole_groups": [
            {
                "count": 2,
                "diameter": 8.5,
                "axis": "z",
                "centers": [[25, 20], [25, -20]],
                "depth_class": "through",
                "blocked": 0,
            },
            {
                "count": 2,
                "diameter": 8.5,
                "axis": "x",
                "centers": [[20, 36], [-20, 36]],
                "depth_class": "through",
                "blocked": 0,
            },
        ]
    }
    out = _out(features=feats)
    assert run_gates([{"type": "holes", "count": 2, "dia": 8.5, "axis": "z"}], out, 0)[0].ok
    assert run_gates([{"type": "holes", "count": 2, "dia": 8.5, "axis": "x"}], out, 0)[0].ok
    assert not run_gates([{"type": "holes", "count": 3, "axis": "z"}], out, 0)[0].ok
    assert not run_gates([{"type": "holes", "count": 1, "dia": 12, "axis": "z"}], out, 0)[0].ok
    pos = {"type": "holes", "count": 1, "positions": [[25, 20], [25, -20]]}
    assert run_gates([pos], out, 0)[0].ok
    assert not run_gates([{**pos, "positions": [[99, 99]]}], out, 0)[0].ok
    # pcd: 6 holes on r=40 circle
    centers = [[40 * math.cos(i * math.pi / 3), 40 * math.sin(i * math.pi / 3)] for i in range(6)]
    out6 = _out(features={"hole_groups": _hole_groups(6, 8.5, "z", centers)})
    assert run_gates([{"type": "holes", "count": 6, "pcd": 80}], out6, 0)[0].ok
    assert not run_gates([{"type": "holes", "count": 6, "pcd": 100}], out6, 0)[0].ok


def test_feature_gate_unavailable_is_inconclusive_but_measured_mismatch_fails():
    unavailable = run_gates(
        [{"type": "holes", "count": 4, "dia": 8.5}],
        _out(features=None, features_available=False, features_error="no artifact"),
        0,
    )[0]
    assert unavailable.ok is None
    assert "measurement unavailable" in unavailable.detail
    assert "no artifact" in unavailable.detail

    measured_empty = run_gates(
        [{"type": "holes", "count": 4, "dia": 8.5}],
        _out(features={}, features_available=True),
        0,
    )[0]
    assert measured_empty.ok is False
    assert measured_empty.detail == ""


def test_fillet_watertight_verify_spec():
    out = _out(
        features={"rounds": [{"radius": 2}]},
        analysis={"watertight": True},
        mesh={"topopt": {"verify_spec": {"status": "PASS", "checks": []}}},
    )
    gates = run_gates(
        [{"type": "fillet"}, {"type": "watertight"}, {"type": "verify_spec", "expect": "PASS"}],
        out,
        0,
    )
    assert all(g.ok for g in gates)
    bad = _out(features={}, analysis={"watertight": False}, issues=[])
    res = run_gates(
        [{"type": "fillet"}, {"type": "watertight"}, {"type": "verify_spec", "expect": "PASS"}],
        bad,
        0,
    )
    assert [g.ok for g in res] == [False, False, False]


def test_fem_topopt_gates():
    out = _out(fem={"worst_safety_factor": 2.1, "cases": [{"safety_factor": 2.1}]})
    assert run_gates([{"type": "fem", "field": "sf", "min": 1.5}], out, 0)[0].ok
    assert not run_gates([{"type": "fem", "field": "sf", "min": 3.0}], out, 0)[0].ok
    top = _out(mesh={"topopt": {"sf_export": 1.8, "volfrac_achieved": 0.45}})
    assert run_gates([{"type": "topopt", "field": "sf_export", "min": 1.5}], top, 0)[0].ok
    assert run_gates([{"type": "volfrac", "min": 0.3, "max": 0.6}], top, 0)[0].ok
    assert not run_gates([{"type": "volfrac", "max": 0.4}], top, 0)[0].ok
    # sf falls back fem -> topopt.sf_verified
    both = _out(fem={}, mesh={"topopt": {"sf_verified": 1.7}})
    assert run_gates([{"type": "fem", "field": "sf", "min": 1.5}], both, 0)[0].ok
    # envelope_within reads analysis bbox vs envelope dims
    env = _out(analysis={"bbox_size": [98, 78, 38]})
    assert run_gates([{"type": "envelope_within", "dims": [100, 80, 40]}], env, 0)[0].ok
    assert not run_gates([{"type": "envelope_within", "dims": [90, 70, 30]}], env, 0)[0].ok


def test_load_path_gate_rejects_disconnected_and_preserves_unknown():
    passed = run_gates(
        [{"type": "load_path", "expect": True}],
        _out(mesh={"topopt": {"load_path_ok": True}}),
        0,
    )[0]
    failed = run_gates(
        [{"type": "load_path", "expect": True}],
        _out(mesh={"topopt": {"load_path_ok": False}}),
        0,
    )[0]
    unknown = run_gates([{"type": "load_path"}], _out(mesh={"topopt": {}}), 0)[0]
    assert passed.ok is True
    assert failed.ok is False
    assert unknown.ok is None
    assert "unavailable" in unknown.detail


def test_dfm_gate():
    dfm = {
        "process": "fdm",
        "summary": "1 high, 2 medium, 0 low",
        "warnings": [
            {"code": "thin_wall", "severity": "high"},
            {"code": "overhang", "severity": "medium"},
            {"code": "bridge", "severity": "medium"},
        ],
    }
    out = _out(dfm=dfm)
    assert run_gates([{"type": "dfm", "max_high": 1, "max_medium": 2}], out, 0)[0].ok
    assert not run_gates([{"type": "dfm", "max_high": 0}], out, 0)[0].ok
    assert not run_gates([{"type": "dfm", "max_medium": 1}], out, 0)[0].ok
    assert run_gates([{"type": "dfm", "process": "fdm"}], out, 0)[0].ok
    assert not run_gates([{"type": "dfm", "process": "cnc3"}], out, 0)[0].ok
    # missing report fails instead of silently passing
    res = run_gates([{"type": "dfm"}], _out(dfm=None), 0)[0]
    assert not res.ok and res.detail


def test_misc_gates():
    out = _out(
        issues=[{"what": "hole blocked", "fix_hint": "clear it"}],
        asset_ids=["a1", "a2"],
        usage={"prompt_tokens": 500},
    )
    assert run_gates([{"type": "issue_contains", "substr": "hole blocked"}], out, 0)[0].ok
    assert run_gates([{"type": "asset_image", "min": 2}], out, 0)[0].ok
    assert not run_gates([{"type": "asset_image", "min": 3}], out, 0)[0].ok
    assert run_gates([{"type": "max_prompt_tokens", "n": 1000}], out, 0)[0].ok
    assert not run_gates([{"type": "max_wall_s", "s": 10}], out, 60.0)[0].ok
    out.extras["execs"] = 3
    assert run_gates([{"type": "execs_min", "n": 2}], out, 0)[0].ok
    m = _out(resolved_mode="cad")
    assert run_gates([{"type": "mode", "expect": "cad"}], m, 0)[0].ok


def test_mesh_gates(tmp_path):
    mesh = trimesh.creation.box(extents=(80, 60, 66))
    p = tmp_path / "m.stl"
    mesh.export(p)
    out = _out(artifact_files=[str(p)])
    assert run_gates([{"type": "contains_point", "xyz": [0, 0, 10]}], out, 0)[0].ok
    assert not run_gates([{"type": "contains_point", "xyz": [0, 0, 200]}], out, 0)[0].ok
    assert run_gates([{"type": "bodies", "max": 1}], out, 0)[0].ok
    vol = run_gates([{"type": "volume", "min_mm3": 10000}], out, 0)[0]
    assert vol.ok
    frac = run_gates([{"type": "volume_fraction", "max": 1.0}], out, 0)[0]
    assert frac.ok
    # no artifact -> fail with detail, not crash
    res = run_gates([{"type": "radial_peaks", "count": 45}], _out(), 0)[0]
    assert res.ok is False and res.detail


def test_radial_peaks_counts_lobes(tmp_path):
    import numpy as np

    cyl = trimesh.creation.cylinder(radius=20, height=6, sections=720)
    v = cyl.vertices.copy()
    ang = np.arctan2(v[:, 1], v[:, 0])
    mod = 1.0 + 0.2 * (np.cos(ang * 8) > 0)  # 8 square-ish teeth
    v[:, 0] *= mod
    v[:, 1] *= mod
    cyl.vertices = v
    p = tmp_path / "gear.stl"
    cyl.export(p)
    out = _out(artifact_files=[str(p)])
    g = run_gates([{"type": "radial_peaks", "count": 8, "tol": 1, "axis": "z"}], out, 0)[0]
    assert g.ok, g.detail
