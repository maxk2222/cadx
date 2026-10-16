"""Tests for cadx.cad.defects — B-Rep defect location + STEP roundtrip gate."""

from build123d import Box, Shell, Solid

from cadx.cad.defects import (
    defect_summary,
    locate_defects,
    step_roundtrip_check,
    summarize_defects,
)


def _open_solid() -> Solid:
    """A solid built on an open (5-face) shell — BRepCheck-invalid."""
    box = Box(10, 10, 10)
    return Solid(Solid._make_solid(Shell(list(box.faces())[:5])))


def test_valid_box_has_no_defects():
    assert locate_defects(Box(10, 10, 10)) == []
    assert defect_summary(Box(10, 10, 10)) == "no B-Rep defects located"


def test_broken_solid_located_with_coords():
    s = _open_solid()
    assert not s.is_valid
    defects = locate_defects(s)
    assert defects
    assert not any("error" in d for d in defects)
    kinds = {d["kind"] for d in defects}
    assert "open_edge" in kinds
    for d in defects:
        assert d["shape_type"]
        assert isinstance(d.get("index"), int)
        assert d["description"]
        center = d["center"]
        assert center is None or (len(center) == 3 and all(isinstance(v, float) for v in center))
    # the open rim sits on the missing top face's boundary (z = +5)
    open_edges = [d for d in defects if d["kind"] == "open_edge"]
    assert open_edges
    assert all(abs(d["center"][2] - 5.0) < 1e-6 for d in open_edges)
    summary = defect_summary(s)
    assert "defect" in summary and "open_edge" in summary


def test_locate_defects_never_raises_on_nonsense():
    for bad in (object(), None, 42):
        out = locate_defects(bad)
        assert len(out) == 1
        assert "error" in out[0]
    assert "failed" in defect_summary(object())


def test_summarize_defects_contract():
    assert summarize_defects([]) == "no B-Rep defects located"
    assert summarize_defects([{"error": "boom"}]) == "boom"


def test_step_roundtrip_clean_box(tmp_path):
    gate_file = tmp_path / "gate.step"
    report = step_roundtrip_check(Box(10, 10, 10), gate_file)
    assert report["checked"] and report["ok"], report
    assert "warning" not in report
    assert report["volume_delta_pct"] < 0.5
    assert not gate_file.exists()  # self-written temp file removed


def test_step_roundtrip_broken_solid_warns(tmp_path):
    report = step_roundtrip_check(_open_solid(), tmp_path / "gate.step")
    assert report["checked"], report
    assert report["ok"] is False
    assert report["warning"].startswith("STEP roundtrip:")
    # volume collapses to 0 (re-imports as an open shell) -> 100% drift
    assert report["volume_delta_pct"] > 0.5


def test_step_roundtrip_skips_huge_shape(tmp_path):
    report = step_roundtrip_check(Box(10, 10, 10), tmp_path / "gate.step", max_faces=1)
    assert report["checked"] is False
    assert "faces" in report["skipped"]


def test_step_roundtrip_existing_file_write_false(tmp_path):
    from build123d import export_step

    target = tmp_path / "model.step"
    export_step(Box(10, 10, 10), str(target))
    report = step_roundtrip_check(Box(10, 10, 10), target, write=False)
    assert report["checked"] and report["ok"], report
    assert target.exists()  # a pre-exported artifact is never removed


def test_pipeline_invalid_result_reports_defect_coords(tmp_path):
    """End-to-end: generated code producing an invalid solid surfaces defect
    coordinates in the ExecResult warnings/features for the repair loop."""
    from cadx.cad.engine import run_pipeline

    code = (
        "from build123d import *\n"
        "PARAMS = {}\n"
        "box = Box(10, 10, 10)\n"
        "result = Solid(Solid._make_solid(Shell(list(box.faces())[:5])))\n"
    )
    res = run_pipeline(code=code, out_dir=tmp_path / "it1", light=True)
    assert res.ok, res.error
    assert res.analysis is not None
    assert res.analysis.is_valid is False
    assert res.features.get("defects"), "expected located defects in features"
    defect_warn = [w for w in res.warnings if "invalid B-Rep" in w]
    assert defect_warn, res.warnings
    assert "@" in defect_warn[0]  # carries coordinates


def test_pipeline_step_gate_no_warning_on_clean_box(tmp_path):
    """Happy path: a clean box produces no STEP roundtrip warning."""
    from cadx.cad.engine import run_pipeline

    res = run_pipeline(
        code="from build123d import *\nPARAMS = {}\nresult = Box(10, 10, 10)\n",
        out_dir=tmp_path / "it1",
        light=True,
    )
    assert res.ok, res.error
    assert not any("STEP roundtrip" in w for w in res.warnings)
    assert not (tmp_path / "it1" / "_run" / "step_gate.step").exists()
