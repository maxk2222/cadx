"""Error diagnosis tests — classification + failing-line extraction."""

from __future__ import annotations

from cadx.cad.errordiag import diag_block, diagnose

CODE = "line1\nline2\nline3\nline4\nline5\nline6\nline7\n"


def test_syntax_error():
    d = diagnose(
        "SyntaxError: invalid syntax (model.py, line 3)",
        'File "model.py", line 3\n    line3\n         ^\nSyntaxError: invalid syntax',
        CODE,
    )
    assert d.category == "syntax"
    assert d.failing_line == 3
    assert ">>>    3 | line3" in d.code_excerpt


def test_forbidden_import():
    d = diagnose(
        "import of 'os' is not allowed",
        None,
        CODE,
    )
    assert d.category == "forbidden-import"
    assert "build123d" in d.hint


def test_occ_fillet():
    d = diagnose(
        "StdFail_NotDone\nBRep_API: command not done.",
        'File "model.py", line 5, in <module>\n    fillet(...)\nStandard_Failure',
        CODE,
    )
    assert d.category == "occ-fillet"
    assert "radius" in d.hint
    assert "line5" in d.code_excerpt


def test_occ_fillet_empty_selection():
    """build123d raises 'Failed creating a fillet...' on an empty/stale edge
    selection — that is NOT the OCC radius error and needs its own hint."""
    d = diagnose(
        "ValueError: Failed creating a fillet with radius of 1, "
        "try a smaller value or use max_fillet()",
        'File "model.py", line 5, in <module>\n    fillet(edges, radius=3)\nValueError',
        CODE,
    )
    assert d.category == "occ-fillet-empty"
    assert "EMPTY" in d.hint
    assert "safe_fillet" in d.hint


def test_removesplitter_api_missing_method():
    """Observed: `body.removeSplitter()` (CadQuery idiom) looped 3x in repair —
    must get a dedicated category + the .clean() hint, not generic api-misuse."""
    d = diagnose(
        "AttributeError: 'Solid' object has no attribute 'removeSplitter'",
        'File "model.py", line 22\nAttributeError',
        CODE,
    )
    assert d.category == "api-missing-method"
    assert "clean()" in d.hint


def test_attribute_error_api_misuse():
    d = diagnose(
        "AttributeError: 'Edge' object has no attribute 'pos'",
        'File "model.py", line 4\nAttributeError',
        CODE,
    )
    assert d.category == "api-misuse"
    assert "center()" in d.hint


def test_contract_missing_result():
    d = diagnose("no `result` assigned", None, CODE)
    assert d.category == "contract"


def test_bd_warehouse_bad_size():
    d = diagnose("ValueError: invalid size 'M4.5' for iso10642", None, CODE)
    assert d.category == "bd-warehouse"


def test_unexpected_kwargs_api_kwargs():
    """T1.1: Rot(a=90, about=Axis.Y) gets a signature hint, not a generic
    TypeError diagnosis."""
    d = diagnose(
        "TypeError: Unexpected keyword arguments: a, about",
        'File "model.py", line 4\nTypeError',
        CODE,
    )
    assert d.category == "api-kwargs"
    assert "Rot" in d.hint and "axis=" in d.hint


def test_timeout():
    d = diagnose("TimeoutError: execution timed out", None, CODE)
    assert d.category == "timeout"


def test_unknown_falls_back_to_runtime():
    d = diagnose("ZeroDivisionError: division by zero", None, CODE)
    assert d.category == "runtime"
    assert d.summary.startswith("[runtime]")


def test_failing_line_from_error_only():
    d = diagnose("line 7: something broke", None, CODE)
    assert d.failing_line == 7
    assert "line7" in d.code_excerpt


def test_diag_block_contains_all_sections():
    d = diagnose("AttributeError", 'File "model.py", line 2\nx.pos()', CODE)
    txt = diag_block(d, "the error", "the traceback", "print output")
    assert "diagnosis:" in txt
    assert "hint:" in txt
    assert "failing code:" in txt
    assert "the error" in txt
    assert "the traceback" in txt
    assert "print output" in txt


def test_diagnose_2d_shape_in_boolean():
    d = diagnose(
        "TypeError: cut() expected a solid tool, got Curve",
        "Traceback:\n  File model.py, line 42\nTypeError: cut() expected a solid tool",
        "result = mesh.cut(Pos(1,1,0) * Ellipse(3, 2))\n",
    )
    assert d.category == "hybrid-2d"
    assert "2D" in d.hint or "Cylinder" in d.hint


def test_bbox_api_misuse_gets_bbox_hint():
    """OCP-style BoundBox attribute guesses must classify as api-bbox with
    the correct build123d API in the hint — observed 5 consecutive repair
    failures (boundingBox/xlen/xmax/xsize/size.x) in one session."""
    from cadx.cad.errordiag import diagnose

    for bad in (
        "'Vector' object has no attribute 'x'",
        "'BoundBox' object has no attribute 'xlen'",
        "'Part' object has no attribute 'boundingBox'",
    ):
        d = diagnose(error=f"AttributeError: {bad}", traceback="", code="")
        assert d.category == "api-bbox", bad
        assert "bbox_dims" in d.hint
