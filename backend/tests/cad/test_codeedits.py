"""Line-addressable code editing primitives."""

from __future__ import annotations

from cadx.cad import codeedits as ce

CODE = """PARAMS = {
    "length": 100.0,  # mm overall
    "screw": "M4-0.7",
}
from build123d import *

with BuildPart() as p:
    Box(PARAMS["length"], 60, 10)
    Hole(4.0)
result = p.part
"""


def test_line_numbered():
    out = ce.line_numbered(CODE, 6, 8)
    assert "6 │" in out and "7 │" in out and "8 │" in out
    assert "with BuildPart" in out


def test_read_lines_basic_and_clamped():
    r = ce.read_lines(CODE, 6, 10)
    assert r["ok"] and r["start"] == 6 and r["end"] == 10
    assert "Box(" in r["content"]
    r2 = ce.read_lines(CODE, 1, 9999)
    assert r2["ok"] and r2["clamped"] and r2["end"] == r2["lines_total"]


def test_read_lines_invalid():
    assert not ce.read_lines(CODE, 0, 5)["ok"]
    assert not ce.read_lines(CODE, 9, 5)["ok"]
    assert not ce.read_lines("", 1, 5)["ok"]


def test_search_literal_and_regex():
    r = ce.search_code(CODE, "Hole", context=1)
    assert r["ok"] and r["count"] == 1 and r["matches"][0]["line"] == 9
    r2 = ce.search_code(CODE, r"Hole\(\d", regex=True)
    assert r2["ok"] and r2["count"] == 1
    assert ce.search_code(CODE, "nonexistent")["count"] == 0
    assert not ce.search_code(CODE, "([", regex=True)["ok"]


def test_replace_lines():
    r = ce.replace_lines(CODE, 8, 9, "    Box(80, 40, 8)\n    Hole(2.0)")
    assert r["ok"]
    assert "Box(80, 40, 8)" in r["code"] and "Hole(2.0)" in r["code"]
    assert "Box(PARAMS" not in r["code"]
    assert not ce.replace_lines(CODE, 99, 100, "x")["ok"]


def test_insert_after_before():
    r = ce.insert_after_line(CODE, 10, "# tail")
    assert r["ok"] and r["code"].splitlines()[10] == "# tail"
    r2 = ce.insert_before_line(CODE, 1, "# header")
    assert r2["ok"] and r2["code"].startswith("# header")
    assert not ce.insert_after_line(CODE, 99, "x")["ok"]
    assert not ce.insert_before_line(CODE, 0, "x")["ok"]


def test_find_replace():
    r = ce.find_replace(CODE, "Hole(4.0)", "Hole(4.5)")
    assert r["ok"] and "Hole(4.5)" in r["code"]
    assert not ce.find_replace(CODE, "PARAMS", "X")["ok"]  # ambiguous
    miss = ce.find_replace(CODE, "Hole(9.9)", "x")
    assert not miss["ok"]


def test_set_param_number_str_bool():
    r = ce.set_param(CODE, "length", 120.0)
    assert r["ok"] and '"length": 120.0,  # mm overall' in r["code"]
    r2 = ce.set_param(CODE, "screw", "M5-0.8")
    assert r2["ok"] and '"screw": "M5-0.8",' in r2["code"]
    missing = ce.set_param(CODE, "nope", 1.0)
    assert not missing["ok"] and "insert_after" in missing["error"]


def test_apply_line_ops_bottom_up():
    r = ce.apply_line_ops(
        CODE,
        [
            {"op": "replace", "start": 2, "end": 2, "code": '    "length": 200.0,'},
            {"op": "insert_after", "line": 10, "code": "# added"},
        ],
    )
    assert r["ok"] and r["applied"] == 2
    lines = r["code"].splitlines()
    assert '"length": 200.0,' in lines[1]
    assert "# added" in lines[-2] or "# added" in lines[-1]


def test_apply_line_ops_bad_range():
    r = ce.apply_line_ops(CODE, [{"op": "replace", "start": 99, "end": 100, "code": "x"}])
    assert not r["ok"]
    r2 = ce.apply_line_ops(CODE, [{"op": "bogus", "line": 1}])
    assert not r2["ok"]


def test_no_op_edit_rejected():
    assert not ce.replace_lines(CODE, 9, 9, "    Hole(4.0)")["ok"]
