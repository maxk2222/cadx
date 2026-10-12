"""Patch application tests — find/replace hunks with contract validation."""

from __future__ import annotations

from cadx.cad.patch import apply_edits, apply_patch, diff_summary
from cadx.llm.schemas import PatchEdit

CODE = """from build123d import *

PARAMS = {
    "length": 40.0,
    "height": 10.0,
}

with BuildPart() as p:
    Box(PARAMS["length"], PARAMS["width"], PARAMS["height"],
        align=(Align.CENTER, Align.CENTER, Align.MIN))
result = p.part
"""


def _e(find: str, replace: str) -> PatchEdit:
    return PatchEdit(find=find, replace=replace)


def test_apply_single_edit():
    out, errs = apply_edits(CODE, [_e('"height": 10.0', '"height": 8.0')])
    assert errs == []
    assert '"height": 8.0' in out
    assert '"length": 40.0' in out


def test_apply_multiple_edits_ordered():
    out, errs = apply_edits(
        CODE,
        [
            _e('"height": 10.0', '"height": 8.0'),
            _e('PARAMS["length"]', 'PARAMS["length"]  # x'),
        ],
    )
    assert errs == []
    assert "# x" in out


def test_find_not_found():
    out, errs = apply_edits(CODE, [_e("nonexistent_xyz", "x")])
    assert out == CODE
    assert any("not found" in e for e in errs)


def test_find_ambiguous():
    code = CODE + '\n# PARAMS["length"] again\n'
    _out, errs = apply_edits(code, [_e('PARAMS["length"]', "X")])
    assert any("2 locations" in e or "matches" in e for e in errs)


def test_empty_find():
    _out, errs = apply_edits(CODE, [_e("   ", "x")])
    assert any("empty" in e for e in errs)


def test_identical_patch_rejected():
    out, errs = apply_patch(CODE, [_e('"height": 10.0', '"height": 10.0')])
    assert out is None
    assert any("identical" in e for e in errs)


def test_contract_breaking_patch_rejected():
    out, errs = apply_patch(
        CODE,
        [_e("result = p.part", "import os\nresult = p.part")],
    )
    assert out is None
    assert any("contract" in e for e in errs)


def test_apply_patch_validates_result():
    out, errs = apply_patch(
        CODE,
        [_e('"height": 10.0', '"height": 8.0'), _e('PARAMS["width"]', "20.0")],
    )
    assert errs == []
    assert out is not None
    assert '"height": 8.0' in out


def test_diff_summary():
    new = CODE.replace('"height": 10.0', '"height": 8.0')
    s = diff_summary(CODE, new)
    assert "height" in s
