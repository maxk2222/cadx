"""Codemod tests: deterministic rewrites of unambiguous API misuse (M4/C5)."""

from __future__ import annotations

import ast

from cadx.cad.codemod import fix_api_misuse


def test_simple_translate_rewrite():
    code = "result = box.translate(1, 2, 3)\n"
    fixed, notes = fix_api_misuse(code)
    assert fixed == "result = box.translate((1, 2, 3))\n"
    assert len(notes) == 1
    assert "line 1" in notes[0]


def test_multiple_hits_back_to_front():
    code = "a = x.translate(1, 2, 3)\nb = y.translate(4, 5, 6)\n"
    fixed, notes = fix_api_misuse(code)
    assert "x.translate((1, 2, 3))" in fixed
    assert "y.translate((4, 5, 6))" in fixed
    assert len(notes) == 2
    assert "line 1" in notes[0] and "line 2" in notes[1]


def test_nested_in_expression():
    code = "part = base - cutter.translate(10, 0, 5) + rib\n"
    fixed, notes = fix_api_misuse(code)
    assert fixed == "part = base - cutter.translate((10, 0, 5)) + rib\n"
    assert len(notes) == 1


def test_comments_and_line_numbers_preserved():
    code = "b = Box(1, 1, 1)  # base\nc = b.translate(5, 0, 0)  # move it\nresult = c\n"
    fixed, _notes = fix_api_misuse(code)
    assert "# base" in fixed and "# move it" in fixed
    assert fixed.splitlines()[1] == "c = b.translate((5, 0, 0))  # move it"
    ast.parse(fixed)  # still valid


def test_multiline_call():
    code = "c = b.translate(\n    5,\n    0,\n    0,\n)\n"
    fixed, notes = fix_api_misuse(code)
    assert len(notes) == 1
    # valid code, and the call now passes ONE argument
    call = next(n for n in ast.walk(ast.parse(fixed)) if isinstance(n, ast.Call))
    assert len(call.args) == 1


def test_correct_code_unchanged():
    for code in [
        "result = box.translate((1, 2, 3))\n",
        "result = box.translate((1, 2, 3), transform=True)\n",
        "result = box.translate(v)\n",
        "result = box.translate(x, y, z, transform=True)\n",  # keywords -> not our rule
        "result = box.translate(*args)\n",
    ]:
        fixed, notes = fix_api_misuse(code)
        assert fixed == code, code
        assert notes == []


def test_syntax_error_unchanged():
    code = "def broken(:\n"
    fixed, notes = fix_api_misuse(code)
    assert fixed == code
    assert notes == []


def test_local_translate_definition_is_ambiguous():
    """A user-defined translate(x, y, z) method makes the call legal — skip."""
    code = (
        "class Mover:\n"
        "    def translate(self, x, y, z):\n"
        "        return self\n"
        "m = Mover().translate(1, 2, 3)\n"
    )
    fixed, notes = fix_api_misuse(code)
    assert fixed == code
    assert notes == []


def test_result_parses_and_runs():
    code = "v = (1, 2)\nr = b.translate(v[0], v[1], 0)\n"
    fixed, _notes = fix_api_misuse(code)
    ast.parse(fixed)
    assert "b.translate((v[0], v[1], 0))" in fixed
