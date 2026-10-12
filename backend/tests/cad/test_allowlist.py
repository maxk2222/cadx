from cadx.cad.allowlist import apply_params_override, check_code, validate_overrides

GOOD = """from build123d import *
PARAMS = {"length": 10.0, "name": "a", "flag": True}
result = Box(PARAMS["length"], 10, 10)
"""


def test_good_code_passes():
    r = check_code(GOOD)
    assert r.ok, r.errors
    assert r.params == {"length": 10.0, "name": "a", "flag": True}


def test_bd_warehouse_import_allowed():
    code = (
        GOOD
        + "from bd_warehouse.fastener import HexNut\nresult = HexNut(size='M3-0.5', fastener_type='iso4032')\n"
    )
    r = check_code(code)
    assert r.ok, r.errors


def test_forbidden_imports():
    for snippet in [
        "import os",
        "import cadquery",
        "from cq_warehouse import fastener",
        "import socket",
    ]:
        r = check_code(GOOD + snippet + "\n")
        assert not r.ok, snippet
        if "cadquery" in snippet or "cq_warehouse" in snippet:
            assert any("build123d" in e for e in r.errors)


def test_forbidden_calls():
    for snippet in ["open('x')", "eval('1')", "exec('x=1')", "__import__('os')", "input('?')"]:
        r = check_code(GOOD + snippet + "\n")
        assert not r.ok, snippet


def test_forbidden_attrs():
    r = check_code(GOOD + "x = ().__class__.__subclasses__()\n")
    assert not r.ok


def test_missing_params_and_result():
    r = check_code("from build123d import *\nresult = Box(1,1,1)\n")
    assert not r.ok and any("PARAMS" in e for e in r.errors)
    r2 = check_code("from build123d import *\nPARAMS = {}\nx = Box(1,1,1)\n")
    assert not r2.ok and any("result" in e for e in r2.errors)


def test_params_must_be_literals():
    r = check_code("PARAMS = {'a': len('x')}\nresult = Box(1,1,1)\n")
    assert not r.ok


def test_override_injection():
    out = apply_params_override(GOOD, {"length": 20.0})
    assert "PARAMS.update(__PARAMS_OVERRIDE__)" in out
    assert out.index("PARAMS.update") > out.index("PARAMS = {")


def test_validate_overrides():
    declared = {"length": 10.0, "name": "a", "flag": True}
    assert validate_overrides({"length": 5.0}, declared) == []
    errs = validate_overrides({"nope": 1, "flag": 3, "name": 5}, declared)
    assert len(errs) == 3
