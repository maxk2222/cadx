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


def test_surrogate_code_is_check_failure_not_crash():
    r = check_code("x = 1\ud800\nresult = Box(1,1,1)\n")
    assert not r.ok
    assert any("parseable" in e for e in r.errors)


def test_lint_discarded_transform_warns():
    code = (
        GOOD
        + "with BuildPart() as p:\n"
        + "    Box(60, 40, 6)\n"
        + "    Box(10, 40, 48).translate((25, 0, 6))\n"
    )
    r = check_code(code)
    assert r.ok  # warnings do not fail the check
    assert any("discarded" in w for w in r.warnings)


def test_lint_locate_inplace_no_warning():
    code = GOOD + "part = Box(1,1,1)\npart.locate(Pos(5,0,0))\nresult = part\n"
    r = check_code(code)
    assert not any("discarded" in w for w in r.warnings)


def test_lint_double_transform_warns():
    """Observed on l_bracket ribs: `wedge = wp.part.moved(Pos(0,y0,0))` then
    `wedge = wedge.translate(0, y0, 0)` — the second transform applied on top
    of the first and the rib floated outside the part."""
    code = (
        GOOD
        + "w = Box(10, 3, 10)\n"
        + "w = w.moved(Pos(0, 17, 0))\n"
        + "w = w.translate(0, 17, 0)\n"
        + "result = w\n"
    )
    r = check_code(code)
    assert r.ok
    assert any("twice" in w and "w" in w for w in r.warnings)
    # a single transform chain is fine — placement via Pos, one translate
    ok = GOOD + "w = Box(10, 3, 10)\nw = w.moved(Pos(0, 17, 0))\nresult = w\n"
    r2 = check_code(ok)
    assert not any("twice" in w for w in r2.warnings)


def test_lint_self_axis_rotation_warns():
    code = (
        GOOD
        + "cap = Cylinder(radius=2, height=10)\n"
        + "cap2 = cap.rotate(Axis.Z, 90)  # no-op: own axis\n"
        + "flat = cap.rotate(Axis.X, 90)  # fine\n"
    )
    r = check_code(code)
    assert any("no-op" in w and "Axis.Z" in w for w in r.warnings)
    # only ONE self-axis warning (Axis.X on a cylinder is legitimate)
    assert sum("no-op" in w for w in r.warnings) == 1


def test_lint_2d_shape_in_boolean_warns():
    code = GOOD + "mesh = Box(10, 10, 5)\nresult = mesh - Pos(0, 0, 0) * Ellipse(4, 2)\n"
    r = check_code(code)
    assert any("2D" in w for w in r.warnings)
    code2 = GOOD + "mesh = Box(10,10,5)\nresult = mesh.cut(Pos(1,1,0) * Ellipse(3, 2))\n"
    r2 = check_code(code2)
    assert any("2D" in w for w in r2.warnings)


def test_lint_rot_bad_kwargs_warns():
    """T1.1: Rot(a=, about=) — the observed grip failure — warns pre-exec."""
    code = GOOD + "bore = Cylinder(2, 10).moved(Rot(a=90, about=Axis.Y))\nresult = bore\n"
    r = check_code(code)
    assert any("keyword" in w and "Rot" in w for w in r.warnings)
    ok = GOOD + "bore = Cylinder(2, 10).moved(Rot(axis=Axis.Y, angle=90))\nresult = bore\n"
    r2 = check_code(ok)
    assert not any("keyword" in w for w in r2.warnings)


def test_lint_hallucinated_method_warns():
    """Observed: `body.removeSplitter()` — CadQuery idiom absent in build123d —
    caused a non-converging repair loop; warn with the real API name."""
    code = GOOD + "body = Box(10, 10, 5)\nbody = body.removeSplitter()\nresult = body\n"
    r = check_code(code)
    assert any("removeSplitter" in w and "clean()" in w for w in r.warnings)
    ok = GOOD + "body = Box(10, 10, 5)\nbody = body.clean()\nresult = body\n"
    r2 = check_code(ok)
    assert not any("removeSplitter" in w for w in r2.warnings)


def test_lint_clean_code_no_warnings():
    code = (
        GOOD
        + "with BuildPart() as p:\n"
        + "    Box(60, 40, 6)\n"
        + "    with Locations(Pos(25, 0, 6)):\n"
        + "        Box(10, 40, 48)\n"
        + "result = p.part\n"
    )
    r = check_code(code)
    assert r.ok and not r.warnings, r.warnings


def test_absurd_literal_warns():
    """Corrupted sampling emits literals like -8e+67 — flag them."""
    code = "PARAMS = {}\nresult = Box(-8e+67, 10, 10)\n"
    res = check_code(code)
    assert res.ok
    assert any("suspicious literal" in w for w in res.warnings)


def test_float_nonnumeric_string_warns():
    code = "PARAMS = {}\nresult = Box(float('.'), 10, 10)\n"
    res = check_code(code)
    assert res.ok
    assert any("float('.'" in w for w in res.warnings)


def test_bare_pos_times_primitive_inside_builder_is_error():
    code = (
        GOOD
        + """with BuildPart() as p:
    Pos(-28, 0, 4) * Box(4, 40, 50, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = p.part
"""
    )
    r = check_code(code)
    assert not r.ok
    assert any("DISCARDED" in e for e in r.errors)


def test_bare_pos_times_primitive_outside_builder_is_warning():
    r = check_code(GOOD + "Pos(1, 2, 3) * Cylinder(2, 10)\n")
    assert r.ok
    assert any("DISCARDED" in w for w in r.warnings)


def test_pos_times_tracked_var_inside_builder_is_error():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(60, 40, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
rib = Pos(0, 10, 0) * Cylinder(1, 5)
with BuildPart() as q:
    Pos(1, 0, 0) * rib
result = p.part
"""
    )
    r = check_code(code)
    assert not r.ok


def test_locations_scoped_primitive_is_fine():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(60, 40, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations(Pos(-28, 0, 4)):
        Box(4, 40, 50, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = p.part
"""
    )
    r = check_code(code)
    assert r.ok, r.errors


def test_assigned_pos_times_primitive_is_fine():
    r = check_code(GOOD + "leg = Pos(-28, 0, 4) * Box(4, 40, 50)\nresult = leg\n")
    assert r.ok, r.errors


def test_add_pos_times_primitive_is_fine():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(60, 40, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    add(Pos(-28, 0, 4) * Box(4, 40, 50))
result = p.part
"""
    )
    r = check_code(code)
    assert r.ok, r.errors


def test_shapelist_filter_is_error():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(10, 10, 10)
    edges = p.edges().filter_by(GeomType.LINE).filter(lambda e: e.center().X > 0)
    fillet(edges, radius=1)
result = p.part
"""
    )
    r = check_code(code)
    assert not r.ok
    assert any("filter" in e for e in r.errors)


def test_list_comprehension_edges_is_fine():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(10, 10, 10)
    edges = [e for e in p.edges().filter_by(GeomType.LINE) if e.center().X > 0]
    fillet(edges, radius=1)
result = p.part
"""
    )
    r = check_code(code)
    assert r.ok, r.errors


def test_center_attr_without_call_is_error():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(10, 10, 10)
    e = [x for x in p.edges() if x.center.X > 0]
result = p.part
"""
    )
    r = check_code(code)
    assert not r.ok
    assert any(".center().X" in e for e in r.errors)


def test_center_call_is_fine():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(10, 10, 10)
    e = [x for x in p.edges() if x.center().X > 0]
result = p.part
"""
    )
    r = check_code(code)
    assert r.ok, r.errors


def test_builder_part_augassign_is_warning():
    code = (
        GOOD
        + """with BuildPart() as p:
    Box(10, 10, 10)
    p.part += Pos(20, 0, 0) * Box(2, 2, 2)
result = p.part
"""
    )
    r = check_code(code)
    assert r.ok, r.errors
    assert any("add(" in w for w in r.warnings)


def test_oversized_fillet_on_box_warns():
    code = """from build123d import *
PARAMS = {"slot_w": 6.5}
L, T = 60.0, 4.0
with BuildPart() as p:
    Box(L, 40, T)
slot = Pos(-28, 0, 27) * Box(6, PARAMS["slot_w"], 15)
slot = fillet(slot.edges().filter_by(Axis.X), radius=PARAMS["slot_w"] / 2)
result = p.part
"""
    r = check_code(code)
    assert r.ok, r.errors
    assert any("oblong_slot" in w for w in r.warnings)


def test_sane_fillet_on_box_no_warning():
    code = """from build123d import *
PARAMS = {"r": 1.5}
with BuildPart() as p:
    Box(40, 20, 10)
b = fillet(Box(40, 20, 10).edges().filter_by(Axis.Z), radius=PARAMS["r"])
result = b
"""
    r = check_code(code)
    assert r.ok, r.errors
    assert not any("oblong_slot" in w for w in r.warnings)
