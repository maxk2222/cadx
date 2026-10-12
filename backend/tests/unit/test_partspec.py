"""Static standard-part validation (AST + PARAMS resolution vs catalog)."""

from cadx.cad.partspec import check_standard_part_calls

GOOD = """from build123d import *
from bd_warehouse.fastener import CounterSunkScrew, ClearanceHole
PARAMS = {"screw": "M4-0.7", "length": 10.0}
with BuildPart() as p:
    Box(40, 20, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    screw = CounterSunkScrew(size=PARAMS["screw"], fastener_type="iso10642",
                             length=PARAMS["length"])
    with Locations(p.faces().sort_by(Axis.Z)[-1]):
        ClearanceHole(fastener=screw, fit="Normal")
result = p.part
"""


def test_valid_call_passes():
    assert check_standard_part_calls(GOOD) == []


def test_invalid_size_flagged():
    code = GOOD.replace('"M4-0.7"', '"M4-0.9"')
    errs = check_standard_part_calls(code)
    assert errs and "size 'M4-0.9' not in norm table" in errs[0]
    assert "line" in errs[0]


def test_invalid_type_flagged():
    code = GOOD.replace('fastener_type="iso10642"', 'fastener_type="din999"')
    errs = check_standard_part_calls(code)
    assert errs and "type 'din999' not valid" in errs[0]


def test_inline_literal_flagged():
    code = GOOD.replace('size=PARAMS["screw"]', 'size="M4-0.9"')
    errs = check_standard_part_calls(code)
    assert errs and "M4-0.9" in errs[0]


def test_invented_class_flagged():
    code = GOOD.replace("CounterSunkScrew", "TitaniumScrew")
    errs = check_standard_part_calls(code)
    assert errs and "not in the bd_warehouse catalog" in errs[0]


def test_nonliteral_skipped():
    code = GOOD.replace('size=PARAMS["screw"]', "size=compute_size()")
    assert check_standard_part_calls(code) == []


def test_parametric_class_not_validated():
    code = (
        "from build123d import *\n"
        "from bd_warehouse.gear import SpurGear\n"
        "PARAMS = {'m': 1.0}\n"
        "result = SpurGear(module=PARAMS['m'], tooth_count=20, "
        "pressure_angle=20, thickness=6)\n"
    )
    assert check_standard_part_calls(code) == []
