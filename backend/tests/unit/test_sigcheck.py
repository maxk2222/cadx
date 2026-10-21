"""Signature gate (M1): introspected build123d/cadlib/bd_warehouse member
index + AST checks over generated code — property-called-as-method (the
real ``Edge.radius()`` failure class), unknown attributes, module-member
hallucinations, arity warnings, and ToolSession surfacing."""

from __future__ import annotations

import inspect
import uuid

import build123d as bd

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.toolspecs import ToolSession
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.cad.allowlist import check_code
from cadx.cad.sigcheck import check_code_signatures
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.runtime_config import RuntimeSettings

GOOD = "from build123d import *\n\nPARAMS = {}\n\nresult = Box(10, 10, 5)\n"

SEED = '''"""seed scaffold."""

from build123d import *

PARAMS = {}
'''


def _ctx(**settings_kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    return AgentCtx(
        llm=FakeLLM([]),
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="test",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
        runtime_settings=RuntimeSettings(**settings_kw),
    )


# --------------------------------------------------------------------------- #
# checker core                                                                 #
# --------------------------------------------------------------------------- #


def test_property_called_as_method_is_error():
    """The real observed failure: Edge.radius is a property, not a method."""
    assert isinstance(inspect.getattr_static(bd.Edge, "radius"), property)
    rep = check_code_signatures(GOOD + "e = result.edges()[0]\nr = e.radius()\n")
    assert any("Edge.radius" in e and "property" in e for e in rep.errors)


def test_property_called_on_class_receiver():
    rep = check_code_signatures(GOOD + "r = Edge.radius()\n")
    assert any("Edge.radius" in e and "property" in e for e in rep.errors)


def test_unknown_attribute_on_inferred_type():
    rep = check_code_signatures(GOOD + "result.frobnicate()\n")
    assert any("has no attribute 'frobnicate'" in e for e in rep.errors)


def test_constructor_assignment_inference():
    rep = check_code_signatures(GOOD + "plate = Cylinder(2, 5)\nplate.bogus()\n")
    assert any("Cylinder" in e and "bogus" in e for e in rep.errors)


def test_loop_target_inference():
    rep = check_code_signatures(GOOD + "for e in result.edges():\n    e.radius()\n")
    assert any("Edge.radius" in e for e in rep.errors)


def test_comprehension_target_inference():
    rep = check_code_signatures(GOOD + "radii = [e.radius() for e in result.edges()]\n")
    assert any("Edge.radius" in e for e in rep.errors)


def test_module_member_hallucination():
    code = (
        "import build123d\nPARAMS = {}\n"
        "result = build123d.Box(1, 1, 1)\nx = build123d.not_a_real_thing()\n"
    )
    rep = check_code_signatures(code)
    assert any("build123d" in e and "not_a_real_thing" in e for e in rep.errors)


def test_cadlib_member_hallucination():
    code = "import cadlib\nPARAMS = {}\nresult = Box(1,1,1)\ncadlib.not_real()\n"
    rep = check_code_signatures(code)
    assert any("cadlib" in e and "not_real" in e for e in rep.errors)


def test_arity_mismatch_is_warning_not_error():
    rep = check_code_signatures(
        "import math\nPARAMS = {}\nresult = Box(1,1,1)\nx = math.sqrt(1, 2)\n"
    )
    assert not rep.errors
    assert any("sqrt" in w for w in rep.warnings)


def test_uninferable_receiver_is_silent():
    """No type info -> no findings (a wrong error teaches bad things)."""
    rep = check_code_signatures(
        GOOD + "v = load_reference('abc')\nv.anything_at_all()\nv.radius()\n"
    )
    assert not rep.errors and not rep.warnings


def test_clean_code_has_no_findings():
    rep = check_code_signatures(
        GOOD
        + "e = result.edges().sort_by(Axis.Z)[0]\n"
        + "r = e.radius\n"
        + "result = result.fillet if False else result\n"
    )
    assert not rep.errors


def test_buildpart_part_inference():
    code = (
        "from build123d import *\nPARAMS = {}\n"
        "with BuildPart() as bp:\n    Box(1, 1, 1)\n"
        "result = bp.part\nresult.frobnicate()\n"
    )
    rep = check_code_signatures(code)
    assert any("frobnicate" in e for e in rep.errors)


def test_shapelist_has_no_filter():
    rep = check_code_signatures(GOOD + "es = result.edges()\nes.filter(lambda x: x)\n")
    assert any("filter" in e for e in rep.errors)


def test_unparseable_code_is_empty_report():
    rep = check_code_signatures("def broken(:\n")
    assert not rep.errors and not rep.warnings


# --------------------------------------------------------------------------- #
# check_code integration                                                       #
# --------------------------------------------------------------------------- #


def test_check_code_sigcheck_off_by_default():
    code = GOOD + "e = result.edges()[0]\nr = e.radius()\n"
    res = check_code(code)
    assert res.ok and not res.sig_errors


def test_check_code_sigcheck_on_appends_findings():
    code = GOOD + "e = result.edges()[0]\nr = e.radius()\n"
    res = check_code(code, sigcheck=True)
    assert not res.ok
    assert any("Edge.radius" in e for e in res.sig_errors)
    assert not res.errors  # contract itself is clean — sig gate owns this


# --------------------------------------------------------------------------- #
# tool session integration                                                     #
# --------------------------------------------------------------------------- #


async def test_apply_surfaces_sig_errors_without_rejecting():
    ctx = _ctx()
    s = ToolSession(ctx, kind="edit", code=SEED + "result = Box(10, 10, 5)\n")
    last = len(s.buffer.splitlines())
    res = await s.execute(
        "insert_after",
        {"line": last, "code": "e = result.edges()[0]\nr = e.radius()\n"},
    )
    assert res["ok"]
    assert any("Edge.radius" in e for e in res["sig_errors"])
    assert "e.radius()" in s.buffer  # edit applied — findings are advisory


async def test_apply_clean_edit_has_no_sig_keys():
    ctx = _ctx()
    s = ToolSession(ctx, kind="edit", code=SEED + "result = Box(10, 10, 5)\n")
    res = await s.execute(
        "find_replace",
        {"find": "result = Box(10, 10, 5)", "replace": "result = Box(10, 10, 6)"},
    )
    assert res["ok"]
    assert "sig_errors" not in res and "sig_warnings" not in res


async def test_apply_silent_when_sigcheck_disabled():
    ctx = _ctx(agent_sigcheck=False)
    s = ToolSession(ctx, kind="edit", code=SEED + "result = Box(10, 10, 5)\n")
    last = len(s.buffer.splitlines())
    res = await s.execute("insert_after", {"line": last, "code": "r = result.edges()[0].radius()"})
    assert res["ok"]
    assert "sig_errors" not in res


async def test_add_block_surfaces_sig_errors():
    ctx = _ctx()
    s = ToolSession(ctx, kind="build", code=SEED)
    res = await s.execute(
        "add_block",
        {"name": "base", "code": "result = Box(10, 10, 5)\nr = result.edge().radius()"},
    )
    assert res["ok"] and res["block"] == "base"
    assert any("radius" in e and "property" in e for e in res["sig_errors"])


async def test_done_includes_sig_errors():
    ctx = _ctx()
    s = ToolSession(
        ctx,
        kind="edit",
        code=SEED + "result = Box(10, 10, 5)\nx = result.frobnicate()\n",
    )
    await s.execute(
        "find_replace",
        {"find": "Box(10, 10, 5)", "replace": "Box(10, 10, 5)  # same"},
    )
    res = await s.execute("done", {"reasoning": "try"})
    assert res["ok"] is False
    assert any("frobnicate" in e for e in res["errors"])


async def test_submit_code_includes_sig_errors():
    ctx = _ctx()
    s = ToolSession(ctx, kind="codegen")
    res = await s.execute(
        "submit_code",
        {
            "code": "from build123d import *\nPARAMS = {}\n"
            "result = Box(1, 1, 1)\nresult.frobnicate()\n"
        },
    )
    assert res["ok"] is False
    assert any("frobnicate" in e for e in res["errors"])


async def test_submit_code_clean_passes_gate():
    ctx = _ctx()
    s = ToolSession(ctx, kind="codegen")
    res = await s.execute(
        "submit_code",
        {"code": "from build123d import *\nPARAMS = {}\nresult = Box(1, 1, 1)\n"},
    )
    assert res["ok"] is True


# --------------------------------------------------------------------------- #
# strict arity for Shape copy-transforms (M4)


def test_translate_three_positional_args_is_error():
    """The observed `translate(x, y, z)` misuse — an ERROR now, not a warning."""
    rep = check_code_signatures(GOOD + "result = result.translate(1, 2, 3)\n")
    assert any("translate" in e and "ONE vector" in e for e in rep.errors)


def test_translate_tuple_is_fine():
    rep = check_code_signatures(GOOD + "result = result.translate((1, 2, 3))\n")
    assert not rep.errors and not rep.warnings


def test_translate_two_args_second_not_bool_is_error():
    rep = check_code_signatures(GOOD + "result = result.translate(5, 10)\n")
    assert any("ONE vector" in e for e in rep.errors)


def test_translate_two_args_bool_flag_is_fine():
    rep = check_code_signatures(GOOD + "result = result.translate((1, 0, 0), True)\n")
    assert not rep.errors


def test_rotate_three_positional_args_is_error():
    """The observed `rotate((x, y, z), Axis, angle)` — Point, Axis, angle is not
    the build123d signature."""
    rep = check_code_signatures(GOOD + "result = result.rotate((0, 0, 0), Axis.Y, 30)\n")
    assert any("rotate" in e and "Axis, angle" in e for e in rep.errors)


def test_rotate_axis_angle_is_fine():
    rep = check_code_signatures(GOOD + "result = result.rotate(Axis.Y, 30)\n")
    assert not rep.errors and not rep.warnings


def test_strict_arity_still_silent_on_uninferable_receiver():
    rep = check_code_signatures(
        GOOD + "v = load_reference('abc')\nv = v.translate(1, 2, 3)\nv = v.rotate(1, 2, 3)\n"
    )
    assert not rep.errors


def test_strict_arity_other_methods_still_warn():
    """Non-Shape methods keep the warning-only policy (e.g. math module)."""
    rep = check_code_signatures(
        "import math\nPARAMS = {}\nresult = Box(1,1,1)\nx = math.sqrt(1, 2)\n"
    )
    assert not rep.errors
    assert any("sqrt" in w for w in rep.warnings)


def test_check_code_sigcheck_translate_error_blocks():
    res = check_code(GOOD + "result = result.translate(1, 2, 3)\n", sigcheck=True)
    assert not res.ok
    assert any("translate" in e for e in res.sig_errors)


# --------------------------------------------------------------------------- #
# keyword-arg binding + arg-type hints (P5)


def test_unknown_kwarg_on_module_call_is_error():
    """The observed `cadlib.oblong_slot(..., part=p)` eval burn — an unknown
    kwarg is a guaranteed TypeError in the sandbox, so the gate flags it."""
    rep = check_code_signatures(
        "import cadlib\nPARAMS = {}\nresult = Box(1,1,1)\ncadlib.oblong_slot(6.5, 15, part=p)\n"
    )
    assert any("oblong_slot" in e and "'part'" in e for e in rep.errors)


def test_unknown_kwarg_on_bare_call_is_error():
    rep = check_code_signatures(
        "from cadlib import through_hole\nPARAMS = {}\nresult = Box(1,1,1)\n"
        "through_hole(5.5, bogus=1)\n"
    )
    assert any("through_hole" in e and "'bogus'" in e for e in rep.errors)


def test_known_kwargs_stay_clean():
    rep = check_code_signatures(
        "import cadlib\nfrom cadlib import through_hole\n"
        "PARAMS = {}\nresult = Box(1,1,1)\n"
        "through_hole(5.5, csk_dia=8.4)\n"
        "through_hole(diameter=5.5)\n"
        "cadlib.oblong_slot(6.5, 15, direction='+y', normal='+z', depth=2)\n"
    )
    assert not rep.errors


def test_varkw_signature_accepts_any_kwarg():
    """`_bounds` must capture **kwargs — such callables take any keyword."""
    from cadx.cad.sigcheck import _bounds

    def _f(a: int, **kw) -> None: ...

    sig, _ = _bounds(inspect.signature(_f))
    assert sig is not None and sig.varkw

    def _g(a: int, b: int = 0) -> None: ...

    sig2, _ = _bounds(inspect.signature(_g))
    assert sig2 is not None and not sig2.varkw


def test_local_varkw_function_calls_not_flagged():
    """`def f(**kw)` — a locally defined callable is never checked at all."""
    rep = check_code_signatures(GOOD + "def f(**kw):\n    return kw\nx = f(anything=1)\n")
    assert not rep.errors


def test_moved_vector_arg_is_location_error():
    """`p.moved(Vector(...))` — arity-legal but a guaranteed TypeError:
    moved/located take a Location, not a Vector."""
    rep = check_code_signatures(
        "from build123d import *\nPARAMS = {}\n"
        "p = Box(10, 10, 5)\nresult = p.moved(Vector(0, 1, 0))\n"
    )
    assert any("moved" in e and "Location" in e and "Vector" in e for e in rep.errors)


def test_located_dotted_vector_arg_is_location_error():
    rep = check_code_signatures(
        "import build123d\nPARAMS = {}\n"
        "p = build123d.Box(10, 10, 5)\nresult = p.located(build123d.Vector(0, 0, 3))\n"
    )
    assert any("located" in e and "Location" in e for e in rep.errors)


def test_moved_pos_arg_is_legal():
    """Pos IS a Location subclass — the correct placement call is not flagged."""
    rep = check_code_signatures(
        "from build123d import *\nPARAMS = {}\np = Box(10, 10, 5)\nresult = p.moved(Pos(0, 1, 0))\n"
    )
    assert not rep.errors


def test_moved_vector_silent_on_uninferable_receiver():
    """No inferred Shape receiver -> no finding (conservative principle)."""
    rep = check_code_signatures(GOOD + "v = load_reference('abc')\nv = v.moved(Vector(0, 1, 0))\n")
    assert not rep.errors


def test_safe_fillet_radius_cannot_overwrite_shape():
    """The helper returns a radius, so assigning it to the model shape is a
    guaranteed downstream geometry failure."""
    code = (
        "from build123d import *\nfrom cadlib import safe_fillet\nPARAMS = {}\n"
        "part = Box(10, 10, 10)\npart = safe_fillet(part.edges(), 2)\n"
    )
    rep = check_code_signatures(code)
    assert any("returns an applied radius, not a Shape" in e for e in rep.errors)
