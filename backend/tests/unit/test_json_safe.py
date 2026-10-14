"""Non-finite floats must never reach Postgres JSON columns (T0.1)."""

import math

from cadx.agent.worker_jobs import _json_safe
from cadx.llm.schemas import ParamDef


def test_paramdef_nonfinite_bounds_become_none() -> None:
    p = ParamDef(name="a", value=1.0, min=float("-inf"), max=float("inf"))
    assert p.min is None
    assert p.max is None


def test_paramdef_nonfinite_value_becomes_zero() -> None:
    p = ParamDef(name="a", value=float("nan"))
    assert p.value == 0.0
    p2 = ParamDef(name="b", value=3.5, min=0.0, max=10.0)
    assert p2.value == 3.5
    assert p2.min == 0.0 and p2.max == 10.0


def test_json_safe_recurses() -> None:
    obj = {"a": [float("nan"), {"b": float("inf")}], "c": -math.inf, "d": "ok", "e": 1}
    assert _json_safe(obj) == {"a": [None, {"b": None}], "c": None, "d": "ok", "e": 1}


def test_json_safe_passthrough() -> None:
    assert _json_safe(None) is None
    assert _json_safe("x") == "x"
    assert _json_safe((1, 2.5)) == [1, 2.5]
