"""Static validation of standard-part calls in generated code.

Walks the AST for constructor calls to catalog classes (``CounterSunkScrew``,
``HexNut``, …) and checks literal ``size=``/``fastener_type=``/type kwargs —
including values taken from the ``PARAMS`` dict — against the norm tables.
Runs before the executor so an invented size string becomes a cheap static
error instead of a sandbox round-trip.

Unresolved (non-literal) arguments are skipped — they can't be checked.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from cadagent.cad import catalog

# call kwargs treated as the "type" selector, validated against class types
_TYPE_KWARGS = ("fastener_type", "bearing_type", "o_ring_type", "bottle_type")

# name suffixes that indicate a standard part — invented names get flagged
_PARTISH_RE = re.compile(r"(Screw|Nut|Washer|Bearing|Insert|Gear|Sprocket|Flange|Thread|Spacer)$")


def _params_literals(tree: ast.Module) -> dict[str, Any]:
    """Literal values of the top-level PARAMS dict (for PARAMS['x'] resolution)."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            out: dict[str, Any] = {}
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if isinstance(v, ast.Constant):
                        out[k.value] = v.value
                    elif (
                        isinstance(v, ast.UnaryOp)
                        and isinstance(v.op, ast.USub)
                        and isinstance(v.operand, ast.Constant)
                        and isinstance(v.operand.value, int | float)
                    ):
                        out[k.value] = -v.operand.value
            return out
    return {}


def _literal(node: ast.AST, params: dict[str, Any]) -> Any:
    """Resolve a call arg to a literal: direct Constant or PARAMS['key']."""
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "PARAMS"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return params.get(node.slice.value)
    return None


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def check_standard_part_calls(code: str) -> list[str]:
    """Return validation errors for standard-part calls in ``code``.

    Unknown classes used like ``XxxScrew``/``XxxNut`` are reported too — the
    name patterns come from the catalog registry.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    params = _params_literals(tree)
    reg = catalog.registry()
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not name:
            continue
        if name not in reg:
            if _PARTISH_RE.search(name):
                errors.append(
                    f"line {node.lineno}: '{name}' looks like a standard part but "
                    "is not in the bd_warehouse catalog — use a real class name"
                )
            continue
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            val = _literal(kw.value, params)
            if val is not None:
                kwargs[kw.arg] = val
        type_val: str | None = None
        for tw in _TYPE_KWARGS:
            if isinstance(kwargs.get(tw), str):
                type_val = kwargs[tw]
        errs = catalog.validate_part_ref(
            name,
            fastener_type=type_val,
            size=kwargs.get("size"),
        )
        for e in errs:
            errors.append(f"line {node.lineno}: {e}")
    return errors
