"""Static pre-checks for LLM-generated code (AST-level, runs before execution).

Allowed import roots: build123d, bd_warehouse, cadlib, math, numpy, typing,
dataclasses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_MODULES = {
    "build123d",
    "bd_warehouse",
    "cadlib",
    "math",
    "numpy",
    "np",
    "typing",
    "dataclasses",
}

FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "breakpoint",
    "input",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "help",
    "exit",
    "quit",
}

FORBIDDEN_ATTRS = {
    "__subclasses__",
    "__builtins__",
    "__loader__",
    "__spec__",
    "__globals__",
    "__code__",
    "__class__",
    "__dict__",
    "__import__",
    "mro",
}

_ALLOWED_LITERAL_TYPES = (int, float, str, bool)


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    params: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# transforms that RETURN a new object — a bare expression statement discards
# the result (verified: translate/rotate return copies; locate/move are in-place)
_COPY_TRANSFORMS = {"translate", "rotate", "moved", "located", "scale", "mirror", "offset"}
# 2D curve/sketch constructors — invalid as boolean operands on solids
_2D_CTORS = {
    "Ellipse",
    "Circle",
    "Rectangle",
    "Polygon",
    "RegularPolygon",
    "Trapezoid",
    "Triangle",
    "Text",
    "SlotOverall",
    "SlotCenterToCenter",
    "SlotArc",
}
# shape constructors whose self-axis rotation is a geometric no-op
_AXIS_SELF = {"Cylinder": "Z", "Cone": "Z"}
_BOOLEAN_CUTTERS = {"cut", "fuse", "intersect"}


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.params_node: ast.Assign | None = None
        self.has_result = False
        # simple ctor tracking: `leg = Cylinder(...)` -> {"leg": "Cylinder"}
        self._ctors: dict[str, str] = {}

    def _err(self, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"line {line}: {msg}")

    def _warn(self, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", "?")
        self.warnings.append(f"line {line}: {msg}")

    def _ctor_name(self, node: ast.AST) -> str | None:
        """Constructor name for `Name(...)` calls and tracked `name = Ctor(...)` vars."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node, ast.Name):
            return self._ctors.get(node.id)
        return None

    def _contains_2d_ctor(self, node: ast.AST) -> str | None:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in _2D_CTORS
            ):
                return sub.func.id
        return None

    def visit_Expr(self, node: ast.Expr) -> None:
        # `Box(...).translate(5,0,0)` as a statement: translate/rotate RETURN a
        # moved copy — discarded here, the original stays unpositioned
        v = node.value
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr in _COPY_TRANSFORMS
        ):
            self._warn(
                node,
                f"`.{v.func.attr}(...)` returns a new object — its result is "
                "discarded. Assign it (`p = p.translate(...)`) or use `Pos *` on "
                "the primitive instead.",
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # `mesh - Pos(...) * Ellipse(...)`: a 2D curve cannot boolean a solid
        if isinstance(node.op, ast.Sub | ast.Add):
            for side in (node.left, node.right):
                bad = self._contains_2d_ctor(side)
                if bad:
                    self._warn(
                        node,
                        f"`{bad}(...)` is a 2D shape — it cannot be a boolean "
                        "operand. Extrude it or use a 3D primitive "
                        "(Cylinder/Box/extrude).",
                    )
        self.generic_visit(node)

    def _check_rotate_self_axis(self, node: ast.Call) -> None:
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "rotate"):
            return
        if not node.args:
            return
        a0 = node.args[0]
        axis = a0.attr if isinstance(a0, ast.Attribute) and a0.attr in ("X", "Y", "Z") else None
        ctor = self._ctor_name(f.value)
        if ctor == "Sphere" and axis:
            self._warn(node, "rotating a Sphere is always a no-op — remove it")
        elif ctor in _AXIS_SELF and axis == _AXIS_SELF[ctor]:
            self._warn(
                node,
                f"rotating a {ctor} about Axis.{axis} is a no-op (its own axis) — "
                "to lay it over, rotate about a perpendicular axis or build it "
                "with `Pos(..., rotation=...)`/Plane.",
            )

    def _check_2d_boolean_method(self, node: ast.Call) -> None:
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in _BOOLEAN_CUTTERS):
            return
        for arg in node.args:
            bad = self._contains_2d_ctor(arg)
            if bad:
                self._warn(
                    node,
                    f"`.{f.attr}({bad}(...))` — a 2D shape cannot be a boolean "
                    "tool on a solid. Extrude it or use a 3D primitive.",
                )

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                hint = (
                    " (cadquery is not available - use build123d / bd_warehouse)"
                    if root in {"cadquery", "cq_warehouse"}
                    else ""
                )
                self._err(node, f"import of '{a.name}' is not allowed{hint}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_MODULES:
            hint = (
                " (cadquery is not available - use build123d / bd_warehouse)"
                if root in {"cadquery", "cq_warehouse"}
                else ""
            )
            self._err(node, f"import from '{node.module}' is not allowed{hint}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            self._err(node, f"call to '{node.func.id}' is not allowed")
        # float('…') with a non-numeric string — observed model corruption
        # (`float('.')`) that parses fine but raises at exec time
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            try:
                float(node.args[0].value)
            except ValueError:
                self._warn(
                    node,
                    f"float({node.args[0].value!r}) raises ValueError at runtime — "
                    "the literal is not numeric (corrupted output?)",
                )
        self._check_rotate_self_axis(node)
        self._check_2d_boolean_method(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # absurd magnitude — models emit e.g. -8e+67 when sampling degenerates;
        # no real part dimension exceeds ~1e5 mm
        if (
            isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
            and abs(node.value) > 1e6
        ):
            self._warn(
                node,
                f"suspicious literal {node.value!r} — far beyond any plausible mm "
                "dimension (corrupted output?)",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"__import__"}:
            self._err(node, f"use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self._err(node, f"attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        ctor = (
            node.value.func.id
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            else None
        )
        for t in node.targets:
            if isinstance(t, ast.Name):
                if t.id == "PARAMS" and isinstance(node.value, ast.Dict):
                    self.params_node = node
                if t.id == "result":
                    self.has_result = True
                if ctor:
                    self._ctors[t.id] = ctor
        self.generic_visit(node)


def _parse_params(tree: ast.Module) -> tuple[dict[str, object], list[str]]:
    """Return (params dict, errors) from the top-level PARAMS literal."""
    errors: list[str] = []
    params: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                    errors.append("PARAMS keys must be string literals")
                    continue
                if isinstance(v, ast.Constant) and isinstance(v.value, _ALLOWED_LITERAL_TYPES):
                    params[k.value] = v.value
                elif (
                    isinstance(v, ast.UnaryOp)
                    and isinstance(v.op, ast.USub)
                    and isinstance(v.operand, ast.Constant)
                    and isinstance(v.operand.value, int | float)
                ):
                    params[k.value] = -v.operand.value
                else:
                    errors.append(f"PARAMS['{k.value}'] must be a literal number/string/bool")
            return params, errors
    return params, ["missing top-level PARAMS = {...} dict literal"]


def check_code(code: str) -> CheckResult:
    """Static check of user code. Returns CheckResult(ok, errors, params)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CheckResult(ok=False, errors=[f"syntax error line {e.lineno}: {e.msg}"])
    except Exception as e:
        # e.g. UnicodeEncodeError from lone surrogates in model output —
        # must surface as a repairable contract failure, not a crash
        return CheckResult(ok=False, errors=[f"code is not parseable: {type(e).__name__}: {e}"])

    checker = _Checker()
    checker.visit(tree)
    params, param_errors = _parse_params(tree)
    errors = checker.errors + param_errors
    if not checker.has_result:
        errors.append("missing assignment to `result` (a build123d Part/Solid/Compound)")
    return CheckResult(ok=not errors, errors=errors, params=params, warnings=checker.warnings)


def apply_params_override(code: str, overrides: dict[str, object]) -> str:
    """Insert `PARAMS.update(__PARAMS_OVERRIDE__)` right after the PARAMS assignment."""
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets)
            and isinstance(node.value, ast.Dict)
            and node.end_lineno is not None
        ):
            indent = "PARAMS.update(__PARAMS_OVERRIDE__)  # injected overrides\n"
            lines.insert(node.end_lineno, indent)
            return "".join(lines)
    raise ValueError("no top-level PARAMS dict literal found")


def validate_overrides(overrides: dict[str, object], declared: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key, value in overrides.items():
        if key not in declared:
            errors.append(f"unknown param '{key}' (declared: {sorted(declared)})")
            continue
        declared_type = type(declared[key])
        if isinstance(declared[key], bool):
            if not isinstance(value, bool):
                errors.append(f"param '{key}' must be bool")
        elif isinstance(declared[key], int | float):
            if not isinstance(value, int | float) or isinstance(value, bool):
                errors.append(f"param '{key}' must be a number")
        elif not isinstance(value, declared_type):
            errors.append(f"param '{key}' must be {declared_type.__name__}")
    return errors
