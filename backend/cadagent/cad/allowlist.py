"""Static pre-checks for LLM-generated code (AST-level, runs before execution).

Allowed import roots: build123d, bd_warehouse, math, numpy, typing, dataclasses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_MODULES = {
    "build123d",
    "bd_warehouse",
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


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.params_node: ast.Assign | None = None
        self.has_result = False

    def _err(self, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"line {line}: {msg}")

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
        for t in node.targets:
            if isinstance(t, ast.Name):
                if t.id == "PARAMS" and isinstance(node.value, ast.Dict):
                    self.params_node = node
                if t.id == "result":
                    self.has_result = True
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

    checker = _Checker()
    checker.visit(tree)
    params, param_errors = _parse_params(tree)
    errors = checker.errors + param_errors
    if not checker.has_result:
        errors.append("missing assignment to `result` (a build123d Part/Solid/Compound)")
    return CheckResult(ok=not errors, errors=errors, params=params)


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
