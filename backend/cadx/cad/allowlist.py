"""Static pre-checks for LLM-generated code (AST-level, runs before execution).

Allowed import roots: build123d, bd_warehouse, cadlib, math, numpy, typing,
dataclasses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import ClassVar

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
    # signature-gate findings (cadx.cad.sigcheck) — populated only when
    # check_code(sigcheck=True); sig_errors count towards `ok`
    sig_errors: list[str] = field(default_factory=list)
    sig_warnings: list[str] = field(default_factory=list)


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

# context managers that add statements to a builder — a bare
# `Pos(...) * Box(...)` inside them adds the primitive UNPOSITIONED and
# discards the moved copy (observed: L-bracket leg landed centred instead of
# at the -X end; the repair loop then spent 24 turns chasing the edge that
# "should" have been there)
_BUILDER_CTX = {"BuildPart", "BuildSketch", "BuildLine"}
# location-ish transforms that produce a new object when multiplied
_LOC_CTORS = {"Pos", "Rot", "Location", "Locations", "Plane"}
# 3D solid constructors — the right operand of `Pos * X` that gets discarded
_3D_CTORS = {
    "Box",
    "Cylinder",
    "Cone",
    "Sphere",
    "Torus",
    "Wedge",
    "Prism",
    "Hole",
    "CounterBoreHole",
    "CounterSinkHole",
    "CounterSunkHole",
    "extrude",
    "revolve",
    "loft",
    "sweep",
}
# ShapeList-returning selections — `.filter(...)` exists on plain lists, not
# on ShapeList (observed: `p.edges().filter_by(...).filter(lambda...)`)
_SHAPELIST_METHODS = {
    "edges",
    "faces",
    "vertices",
    "solids",
    "wires",
    "compounds",
    "filter_by",
    "sort_by",
    "group_by",
}

# Method names models borrow from CadQuery/OpenSCAD that don't exist in
# build123d — observed: `body.removeSplitter()` (AttributeError, repair loop
# repeated it 3x). Warn so codegen feedback names the real API.
_HALLUCINATED_METHODS = {
    "removeSplitter": ".clean() merges coplanar faces/edges left over by booleans",
    "unify": ".clean() merges coplanar faces/edges left over by booleans",
    "makeSolid": "Solid(...)/Solid.make_solid(...) turns a shell into a solid",
}


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.params_node: ast.Assign | None = None
        self.has_result = False
        # simple ctor tracking: `leg = Cylinder(...)` -> {"leg": "Cylinder"}
        self._ctors: dict[str, str] = {}
        # `name = Box(L, W, T)` / `name = Pos(...) * Box(L, W, T)` -> resolved dims
        self._ctor_dims: dict[str, list[float]] = {}
        # `L = PARAMS["foot_length"]` -> numeric value (filled post-PARAMS parse)
        self._param_aliases: dict[str, float] = {}
        self._param_values: dict[str, object] = {}
        # vars whose last assignment was a copy-transform — a second
        # transform on top double-applies the placement (observed: wedge
        # moved() then translate()d again → ribs floated outside the part)
        self._transformed: set[str] = set()
        # vars assigned from a builder-additive call inside BuildPart —
        # extrude()/Box()/add() already fuse the solid; passing the var to
        # add() or subtracting a moved copy leaves the un-positioned
        # original in the body (observed: ghost pillar at the part centre)
        self._builder_fused: set[str] = set()
        # `with BuildPart() as p` -> "p"; >0 while inside that block
        self._builder_vars: set[str] = set()
        self._builder_depth = 0
        # >0 inside for/while bodies — self-referential boolean accumulation
        # (`arch = arch + seg` per segment) runs one OCC fuse per iteration and
        # re-sews the grown solid every time (observed: 40-cylinder arch →
        # >480 s pipeline hang)
        self._loop_depth = 0

    # ---- constant evaluation over literals, PARAMS subs and simple arith ----
    def _const(self, node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if isinstance(node.value, bool):
                return None
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            v = self._const(node.operand)
            return -v if (v is not None and isinstance(node.op, ast.USub)) else v
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, ast.Add | ast.Sub | ast.Mult | ast.Div
        ):
            a, b = self._const(node.left), self._const(node.right)
            if a is None or b is None:
                return None
            try:
                return {"Add": a + b, "Sub": a - b, "Mult": a * b, "Div": a / b}[
                    type(node.op).__name__
                ]
            except ZeroDivisionError:
                return None
        if isinstance(node, ast.Subscript):
            # PARAMS["x"]
            src = node.value
            key = node.slice
            if (
                isinstance(src, ast.Name)
                and src.id == "PARAMS"
                and isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            ):
                v = self._param_values.get(key.value)
                if isinstance(v, int | float) and not isinstance(v, bool):
                    return float(v)
            return None
        if isinstance(node, ast.Name):
            return self._param_aliases.get(node.id)
        return None

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

    def _has_loc_ctor(self, node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in _LOC_CTORS
            ):
                return True
        return False

    def _has_3d_ctor(self, node: ast.AST) -> str | None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in _3D_CTORS:
                    return sub.func.id
            elif isinstance(sub, ast.Name) and self._ctors.get(sub.id) in _3D_CTORS:
                return self._ctors[sub.id]
        return None

    def _receiver_is_shapelist(self, node: ast.AST) -> str | None:
        """`x.edges().filter(...)`: is `x.edges()` built on a ShapeList chain?"""
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in _SHAPELIST_METHODS
            ):
                return sub.func.attr
        return None

    def _sketch_plane(self, ctx: ast.AST) -> str | None:
        """'XZ'/'YZ' when ctx is ``BuildSketch(<...Plane.XZ/Plane.YZ...>)``."""
        if not (
            isinstance(ctx, ast.Call)
            and isinstance(ctx.func, ast.Name)
            and ctx.func.id == "BuildSketch"
        ):
            return None
        for sub in ast.walk(ctx):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr in ("XZ", "YZ")
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "Plane"
            ):
                return sub.attr
        return None

    def _check_onesided_extrude(self, node: ast.With) -> None:
        """extrude() after ``with BuildSketch(Plane.XZ/YZ)`` in the same
        BuildPart block is one-sided (XZ extrudes -Y, YZ extrudes -X). A body
        meant centred on that axis needs `both=True` — observed: the bracket
        gusset extruded 10 mm one-sided instead of centred in Y."""
        side: str | None = None
        for stmt in node.body:
            if isinstance(stmt, ast.With):
                sketched = False
                for item in stmt.items:
                    p = self._sketch_plane(item.context_expr)
                    if p is not None:
                        side = p
                        sketched = True
                    elif (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "BuildSketch"
                    ):
                        sketched = True
                        side = None  # sketch on another plane — no longer applies
                if sketched:
                    continue  # extrudes live in sibling statements, not in the sketch
            if side is None:
                continue
            axis = "Y" if side == "XZ" else "X"
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "extrude"
                    and all(k.arg != "both" for k in sub.keywords)
                ):
                    self._warn(
                        sub,
                        f"extrude after BuildSketch(Plane.{side}) is one-sided "
                        f"(-{axis}); a body centred in {axis} needs "
                        "extrude(amount=w/2, both=True)",
                    )

    def visit_With(self, node: ast.With) -> None:
        entered: list[str] = []
        in_part = False
        for item in node.items:
            ctx = item.context_expr
            if (
                isinstance(ctx, ast.Call)
                and isinstance(ctx.func, ast.Name)
                and ctx.func.id in _BUILDER_CTX
            ):
                self._builder_depth += 1
                if ctx.func.id == "BuildPart":
                    in_part = True
                if isinstance(item.optional_vars, ast.Name):
                    self._builder_vars.add(item.optional_vars.id)
                    entered.append(item.optional_vars.id)
        if in_part:
            self._check_onesided_extrude(node)
        self.generic_visit(node)
        self._builder_depth -= 1
        for v in entered:
            self._builder_vars.discard(v)

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def _loop_fuse_warn(self, node: ast.AST, expr: str) -> None:
        self._warn(
            node,
            f"`{expr}` accumulates booleans inside a loop — "
            "each iteration runs a full OCC fuse on the grown solid "
            "(~seconds→minutes for >10 segments; observed >480 s). "
            "Collect the pieces in a list and fuse once after the loop "
            "(`parts[0].fuse(*parts[1:])`), or model swept profiles with "
            "`sweep()`/`loft()` instead of stacked primitives.",
        )

    def _check_loop_fuse(self, node: ast.AST, name: str, value: ast.AST) -> None:
        """`x = x + seg` / `x = a if c else x + seg` inside a loop —
        each iteration performs a full OCC boolean on the grown solid."""
        if self._loop_depth == 0:
            return
        for sub in ast.walk(value):
            if (
                isinstance(sub, ast.BinOp)
                and isinstance(sub.op, ast.Add)
                and (
                    (isinstance(sub.left, ast.Name) and sub.left.id == name)
                    or (isinstance(sub.right, ast.Name) and sub.right.id == name)
                )
            ):
                self._loop_fuse_warn(node, f"{name} = {name} + ...")
                return

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
        # `Pos(...) * Box(...)` as a statement: the moved copy is discarded.
        # Inside a builder it is worse — the primitive registers at the
        # CURRENT location the moment it is constructed, so the part silently
        # keeps an unpositioned body (observed: L-bracket leg at centre).
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult):
            prim = self._has_3d_ctor(v)
            if prim and self._has_loc_ctor(v):
                msg = (
                    f"`Pos/Rot * {prim}(...)` as a bare statement returns a moved "
                    "copy that is DISCARDED. "
                    + (
                        "Inside the builder the primitive is already added "
                        "unpositioned — use `with Locations(Pos(...)): "
                        f"{prim}(...)` or `add(Pos(...) * {prim}(...))`."
                        if self._builder_depth > 0
                        else "Assign the result (`part = Pos(...) * X(...)`) or "
                        "use it inside a boolean/add call."
                    )
                )
                if self._builder_depth > 0:
                    self._err(node, msg)
                else:
                    self._warn(node, msg)
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
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "add"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self._builder_fused
        ):
            self._warn(
                node,
                f"`{node.args[0].id}` was already fused into the BuildPart by "
                "the call that created it — add()ing it again leaves an "
                "un-positioned copy in the body. Build the cutter OUTSIDE "
                "the part, or subtract it in the original call with "
                "mode=Mode.SUBTRACT.",
            )
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
        if isinstance(node.func, ast.Attribute) and node.func.attr in _HALLUCINATED_METHODS:
            self._warn(
                node,
                f"`.{node.func.attr}()` does not exist in build123d (CadQuery/"
                f"OpenSCAD idiom) — {_HALLUCINATED_METHODS[node.func.attr]}; "
                "or remove the call.",
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "filter":
            src = self._receiver_is_shapelist(node.func.value)
            if src:
                self._err(
                    node,
                    f"`.{src}()` returns a ShapeList — it has no `.filter()` "
                    "(that's the Python list API). Use `.filter_by(lambda x: ...)` "
                    "or a list comprehension `[e for e in ... if ...]`.",
                )
        self._check_rotate_self_axis(node)
        self._check_2d_boolean_method(node)
        self._check_ctor_kwargs(node)
        self._check_oversized_fillet(node)
        self._check_builder_inplace_transform(node)
        self.generic_visit(node)

    def _check_builder_inplace_transform(self, node: ast.Call) -> None:
        """`Box(...).locate(Pos(...))` inside BuildPart silently keeps the
        builder's copy at the origin: `locate`/`move` are in-place on the
        returned object, but the builder fused its own copy at registration
        time. Observed: leg located to z=4..54 while the fused part kept it
        at z=0..50 — dimension gate fails with no exec error."""
        if not (
            self._builder_depth > 0
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("locate", "move")
        ):
            return
        recv = node.func.value
        # warn only when the receiver belongs to the builder: a fresh
        # primitive call (`Box(...)`) or a var that was assigned a
        # builder-registered result — not an outside object being staged
        tracked = isinstance(recv, ast.Call) or (
            isinstance(recv, ast.Name) and recv.id in self._builder_fused
        )
        if not tracked:
            return
        self._warn(
            node,
            f"`.{node.func.attr}()` inside `with BuildPart()` moves only the "
            "returned object — the builder already registered a copy at the "
            "original location, so the fused part keeps the un-moved shape. "
            "Use `Pos(...) * Box(...)` / `Locations(...)` instead, or assign "
            "the moved copy back with `part = obj.located(...)`.",
        )

    def _check_oversized_fillet(self, node: ast.Call) -> None:
        """fillet(slot.edges()...) on a known Box with r >= min(dims)/2 is
        geometrically impossible — the tell-tale hand-rolled oblong slot
        (observed: 'Failed creating a fillet with radius 3.25' on a 6.5-wide
        box). Warn + point at cadlib.oblong_slot."""
        if not (
            isinstance(node.func, ast.Name) and node.func.id in ("fillet", "chamfer") and node.args
        ):
            return
        r = self._const(node.args[1]) if len(node.args) >= 2 else None
        if r is None:
            r = next(
                (
                    v
                    for k in node.keywords
                    if k.arg in ("radius", "length")
                    for v in [self._const(k.value)]
                ),
                None,
            )
        if r is None:
            return
        sel = node.args[0]
        src: str | None = None
        for sub in ast.walk(sel):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "edges"
                and isinstance(sub.value, ast.Name)
                and sub.value.id in self._ctor_dims
            ):
                src = sub.value.id
                break
        if src is None:
            return
        dims = [d for d in self._ctor_dims[src] if d > 0]
        if dims and r >= min(dims) / 2:
            self._warn(
                node,
                f"`{node.func.id}(radius={r:g})` on `{src}` (smallest side "
                f"{min(dims):g} mm) exceeds the half-width — this raises "
                "'Failed creating a fillet'/'no suitable edges'. For rounded "
                "slot ends use cadlib.oblong_slot() or build the slot as a "
                "SlotOverall sketch instead of filleting a box.",
            )

    # kwargs a constructor call is allowed to use — anything else is certain
    # to fail with "Unexpected keyword arguments" at exec time (observed:
    # Rot(a=90, about=Axis.Y))
    _CTOR_KWARGS: ClassVar[dict[str, frozenset[str]]] = {
        "Rot": frozenset({"axis", "angle", "ordering", "X", "Y", "Z"}),
        "Rotation": frozenset({"axis", "angle", "ordering", "X", "Y", "Z"}),
        "Pos": frozenset({"x", "y", "z", "X", "Y", "Z"}),
        "Cylinder": frozenset(
            {"radius", "height", "arc_size", "align", "mode", "plane", "rotation"}
        ),
        "Box": frozenset({"length", "width", "height", "align", "mode", "plane", "rotation"}),
    }

    def _check_ctor_kwargs(self, node: ast.Call) -> None:
        if not (isinstance(node.func, ast.Name) and node.func.id in self._CTOR_KWARGS):
            return
        bad = [
            k.arg for k in node.keywords if k.arg and k.arg not in self._CTOR_KWARGS[node.func.id]
        ]
        if bad:
            self._warn(
                node,
                f"{node.func.id}() has no keyword(s) {bad} — this raises TypeError "
                "'Unexpected keyword arguments' at runtime; use positional args "
                "or the documented signature (e.g. Rot(axis=Axis.Y, angle=90))",
            )

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
        # `e.center.X`: .center is a METHOD, not a property — the result is a
        # bound method and raises "function object has no attribute 'X'"
        if (
            node.attr in ("X", "Y", "Z")
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "center"
        ):
            self._err(node, "`.center.X` — `.center()` is a method: use `.center().X`")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.op, ast.Add)
            and self._loop_depth > 0
        ):
            self._loop_fuse_warn(node, f"{node.target.id} += ...")
        # `p.part += x` inside `with BuildPart() as p` bypasses the builder —
        # works, but skips BuildPart bookkeeping; the idiomatic call is add()
        if (
            isinstance(node.target, ast.Attribute)
            and node.target.attr == "part"
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id in self._builder_vars
        ):
            mode = "Mode.SUBTRACT" if isinstance(node.op, ast.Sub) else None
            self._warn(
                node,
                f"`{node.target.value.id}.part "
                f"{'-=' if isinstance(node.op, ast.Sub) else '+='} ...` mutates "
                f"the builder result directly — prefer `add(x"
                f"{', mode=' + mode if mode else ''})` inside the `with` block "
                "or run the boolean in algebra mode after `p.part`.",
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: C901 — per-name bookkeeping
        ctor = (
            node.value.func.id
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            else None
        )
        for t in node.targets:
            if isinstance(t, ast.Name) and self._loop_depth > 0:
                self._check_loop_fuse(node, t.id, node.value)
        for t in node.targets:
            if isinstance(t, ast.Name):
                v = node.value
                is_transform = (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and v.func.attr in _COPY_TRANSFORMS
                )
                if is_transform:
                    if (
                        isinstance(v.func.value, ast.Name)
                        and v.func.value.id == t.id
                        and t.id in self._transformed
                    ):
                        self._warn(
                            node,
                            f"`{t.id}` is transformed twice — translate/rotate/moved "
                            "each return an already-positioned copy, so the second "
                            "transform applies ON TOP of the first (double offset). "
                            "Compose the placement once with Pos/Rot instead.",
                        )
                    self._transformed.add(t.id)
                else:
                    self._transformed.discard(t.id)
                if self._builder_depth > 0:
                    if (
                        isinstance(v, ast.Call)
                        and isinstance(v.func, ast.Name)
                        and (v.func.id in _3D_CTORS or v.func.id == "add")
                    ):
                        self._builder_fused.add(t.id)
                    elif not is_transform:
                        self._builder_fused.discard(t.id)
                if t.id == "PARAMS" and isinstance(node.value, ast.Dict):
                    self.params_node = node
                if t.id == "result":
                    self.has_result = True
                if ctor:
                    self._ctors[t.id] = ctor
                else:
                    # `rib = Pos(...) * Cylinder(...)` — track the wrapped
                    # primitive so `Pos * rib` can be flagged too
                    for sub in ast.walk(node.value):
                        if (
                            isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Name)
                            and sub.func.id in _3D_CTORS
                        ):
                            self._ctors[t.id] = sub.func.id
                            break
                # `L = PARAMS["x"]` / `L = a*b` — track resolvable constants
                c = self._const(node.value)
                if c is not None:
                    self._param_aliases[t.id] = c
                # `s = Box(L,W,T)` / `s = Pos(...)*Box(L,W,T)` — remember dims
                # for the oversized-fillet heuristic
                box = self._find_call(node.value, "Box")
                if box is not None:
                    dims = self._box_dims(box)
                    if dims:
                        self._ctor_dims[t.id] = dims
                        self._ctors[t.id] = "Box"
        self.generic_visit(node)

    def _find_call(self, node: ast.AST, name: str) -> ast.Call | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return node
        if isinstance(node, ast.BinOp):
            return self._find_call(node.left, name) or self._find_call(node.right, name)
        return None

    def _box_dims(self, node: ast.Call) -> list[float] | None:
        vals = [self._const(a) for a in node.args[:3]]
        if all(v is not None for v in vals) and len(vals) == 3:
            return [float(v) for v in vals]  # type: ignore[misc]
        kw = {k.arg: self._const(k.value) for k in node.keywords}
        dims = [kw.get("length"), kw.get("width"), kw.get("height")]
        if all(d is not None for d in dims):
            return [float(d) for d in dims]  # type: ignore[misc]
        return None


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


def check_code(code: str, *, sigcheck: bool = False) -> CheckResult:
    """Static check of user code. Returns CheckResult(ok, errors, params).

    ``sigcheck`` additionally runs the introspected API-signature gate
    (``cadx.cad.sigcheck``): unknown attributes / property-called-as-method
    on inferred types land in ``sig_errors``, arity mismatches in
    ``sig_warnings``. Off by default so the executor-side contract check
    keeps its existing behaviour — the agent worker opts in via
    ``runtime_settings.agent_sigcheck``."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CheckResult(ok=False, errors=[f"syntax error line {e.lineno}: {e.msg}"])
    except Exception as e:
        # e.g. UnicodeEncodeError from lone surrogates in model output —
        # must surface as a repairable contract failure, not a crash
        return CheckResult(ok=False, errors=[f"code is not parseable: {type(e).__name__}: {e}"])

    checker = _Checker()
    params, param_errors = _parse_params(tree)
    # PARAMS values must be known before the walk — constant resolution
    # (ctor dims for the fillet heuristic, param aliases) depends on them
    checker._param_values = params
    checker.visit(tree)
    errors = checker.errors + param_errors
    if not checker.has_result:
        errors.append("missing assignment to `result` (a build123d Part/Solid/Compound)")
    if "load_mesh_solid" in code and not mesh_solid_flows_to_result(code):
        checker.warnings.append(
            "load_mesh_solid() result never reaches `result` — in hybrid mode the "
            "organic mesh body must be part of the final geometry "
            "(mesh_fuse/mesh_cut/booleans), not a parked variable. A parametric "
            "replacement is rejected by the hybrid guard."
        )
    sig_errors: list[str] = []
    sig_warnings: list[str] = []
    if sigcheck:
        try:
            from cadx.cad.sigcheck import check_tree

            rep = check_tree(tree)
            sig_errors, sig_warnings = rep.errors, rep.warnings
        except Exception:
            # the signature gate must never break the contract check —
            # sigcheck.check_tree already logs + swallows internal failures
            sig_errors, sig_warnings = [], []
    return CheckResult(
        ok=not errors and not sig_errors,
        errors=errors,
        params=params,
        warnings=checker.warnings,
        sig_errors=sig_errors,
        sig_warnings=sig_warnings,
    )


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


def mesh_solid_flows_to_result(code: str) -> bool:
    """Hybrid-contract check: does a `load_mesh_solid()` result actually reach
    the geometry, or is it parked in a dead variable?

    Observed evasion: `mesh = load_mesh_solid()` inside try/except with the
    name never referenced again — the substring check passes while `result`
    is a fully parametric replacement. Returns True when:
      - the call is used inline in an expression (not a bare-name assign), or
      - every name it is assigned to is loaded somewhere else in the file.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return "load_mesh_solid" in code

    def _has_call(node: ast.AST) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "load_mesh_solid"
            for sub in ast.walk(node)
        )

    mesh_vars: set[str] = set()
    assign_lines: set[int] = set()
    inline_use = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # bare parking: `mesh = load_mesh_solid()` — the RHS is exactly the
            # call; `x = load_mesh_solid() + Box` uses it inline instead
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "load_mesh_solid"
            ):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        mesh_vars.add(t.id)
                        assign_lines.add(node.lineno)
            elif _has_call(node.value):
                inline_use = True
        elif (
            isinstance(node, ast.Expr | ast.Return | ast.AnnAssign | ast.AugAssign)
            and getattr(node, "value", None) is not None
            and _has_call(node.value)
        ):
            inline_use = True
    if inline_use:
        return True
    if not mesh_vars:
        return False
    loaded = {
        sub.id
        for sub in ast.walk(tree)
        if isinstance(sub, ast.Name)
        and sub.id in mesh_vars
        and isinstance(sub.ctx, ast.Load)
        and getattr(sub, "lineno", -1) not in assign_lines
    }
    return bool(loaded)
