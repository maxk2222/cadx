"""Signature gate — AST-level API check for generated CAD code.

Catches the top open-weights codegen failure class BEFORE sandbox
execution: hallucinated or mis-remembered build123d members — the
``Edge.radius()`` case (``radius`` is a property, not a method), unknown
methods and wrong attribute names. Like ``cadx.agent.quickref`` it
introspects the INSTALLED packages (build123d / cadlib / bd_warehouse /
math / numpy), but instead of digest strings it builds a real member
index: name -> kind (class|function|method|property|attribute|module)
plus call-arity bounds. The index is built lazily on the first check
(~1 s, imports dominate) and cached per-process.

Conservative by design: a receiver whose type cannot be inferred emits
NOTHING — a wrong error would teach the model bad things. Missing members
on inferred types are errors; arity mismatches are warnings only (too many
false-positive risks with *args/**kwargs/overloads).
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import re
import threading
from dataclasses import dataclass, field
from types import ModuleType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from cadx.logging import get_logger

log = get_logger(__name__)

_MemberKind = Literal["class", "function", "method", "property", "attribute", "module"]

# modules indexed eagerly on first check; submodule members resolve lazily
_ROOT_MODULES = ("build123d", "cadlib", "bd_warehouse", "math", "numpy")
#: import roots the checker may resolve/index on demand (mirrors allowlist)
_LAZY_ROOTS = frozenset(
    {"build123d", "bd_warehouse", "cadlib", "math", "numpy", "np", "typing", "dataclasses"}
)

# Shape/ShapeList selector methods -> element type of the ShapeList they return
_PLURAL_SELECTORS = {
    "edges": "Edge",
    "faces": "Face",
    "vertices": "Vertex",
    "solids": "Solid",
    "wires": "Wire",
    "compounds": "Compound",
    "shells": "Shell",
}
_SINGLE_SELECTORS = {
    "edge": "Edge",
    "face": "Face",
    "vertex": "Vertex",
    "solid": "Solid",
    "wire": "Wire",
    "compound": "Compound",
    "shell": "Shell",
}
# ShapeList methods returning a ShapeList of the same element type
_SHAPELIST_PRESERVING = {
    "filter_by",
    "filter_by_position",
    "sort_by",
    "sort_by_distance",
    "reverse",
}
# ShapeList attributes/properties yielding a single element
_SHAPELIST_ELEM_ATTRS = {"first", "last"}
# copy-transforms returning the receiver's own type
_SAME_TYPE_METHODS = {
    "located",
    "moved",
    "copy",
    "translate",
    "rotate",
    "scale",
    "transform_geometry",
}
# Shape methods with unambiguous signatures — an arity violation is an ERROR
# (not a warning): observed LLM output `translate(x, y, z)` / `rotate(origin,
# axis, angle)` costs a full repair session each. Still gated on an inferred
# Shape receiver — the conservative principle is unchanged.
_STRICT_ARITY = {"translate", "rotate", "moved", "located", "scale"}
_STRICT_FIX = {
    "translate": "translate takes ONE vector: .translate((x, y, z))",
    "rotate": (
        "rotate takes (Axis, angle): .rotate(Axis.Y, 30) or .rotate(Axis((x, y, z), (0, 1, 0)), 30)"
    ),
}
# builtins applied to a ShapeList
_LIST_YIELDING = {"sorted", "list", "tuple", "reversed"}  # -> same ShapeList
_ELEM_YIELDING = {"min", "max"}  # -> single element

_SELF_ATTR_RE = re.compile(r"\bself\.(\w+)\s*=")
_ANN_NAME_RE = re.compile(r"^['\"]?(\w+)(?:\[['\"]?(\w+)['\"]?[^\]]*\])?['\"]?$")


@dataclass(frozen=True)
class Sig:
    """Call-arity bounds of a callable in bound form (self/cls removed).

    ``max_args`` is None for unbounded (*args). ``req_kw`` holds required
    keyword-only params; ``params`` every named param for coverage counts."""

    min_args: int
    max_args: int | None
    req_kw: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Member:
    """One indexed member of a module or class."""

    kind: _MemberKind
    sig: Sig | None = None
    ret: str | None = None  # value/return class short name when resolvable
    ret_elem: str | None = None  # element class for `ShapeList[T]` returns
    target: str | None = None  # class kind: registered name; module kind: dotted name
    needs_self: bool = False  # plain instance fn — a call ON the class passes self


@dataclass
class SigReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# signature helpers                                                            #
# --------------------------------------------------------------------------- #


def _call_sig(obj: Any, *, ctor: bool = False) -> inspect.Signature | None:
    """inspect.signature with the ``(*args, **kwargs)`` -> ``__init__`` fallback
    used by classes on metaclasses (same trick as quickref._signature)."""
    targets: tuple[Any, ...] = (obj, getattr(obj, "__init__", None)) if ctor else (obj,)
    for target in targets:
        if target is None:
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if params and all(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        ):
            continue  # uninformative — try __init__
        return sig
    return None


def _bounds(sig: inspect.Signature | None) -> tuple[Sig | None, bool]:
    """(bounds, had_self) — drops a leading self/cls positional."""
    if sig is None:
        return None, False
    params = list(sig.parameters.values())
    had_self = bool(params) and params[0].name in ("self", "cls")
    if had_self:
        params = params[1:]
    min_a = max_a = 0
    unbounded = False
    req_kw: set[str] = set()
    names: set[str] = set()
    for p in params:
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            unbounded = True
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            names.add(p.name)
            if p.default is inspect.Parameter.empty:
                req_kw.add(p.name)
        else:
            names.add(p.name)
            max_a += 1
            if p.default is inspect.Parameter.empty:
                min_a += 1
    return (
        Sig(min_a, None if unbounded else max_a, frozenset(req_kw), frozenset(names)),
        had_self,
    )


def _ann_name(ann: Any) -> tuple[str | None, str | None]:
    """(class short name, element name) from a return/property annotation —
    handles plain types, 'Name' strings, 'Name[T]' generics and ``X | None``."""
    if ann is inspect.Signature.empty or ann is None:
        return None, None
    if isinstance(ann, type):
        return ann.__name__, None
    if isinstance(ann, str):
        m = _ANN_NAME_RE.match(ann.strip())
        if m:
            return m.group(1), m.group(2)
        # "Part | None"-style string unions: take the first plain name
        for part in re.split(r"[|]", ann):
            m = _ANN_NAME_RE.match(part.strip())
            if m and m.group(1) != "None":
                return m.group(1), None
        return None, None
    origin = get_origin(ann)
    if origin in (Union, UnionType):
        for arg in get_args(ann):
            if arg is type(None):
                continue
            return _ann_name(arg)
        return None, None
    if origin is not None:
        base = getattr(origin, "__name__", None)
        args = get_args(ann)
        elem = args[0].__name__ if args and isinstance(args[0], type) else None
        return base, elem
    return None, None


def _ret_names(sig: inspect.Signature | None) -> tuple[str | None, str | None]:
    if sig is None:
        return None, None
    try:
        return _ann_name(sig.return_annotation)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# module resolution (mirrors agent/quickref.py's approach)                     #
# --------------------------------------------------------------------------- #


def _import_cadlib() -> Any:
    """cadlib is the reviewed helper package inside cadx (sandboxed builds
    import it as a bare ``cadlib`` module)."""
    try:
        from cadx.cad import cadlib  # backend package layout
    except ImportError:
        cadlib = importlib.import_module("cadlib")  # sandbox layout
    return cadlib


def _resolve_module(dotted: str) -> ModuleType | None:
    if dotted == "cadlib":
        try:
            return _import_cadlib()
        except Exception:
            return None
    if dotted.startswith("cadlib."):
        # generated code imports the bare ``cadlib`` root — submodule chains
        # resolve inside cadx.cad.cadlib on the host
        try:
            return importlib.import_module(f"cadx.cad.{dotted}")
        except ImportError:
            try:
                return importlib.import_module(dotted)
            except ImportError:
                return None
    if dotted.split(".")[0] == "np":
        dotted = "numpy" + dotted[2:]
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def _self_attrs(cls: type) -> list[str]:
    """``self.X =`` assignments in ``__init__`` — instance attributes that
    never appear in ``dir(cls)`` (build123d's Shape.label/.color/...)."""
    init = cls.__dict__.get("__init__")
    if init is None:
        return []
    try:
        src = inspect.getsource(init)
    except (OSError, TypeError):
        return []
    return _SELF_ATTR_RE.findall(src)


class _Index:
    """Per-process member index over the allowed modules. Module tables are
    built eagerly for the roots; submodules index on first reference."""

    def __init__(self) -> None:
        self.modules: dict[str, ModuleType] = {}
        self.members: dict[str, dict[str, Member]] = {}
        self.bad_modules: set[str] = set()
        self.classes: dict[str, type] = {}  # short name -> class (first wins)
        self.class_members: dict[type, dict[str, Member]] = {}
        for root in _ROOT_MODULES:
            self.module_members(root)

    # -- module members ------------------------------------------------------ #

    def module_members(self, dotted: str) -> dict[str, Member] | None:
        if dotted in self.members:
            return self.members[dotted]
        if dotted in self.bad_modules:
            return None
        if dotted.split(".")[0] not in _LAZY_ROOTS:
            self.bad_modules.add(dotted)
            return None
        mod = _resolve_module(dotted)
        if mod is None:
            self.bad_modules.add(dotted)
            return None
        real = mod.__name__
        self.modules[real] = mod
        self.modules[dotted] = mod
        table: dict[str, Member] = {}
        for name in dir(mod):
            if name.startswith("_"):
                continue
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            table[name] = self._mod_member(name, obj)
        self.members[dotted] = table
        self.members.setdefault(real, table)
        return table

    def module_member(self, dotted: str, name: str) -> Member | None:
        table = self.module_members(dotted)
        return table.get(name) if table else None

    def _mod_member(self, name: str, obj: Any) -> Member:
        if inspect.ismodule(obj):
            return Member("module", target=obj.__name__)
        if inspect.isclass(obj):
            self._register_class(obj)
            sig, _ = _bounds(_call_sig(obj, ctor=True))
            return Member("class", sig=sig, target=obj.__name__)
        if callable(obj):
            s = _call_sig(obj)
            sig, _ = _bounds(s)
            ret, ret_elem = _ret_names(s)
            return Member("function", sig=sig, ret=ret, ret_elem=ret_elem)
        return Member("attribute", ret=type(obj).__name__)

    # -- class members -------------------------------------------------------- #

    def _register_class(self, cls: type) -> None:
        self.classes.setdefault(cls.__name__, cls)
        if cls in self.class_members:
            return
        table: dict[str, Member] = {}
        for name in dir(cls):
            if name.startswith("__") and name.endswith("__"):
                continue
            try:
                static = inspect.getattr_static(cls, name)
            except AttributeError:
                continue
            table[name] = self._class_member(cls, name, static)
        for name in getattr(cls, "__members__", {}):  # enum members
            table.setdefault(name, Member("attribute", ret=cls.__name__))
        for name in _self_attrs(cls):
            table.setdefault(name, Member("attribute"))
        self.class_members[cls] = table

    def _class_member(self, cls: type, name: str, static: Any) -> Member:
        if isinstance(static, property):
            ret = ret_elem = None
            with contextlib.suppress(TypeError, ValueError):
                ret, ret_elem = _ann_name(inspect.signature(static.fget).return_annotation)
            return Member("property", ret=ret, ret_elem=ret_elem)
        if isinstance(static, classmethod | staticmethod):
            s = _call_sig(static.__func__)
            sig, _ = _bounds(s)  # cls/self dropped; class callers stay bound
            ret, ret_elem = _ret_names(s)
            return Member("method", sig=sig, ret=ret, ret_elem=ret_elem)
        if inspect.isclass(static):
            self._register_class(static)
            sig, _ = _bounds(_call_sig(static, ctor=True))
            return Member("class", sig=sig, target=static.__name__)
        if callable(static):
            s = _call_sig(getattr(cls, name, static))
            sig, needs_self = _bounds(s)
            ret, ret_elem = _ret_names(s)
            return Member("method", sig=sig, ret=ret, ret_elem=ret_elem, needs_self=needs_self)
        return Member("attribute", ret=type(static).__name__)

    def class_table(self, name: str) -> tuple[type, dict[str, Member]] | None:
        cls = self.classes.get(name)
        if cls is None:
            return None
        self._register_class(cls)
        return cls, self.class_members[cls]


_INDEX: _Index | None = None
_INDEX_LOCK = threading.Lock()


def _index() -> _Index:
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = _Index()
    return _INDEX


# --------------------------------------------------------------------------- #
# AST pass                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Ref:
    """What an expression evaluates to, as far as the checker can tell."""

    kind: str  # "module" | "class" | "instance" | "shapelist"
    name: str  # dotted module name, class short name, or "@local:<cls>"
    elem: str | None = None  # element class for shapelist


@dataclass
class _Owner:
    """A resolved attribute receiver: its member table + display label."""

    members: dict[str, Member]
    name: str  # raw: dotted module or class short name
    label: str  # display: "Box" / "module 'build123d'"
    recv: str  # "module" | "classobj" | "instance"
    obj: Any


class _Visitor(ast.NodeVisitor):
    def __init__(self, idx: _Index) -> None:
        self.idx = idx
        self.errors: list[str] = []
        self.warnings: list[str] = []
        # local name -> ("module", dotted) | ("object", module, member)
        self.names: dict[str, tuple[str, ...]] = {}
        self.star: list[str] = []  # star-imported modules
        self.types: dict[str, str] = {}  # var -> class short name (instances)
        self.elem: dict[str, str] = {}  # ShapeList var -> element class
        self.class_vars: dict[str, str] = {}  # var -> class short name (class obj)
        self.dyn_attrs: dict[str, set[str]] = {}  # var -> attrs assigned in code
        self.local_defs: set[str] = set()  # functions/classes defined in the code
        self._call_funcs: set[int] = set()  # Attribute nodes used as callables

    # -- diagnostics --------------------------------------------------------- #

    def _err(self, node: ast.AST, msg: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def _warn(self, node: ast.AST, msg: str) -> None:
        self.warnings.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # -- name resolution ------------------------------------------------------ #

    def _resolve_name(self, name: str) -> tuple[str, ...] | None:
        ref = self.names.get(name)
        if ref is not None:
            return ref
        for mod in self.star:
            table = self.idx.module_members(mod) or {}
            if name in table:
                ref = ("object", mod, name)
                self.names[name] = ref
                return ref
        return None

    def _member_value_ref(self, m: Member, name: str) -> _Ref | None:
        """What READING a member yields (properties/attrs -> their value type)."""
        if m.kind == "class":
            return _Ref("class", m.target or name)
        if m.kind == "module":
            return _Ref("module", m.target or "")
        if m.kind in ("property", "attribute") and m.ret:
            if m.ret == "ShapeList":
                return _Ref("shapelist", "ShapeList", m.ret_elem)
            return _Ref("instance", m.ret)
        return None  # methods/functions as values are callables — untyped

    def _call_result(self, m: Member, name: str) -> _Ref | None:
        """What CALLING a member yields."""
        if m.kind == "class":
            return _Ref("instance", m.target or name)
        if m.ret == "ShapeList":
            return _Ref("shapelist", "ShapeList", m.ret_elem)
        if m.ret:
            return _Ref("instance", m.ret)
        return None

    @staticmethod
    def _is_local(ref: _Ref | None) -> bool:
        return ref is not None and ref.name.startswith("@")

    # -- expression -> ref (type inference) ------------------------------------ #

    def _ref_of(self, node: ast.AST) -> _Ref | None:  # noqa: C901
        if isinstance(node, ast.Name):
            n = node.id
            if n in self.types:
                return _Ref("instance", self.types[n], self.elem.get(n))
            if n in self.class_vars:
                return _Ref("class", self.class_vars[n])
            ref = self._resolve_name(n)
            if ref is None:
                return None
            if ref[0] == "module":
                return _Ref("module", ref[1])
            m = self.idx.module_member(ref[1], ref[2])
            return self._member_value_ref(m, ref[2]) if m else None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.attr in self.dyn_attrs.get(
                node.value.id, ()
            ):
                return None
            recv = self._ref_of(node.value)
            if recv is None or self._is_local(recv):
                return None
            if recv.kind == "shapelist" and node.attr in _SHAPELIST_ELEM_ATTRS:
                return _Ref("instance", recv.elem) if recv.elem else None
            owner = self._owner_from_ref(recv)
            if owner is None:
                return None
            m = owner.members.get(node.attr)
            if m is None:
                # submodule fallback: `bd_warehouse.fastener` before its import
                if owner.recv == "module":
                    full = f"{recv.name}.{node.attr}"
                    if self.idx.module_members(full) is not None:
                        return _Ref("module", full)
                return None
            return self._member_value_ref(m, node.attr)
        if isinstance(node, ast.Call):
            return self._call_ref(node)
        if isinstance(node, ast.Subscript):
            r = self._ref_of(node.value)
            if r is None or self._is_local(r):
                return None
            if r.kind == "shapelist":
                if isinstance(node.slice, ast.Slice):
                    return r
                return _Ref("instance", r.elem) if r.elem else None
            return None
        if isinstance(node, ast.BinOp):
            # `Pos(...) * Box(...)` -> the right operand's type;
            # `part - cutter` -> the left operand's type
            order = (
                (node.right, node.left)
                if isinstance(node.op, ast.Mult)
                else (node.left, node.right)
            )
            for side in order:
                r = self._ref_of(side)
                if r is not None and not self._is_local(r):
                    return r
            return None
        return None

    def _call_name_ref(self, node: ast.Call, n: str) -> _Ref | None:
        """Return type of a bare ``name(...)`` call."""
        if n in self.class_vars:
            return _Ref("instance", self.class_vars[n])
        if n in self.types:
            return None  # calling an instance variable — result uninferable
        if n in self.local_defs:
            # locally defined function/class — mark the result as local so
            # the selector tables never fire on it (a local class may define
            # its own `.edges()` with different semantics)
            return _Ref("instance", f"@local:{n}")
        ref = self._resolve_name(n)
        if ref is None:
            if node.args:
                r = self._ref_of(node.args[0])
                if r is not None and r.kind == "shapelist":
                    if n in _LIST_YIELDING:
                        return r
                    if n in _ELEM_YIELDING and r.elem:
                        return _Ref("instance", r.elem)
            return None
        if ref[0] != "object":
            return None
        m = self.idx.module_member(ref[1], ref[2])
        return self._call_result(m, ref[2]) if m else None

    def _call_ref(self, node: ast.Call) -> _Ref | None:
        f = node.func
        if isinstance(f, ast.Name):
            return self._call_name_ref(node, f.id)
        if isinstance(f, ast.Attribute):
            recv = self._ref_of(f.value)
            if self._is_local(recv):
                return None
            inferable = recv is None or recv.kind in ("instance", "shapelist")
            if inferable:
                if f.attr in _PLURAL_SELECTORS:
                    return _Ref("shapelist", "ShapeList", _PLURAL_SELECTORS[f.attr])
                if f.attr in _SINGLE_SELECTORS:
                    return _Ref("instance", _SINGLE_SELECTORS[f.attr])
            if recv is None:
                return None
            if recv.kind == "shapelist":
                if f.attr in _SHAPELIST_PRESERVING:
                    return recv
                if f.attr in _SHAPELIST_ELEM_ATTRS:
                    return _Ref("instance", recv.elem) if recv.elem else None
            if f.attr in _SAME_TYPE_METHODS and recv.kind in ("instance", "shapelist"):
                return recv
            owner = self._owner_from_ref(recv)
            if owner is None:
                return None
            m = owner.members.get(f.attr)
            return self._call_result(m, f.attr) if m else None
        return None

    def _owner_from_ref(self, ref: _Ref) -> _Owner | None:
        if ref.kind == "module":
            members = self.idx.module_members(ref.name)
            if members is None:
                return None
            return _Owner(
                members,
                ref.name,
                f"module '{ref.name}'",
                "module",
                self.idx.modules.get(ref.name),
            )
        entry = self.idx.class_table(ref.name)
        if entry is None:
            return None
        cls, members = entry
        recv = "classobj" if ref.kind == "class" else "instance"
        return _Owner(members, ref.name, ref.name, recv, cls)

    def _owner_of(self, expr: ast.AST) -> _Owner | None:
        ref = self._ref_of(expr)
        if ref is None or self._is_local(ref):
            return None
        return self._owner_from_ref(ref)

    def _exists(self, owner: _Owner, attr: str) -> bool:
        """Ground-truth fallback for members outside the static table:
        __getattr__-provided names and not-yet-imported submodules."""
        if owner.recv == "module":
            try:
                if owner.obj is not None and hasattr(owner.obj, attr):
                    return True
            except Exception:
                pass
            return self.idx.module_members(f"{owner.name}.{attr}") is not None
        try:
            return hasattr(owner.obj, attr)
        except Exception:
            return False

    # -- call checks ----------------------------------------------------------- #

    def _is_shape_receiver(self, owner: _Owner) -> bool:
        """Strict arity applies only when the receiver resolves to a Shape
        subclass — the signatures there are fixed and unambiguous."""
        cls = self.idx.classes.get("Shape")
        return (
            cls is not None
            and isinstance(owner.obj, type)
            and owner.recv in ("instance", "classobj")
            and issubclass(owner.obj, cls)
        )

    def _check_arity(
        self,
        node: ast.Call,
        m: Member,
        class_receiver: bool,
        label: str,
        *,
        strict: bool = False,
        fix: str | None = None,
    ) -> None:
        sig = m.sig
        if sig is None:
            return
        if any(isinstance(a, ast.Starred) for a in node.args):
            return
        if any(k.arg is None for k in node.keywords):
            return
        emit = self._err if strict else self._warn
        n_pos = len(node.args)
        min_req, max_pos = sig.min_args, sig.max_args
        if class_receiver and m.needs_self:
            # unbound call on the class object — self occupies a positional slot
            min_req += 1
            max_pos = None if max_pos is None else max_pos + 1
        kw = {k.arg for k in node.keywords if k.arg}
        if max_pos is not None and n_pos > max_pos:
            emit(
                node,
                f"{label}() takes at most {max_pos} positional arg(s), {n_pos} given"
                + (f" — {fix}" if fix else ""),
            )
            return
        missing_kw = sig.req_kw - kw
        given = n_pos + len(kw & sig.params)
        required = min_req + len(sig.req_kw)
        if missing_kw:
            emit(
                node,
                f"{label}() is missing required keyword arg(s) {sorted(missing_kw)}"
                + (f" — {fix}" if fix else ""),
            )
        elif given < required:
            emit(
                node,
                f"{label}() expects at least {required} arg(s), {given} given"
                + (f" — {fix}" if fix else ""),
            )

    def _check_attr_call(self, node: ast.Call, f: ast.Attribute) -> None:
        if isinstance(f.value, ast.Name) and f.attr in self.dyn_attrs.get(f.value.id, ()):
            return
        owner = self._owner_of(f.value)
        if owner is None:
            return
        m = owner.members.get(f.attr)
        if m is None:
            if self._exists(owner, f.attr):
                return
            self._err(node, f"{owner.label} has no attribute '{f.attr}'")
            return
        if m.kind == "property":
            self._err(
                node,
                f"{owner.label}.{f.attr} is a property — drop the parentheses",
            )
            return
        if m.kind == "attribute":
            self._err(node, f"{owner.label}.{f.attr} is not callable")
            return
        if m.kind == "module":
            self._err(node, f"{owner.label}.{f.attr} is a module, not callable")
            return
        label = f"{owner.label}.{f.attr}"
        strict = f.attr in _STRICT_ARITY and self._is_shape_receiver(owner)
        # translate(vec, <not a bool literal>) — the LLM meant translate(x, y, z)
        # with a missing third component; a real transform flag is a bool
        if (
            strict
            and f.attr == "translate"
            and len(node.args) == 2
            and not (
                isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, bool)
            )
        ):
            self._err(node, f"{label}() with 2 positional args — {_STRICT_FIX['translate']}")
            return
        # rotate((x, y, z), Axis.Y, a) — the CadQuery origin/axis/angle idiom;
        # arity fits (axis, angle, transform) but a literal tuple is never an Axis
        if (
            strict
            and f.attr == "rotate"
            and node.args
            and isinstance(node.args[0], ast.Tuple | ast.List)
        ):
            self._err(node, f"{label}() — {_STRICT_FIX['rotate']}")
            return
        self._check_arity(
            node,
            m,
            owner.recv == "classobj",
            label,
            strict=strict,
            fix=_STRICT_FIX.get(f.attr) if strict else None,
        )

    def _check_name_call(self, node: ast.Call, name: str) -> None:
        if name in self.local_defs or name in self.types or name in self.class_vars:
            return
        ref = self._resolve_name(name)
        if ref is None or ref[0] != "object":
            return
        m = self.idx.module_member(ref[1], ref[2])
        if m is None or m.kind == "module":
            return
        if m.kind == "attribute":
            self._err(node, f"{name} is a constant of '{ref[1]}', not callable")
            return
        self._check_arity(node, m, False, name)

    # -- visitors -------------------------------------------------------------- #

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute):
            self._call_funcs.add(id(f))
            self._check_attr_call(node, f)
        elif isinstance(f, ast.Name):
            self._check_name_call(node, f.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) in self._call_funcs or isinstance(node.ctx, ast.Store | ast.Del):
            self.generic_visit(node)
            return
        if isinstance(node.value, ast.Name) and node.attr in self.dyn_attrs.get(node.value.id, ()):
            self.generic_visit(node)
            return
        owner = self._owner_of(node.value)
        if owner is not None:
            m = owner.members.get(node.attr)
            if m is None:
                # instance reads stay silent — __init__-assigned attributes
                # (Shape.label, …) are not all visible statically; class and
                # module attributes are, so a miss there is a real error
                if owner.recv in ("module", "classobj") and not self._exists(owner, node.attr):
                    self._err(node, f"{owner.label} has no attribute '{node.attr}'")
            elif m.kind == "method":
                self._warn(
                    node,
                    f"{owner.label}.{node.attr} is a method — did you mean `.{node.attr}()`?",
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.asname:
                self.names[a.asname] = ("module", a.name)
            else:
                self.names[a.name.split(".")[0]] = ("module", a.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for a in node.names:
            if a.name == "*":
                if mod:
                    self.star.append(mod)
                continue
            self.names[a.asname or a.name] = ("object", mod, a.name)
        self.generic_visit(node)

    # -- binding tracking ------------------------------------------------------- #

    def _bind(self, name: str, ref: _Ref | None) -> None:
        self.names.pop(name, None)
        if ref is None:
            self.types.pop(name, None)
            self.elem.pop(name, None)
            self.class_vars.pop(name, None)
        elif ref.kind == "class":
            self.class_vars[name] = ref.name
            self.types.pop(name, None)
            self.elem.pop(name, None)
        elif ref.kind == "module":
            self.names[name] = ("module", ref.name)
            self.types.pop(name, None)
            self.elem.pop(name, None)
            self.class_vars.pop(name, None)
        else:
            self.types[name] = ref.name
            if ref.elem:
                self.elem[name] = ref.elem
            else:
                self.elem.pop(name, None)
            self.class_vars.pop(name, None)

    def _bind_target(self, target: ast.AST, type_name: str | None) -> None:
        if isinstance(target, ast.Name):
            self.names.pop(target.id, None)
            self.class_vars.pop(target.id, None)
            if type_name:
                self.types[target.id] = type_name
            else:
                self.types.pop(target.id, None)
            self.elem.pop(target.id, None)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._bind_target(elt, None)

    def _iter_elem(self, it: ast.AST) -> str | None:
        ref = self._ref_of(it)
        if ref is not None and ref.kind == "shapelist":
            return ref.elem
        return None

    def _bind_loop(self, target: ast.AST, it: ast.AST) -> None:
        # `for e in x.edges()` / `for i, e in enumerate(edges)` / zip(...)
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "enumerate"
            and it.args
            and isinstance(target, ast.Tuple | ast.List)
        ):
            elts = target.elts
            elem = self._iter_elem(it.args[0])
            for t in elts[:-1]:
                self._bind_target(t, None)
            if elts:
                self._bind_target(elts[-1], elem)
            return
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "zip"
            and isinstance(target, ast.Tuple | ast.List)
        ):
            for t, arg in zip(target.elts, it.args, strict=False):
                self._bind_target(t, self._iter_elem(arg))
            return
        self._bind_target(target, self._iter_elem(it))

    def visit_For(self, node: ast.For) -> None:
        self._bind_loop(node.target, node.iter)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._bind_loop(node.target, node.iter)
        self.generic_visit(node)

    def _visit_comp(self, node: ast.AST) -> None:
        for gen in node.generators:  # type: ignore[attr-defined]
            self._bind_loop(gen.target, gen.iter)
        self.generic_visit(node)

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_DictComp = _visit_comp
    visit_GeneratorExp = _visit_comp

    def visit_With(self, node: ast.With) -> None:
        # `with BuildPart() as bp:` — builders return themselves from __enter__
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self._bind(item.optional_vars.id, self._ref_of(item.context_expr))
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                self.dyn_attrs.setdefault(t.value.id, set()).add(t.attr)
        ref = self._ref_of(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self._bind(t.id, ref)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            ref = self._ref_of(node.value) if node.value is not None else None
            if ref is None and isinstance(node.annotation, ast.Name):
                ann = self._resolve_name(node.annotation.id)
                if ann and ann[0] == "object":
                    m = self.idx.module_member(ann[1], ann[2])
                    if m is not None and m.kind == "class":
                        ref = _Ref("instance", m.target or ann[2])
            self._bind(node.target.id, ref)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, self._ref_of(node.value))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.names.pop(t.id, None)
                self.types.pop(t.id, None)
                self.elem.pop(t.id, None)
                self.class_vars.pop(t.id, None)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_defs.add(node.name)
        self.generic_visit(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.local_defs.add(node.name)
        a = node.args
        arg_objs = [*a.posonlyargs, *a.args, *a.kwonlyargs]
        argnames = {x.arg for x in arg_objs}
        for x in (a.vararg, a.kwarg):
            if x is not None:
                argnames.add(x.arg)
        saved = {n: (self.types.get(n), self.elem.get(n), self.class_vars.get(n)) for n in argnames}
        for x in arg_objs:
            self.types.pop(x.arg, None)
            self.elem.pop(x.arg, None)
            self.class_vars.pop(x.arg, None)
            if isinstance(x.annotation, ast.Name):  # `def f(p: Part)` — typed params
                ref = self._resolve_name(x.annotation.id)
                if ref and ref[0] == "object":
                    m = self.idx.module_member(ref[1], ref[2])
                    if m is not None and m.kind == "class":
                        self.types[x.arg] = m.target or ref[2]
        for x in (a.vararg, a.kwarg):
            if x is not None:
                self.types.pop(x.arg, None)
        self.generic_visit(node)
        for n, (t, e, c) in saved.items():
            if t is not None:
                self.types[n] = t
            if e is not None:
                self.elem[n] = e
            if c is not None:
                self.class_vars[n] = c

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #


def check_tree(tree: ast.Module) -> SigReport:
    """Signature findings for a parsed module — never raises."""
    try:
        v = _Visitor(_index())
        v.visit(tree)
        return SigReport(v.errors, v.warnings)
    except Exception as e:
        log.warning("sigcheck.check_failed", error=str(e))
        return SigReport()


def check_code_signatures(code: str) -> SigReport:
    """Signature findings for a code string. Unparseable code yields an empty
    report — the syntax error itself is the contract checker's job."""
    try:
        tree = ast.parse(code)
    except Exception:
        return SigReport()
    return check_tree(tree)
