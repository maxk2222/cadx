"""quickref — runtime-introspected API signature digests for the CAD stack.

The generated-code model keeps hallucinating method names and signatures
that drifted between build123d/bd_warehouse releases, and the hand-maintained
reference prompts go stale silently. This module introspects the INSTALLED
packages and renders compact signature digests that the model can pull on
demand via the ``quickref`` agent tool — or that get pinned into the codegen
user context when ``runtime_settings.agent_quickref_context`` is on.

Host-side only: this runs in the worker process where build123d/bd_warehouse
are importable — never inside the exec sandbox. Package imports are deferred
into the render functions so importing this module stays cheap (a cold
``import build123d`` costs seconds).
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import inspect
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

#: named digest topics served by quickref()
TOPICS: tuple[str, ...] = ("build123d", "cadlib", "bd_warehouse")

#: import roots allowed for the dotted-path fallback (e.g. "build123d.extrude")
_RESOLVABLE_ROOTS = frozenset(TOPICS)

# in-process cache: (topic, (build123d, bd_warehouse, cadlib) versions) -> digest
_CACHE: dict[tuple[str, tuple[str, str, str]], str] = {}

# --------------------------------------------------------------------------- #
# build123d topic — the curated core the model actually writes                #
# --------------------------------------------------------------------------- #

_BUILD123D_CORE: tuple[str, ...] = (
    # primitives
    "Box",
    "Cylinder",
    "Cone",
    "Sphere",
    # builders
    "BuildPart",
    "BuildSketch",
    "BuildLine",
    # sketch → solid operations
    "extrude",
    "revolve",
    "loft",
    "sweep",
    # finishing operations
    "fillet",
    "chamfer",
    "mirror",
    "offset",
    "thicken",
    # hole operations
    "Hole",
    "CounterSinkHole",
    # positioning
    "Pos",
    "Rot",
    "Location",
    "Locations",
    "GridLocations",
    "PolarLocations",
    # geometry references
    "Plane",
    "Axis",
    # enums
    "Mode",
    "Align",
    "Select",
)

# Stable signature hints for classes whose __init__ is (*args, **kwargs) —
# their real contract lives in docstrings/wrappers. These are checked-in data,
# not prompt text; keep them terse.
_SIG_HINTS: dict[str, str] = {
    "Pos": "Pos(X=0, Y=0, Z=0)  # or Pos(vector)",
    "Rot": "Rot(X=0, Y=0, Z=0)  # or Rot(Z=45), axis order Intrinsic.XYZ",
    "Location": "Location((x, y, z))  # or Location((x, y, z), (rx, ry, rz) deg)",
    "Plane": "Plane.XY | .XZ | .YZ | .front | .top … or Plane(origin=…, x_dir=…, z_dir=…)",
    "Axis": "Axis.X | .Y | .Z  or Axis(origin, direction)",
}

# boilerplate params hidden from signatures (they're uniform across the API)
_SKIP_PARAMS = frozenset({"self", "cls"})
_COMPACT_SKIP_PARAMS = frozenset({"self", "cls", "rotation"})
_BDW_SKIP_PARAMS = frozenset({"self", "cls", "rotation", "align", "mode", "hand", "simple"})

# abstract/base classes that carry no useful constructor for the model
_BDW_BASE_CLASSES = frozenset({"Screw", "Nut", "Washer", "Bearing", "Thread", "Flange"})

_BDW_MODULES: tuple[tuple[str, str], ...] = (
    ("bd_warehouse.fastener", "Fasteners + hole ops"),
    ("bd_warehouse.thread", "Threads"),
    ("bd_warehouse.bearing", "Bearings"),
    ("bd_warehouse.flange", "Flanges"),
    ("bd_warehouse.pipe", "Pipes"),
    ("bd_warehouse.sprocket", "Sprockets"),
    ("bd_warehouse.gear", "Gears"),
    ("bd_warehouse.open_builds", "OpenBuilds parts"),
)

_QUALIFIED_NAME = re.compile(r"\b(?:[a-z_][\w]*\.)+([A-Za-z_]\w*)")
_DOC_HEADER = re.compile(r"^[\w ]+(Object|Operation|Context|Builder|Input|Output):")
_DOC_SECTION = re.compile(r"^(Args|Attributes|Returns|Raises|Examples?|Notes?|Warnings?)\b")
_LITERAL_VALUES = re.compile(r"'([^']+)'")

_MAX_ENTRY_CHARS = 400


# --------------------------------------------------------------------------- #
# formatting helpers                                                          #
# --------------------------------------------------------------------------- #


def _doc_line(obj: Any) -> str:
    """First meaningful docstring line — skips 'Part Object: X' boilerplate
    headers and section leads (Args:/Returns:) to land on the description."""
    doc = inspect.getdoc(obj) or ""
    lines = [ln.strip() for ln in doc.split("\n") if ln.strip()]
    candidates = [ln for ln in lines if not _DOC_SECTION.match(ln)]
    if not candidates:
        return ""
    first = candidates[0]
    if (_DOC_HEADER.match(first) or first == getattr(obj, "__name__", "")) and len(candidates) > 1:
        first = candidates[1]
    return first[:160] + ("…" if len(first) > 160 else "")


def _fmt_ann(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return ""
    text = ann if isinstance(ann, str) else getattr(ann, "__name__", repr(ann))
    text = text.strip("'\"")
    text = _QUALIFIED_NAME.sub(r"\1", text)  # build123d.topology.one_d.Edge → Edge
    text = re.sub(r"<class '([^']+)'>", r"\1", text)
    return text[:80] + ("…" if len(text) > 80 else "")


def _fmt_default(value: Any) -> str:
    if isinstance(value, enum.Enum):
        text = f"{type(value).__name__}.{value.name}"
    elif isinstance(value, tuple):
        items = [
            f"{type(v).__name__}.{v.name}" if isinstance(v, enum.Enum) else repr(v) for v in value
        ]
        if len(set(items)) == 1 and len(items) > 1 and isinstance(value[0], enum.Enum):
            text = f"{items[0]}\u00d7{len(items)}"
        else:
            text = "(" + ", ".join(items) + ")"
    elif isinstance(value, str):
        text = repr(value)
    else:
        text = repr(value)
        text = re.sub(r"<(\w+)\.(\w+)[^>]*>", r"\1.\2", text)  # <Mode.ADD: 'add'> → Mode.ADD
    return text[:40] + ("…" if len(text) > 40 else "")


def _signature(
    obj: Any,
    *,
    compact: bool = False,
    skip: frozenset[str] = _SKIP_PARAMS,
    show_defaults: bool = True,
) -> str:
    """Compact ``(params)`` for a callable/class.

    Classes built on metaclasses report ``(*args, **kwargs)`` — in that case
    fall back to ``__init__``. ``compact`` drops type annotations and the
    uniform ``rotation`` boilerplate (context blocks where every char
    counts); ``show_defaults=False`` additionally drops ``=default`` tails."""
    sig: inspect.Signature | None = None
    skip = skip | _COMPACT_SKIP_PARAMS if compact else skip
    for target in (obj, getattr(obj, "__init__", None)):
        if target is None:
            continue
        try:
            cand = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        params = [p for p in cand.parameters.values() if p.name not in skip]
        if params and all(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        ):
            continue  # uninformative *args/**kwargs — try __init__ next
        sig = cand
        break
    if sig is None:
        return "(…)"
    parts: list[str] = []
    for p in sig.parameters.values():
        if p.name in skip:
            continue
        prefix = "*" if p.kind is inspect.Parameter.VAR_POSITIONAL else ""
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"
        text = prefix + p.name
        if not compact:
            ann = _fmt_ann(p.annotation)
            if ann:
                text += f": {ann}"
        if show_defaults and p.default is not inspect.Parameter.empty:
            text += f"={_fmt_default(p.default)}"
        parts.append(text)
    out = "(" + ", ".join(parts) + ")"
    return out[:_MAX_ENTRY_CHARS] + "…)" if len(out) > _MAX_ENTRY_CHARS else out


def _entry(name: str, obj: Any, *, compact: bool = False, hint: str | None = None) -> str:
    """One digest line: ``Name(sig)  # doc`` or ``Name = MEMBER|…`` for enums."""
    if _is_enum(obj):
        members = " | ".join(m.name for m in obj)
        line = f"{name} = {members}"
    elif hint is not None:
        line = hint
    else:
        line = f"{name}{_signature(obj, compact=compact, show_defaults=not compact)}"
    doc = _doc_line(obj)
    if doc and not compact:
        line += f"  # {doc}"
    return line


def _truncate_lines(text: str, limit: int, note: str) -> str:
    if len(text) <= limit:
        return text
    out = text[:limit]
    out = out[: out.rfind("\n")]  # don't leave a half line
    return out + "\n" + note


# --------------------------------------------------------------------------- #
# topic digests                                                               #
# --------------------------------------------------------------------------- #


def _is_enum(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


def _build123d_digest(compact: bool = False) -> str:
    import build123d  # deferred: cold import costs seconds

    # compact digests put the gotcha-prone surface first — a truncation then
    # only drops trailing primitive classes. Single-line enum entries lead:
    # Mode/Align/Select are unguessable vocabulary and cost ~1 line each,
    # so they survive even when the budget cuts mid-digest.
    names = _BUILD123D_CORE
    if compact:
        enums = [n for n in names if _is_enum(getattr(build123d, n, None))]
        hints = [n for n in names if n in _SIG_HINTS and n not in enums]
        names = tuple(enums) + tuple(hints) + tuple(n for n in names if n not in (*enums, *hints))
    lines = [f"build123d {dist_version('build123d')} — core API digest (runtime introspection)"]
    for name in names:
        obj = getattr(build123d, name, None)
        if obj is None:
            lines.append(f"{name}  # NOT AVAILABLE in this version")
            continue
        lines.append(_entry(name, obj, compact=compact, hint=_SIG_HINTS.get(name)))
    return "\n".join(lines)


def _import_cadlib() -> Any:
    """cadlib is the reviewed helper package inside cadx (sandboxed builds
    import it as a bare ``cadlib`` module)."""
    try:
        from cadx.cad import cadlib  # backend package layout
    except ImportError:
        cadlib = importlib.import_module("cadlib")  # sandbox layout
    return cadlib


# prompt-facing core surface — the context block truncates the tail, so the
# helpers the keyword map and most generated code rely on come first
_CADLIB_FIRST: tuple[str, ...] = (
    "through_hole",
    "hole_grid",
    "safe_fillet",
    "safe_chamfer",
    "finish_edges",
    "edges_at",
    "profile_body",
    "closed_polyline",
    "strut",
    "rib",
    "largest_solid",
)


def _cadlib_digest(compact: bool = False) -> str:
    cadlib = _import_cadlib()
    lines = [
        "cadlib — reviewed helpers (import: `from cadlib import <name>`); "
        "cutouts run inside `with BuildPart() as p:`, shapes return a Part to place"
    ]
    names = list(
        getattr(cadlib, "__all__", None) or sorted(n for n in dir(cadlib) if not n.startswith("_"))
    )
    if compact:
        names = [n for n in _CADLIB_FIRST if n in names] + [
            n for n in names if n not in _CADLIB_FIRST
        ]
    for name in names:
        obj = getattr(cadlib, name, None)
        if obj is None or not callable(obj):
            continue
        lines.append(_entry(name, obj, compact=compact))
    return "\n".join(lines)


def _bdw_sizes(cls: Any, fastener_type: str) -> list[str]:
    try:
        return sorted(cls.sizes(fastener_type))
    except Exception:
        return []


def _bdw_class_entry(
    cls: Any, *, compact: bool = False, size_sink: dict[tuple[str, ...], set[str]] | None = None
) -> str:
    """One digest line per class: ``Name(params)`` with Literal selector
    values inlined. ``size_sink`` collects the ``sizes()`` result per type for
    module-level dedup (many classes share ISO size tables)."""
    sig = _signature(cls, compact=True, skip=_BDW_SKIP_PARAMS)
    line = f"{cls.__name__}{sig}"
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    for pname, p in params.items():
        ann = str(p.annotation)
        if pname in _BDW_SKIP_PARAMS or "Literal[" not in ann:
            continue
        values = _LITERAL_VALUES.findall(ann)
        if not values:
            continue
        line += f"  [{pname}: {'|'.join(values)}]"
        if size_sink is not None:
            for v in values[:8]:
                sizes = _bdw_sizes(cls, v)
                if sizes:
                    size_sink.setdefault(tuple(sizes), set()).add(v)
    doc = _doc_line(cls)
    # bd_warehouse docstrings often just echo the class name ("Hex Nut") —
    # those add zero information, so only keep docs that say something new
    if doc and not compact and doc.lower().replace(" ", "") != cls.__name__.lower():
        line += f"  # {doc[:80]}"
    return line


def _bdw_module_digest(mod: Any, label: str, *, compact: bool = False) -> list[str]:
    """Digest lines for one bd_warehouse module: class signatures + a deduped
    sizes[] table per module (many classes share ISO size lists)."""
    classes = [
        cls
        for name, cls in inspect.getmembers(mod, inspect.isclass)
        if cls.__module__ == mod.__name__ and not name.startswith("_")
    ]
    classes = [c for c in classes if c.__name__ not in _BDW_BASE_CLASSES]
    if not classes:
        return []
    lines = [f"## {label}  ({mod.__name__})"]
    size_sets: dict[tuple[str, ...], set[str]] = {}
    for cls in classes:
        lines.append(
            _bdw_class_entry(cls, compact=compact, size_sink=None if compact else size_sets)
        )
    for size_key, types in size_sets.items():
        shown = ", ".join(size_key[:20]) + (", …" if len(size_key) > 20 else "")
        lines.append(f"  sizes[{', '.join(sorted(types))}]: {shown}")
    return lines


def _bdw_digest(compact: bool = False) -> str:
    try:
        importlib.import_module("bd_warehouse.fastener")
    except ImportError:
        return "bd_warehouse is not installed in this environment."
    lines = [
        f"bd_warehouse {dist_version('bd_warehouse')} — standard parts "
        "(size strings like 'M6-1'; probe valid ones via Class.sizes(type), the "
        "search_catalog/lookup_part tools, or quickref('bd_warehouse.<module>'))"
    ]
    labels = dict(_BDW_MODULES)
    for mod_name, label in _BDW_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        lines.extend(_bdw_module_digest(mod, labels.get(mod_name, label), compact=compact))
    return _truncate_lines(
        "\n".join(lines),
        12_000,
        "… (truncated — use quickref('bd_warehouse.<module>') for per-module digests)",
    )


# --------------------------------------------------------------------------- #
# dotted-path fallback                                                        #
# --------------------------------------------------------------------------- #


def _resolve_root(root: str) -> Any:
    if root == "cadlib":
        return _import_cadlib()
    return importlib.import_module(root)


def _dotted_digest(topic: str) -> str:
    """Resolve '<module>.<name>[.<name>…]' inside build123d/bd_warehouse/cadlib
    and return that object's signature + docstring."""
    parts = topic.split(".")
    try:
        obj = _resolve_root(parts[0])
    except ImportError:
        return f"cannot resolve '{topic}' — package '{parts[0]}' is not installed"
    resolved = parts[0]
    for attr in parts[1:]:
        if not hasattr(obj, attr) and inspect.ismodule(obj):
            # namespace-package submodules aren't attributes until imported
            with contextlib.suppress(ImportError):
                obj = importlib.import_module(f"{resolved}.{attr}")
                resolved += f".{attr}"
                continue
        if not hasattr(obj, attr):
            what = "module" if inspect.ismodule(obj) else "object"
            avail = (
                ", ".join(n for n in dir(obj) if not n.startswith("_"))[:800]
                if inspect.ismodule(obj)
                else ""
            )
            hint = f" Available: {avail}" if avail else ""
            return (
                f"cannot resolve '{topic}' — {what} '{resolved}' has no attribute '{attr}'.{hint}"
            )
        obj = getattr(obj, attr)
        resolved += f".{attr}"
    if inspect.ismodule(obj):
        if obj.__name__.startswith("bd_warehouse"):
            label = dict(_BDW_MODULES).get(obj.__name__, obj.__name__)
            digest = _bdw_module_digest(obj, label)
            if digest:
                return "\n".join(digest)
        names = [n for n in dir(obj) if not n.startswith("_")]
        doc = _doc_line(obj)
        out = f"{resolved} — module" + (f"  # {doc}" if doc else "")
        return out + "\npublic names: " + ", ".join(names[:60])
    doc = inspect.getdoc(obj) or "no docstring"
    return f"{resolved}{_signature(obj)}\n{doc[:1500]}"


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #

_BUILDERS: dict[str, Any] = {
    "build123d": _build123d_digest,
    "cadlib": _cadlib_digest,
    "bd_warehouse": _bdw_digest,
}


def _versions() -> tuple[str, str, str]:
    """Cache key component: installed versions of the introspected packages.

    cadlib ships inside cadx (not a separate dist) so it is fingerprinted by
    the newest source mtime instead."""

    def _v(dist: str) -> str:
        try:
            return dist_version(dist)
        except PackageNotFoundError:
            return "-"

    cadlib_stamp = "-"
    try:
        cadlib = _import_cadlib()
        if getattr(cadlib, "__file__", None):
            src = Path(cadlib.__file__).parent
            cadlib_stamp = str(max(int(p.stat().st_mtime) for p in src.glob("*.py")))
    except Exception:
        pass
    return (_v("build123d"), _v("bd_warehouse"), cadlib_stamp)


def quickref(topic: str) -> str:
    """Signature digest for a named topic or a dotted path.

    Named topics: build123d | cadlib | bd_warehouse. Dotted paths like
    'build123d.extrude' or 'bd_warehouse.fastener.CounterSunkScrew' resolve
    inside those packages. Unknown topics return an error string listing the
    valid choices (tool results are data, not exceptions)."""
    topic = topic.strip()
    key = (topic, _versions())
    if key in _CACHE:
        return _CACHE[key]
    builder = _BUILDERS.get(topic)
    if builder is not None:
        text = builder()
    elif "." in topic and topic.split(".", 1)[0] in _RESOLVABLE_ROOTS:
        text = _dotted_digest(topic)
    else:
        text = (
            f"unknown quickref topic '{topic}'. Valid topics: {' | '.join(TOPICS)} — "
            "or a dotted path like 'build123d.extrude', 'cadlib.through_hole' or "
            "'bd_warehouse.fastener.CounterSunkScrew'."
        )
    _CACHE[key] = text
    return text


def context_block(topics: tuple[str, ...] = ("cadlib", "build123d"), max_chars: int = 2200) -> str:
    """Compact signature-only digest for prompt-context injection.

    Rendered without doc lines to stay small; the model can pull full detail
    per object via the quickref tool. Each topic gets an even share of the
    budget — a growing cadlib digest must not starve the build123d enums."""
    note = "… (truncated — call quickref(<topic>) for the full digest)"
    sections: list[str] = []
    for topic in topics:
        builder = _BUILDERS.get(topic)
        if builder is None:
            continue
        try:
            sections.append(builder(compact=True))
        except Exception as e:  # a missing package must not break codegen
            log.warning("quickref.context_block_failed", topic=topic, error=str(e))
    if not sections:
        return ""
    header = (
        "API QUICKREF — exact signatures introspected from the INSTALLED packages\n"
        "(names/sigs below are ground truth; call the quickref tool for full "
        "detail incl. bd_warehouse type/size tables):\n"
    )
    share = max(200, (max_chars - len(header) - len(sections) * (len(note) + 2)) // len(sections))
    return header + "\n".join(_truncate_lines(s, share, note) for s in sections)


# --------------------------------------------------------------------------- #
# error-triggered symbol lookup (used by agent/docfetch.py)                    #
# --------------------------------------------------------------------------- #


def _clip(text: str, limit: int) -> str:
    """Hard char cap for single injections — unlike _truncate_lines this also
    handles one long line (member listings have no newline to cut at)."""
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _resolve_bare(name: str) -> tuple[str, Any] | None:
    """Resolve a bare or dotted name inside build123d, then cadlib.

    Root names resolve to the package itself ("build123d" → the module) and a
    leading root segment is skipped ("build123d.Edge" works as a name)."""
    parts = name.split(".")
    roots: list[tuple[str, Any]] = []
    try:
        import build123d  # deferred: cold import costs seconds

        roots.append(("build123d", build123d))
    except ImportError:
        pass
    with contextlib.suppress(Exception):
        roots.append(("cadlib", _import_cadlib()))
    for label, root in roots:
        rest = parts[1:] if parts[0] == label else parts
        if not rest:
            return label, root
        obj = getattr(root, rest[0], None)
        if obj is None:
            continue
        qual = f"{label}.{rest[0]}"
        for attr in rest[1:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
            qual += f".{attr}"
        if obj is not None:
            return qual, obj
    return None


def _member_kind(obj: Any, name: str) -> str:
    """property | callable | attribute | missing — getattr_static so a
    property is classified without being evaluated."""
    try:
        static = inspect.getattr_static(obj, name)
    except AttributeError:
        return "missing"
    if isinstance(static, property):
        return "property"
    return "callable" if callable(getattr(obj, name, None)) else "attribute"


def _member_label(obj: Any, name: str) -> str:
    kind = _member_kind(obj, name)
    if kind == "property":
        return f"{name} [prop]"
    if kind == "callable":
        return name + "()"
    return name


def _members_digest(qual: str, obj: Any, *, highlight: str | None, max_chars: int) -> str:
    """``<qual> members: a(), b [prop], c, …`` — the failing member is pulled
    to the front with its kind spelled out, so truncation can't hide it
    (the observed failure: ``Edge.radius()`` — radius is a PROPERTY)."""
    names = [n for n in dir(obj) if not n.startswith("_")]
    labels = [_member_label(obj, n) for n in names]
    head = ""
    if highlight:
        if highlight in names:
            kind = _member_kind(obj, highlight)
            if kind == "missing":  # dir() lists it but getattr_static misses
                kind = "attribute"
            note = " — read it WITHOUT ()" if kind == "property" else ""
            head = f"{qual}.{highlight} is a {kind}{note}\n"
            labels = [_member_label(obj, highlight)] + [
                lbl for n, lbl in zip(names, labels, strict=True) if n != highlight
            ]
        else:
            low = highlight.lower()
            close = [n for n in names if n.lower() == low or (len(low) >= 4 and low in n.lower())][
                :8
            ]
            head = f"{qual} has no member '{highlight}'"
            if close:
                head += f" — did you mean: {', '.join(close)}?"
            head += "\n"
            labels = [_member_label(obj, n) for n in close] + [
                lbl for n, lbl in zip(names, labels, strict=True) if n not in close
            ]
    return _clip(head + f"{qual} members: " + ", ".join(labels), max_chars)


def symbol_digest(name: str, owner: str | None = None, *, max_chars: int = 400) -> str:
    """Compact (≤ ``max_chars``) digest for a failing symbol — the doc payload
    attached to exec-error results by docfetch.

    ``owner`` + ``name``: owner resolves in build123d then cadlib and the
    digest lists its members with ``name``'s kind marked (property/method/
    attribute/missing). Bare ``name``: resolves in build123d then cadlib —
    callables get a signature entry, classes additionally a member listing.
    Returns "" when nothing resolves — doc injection is best-effort only."""
    name = name.strip()
    ident = r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*"
    if not re.fullmatch(ident, name):
        return ""
    if owner:
        owner = owner.strip()
        if not re.fullmatch(ident, owner):
            return ""
        resolved = _resolve_bare(owner)
        if resolved is None:
            return ""
        qual, obj = resolved
        return _members_digest(qual, obj, highlight=name.rsplit(".", 1)[-1], max_chars=max_chars)
    resolved = _resolve_bare(name)
    if resolved is None:
        return ""
    qual, obj = resolved
    if inspect.ismodule(obj):
        return _members_digest(qual, obj, highlight=None, max_chars=max_chars)
    if inspect.isclass(obj):
        head = _entry(name.rsplit(".", 1)[-1], obj)
        members = _members_digest(qual, obj, highlight=None, max_chars=max_chars)
        return _clip(head + "\n" + members, max_chars)
    return _clip(_entry(name, obj), max_chars)
