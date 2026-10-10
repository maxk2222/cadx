"""Catalog of bd_warehouse standard parts for the UI, prompts and validation.

Two entry kinds:
- ``normed``:     classes backed by norm tables — ``types`` maps each
                  fastener/bearing/o-ring/bottle type to its valid size strings.
- ``parametric``: classes without size tables (gears, IsoThread, open_builds…)
                  — ``params`` lists the constructor signature instead.

Every class is also registered in a name->class registry used by validation
(`resolve_class`, `validate_part_ref`) and the prompt excerpt (`catalog_excerpt`).
"""

from __future__ import annotations

import contextlib
import inspect
import typing
from typing import Any

_CATALOG: dict[str, Any] | None = None
_REGISTRY: dict[str, type] = {}

# common metric coarse-pitch size strings, used to map "M4" -> "M4-0.7"
_METRIC_SIZE = {
    2: "M2-0.4",
    3: "M3-0.5",
    4: "M4-0.7",
    5: "M5-0.8",
    6: "M6-1",
    8: "M8-1.25",
    10: "M10-1.5",
    12: "M12-1.75",
    16: "M16-2",
    20: "M20-2.5",
}

# kwargs that select a norm table row; validated against the class's types()
_TYPE_KWARGS = ("fastener_type", "bearing_type", "o_ring_type", "bottle_type")

# object attributes worth reporting by the lookup endpoint
_LOOKUP_ATTRS = (
    "clearance_hole_diameters",
    "tap_hole_diameters",
    "head_diameter",
    "head_height",
    "head_angle",
    "socket_diameter",
    "socket_depth",
    "nut_diameter",
    "nut_thickness",
    "washer_diameter",
    "washer_thickness",
    "thread_diameter",
    "size",
    "fastener_type",
    "bearing_type",
    "hole_locations",
)


def _doc(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.splitlines()[0].strip() if doc else ""


def _literal_values(annotation: Any) -> list[str]:
    """Values of a ``Literal[...]`` annotation (direct or inside Optional etc.)."""
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) is typing.Literal:
        return [str(a) for a in args]
    out: list[str] = []
    for a in args:
        out.extend(_literal_values(a))
    return out


def _sig_params(cls: type) -> list[str]:
    """Constructor params as 'name=default' strings (short, LLM/UI friendly)."""
    try:
        sig = inspect.signature(getattr(cls, "__init__"))  # wrapped ctors hide cls signature
    except Exception:
        return []
    out: list[str] = []
    for name, p in sig.parameters.items():
        if name == "self" or name.startswith("_"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            out.append(name)
        else:
            d = repr(p.default)
            if len(d) > 24:
                d = d[:21] + "..."
            out.append(f"{name}={d}")
        if len(out) >= 12:
            break
    return out


def _sizes_dict(cls: type) -> dict[str, list[str]] | None:
    """{type: [sizes]} for normed classes; None when no size table exists.

    Classes fall into three shapes:
      a) types() + sizes(type)          — fasteners, bearings, o-rings
      b) sizes(<Literal-annotated arg>) — e.g. PlasticBottleThread(bottle_type)
      c) sizes()                      — retaining rings, shaft keys, BSPP…
    """
    types: list[str] = []
    try:
        if hasattr(cls, "types"):
            types = sorted(str(t) for t in cls.types())
    except Exception:
        types = []

    sizes_fn = getattr(cls, "sizes", None)
    if sizes_fn is None:
        return None

    entry: dict[str, list[str]] = {}
    if types:
        for t in types:
            try:
                entry[t] = [str(s) for s in cls.sizes(t)]
            except Exception:
                entry[t] = []
        return entry or None

    # no types() — check whether sizes() takes a Literal-annotated argument
    try:
        sig = inspect.signature(sizes_fn)
        lit_params = [
            _literal_values(p.annotation)
            for n, p in sig.parameters.items()
            if n != "cls" and p.annotation is not inspect.Parameter.empty
        ]
        lit_values = next((v for v in lit_params if v), [])
    except Exception:
        lit_values = []
    if lit_values:
        for t in lit_values:
            try:
                entry[t] = [str(s) for s in cls.sizes(t)]
            except Exception:
                entry[t] = []
        return entry or None

    try:
        return {"default": [str(s) for s in cls.sizes()]}
    except Exception:
        return None


def _holes_for(cls: type) -> list[str]:
    """Which hole/seat helpers pair with this part class."""
    from bd_warehouse.bearing import Bearing
    from bd_warehouse.fastener import HeatSetNut, Nut, Screw
    from bd_warehouse.retaining_ring import RetainingRing

    if issubclass(cls, HeatSetNut):
        return ["InsertHole"]
    if issubclass(cls, Screw):
        return ["ClearanceHole", "TapHole", "ThreadedHole"]
    if issubclass(cls, Nut):
        return ["ClearanceHole"]  # captive_nut=True
    if issubclass(cls, Bearing):
        return ["PressFitHole"]
    if issubclass(cls, RetainingRing):
        return ["RetainingRingGroove"]
    return []


def _normed_entry(cls: type, sizes: dict[str, list[str]], category: str) -> dict[str, Any]:
    return {
        "kind": "normed",
        "category": category,
        "doc": _doc(cls),
        "types": sizes,
        "holes": _holes_for(cls),
    }


def _parametric_entry(cls: type, category: str) -> dict[str, Any]:
    return {
        "kind": "parametric",
        "category": category,
        "doc": _doc(cls),
        "params": _sig_params(cls),
        "holes": _holes_for(cls),
    }


def _register(cls: type, entry: dict[str, Any], cat: dict[str, Any]) -> None:
    cat[cls.__name__] = entry
    _REGISTRY[cls.__name__] = cls


def build_catalog() -> dict[str, Any]:
    """Collect standard-part classes -> catalog entries. Cached process-wide."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    from bd_warehouse import (
        bearing,
        bushing,
        fastener,
        flange,
        gear,
        o_rings,
        open_builds,
        pipe,
        retaining_ring,
        shaft_key,
        sprocket,
        thread,
    )
    from build123d import Compound

    try:
        from bd_warehouse.fastener import BasePartObject
    except Exception:
        BasePartObject = object  # type: ignore[assignment,misc]

    catalog: dict[str, Any] = {}

    def normed(category: str, classes: list[type]) -> None:
        cat = catalog.setdefault(category, {})
        for cls in classes:
            sizes = _sizes_dict(cls)
            if sizes:
                _register(cls, _normed_entry(cls, sizes, category), cat)
            else:
                _register(cls, _parametric_entry(cls, category), cat)

    def parametric(category: str, classes: list[type]) -> None:
        cat = catalog.setdefault(category, {})
        for cls in classes:
            _register(cls, _parametric_entry(cls, category), cat)

    def subclasses(base: type) -> list[type]:
        out: list[type] = []
        subs: list[type] = base.__subclasses__()
        for c in subs:
            out.append(c)
            out.extend(subclasses(c))
        return out

    # --- normed categories ---------------------------------------------------
    normed("screws", subclasses(fastener.Screw))
    normed("nuts", subclasses(fastener.Nut))
    normed("washers", subclasses(fastener.Washer))
    normed("bearings", subclasses(bearing.Bearing))
    normed("retaining_rings", subclasses(retaining_ring.RetainingRing))
    normed("o_rings", [o_rings.ORing])
    normed("shaft_keys", [shaft_key.ShaftKey])
    normed(
        "threads",
        [
            thread.BSPPThread,
            thread.WhitworthThread,
            thread.AcmeThread,
            thread.MetricTrapezoidalThread,
            thread.PlasticBottleThread,
        ],
    )

    # --- parametric categories ------------------------------------------------
    parametric("threads", [thread.IsoThread])
    parametric(
        "gears",
        [
            gear.SpurGear,
            gear.HelicalGear,
            gear.RackGear,
            gear.WormGear,
            gear.WormWheel,
        ],
    )
    parametric("sprockets", [sprocket.Sprocket])
    parametric("bushings", [bushing.HexFlangedEccentricBushing])
    parametric("pipes", [pipe.Pipe])
    parametric("flanges", [*subclasses(flange.Flange), flange.LappedFlangeStub])
    ob_classes = {
        c
        for _, c in inspect.getmembers(open_builds, inspect.isclass)
        if (c.__module__ or "").startswith("bd_warehouse")
        and issubclass(c, (BasePartObject, Compound))
        and not c.__name__.startswith("_")
    }
    parametric("open_builds", sorted(ob_classes, key=lambda c: c.__name__))

    # --- machining helpers (context objects, not parts) -----------------------
    parametric(
        "machining",
        [
            fastener.ClearanceHole,
            fastener.TapHole,
            fastener.ThreadedHole,
            fastener.InsertHole,
            bearing.PressFitHole,
            retaining_ring.RetainingRingGroove,
            shaft_key.Keyway,
        ],
    )

    _CATALOG = {k: v for k, v in catalog.items() if v}
    return _CATALOG


def registry() -> dict[str, type]:
    """class name -> bd_warehouse class (builds the catalog on first use)."""
    build_catalog()
    return dict(_REGISTRY)


def resolve_class(name: str) -> type | None:
    """Look up a catalog class by name."""
    build_catalog()
    return _REGISTRY.get(name)


def find_entry(class_name: str) -> dict[str, Any] | None:
    cat = build_catalog()
    for classes in cat.values():
        if class_name in classes:
            return dict(classes[class_name])
    return None


def valid_types(class_name: str) -> list[str]:
    e = find_entry(class_name)
    if not e or e.get("kind") != "normed":
        return []
    return [t for t in e["types"] if t != "default"]


def valid_sizes(class_name: str, type_: str | None = None) -> list[str]:
    e = find_entry(class_name)
    if not e or e.get("kind") != "normed":
        return []
    types: dict[str, list[str]] = e["types"]
    if type_ and type_ in types:
        return types[type_]
    out: list[str] = []
    for sizes in types.values():
        out.extend(sizes)
    return sorted(set(out))


def validate_part_ref(
    class_name: str,
    *,
    fastener_type: str | None = None,
    size: str | int | float | None = None,
) -> list[str]:
    """Validate a standard-part reference against the catalog. Returns errors."""
    e = find_entry(class_name)
    if e is None:
        return [f"unknown standard part '{class_name}' — not in the bd_warehouse catalog"]
    if e.get("kind") != "normed":
        return []
    errors: list[str] = []
    types: dict[str, list[str]] = e["types"]
    real_types = [t for t in types if t != "default"]
    if fastener_type and real_types and fastener_type not in types:
        errors.append(
            f"{class_name}: type '{fastener_type}' not valid (valid: {', '.join(real_types[:12])})"
        )
    if size is not None:
        pool = types.get(fastener_type or "")
        if pool is None:
            pool = [s for sizes in types.values() for s in sizes]
        if str(size) not in pool:
            shown = ", ".join(pool[:15])
            errors.append(
                f"{class_name}: size '{size}' not in norm table "
                f"(e.g. {shown}{', …' if len(pool) > 15 else ''})"
            )
    return errors


def nominal_hole_diameter(metric_mm: int, kind: str = "clearance") -> float | None:
    """Normal clearance / Soft tap drill diameter for M{metric_mm} from the
    bd_warehouse tables — replaces hard-coded plausibility dicts."""
    size = _METRIC_SIZE.get(metric_mm)
    if not size:
        return None
    try:
        from bd_warehouse.fastener import SocketHeadCapScrew

        s = SocketHeadCapScrew(size=size, fastener_type="iso4762", length=10)
        if kind == "tap":
            return float(s.tap_hole_diameters["Soft"])
        return float(s.clearance_hole_diameters["Normal"])
    except Exception:
        return None


def lookup_part(
    class_name: str,
    *,
    fastener_type: str | None = None,
    size: str | None = None,
    length: float | None = None,
) -> dict[str, Any]:
    """Instantiate a catalog part and return its published dimension data.

    Raises ValueError on unknown class / invalid refs / construction failures.
    """
    cls = resolve_class(class_name)
    if cls is None:
        raise ValueError(f"unknown class '{class_name}'")
    errors = validate_part_ref(class_name, fastener_type=fastener_type, size=size)
    if errors:
        raise ValueError("; ".join(errors))

    params = set()
    with contextlib.suppress(Exception):
        sig = inspect.signature(cls.__init__)  # wrapped ctors hide cls signature
        params = set(sig.parameters)

    kwargs: dict[str, Any] = {}
    if "size" in params:
        kwargs["size"] = size
    for tw in _TYPE_KWARGS:
        if tw in params and fastener_type:
            kwargs[tw] = fastener_type
    if "length" in params:
        kwargs["length"] = length if length is not None else 10.0
    try:
        obj = cls(**{k: v for k, v in kwargs.items() if v is not None})
    except Exception as e:
        raise ValueError(f"{class_name} construction failed: {e}") from e

    data: dict[str, Any] = {"class": class_name}
    for attr in _LOOKUP_ATTRS:
        try:
            v = getattr(obj, attr, None)
        except Exception:
            continue
        if v is None:
            continue
        if attr == "hole_locations":
            data["hole_count"] = len(v) if hasattr(v, "__len__") else 1
            continue
        try:
            import json

            json.dumps(v, default=str)
            data[attr] = v
        except Exception:
            data[attr] = str(v)
    return data


def catalog_excerpt(class_names: list[str], *, max_chars: int = 6000, max_sizes: int = 40) -> str:
    """Compact, prompt-ready excerpt limited to the named classes.

    Unknown names are reported explicitly so the model can fix its plan.
    """
    if not class_names:
        return ""
    build_catalog()
    lines = ["STANDARD PARTS — valid catalog values (use exact type/size strings):"]
    seen: set[str] = set()
    for name in class_names:
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entry = find_entry(name)
        if entry is None:
            lines.append(f"- {name}: NOT IN CATALOG — pick a class from the reference")
            continue
        if entry["kind"] == "parametric":
            lines.append(
                f"- {name} ({entry['category']}, parametric): {', '.join(entry['params'][:10])}"
            )
            continue
        tys = entry["types"]
        parts = []
        for t, sizes in list(tys.items())[:4]:
            label = t if t != "default" else "sizes"
            shown = ", ".join(sizes[:max_sizes])
            extra = f" (+{len(sizes) - max_sizes})" if len(sizes) > max_sizes else ""
            parts.append(f"{label}=[{shown}{extra}]")
        holes = f" | holes: {', '.join(entry['holes'])}" if entry.get("holes") else ""
        lines.append(f"- {name} ({entry['category']}): {' '.join(parts)}{holes}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n… (catalog excerpt truncated)"
    return out


def class_listing() -> str:
    """One line per category with all class names — for the PLAN prompt."""
    cat = build_catalog()
    lines = ["Available standard-part classes (exact names):"]
    for category, classes in cat.items():
        lines.append(f"- {category}: {', '.join(sorted(classes))}")
    return "\n".join(lines)


def search(query: str, *, limit: int = 25) -> list[dict[str, Any]]:
    """Normalized search over class names, types and size strings."""
    q = query.strip().lower()
    if not q:
        return []
    cat = build_catalog()
    hits: list[dict[str, Any]] = []
    for classes in cat.values():
        for cls_name, e in classes.items():
            score = 0
            if q in cls_name.lower():
                score += 3
            matched_sizes: list[str] = []
            for t, sizes in (e.get("types") or {}).items():
                if t != "default" and q in t.lower():
                    score += 2
                for s in sizes:
                    if q in s.lower():
                        matched_sizes.append(s)
                        score += 1
            if score:
                hits.append(
                    {
                        "class": cls_name,
                        "category": e["category"],
                        "kind": e["kind"],
                        "score": score,
                        "matched_sizes": matched_sizes[:10],
                    }
                )
    hits.sort(key=lambda h: (-h["score"], h["class"]))
    return hits[:limit]
