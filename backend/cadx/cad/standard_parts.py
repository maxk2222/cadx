"""Validated code recipes for registered bd_warehouse parts."""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any

from build123d import Rot

from cadx.cad import catalog
from cadx.llm.schemas import PartSpec, StandardPartSpec

_AXIS_ROTATIONS = {
    "+x": (0.0, 90.0, 0.0),
    "-x": (0.0, -90.0, 0.0),
    "+y": (-90.0, 0.0, 0.0),
    "-y": (90.0, 0.0, 0.0),
    "+z": (0.0, 0.0, 0.0),
    "-z": (180.0, 0.0, 0.0),
}
# head-datum catalog classes (screws: head face at local z=0, body extends
# -Z) — the declared axis is the shank direction, so the transform maps
# local -Z onto the axis and a standing screw seats on its head
_DOWN_AXIS_ROTATIONS = {
    "+x": (0.0, -90.0, 0.0),
    "-x": (0.0, 90.0, 0.0),
    "+y": (90.0, 0.0, 0.0),
    "-y": (-90.0, 0.0, 0.0),
    "+z": (180.0, 0.0, 0.0),
    "-z": (0.0, 0.0, 0.0),
}
_THREAD_SIZE_RE = re.compile(r"^M(?P<diameter>\d+(?:\.\d+)?)[x-](?P<pitch>\d+(?:\.\d+)?)$", re.I)
# Constructor params that take a host/object, not a scalar — the class is a
# feature op on an existing part (hole for a fastener, seat for a bearing…)
# and can never be emitted as a standalone standard part.
_CONTEXT_REQUIRED_PARAMS = {"fastener", "bearing", "ring", "key"}


def _part_axis(part: StandardPartSpec) -> str:
    if part.axis:
        return part.axis
    # frame contract: standalone parts rest on XY with their axis up — a
    # screw/pipe lies flat only when the spec explicitly asks for ±x/±y
    return "+z"


def _literal(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("standard part values must be finite")
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        items = ", ".join(_literal(item) for item in value)
        return f"({items},)" if isinstance(value, tuple) else f"[{items}]"
    if value is None or isinstance(value, str | int | float | bool):
        return repr(value)
    raise ValueError(f"unsupported literal value type: {type(value).__name__}")


_RECIPE_LEVEL_PARAMS = {"thread_length", "thread_length_mm", "full_thread"}


def _ctor_params(part: StandardPartSpec) -> dict[str, Any]:
    """`parameters` minus recipe-level keys (thread overrides are method
    overrides, never constructor kwargs)."""
    return {k: v for k, v in dict(part.parameters).items() if k not in _RECIPE_LEVEL_PARAMS}


def _thread_length_request(part: StandardPartSpec) -> float | str | None:
    """Recipe-level thread-reach override → mm value or "full" sentinel.

    `thread_length_mm`/`full_thread` are StandardPartSpec fields (and may
    also arrive inside `parameters` when a model writes them there) — they
    map to a `calculate_thread_length` method override, never a ctor kwarg.
    """
    params = part.parameters or {}
    raw = params.get("thread_length", params.get("thread_length_mm"))
    if raw is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 0.0
        if val <= 0:
            raise ValueError(f"thread_length {raw!r} must be a positive mm value")
        return val
    if part.thread_length_mm is not None:
        return float(part.thread_length_mm)
    for flag in (part.full_thread, params.get("full_thread")):
        if flag is None:
            continue
        if flag in (True, "true", "True", "1", 1):
            return "full"
        if flag in (False, "false", "False", "0", 0):
            return None
        raise ValueError(f"full_thread {flag!r} must be a boolean")
    return None


def _part_values(part: StandardPartSpec) -> tuple[type, dict[str, Any], dict[str, Any]]:
    if not part.class_name:
        raise ValueError("standard part class_name is required")
    cls = catalog.resolve_class(part.class_name)
    entry = catalog.find_entry(part.class_name)
    if cls is None or entry is None:
        raise ValueError(f"unknown standard part '{part.class_name}'")
    schema = entry.get("parameters") or {}
    required = {name for name, row in schema.items() if row.get("required")}
    category = entry.get("category")
    if category == "machining" or required & _CONTEXT_REQUIRED_PARAMS:
        raise ValueError(f"{part.class_name} is a context helper, not a standalone part")
    parameters = _ctor_params(part)
    if category == "pipes" and part.length_mm is None:
        raise ValueError("Pipe requires length_mm")
    if part.length_mm is not None and category != "pipes" and "length" not in schema:
        raise ValueError(f"{part.class_name} does not accept length_mm")
    if part.length_mm is not None and category != "pipes":
        if "length" in parameters and float(parameters["length"]) != part.length_mm:
            raise ValueError("length_mm conflicts with parameters.length")
        parameters["length"] = part.length_mm
    if "simple" in parameters:
        raise ValueError("use the threaded field instead of parameters.simple")
    values = catalog.constructor_kwargs(
        part.class_name,
        fastener_type=part.fastener_type or None,
        size=part.size or None,
        parameters=parameters,
    )
    schema = entry.get("parameters") or {}
    if _thread_length_request(part) is not None and not hasattr(
        cls, "calculate_thread_length"
    ):
        raise ValueError(
            f"{part.class_name}: thread_length/full_thread only applies to "
            "screw parts with a norm-derived thread reach"
        )
    return cls, values, schema


def _thread_data(
    part: StandardPartSpec, cls: type, values: dict[str, Any]
) -> dict[str, Any] | None:
    entry = catalog.find_entry(part.class_name) or {}
    category = entry.get("category")
    threadable = category == "threads" or (
        category in {"screws", "nuts"} and "simple" in (entry.get("parameters") or {})
    )
    if not threadable:
        return None
    threaded = part.threaded if part.threaded is not None else True
    if not threaded:
        return None
    report: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        report = catalog.lookup_part(
            part.class_name,
            fastener_type=part.fastener_type or None,
            size=part.size or None,
            length=part.length_mm,
            parameters={k: v for k, v in values.items() if k not in {"simple"}},
            simple=True,
        )
    diameter = (
        report.get("thread_diameter") or report.get("major_diameter") or report.get("diameter")
    )
    pitch = report.get("thread_pitch") or report.get("pitch")
    size = str(part.size or values.get("size") or "")
    if diameter is None or pitch is None:
        match = _THREAD_SIZE_RE.match(size)
        if match:
            diameter = float(match.group("diameter"))
            pitch = float(match.group("pitch"))
    if diameter is None or pitch is None:
        return None
    external = category == "screws" or bool(values.get("external", category == "threads"))
    hand = str(values.get("hand") or "right")
    requested = _thread_length_request(part) if category == "screws" else None
    if category == "screws":
        total_length = float(part.length_mm or values.get("length") or 0.0)
        length = float(report.get("thread_length") or total_length)
        start = -total_length
        if requested is not None:
            # the recipe emits a `calculate_thread_length` override — report
            # the EFFECTIVE reach so is_standard_only/compare stays honest
            max_len = total_length
            try:
                proto = cls(**{k: v for k, v in values.items() if k != "simple"})
                max_len = float(proto.max_thread_length or total_length)
            except Exception:
                pass
            want = max_len if requested == "full" else float(requested)
            length = min(want, max_len)
            if length <= 0:
                raise ValueError(
                    f"{part.class_name}: thread_length {want:g} exceeds the "
                    f"available shank {max_len:g} mm"
                )
    elif category == "nuts":
        length = float(report.get("nut_thickness") or report.get("head_height") or 0.0)
        start = 0.0
    else:
        length = float(values.get("length") or part.length_mm or 0.0)
        start = 0.0
    if length <= 0:
        return None
    out = {
        "kind": "thread",
        "thread_spec": size,
        "major_diameter": float(diameter),
        "pitch": float(pitch),
        "length": length,
        "start": start,
        "external": external,
        "hand": hand,
    }
    if requested is not None:
        out["thread_length_requested"] = "full" if requested == "full" else float(requested)
        out["thread_length_default"] = float(report.get("thread_length") or 0.0)
        out["max_thread_length"] = float(max_len)
    return out


def validate_standard_parts(parts: list[StandardPartSpec]) -> list[dict[str, Any]]:
    if not parts:
        raise ValueError("at least one standard part is required")
    reports: list[dict[str, Any]] = []
    for part in parts:
        cls, values, _schema = _part_values(part)
        placements = part.placements_mm or [part.position_mm]
        if len(placements) != part.quantity:
            raise ValueError(
                f"{part.class_name}: quantity={part.quantity} requires "
                f"{part.quantity} placements_mm entries"
            )
        reports.append(
            {
                "part": part,
                "class": cls,
                "values": values,
                "placements": placements,
                "thread": _thread_data(part, cls, values),
            }
        )
    return reports


def standard_part_bbox(part: StandardPartSpec) -> list[float]:
    _cls, _values, _schema = _part_values(part)
    measured = catalog.lookup_part(
        part.class_name,
        fastener_type=part.fastener_type or None,
        size=part.size or None,
        length=part.length_mm,
        parameters=_ctor_params(part),
        simple=True,
    )
    local = measured.get("bbox_size_mm")
    if not isinstance(local, list) or len(local) != 3:
        raise ValueError(f"{part.class_name}: catalog geometry has no measurable bounding box")
    x, y, z = (float(value) for value in local)
    if part.class_name == "CounterSunkScrew":
        nominal_diameter = (measured.get("norm_dimensions") or {}).get("dk")
        if nominal_diameter is not None:
            x = y = float(nominal_diameter)
    axis = _part_axis(part).lstrip("+-")
    if axis == "x":
        return [z, y, x]
    if axis == "y":
        return [x, z, y]
    return [x, y, z]


# anatomy a fastener class supplies by definition — an intake spec that lists
# "Torx drive recess" or "threaded shank" as its own feature must not drop the
# recipe path: the catalog part already carries that geometry (a wrong
# description surfaces later via verify_spec on the measured solid, honestly)
_INTRINSIC_FASTENER_FEATURES = re.compile(
    r"head|counter\s*sunk|drive|recess|socket|torx|hex|allen|phillips|"
    r"thread|shank|tip|washer",
    re.I,
)
_FASTENER_CATEGORIES = {"screws", "nuts", "washers", "standoffs", "inserts"}


def _is_intrinsic_fastener_feature(feature: Any) -> bool:
    text = f"{_get(feature, 'name') or ''} {_get(feature, 'type') or ''}"
    return bool(_INTRINSIC_FASTENER_FEATURES.search(text))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """dict- and attribute-tolerant field access — the spec reaches callers
    both as a PartSpec and as a plain spec_dict through the executor."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def covered_feature_indices(spec: Any) -> set[int]:
    """Indices into ``spec.features`` that a catalog standard part supplies —
    a screw's head/shank/drive recess are intrinsic catalog geometry, not
    independent machining ops to cut, anchor or measure."""
    parts = _get(spec, "standard_parts") or []
    if not parts:
        return set()
    has_fastener = any(
        (catalog.find_entry(_get(part, "class_name")) or {}).get("category")
        in _FASTENER_CATEGORIES
        for part in parts
    )
    covered: set[int] = set()
    for i, feature in enumerate(_get(spec, "features") or []):
        name = _get(feature, "name")
        if any(name and name in (_get(part, "covers_features") or []) for part in parts) or (
            has_fastener and name and _is_intrinsic_fastener_feature(feature)
        ):
            covered.add(i)
    return covered


def is_standard_only(spec: PartSpec) -> bool:
    if not spec.standard_parts or any(question.blocking for question in spec.open_questions):
        return False
    covered = covered_feature_indices(spec)
    if any(
        feature.name and i not in covered for i, feature in enumerate(spec.features)
    ):
        return False
    try:
        reports = validate_standard_parts(spec.standard_parts)
    except (TypeError, ValueError):
        return True
    for row in reports:
        thread = row["thread"]
        if thread is None:
            continue
        for i, feature in enumerate(spec.features):
            if i not in covered:
                continue
            feature_kind = str(feature.type or "").lower()
            if "thread" not in feature_kind and not feature.thread_spec:
                continue
            requested_pitch = feature.dims.get("thread_pitch")
            # dims on covered features are sanitized away — a conflicting
            # pitch must still be caught via the free-text thread_spec
            if requested_pitch is None:
                m = _THREAD_SIZE_RE.match(str(feature.thread_spec or "").strip())
                if m:
                    requested_pitch = float(m.group("pitch"))
            requested_length = feature.dims.get("thread_length") or feature.dims.get("length")
            if requested_pitch and abs(float(requested_pitch) - thread["pitch"]) > 1e-6:
                return False
            # a request beyond the mechanical maximum is honoured by clamping
            # ("Vollgewinde" = 100 mm on an M10x100 means max_thread_length
            # 94.5) — only an achievable shortfall fails the fast path
            want_len = float(requested_length) if requested_length else 0.0
            if want_len and thread.get("max_thread_length"):
                want_len = min(want_len, float(thread["max_thread_length"]))
            if want_len and abs(want_len - thread["length"]) > max(
                0.1, thread["length"] * 0.01
            ):
                return False
    return True


def _measure_placement(
    cls: type, values: dict[str, Any], schema: dict[str, Any], axis: str
) -> tuple[tuple[float, float, float], float, float, bool]:
    """(rotation_xyz, z_lift, local_min_z, head_down) for one catalog part.

    Catalog conventions differ: fasteners are built head-datum-first (body
    extends local -Z) while washers/nuts sit on/centre the XY plane. Measure
    the prototype once so the emitted transform points the declared axis
    correctly AND seats the part on XY — a head-up flip is the difference
    between "standing on the head" and "impaled on its own tip". `head_down`
    marks that convention: local +Z maps onto the NEGATIVE declared axis, so
    feature distances measured along the axis run opposite to local z."""
    rotation = _AXIS_ROTATIONS[axis]
    head_down = False
    try:
        proto_kwargs = {k: v for k, v in values.items() if k != "simple"}
        if "simple" in schema:
            proto_kwargs["simple"] = True
        proto = cls(**proto_kwargs)
        bb = proto.bounding_box()
        if bb.max.Z <= 1e-6 and bb.min.Z < -1e-6:
            rotation = _DOWN_AXIS_ROTATIONS[axis]
            head_down = True
        z_lift = -float((Rot(*rotation) * proto).bounding_box().min.Z)
        return rotation, z_lift, float(bb.min.Z), head_down
    except Exception:
        return rotation, 0.0, 0.0, False


def build_standard_part_code(parts: list[StandardPartSpec]) -> str:  # noqa: C901
    reports = validate_standard_parts(parts)
    imports: dict[str, list[str]] = {}
    uses_pipe = any(row["part"].class_name == "Pipe" for row in reports)
    # parts needing a generated subclass: CounterSunkScrew always (norm
    # head profile), plus any screw with a thread-length request — bd_warehouse
    # computes the reach in `calculate_thread_length`, not a ctor kwarg.
    # Parts sharing (class, methods, request) share one subclass named after
    # the catalog class; a second distinct group gets a numeric suffix.
    def _part_methods(part: StandardPartSpec, cls: type) -> set[str]:
        methods: set[str] = set()
        if part.class_name == "CounterSunkScrew":
            methods.add("head_profile")
        if _thread_length_request(part) is not None and hasattr(
            cls, "calculate_thread_length"
        ):
            methods.add("calculate_thread_length")
        return methods

    group_names: dict[tuple[str, frozenset[str], str], str] = {}
    for row in reports:
        cls = row["class"]
        methods = _part_methods(row["part"], cls)
        if not methods:
            continue
        key = (
            cls.__name__,
            frozenset(methods),
            repr(_thread_length_request(row["part"])),
        )
        if key in group_names:
            continue
        taken = set(group_names.values())
        name = cls.__name__
        i = 2
        while name in taken:
            name = f"{cls.__name__}_{i}"
            i += 1
        group_names[key] = name

    subclassed: set[str] = {
        cls.__name__
        for row in reports
        for cls in [row["class"]]
        if _part_methods(row["part"], cls)
    }
    for row in reports:
        cls = row["class"]
        module = str(cls.__module__)
        if not module.startswith("bd_warehouse."):
            raise ValueError(f"{row['part'].class_name} is not a registered bd_warehouse class")
        import_name = (
            f"{cls.__name__} as _Catalog{cls.__name__}"
            if cls.__name__ in subclassed
            else cls.__name__
        )
        imports.setdefault(module, []).append(import_name)

    params: dict[str, Any] = {}
    constructors: list[str] = []
    placements: list[str] = []
    design_features: list[dict[str, Any]] = []
    bom: list[dict[str, Any]] = []
    subclass_bodies: dict[str, list[str]] = {}
    sub_name_map: dict[str, str] = {}
    subclass_method_names: dict[str, set[str]] = {}
    counter = 0
    for row in reports:
        part: StandardPartSpec = row["part"]
        axis = _part_axis(part)
        cls: type = row["class"]
        values: dict[str, Any] = row["values"]
        schema: dict[str, Any] = (catalog.find_entry(part.class_name) or {}).get("parameters") or {}
        thread = row["thread"]
        threaded = thread is not None
        rotation, z_lift, proto_min_z, head_down = _measure_placement(
            cls, values, schema, axis
        )
        kwargs: list[str] = []
        for name, value in values.items():
            if name == "simple":
                continue
            params_key = f"part_{counter}_{name}"
            params[params_key] = value
            kwargs.append(f"{name}=PARAMS[{_literal(params_key)}]")
        if part.class_name == "Pipe":
            path_key = f"part_{counter}_path_length"
            params[path_key] = float(part.length_mm or 0.0)
            kwargs.append(
                f"path=Edge.make_line(Vector(0, 0, 0), Vector(0, 0, PARAMS[{_literal(path_key)}]))"
            )
        thread_req = _thread_length_request(part)
        thread_expr: str | None = None
        if thread_req == "full":
            thread_expr = "self.max_thread_length"
        elif isinstance(thread_req, int | float):
            tl_key = f"part_{counter}_thread_len_req"
            params[tl_key] = float(thread_req)
            thread_expr = f"min(PARAMS[{_literal(tl_key)}], self.max_thread_length)"
        methods = _part_methods(part, cls)
        group_key = (cls.__name__, frozenset(methods), repr(thread_req))
        sub_name = group_names.get(group_key) or (
            f"_Catalog{cls.__name__}" if cls.__name__ in subclassed else cls.__name__
        )
        if methods:
            sub_name_map[sub_name] = cls.__name__
            subclass_method_names[sub_name] = methods
        if thread_expr is not None and methods and sub_name not in subclass_bodies:
            subclass_bodies[sub_name] = [
                "    def calculate_thread_length(self):\n"
                f"        return {thread_expr}"
            ]
        if "simple" in schema:
            simple_value = '(CADX_DETAIL_MODE == "preview")' if threaded else "True"
            kwargs.append(f"simple={simple_value}")
        if part.class_name == "CounterSunkScrew":
            measured = catalog.lookup_part(
                part.class_name,
                fastener_type=part.fastener_type or None,
                size=part.size or None,
                length=part.length_mm,
                parameters=_ctor_params(part),
                simple=True,
            )
            for mismatch in measured.get("geometry_mismatches") or []:
                design_features.append(
                    {
                        "kind": "catalog_geometry_adjustment",
                        "part_class": part.class_name,
                        **mismatch,
                        "generated_geometry": "unfilleted norm-table profile",
                    }
                )
        for position in row["placements"]:
            ident = f"_standard_part_{counter}"
            constructor = f"{sub_name}({', '.join(kwargs)})"
            if thread is not None and "simple" not in schema:
                dia_key = f"part_{counter}_thread_diameter"
                length_key = f"part_{counter}_thread_length"
                params[dia_key] = thread["major_diameter"]
                params[length_key] = thread["length"]
                constructors.append(
                    f'if CADX_DETAIL_MODE == "preview":\n'
                    f"    {ident} = Pos(0.0, 0.0, {proto_min_z}) * Cylinder("
                    f"radius=PARAMS[{_literal(dia_key)}] / 2, "
                    f"height=PARAMS[{_literal(length_key)}], "
                    "align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
                    "else:\n"
                    f"    {ident} = {constructor}"
                )
            else:
                constructors.append(f"{ident} = {constructor}")
            placed = (
                f"Pos({position[0]}, {position[1]}, {position[2] + z_lift}) * "
                f"Rot({rotation[0]}, {rotation[1]}, {rotation[2]}) * {ident}"
            )
            placements.append(placed)
            if thread is not None:
                feature = {
                    **thread,
                    "axis": axis,
                    # the renderer draws position + axis*distance in the FINAL
                    # frame — bake the placement transform into the record or
                    # the helix overlay floats in catalog-frame space
                    "position_mm": [position[0], position[1], position[2] + z_lift],
                }
                if head_down:
                    # head-datum: local -Z maps onto the declared axis, so a
                    # local thread span [start, start+length] lands at axis
                    # distance [-start-length, -start] from the head datum
                    feature["start"] = -float(thread["start"]) - float(thread["length"])
                design_features.append(feature)
            counter += 1
        bom.append(
            {
                "part_class": part.class_name,
                "type": part.fastener_type or None,
                "size": part.size or None,
                "count": part.quantity,
                "name": part.name or None,
            }
        )

    import_lines = [
        f"from {module} import {', '.join(sorted(set(names)))}"
        for module, names in sorted(imports.items())
    ]
    head_profile_lines = [
        "    def head_profile(self):",
        "        a, k, dk = (self.screw_data[key] for key in ('a', 'k', 'dk'))",
        "        with BuildSketch(Plane.XZ) as profile:",
        "            Trapezoid(dk, k, 90 - a / 2, align=(Align.CENTER, Align.MAX), rotation=180)",
        "            split(bisect_by=Plane.YZ)",
        "        return profile.sketch.face()",
    ]
    subclass_defs: list[str] = []
    for sub_name in sorted(sub_name_map):
        subclass_defs.append(f"class {sub_name}(_Catalog{sub_name_map[sub_name]}):")
        if "head_profile" in subclass_method_names[sub_name]:
            subclass_defs += head_profile_lines
        subclass_defs += subclass_bodies.get(sub_name, [])
        subclass_defs.append("")
    fuse_helper = [
        "def _cadx_fuse_solids(obj):",
        '    if CADX_DETAIL_MODE != "export":',
        "        return obj",
        "    try:",
        "        solids = obj.solids()",
        "    except Exception:",
        "        return obj",
        "    # a single N-way boolean over dozens of separate thread turns blows",
        "    # up in OCC (pairwise section work is O(n²); long full threads have",
        "    # segfaulted the executor) — fuse contiguous chunks of turns first",
        "    # so every operation stays small and the helix faces survive",
        "    while len(solids) > 1:",
        "        try:",
        "            merged = []",
        "            for i in range(0, len(solids), 8):",
        "                group = solids[i : i + 8]",
        "                merged.append(group[0] if len(group) == 1 else group[0].fuse(*group[1:]))",
        "            solids = merged",
        "        except Exception:",
        "            return obj",
        "    return solids[0]",
        "",
    ]
    body_lines = [
        "from build123d import Align, BuildSketch, Compound, Cylinder, Plane, Pos, Rot, Trapezoid, split"
        + (", Edge, Vector" if uses_pipe else ""),
        *import_lines,
        *subclass_defs,
        "",
        f"PARAMS = {_literal(params)}",
        f"CADX_FEATURES = {_literal(design_features)}",
        f"CADX_BOM = {_literal(bom)}",
        *fuse_helper,
        *constructors,
    ]
    if len(placements) == 1:
        body_lines.append(f"result = _cadx_fuse_solids({placements[0]})")
    else:
        fused = ", ".join(f"_cadx_fuse_solids({p})" for p in placements)
        body_lines.append(f"result = Compound(children=[{fused}])")
    return "\n".join(body_lines) + "\n"


def has_thread_design(spec: Any) -> bool:
    return expected_thread_count(spec) > 0


def _thread_spec_clearance_note(feature: Any, size_mm: float) -> bool:
    """True when an ``M<size>`` thread_spec really names the bolt going
    through, not a thread to build: the declared bore is closer to the
    metric clearance diameter than to the tap drill (M8 clearance ≈8.5,
    M8 tap ≈6.8). With no declared bore the wording decides."""
    dims = _get(feature, "dims") or {}
    dia = next(
        (
            float(dims[k])
            for k in ("diameter", "dia", "d", "bore", "hole_d")
            if isinstance(dims.get(k), int | float) and float(dims[k]) > 0
        ),
        None,
    )
    if dia is None:
        return False
    from cadx.cad import catalog

    size = int(size_mm)
    tap = catalog.nominal_hole_diameter(size, "tap")
    clearance = catalog.nominal_hole_diameter(size, "clearance")
    if tap is None or clearance is None or abs(clearance - tap) < 1e-6:
        return False
    return abs(dia - clearance) < abs(dia - tap)


def expected_thread_count(spec: Any) -> int:
    expected = 0
    covered: set[str] = set()
    for part in getattr(spec, "standard_parts", []) or []:
        covered.update(part.covers_features)
        cls, values, _schema = _part_values(part)
        if part.threaded is not False and _thread_data(part, cls, values) is not None:
            expected += int(part.quantity)
    for feature in getattr(spec, "features", []) or []:
        if feature.name in covered:
            continue
        kind = str(getattr(feature, "type", "") or "").lower()
        thread_spec = str(getattr(feature, "thread_spec", "") or "").strip()
        # a real thread callout starts with the size ("M12x1.75", "M4") —
        # fit notes like "clearance for M8" also fill thread_spec but are
        # plain clearance holes, not threads to build
        is_thread = "thread" in kind or "tapped" in kind
        m = re.match(r"M\s*(\d+(?:\.\d+)?)", thread_spec)
        if not is_thread and m:
            is_thread = not _thread_spec_clearance_note(feature, float(m.group(1)))
        if is_thread:
            expected += max(1, int(getattr(feature, "count", 1) or 1))
    return expected
