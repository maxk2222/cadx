"""Validated code recipes for registered bd_warehouse parts."""

from __future__ import annotations

import contextlib
import math
import re
from typing import Any

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
_THREAD_SIZE_RE = re.compile(r"^M(?P<diameter>\d+(?:\.\d+)?)[x-](?P<pitch>\d+(?:\.\d+)?)$", re.I)
# Constructor params that take a host/object, not a scalar — the class is a
# feature op on an existing part (hole for a fastener, seat for a bearing…)
# and can never be emitted as a standalone standard part.
_CONTEXT_REQUIRED_PARAMS = {"fastener", "bearing", "ring", "key"}


def _part_axis(part: StandardPartSpec) -> str:
    if part.axis:
        return part.axis
    category = (catalog.find_entry(part.class_name) or {}).get("category")
    return "+x" if category in {"screws", "threads", "pipes", "shaft_keys"} else "+z"


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
    parameters = dict(part.parameters)
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
    schema = (entry.get("parameters") or {})
    return cls, values, schema


def _thread_data(part: StandardPartSpec, cls: type, values: dict[str, Any]) -> dict[str, Any] | None:
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
        report.get("thread_diameter")
        or report.get("major_diameter")
        or report.get("diameter")
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
    if category == "screws":
        total_length = float(part.length_mm or values.get("length") or 0.0)
        length = float(report.get("thread_length") or total_length)
        start = -total_length
    elif category == "nuts":
        length = float(report.get("nut_thickness") or report.get("head_height") or 0.0)
        start = 0.0
    else:
        length = float(values.get("length") or part.length_mm or 0.0)
        start = 0.0
    if length <= 0:
        return None
    return {
        "kind": "thread",
        "thread_spec": size,
        "major_diameter": float(diameter),
        "pitch": float(pitch),
        "length": length,
        "start": start,
        "external": external,
        "hand": hand,
    }


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
        parameters=part.parameters,
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


def is_standard_only(spec: PartSpec) -> bool:
    if not spec.standard_parts or any(question.blocking for question in spec.open_questions):
        return False
    feature_names = {feature.name for feature in spec.features if feature.name}
    covered = {name for part in spec.standard_parts for name in part.covers_features}
    if feature_names - covered:
        return False
    try:
        reports = validate_standard_parts(spec.standard_parts)
    except (TypeError, ValueError):
        return True
    for row in reports:
        part: StandardPartSpec = row["part"]
        thread = row["thread"]
        if thread is None:
            continue
        for feature in spec.features:
            if feature.name not in part.covers_features:
                continue
            feature_kind = str(feature.type or "").lower()
            if "thread" not in feature_kind and not feature.thread_spec:
                continue
            requested_pitch = feature.dims.get("thread_pitch")
            requested_length = feature.dims.get("thread_length") or feature.dims.get("length")
            if requested_pitch and abs(float(requested_pitch) - thread["pitch"]) > 1e-6:
                return False
            if requested_length and abs(float(requested_length) - thread["length"]) > max(
                0.1, thread["length"] * 0.01
            ):
                return False
    return True


def build_standard_part_code(parts: list[StandardPartSpec]) -> str:
    reports = validate_standard_parts(parts)
    imports: dict[str, list[str]] = {}
    uses_pipe = any(row["part"].class_name == "Pipe" for row in reports)
    needs_countersunk_profile = any(
        row["part"].class_name == "CounterSunkScrew" for row in reports
    )
    for row in reports:
        cls = row["class"]
        module = str(cls.__module__)
        if not module.startswith("bd_warehouse."):
            raise ValueError(f"{row['part'].class_name} is not a registered bd_warehouse class")
        import_name = (
            f"{cls.__name__} as _CatalogCounterSunkScrew"
            if cls.__name__ == "CounterSunkScrew"
            else cls.__name__
        )
        imports.setdefault(module, []).append(import_name)

    params: dict[str, Any] = {}
    constructors: list[str] = []
    placements: list[str] = []
    design_features: list[dict[str, Any]] = []
    bom: list[dict[str, Any]] = []
    counter = 0
    for row in reports:
        part: StandardPartSpec = row["part"]
        axis = _part_axis(part)
        cls: type = row["class"]
        values: dict[str, Any] = row["values"]
        schema: dict[str, Any] = (catalog.find_entry(part.class_name) or {}).get("parameters") or {}
        thread = row["thread"]
        threaded = thread is not None
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
                "path=Edge.make_line(Vector(0, 0, 0), "
                f"Vector(0, 0, PARAMS[{_literal(path_key)}]))"
            )
        if "simple" in schema:
            simple_value = '(CADX_DETAIL_MODE == "preview")' if threaded else "True"
            kwargs.append(f"simple={simple_value}")
        if part.class_name == "CounterSunkScrew":
            measured = catalog.lookup_part(
                part.class_name,
                fastener_type=part.fastener_type or None,
                size=part.size or None,
                length=part.length_mm,
                parameters=part.parameters,
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
            constructor = f"{cls.__name__}({', '.join(kwargs)})"
            if thread is not None and "simple" not in schema:
                dia_key = f"part_{counter}_thread_diameter"
                length_key = f"part_{counter}_thread_length"
                params[dia_key] = thread["major_diameter"]
                params[length_key] = thread["length"]
                constructors.append(
                    f"if CADX_DETAIL_MODE == \"preview\":\n"
                    f"    {ident} = Cylinder("
                    f"radius=PARAMS[{_literal(dia_key)}] / 2, "
                    f"height=PARAMS[{_literal(length_key)}], "
                    "align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
                    "else:\n"
                    f"    {ident} = {constructor}"
                )
            else:
                constructors.append(f"{ident} = {constructor}")
            rotation = _AXIS_ROTATIONS[axis]
            placed = (
                f"Pos({position[0]}, {position[1]}, {position[2]}) * "
                f"Rot({rotation[0]}, {rotation[1]}, {rotation[2]}) * {ident}"
            )
            placements.append(placed)
            if thread is not None:
                design_features.append(
                    {
                        **thread,
                        "axis": axis,
                        "position_mm": list(position),
                    }
                )
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
    countersunk_profile = (
        [
            "class CounterSunkScrew(_CatalogCounterSunkScrew):",
            "    def head_profile(self):",
            "        a, k, dk = (self.screw_data[key] for key in ('a', 'k', 'dk'))",
            "        with BuildSketch(Plane.XZ) as profile:",
            "            Trapezoid(dk, k, 90 - a / 2, align=(Align.CENTER, Align.MAX), rotation=180)",
            "            split(bisect_by=Plane.YZ)",
            "        return profile.sketch.face()",
            "",
        ]
        if needs_countersunk_profile
        else []
    )
    fuse_helper = [
        "def _cadx_fuse_solids(obj):",
        '    if CADX_DETAIL_MODE != "export":',
        "        return obj",
        "    try:",
        "        solids = obj.solids()",
        "        if len(solids) > 1:",
        "            return solids[0].fuse(*solids[1:])",
        "    except Exception:",
        "        pass",
        "    return obj",
        "",
    ]
    body_lines = [
        "from build123d import Align, BuildSketch, Compound, Cylinder, Plane, Pos, Rot, Trapezoid, split"
        + (", Edge, Vector" if uses_pipe else ""),
        *import_lines,
        *countersunk_profile,
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
        if "thread" in kind or getattr(feature, "thread_spec", ""):
            expected += max(1, int(getattr(feature, "count", 1) or 1))
    return expected
