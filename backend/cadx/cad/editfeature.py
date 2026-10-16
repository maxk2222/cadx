"""Conservative verified parametric edits of recognised B-Rep features.

Port of build123d-mcp's ``edit_features`` tool (scope v1: plain through-holes
only). Enlarging a hole re-cuts a larger cylinder at the recorded
opening/axis/depth; shrinking first defeatures the old bore
(``BRepAlgoAPI_Defeaturing``), then cuts the smaller one. Every edit is
transactional: the edited solid is only returned when the full verification
gate passes — predicted annulus volume vs measured volume delta, unchanged
outer envelope, unchanged other holes, and clean re-recognition of the
resized feature. Anything else is a controlled refusal, never a guess.

Deliberately no ``cadx`` imports — pure build123d/OCP + the optional
``quiddity`` recogniser (same constraint as compare.py/defects.py: this is
trusted code that may also run inside the executor where imports stay lazy).
All public entry points are defensive: they return dicts, never raise.
"""

from __future__ import annotations

import math
from typing import Any

# faceted hybrid/mesh solids carry thousands of tiny faces — analytic hole
# recognition and defeaturing are neither reliable nor bounded there; same
# cap as compare.MAX_FACES_FOR_BOOLEAN / defects.STEP_GATE_MAX_FACES
MAX_FACES_FOR_EDIT = 20_000

_POS_TOL_MM = 0.01  # opening point / cylinder-geometry agreement
_AXIS_TOL = 0.001  # unit axis-vector agreement
_BBOX_TOL_MM = 0.01  # outer envelope must not move at all
_REGION_TOL_MM = 0.02  # predicted vs measured changed-region bbox
_SAME_DIA_TOL = 1e-6  # "unchanged diameter" epsilon

_DEPENDENT_KEYS = ("cbore", "spotface", "csink")

# quiddity emits counterbores/spotfaces on the hole record itself, but
# countersinks arrive as their own coaxial feature family — correlate those
# back onto the bore so a dependent hole is never guessed as editable
_DEPENDENT_FAMILIES = ("countersinks", "counterbores", "spotfaces", "profiled_bores")


def _evidence(shape: Any) -> Any | None:
    """quiddity recognition evidence for ``shape``; None when unavailable."""
    try:
        import quiddity.evidence as qev
    except Exception:
        return None
    try:
        if len(shape.faces()) > MAX_FACES_FOR_EDIT:
            return None
        return qev.build_recognition_evidence(shape)
    except Exception:
        return None


def _vec(v: Any) -> list[float]:
    return [float(c) for c in v]


def _scan_holes(shape: Any) -> tuple[Any | None, list[tuple[Any, dict[str, Any], frozenset]]]:
    """(evidence, [(feature_ref, record_dict, constituent FaceRefs)]) — hole
    features only, in recognition order. The order is deterministic for a
    given shape, so the list index is the stable ``feature_id``."""
    ev = _evidence(shape)
    if ev is None:
        return None, []
    holes: list[tuple[Any, dict[str, Any], frozenset]] = []
    for feature in getattr(ev, "features", ()):
        try:
            if ev.family(feature) != "holes":
                continue
            record = ev.record(feature).to_dict()
            holes.append((feature, record, frozenset(ev.constituent_faces(feature))))
        except Exception:
            continue
    return ev, holes


def _dominant(axis: list[float]) -> str:
    i = max(range(3), key=lambda k: abs(axis[k])) if axis else 0
    return f"{'+' if axis[i] >= 0 else '-'}{'xyz'[i]}"


def _hole_dependents(ev: Any, record: dict[str, Any]) -> list[str]:
    """Dependent features on this hole: quiddity's own cbore/spotface/csink
    record fields plus coaxial features of dependent families (countersinks
    are emitted as a separate feature, not folded into the hole record)."""
    deps = {k for k in _DEPENDENT_KEYS if record.get(k) is not None}
    axis = _vec(record.get("axis") or (0.0, 0.0, 0.0))
    opening = _vec(record.get("location") or (0.0, 0.0, 0.0))
    depth = float(record.get("depth") or 0.0)
    radius = float(record.get("diameter") or 0.0) / 2
    for feat in getattr(ev, "features", ()):
        try:
            if ev.family(feat) not in _DEPENDENT_FAMILIES:
                continue
            rec = ev.record(feat).to_dict()
        except Exception:
            continue
        d_axis = _vec(rec.get("axis") or (0.0, 0.0, 0.0))
        d_loc = _vec(rec.get("location") or (0.0, 0.0, 0.0))
        if len(d_axis) != 3 or len(d_loc) != 3:
            continue
        # parallel axis (either direction) and coaxial with the bore
        if abs(sum(a * b for a, b in zip(axis, d_axis, strict=False))) < 1.0 - _AXIS_TOL:
            continue
        off = [d_loc[i] - opening[i] for i in range(3)]
        along = sum(o * a for o, a in zip(off, axis, strict=False))
        lateral = math.sqrt(max(0.0, sum(o * o for o in off) - along * along))
        if lateral > max(_POS_TOL_MM * 10, radius * 0.25):
            continue
        # near the bore's axial span (opening..exit, plus the dependent's
        # own depth on either side — a csink sits proud of the opening)
        margin = max(1.0, float(rec.get("depth") or 0.0) + radius)
        if -margin <= along <= depth + margin:
            deps.add(str(ev.family(feat)))
    return sorted(deps)


def _public_feature(
    index: int, record: dict[str, Any], n_faces: int, dependents: list[str]
) -> dict[str, Any]:
    through = record.get("bottom") == "through"
    refusal: str | None = None
    if not through:
        refusal = f"not a through hole (bottom={record.get('bottom')!r})"
    elif dependents:
        refusal = f"hole has dependent geometry ({', '.join(dependents)})"
    elif n_faces != 1:
        refusal = f"bore spans {n_faces} faces — not a single cylindrical wall"
    axis = _vec(record.get("axis") or (0.0, 0.0, 0.0))
    return {
        "feature_id": index,
        "kind": "through_hole" if through else "hole",
        "diameter": round(float(record.get("diameter") or 0.0), 6),
        "axis": axis,
        "direction": _dominant(axis),
        "opening": _vec(record.get("location") or (0.0, 0.0, 0.0)),
        "depth": round(float(record.get("depth") or 0.0), 6),
        "has_dependents": bool(dependents),
        "dependents": dependents,
        "editable": refusal is None,
        "refusal": refusal,
    }


def find_editable_features(shape: Any) -> list[dict[str, Any]]:
    """Recognised hole features of ``shape`` — each a dict with a stable
    ``feature_id`` (index in recognition order), measured geometry and an
    ``editable`` flag. Plain through-holes are editable; holes with
    dependents (countersink/counterbore/spotface) or multi-face bores are
    still listed but flagged with a ``refusal`` reason. Returns ``[]`` for
    over-complex inputs (> MAX_FACES_FOR_EDIT faces) or missing quiddity —
    callers that need the distinction use :func:`edit_feature`'s refusal.
    """
    if shape is None:
        return []
    try:
        if len(shape.faces()) > MAX_FACES_FOR_EDIT:
            return []
    except Exception:
        return []
    try:
        ev, holes = _scan_holes(shape)
    except Exception:
        return []
    if ev is None:
        return []
    return [
        _public_feature(i, rec, len(faces), _hole_dependents(ev, rec))
        for i, (_, rec, faces) in enumerate(holes)
    ]


def _solid_from_occt(shape: Any) -> Any:
    """Single build123d Solid out of a TopoDS result — reject mixed
    defeaturing output instead of silently taking one solid."""
    from build123d import Solid
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID
    from OCP.TopoDS import TopoDS, TopoDS_Iterator

    to_solid = getattr(TopoDS, "Solid_s", None) or TopoDS.Solid

    if shape.ShapeType() == TopAbs_SOLID:
        return Solid(to_solid(shape))
    if shape.ShapeType() != TopAbs_COMPOUND:
        raise ValueError("defeaturing did not produce a solid")
    solids = []
    stack = [shape]
    while stack:
        iterator = TopoDS_Iterator(stack.pop())
        while iterator.More():
            child = iterator.Value()
            if child.ShapeType() == TopAbs_SOLID:
                solids.append(Solid(to_solid(child)))
            elif child.ShapeType() == TopAbs_COMPOUND:
                stack.append(child)
            else:
                raise ValueError("defeaturing produced mixed topology")
            iterator.Next()
    if len(solids) != 1:
        raise ValueError(f"defeaturing produced {len(solids)} solids, expected one")
    return solids[0]


def _bounds(shape: Any) -> tuple[float, ...]:
    box = shape.bounding_box()
    return (*box.min, *box.max)


def _annulus_bounds(
    opening: list[float], axis: list[float], depth: float, radius: float
) -> tuple[float, ...]:
    exit_point = [opening[i] + axis[i] * depth for i in range(3)]
    radial = [radius * math.sqrt(max(0.0, 1 - axis[i] ** 2)) for i in range(3)]
    return tuple(min(opening[i], exit_point[i]) - radial[i] for i in range(3)) + tuple(
        max(opening[i], exit_point[i]) + radial[i] for i in range(3)
    )


def _same_holes(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    """Match occurrences, including duplicates, with small recognition tolerance."""
    if len(expected) != len(actual):
        return False
    unused = actual.copy()
    for old in expected:
        match = next(
            (
                new
                for new in unused
                if old.get("bottom") == new.get("bottom")
                and abs(float(old["diameter"]) - float(new["diameter"])) <= _POS_TOL_MM
                and abs(float(old["depth"]) - float(new["depth"])) <= _POS_TOL_MM
                and all(
                    abs(a - b) <= _POS_TOL_MM
                    for a, b in zip(old["location"], new["location"], strict=False)
                )
                and all(
                    abs(a - b) <= _AXIS_TOL for a, b in zip(old["axis"], new["axis"], strict=False)
                )
                and old.get("cbore") == new.get("cbore")
                and old.get("spotface") == new.get("spotface")
                and old.get("csink") == new.get("csink")
            ),
            None,
        )
        if match is None:
            return False
        unused.remove(match)
    return True


def edit_feature(  # noqa: C901 — the gate is deliberately one transaction
    shape: Any,
    feature_id: int | str,
    parameter: str = "diameter",
    value: float | int | str | None = None,
) -> dict[str, Any]:
    """Resize one plain through hole, checking predicted change before commit.

    Returns ``{"ok": True, "shape": <edited solid>, "report": {...}}`` when the
    whole gate passes; ``{"ok": False, "refusal": str, "report": {...}}``
    otherwise — a refusal never carries mutated geometry.
    """
    report: dict[str, Any] = {
        "feature_id": feature_id,
        "parameter": parameter,
        "value": value,
        "committed": False,
    }

    def refuse(reason: str, **extra: Any) -> dict[str, Any]:
        report.update(extra)
        return {"ok": False, "refusal": reason, "report": report}

    try:
        faces_count = len(shape.faces())
    except Exception as exc:
        return refuse(f"not a B-Rep shape: {type(exc).__name__}: {exc}")
    if faces_count > MAX_FACES_FOR_EDIT:
        return refuse(
            f"input has {faces_count} faces (> {MAX_FACES_FOR_EDIT}) — "
            "faceted solids break the analytic assumptions; edit refused",
            face_count=faces_count,
        )
    try:
        valid = bool(shape.is_valid) and len(shape.solids()) == 1
    except Exception as exc:
        return refuse(f"cannot inspect shape: {type(exc).__name__}: {exc}")
    if not valid:
        return refuse("source must be one valid solid")

    if str(parameter).strip().lower() != "diameter":
        return refuse(f"unsupported parameter {parameter!r} — only 'diameter' is editable")

    try:
        ev, holes = _scan_holes(shape)
    except Exception as exc:
        return refuse(f"feature recognition failed: {type(exc).__name__}: {exc}")
    if ev is None:
        return refuse("feature recognition unavailable (quiddity not installed)")
    try:
        index = int(str(feature_id).strip())
    except (TypeError, ValueError):
        return refuse(f"feature_id must be an integer index, got {feature_id!r}")
    if not 0 <= index < len(holes):
        return refuse(
            f"unknown feature_id {feature_id!r} — {len(holes)} hole feature(s) recognised"
        )

    try:
        diameter = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return refuse(f"value must be a positive finite number, got {value!r}")
    if isinstance(value, bool) or not math.isfinite(diameter) or diameter <= 0:
        return refuse(f"diameter must be a positive finite number, got {value!r}")

    _feature, record, constituent = holes[index]
    record = dict(record)
    dependents = _hole_dependents(ev, record)
    if record.get("bottom") != "through" or dependents or len(constituent) != 1:
        return refuse(
            "hole has dependent or non-cylindrical geometry; edit refused",
            dependents=dependents,
            bottom=record.get("bottom"),
            constituent_faces=len(constituent),
        )

    old_diameter = float(record["diameter"])
    if abs(diameter - old_diameter) <= _SAME_DIA_TOL:
        return refuse("new diameter equals the current diameter")

    axis = _vec(record["axis"])
    opening = _vec(record["location"])
    depth = float(record["depth"])
    old_radius = old_diameter / 2
    new_radius = diameter / 2
    expected_delta = math.pi * (old_radius**2 - new_radius**2) * depth
    vol_tol = max(0.01, abs(expected_delta) * 0.001)
    report.update(
        {
            "old_diameter": round(old_diameter, 6),
            "new_diameter": diameter,
            "opening": opening,
            "axis": axis,
            "depth": depth,
        }
    )

    try:
        if diameter > old_diameter:
            base = shape
        else:
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

            remover = BRepAlgoAPI_Defeaturing()
            remover.SetShape(shape.wrapped)
            resolver = getattr(ev, "caller_face", ev.face)
            for ref in constituent:
                remover.AddFaceToRemove(resolver(ref).wrapped)
            remover.Build()
            if not remover.IsDone():
                return refuse("defeaturing the old bore failed")
            base = _solid_from_occt(remover.Shape())
            filled = float(base.volume) - float(shape.volume)
            expected_fill = math.pi * old_radius**2 * depth
            if abs(filled - expected_fill) > max(0.01, expected_fill * 0.001):
                return refuse(
                    "defeaturing changed material beyond the old bore",
                    filled_volume=round(filled, 6),
                    expected_fill=round(expected_fill, 6),
                )

        # Start just outside the opening but stop at the recorded exit
        # exactly — cutting deeper risks biting into a far wall after a
        # through hole opens into a cavity.
        from build123d import Align, Cylinder, Plane

        lead = min(0.01, depth * 0.001)
        start = tuple(opening[i] - axis[i] * lead for i in range(3))
        cutter = Plane(origin=start, z_dir=axis) * Cylinder(
            new_radius, depth + lead, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        candidate = base - cutter
        if not candidate.is_valid or len(candidate.solids()) != 1:
            return refuse("edited shape is invalid or not a single solid")

        # --- verification gate ------------------------------------------------
        actual_delta = float(candidate.volume) - float(shape.volume)
        report["predicted"] = {"volume_delta": round(expected_delta, 6)}
        report["measured"] = {"volume_delta": round(actual_delta, 6)}
        if abs(actual_delta - expected_delta) > vol_tol:
            return refuse(
                f"volume mismatch: predicted {expected_delta:.6f}, measured {actual_delta:.6f} mm³"
            )

        expected_added = max(expected_delta, 0.0)
        expected_removed = max(-expected_delta, 0.0)
        added_shape = candidate - shape
        removed_shape = shape - candidate
        added = float(added_shape.volume)
        removed = float(removed_shape.volume)
        report["predicted"]["added_volume"] = round(expected_added, 6)
        report["predicted"]["removed_volume"] = round(expected_removed, 6)
        report["measured"]["added_volume"] = round(added, 6)
        report["measured"]["removed_volume"] = round(removed, 6)
        if abs(added - expected_added) > vol_tol or abs(removed - expected_removed) > vol_tol:
            return refuse("changed material differs from the predicted hole annulus")

        predicted_bounds = _annulus_bounds(opening, axis, depth, max(old_radius, new_radius))
        measured_bounds = _bounds(added_shape if expected_added else removed_shape)
        report["predicted_region_bbox"] = [round(v, 6) for v in predicted_bounds]
        report["measured_region_bbox"] = [round(v, 6) for v in measured_bounds]
        if any(
            abs(a - b) > _REGION_TOL_MM
            for a, b in zip(predicted_bounds, measured_bounds, strict=False)
        ):
            return refuse("changed region differs from the predicted hole annulus")
        if any(
            abs(a - b) > _BBOX_TOL_MM
            for a, b in zip(_bounds(shape), _bounds(candidate), strict=False)
        ):
            return refuse("outer envelope moved during hole edit")
        report["outer_envelope_unchanged"] = True

        # re-recognition: the resized bore must come back at the new diameter
        # on the same opening/axis/depth, and every OTHER hole unchanged
        _, after_holes = _scan_holes(candidate)
        after_records = [dict(rec) for _, rec, _ in after_holes]
        changed = [h for h in after_records if abs(float(h["diameter"]) - diameter) <= _POS_TOL_MM]
        if not changed:
            return refuse("resized hole was not recognised in the result")
        replacement = next(
            (
                h
                for h in changed
                if all(
                    abs(a - b) <= _POS_TOL_MM for a, b in zip(h["location"], opening, strict=False)
                )
                and abs(float(h["depth"]) - depth) <= _POS_TOL_MM
                and h.get("bottom") == "through"
                and all(abs(a - b) <= _AXIS_TOL for a, b in zip(h["axis"], axis, strict=False))
            ),
            None,
        )
        if replacement is None:
            return refuse("resized hole moved or changed depth")
        after_records.remove(replacement)
        before_records = [dict(rec) for _, rec, _ in holes]
        del before_records[index]
        report["other_holes_unchanged"] = _same_holes(before_records, after_records)
        if not report["other_holes_unchanged"]:
            return refuse("another hole or mounting interface changed")

        report["committed"] = True
        return {"ok": True, "shape": candidate, "report": report}
    except Exception as exc:  # OCC failures must not leak half-edits
        return refuse(f"{type(exc).__name__}: {exc}")
