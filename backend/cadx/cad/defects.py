"""B-Rep defect location + STEP roundtrip gate for the executor pipeline.

``locate_defects`` turns a bare "B-Rep is not valid" into WHERE it is broken:
each defect carries a kind, the sub-shape type/index and a 3D centre, so the
repair loop can target a specific face/edge instead of guessing. ``defect_summary``
renders the same data as a one-line log/feedback string.

``step_roundtrip_check`` is the export gate: a shape can be valid in memory yet
break on STEP serialisation (unsewn shells, non-manifold leftovers). It writes
the part to STEP, re-imports it and compares validity + volume — the pipeline
appends the report's warning to ``ExecResult.warnings``.

Runs host-side in the executor (never inside the sandboxed runner), but keeps
to build123d/OCP + stdlib imports like the neighbouring trusted CAD modules.
All entry points are defensive: they return error dicts, never raise.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from pathlib import Path
from typing import Any

MAX_DEFECTS = 20
# faceted hybrid solids make STEP export quadratic-slow — bound the gate
STEP_GATE_MAX_FACES = 20_000
# warn when the re-imported volume differs from the in-memory solid
STEP_GATE_VOLUME_TOL = 0.005  # 0.5%

# BRepCheck sub-shape walk order, most actionable first
_SUBSHAPE_KINDS = (
    ("FACE", "face"),
    ("EDGE", "edge"),
    ("WIRE", "wire"),
    ("SHELL", "shell"),
    ("SOLID", "solid"),
    ("VERTEX", "vertex"),
)


def _pnt(p: Any) -> list[float]:
    return [round(float(p.X()), 3), round(float(p.Y()), 3), round(float(p.Z()), 3)]


def _center(sub: Any) -> list[float] | None:
    """Geometric centre of a TopoDS sub-shape, best effort."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    try:
        st = sub.ShapeType()
        props = GProp_GProps()
        if st == TopAbs_VERTEX:
            return _pnt(BRep_Tool.Pnt_s(TopoDS.Vertex(sub)))
        if st == TopAbs_EDGE:
            BRepGProp.LinearProperties_s(sub, props)
        elif st in (TopAbs_FACE, TopAbs_SHELL):
            BRepGProp.SurfaceProperties_s(sub, props)
        elif st == TopAbs_SOLID:
            try:
                BRepGProp.VolumeProperties_s(sub, props)
            except Exception:
                BRepGProp.SurfaceProperties_s(sub, props)
        else:  # wire/compound/etc — linear props still yield a centre
            BRepGProp.LinearProperties_s(sub, props)
        return _pnt(props.CentreOfMass())
    except Exception:
        return None


def _status_names(analyzer: Any, sub: Any) -> list[str]:
    """BRepCheck status names for an invalid sub-shape — best effort, the
    NCollection list binding is not registered on every OCP build."""
    try:
        return [
            getattr(s, "name", str(s)).replace("BRepCheck_", "")
            for s in analyzer.Result(sub).Status()
        ]
    except Exception:
        return []


def _edge_topological_defects(
    wrapped: Any, max_entries: int
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Open (1-face) and non-manifold (>2-face) B-Rep edges, each with the
    edge midpoint — the coordinates a repair prompt needs to sew a gap or
    redo a boolean."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
    )
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    edge_face_map = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapesAndAncestors_s(wrapped, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    out: list[dict[str, Any]] = []
    edges: list[Any] = []
    for i in range(1, edge_face_map.Extent() + 1):
        edge = TopoDS.Edge(edge_face_map.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):  # seam/pole edges aren't boundaries
            continue
        faces = edge_face_map.FindFromIndex(i).Extent()
        if faces == 0 or faces == 2:  # free wire (PMI) / manifold
            continue
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        kind = "open_edge" if faces == 1 else "nonmanifold_edge"
        edges.append(edge)
        out.append(
            {
                "kind": kind,
                "shape_type": "edge",
                "index": i - 1,
                "center": _pnt(props.CentreOfMass()),
                "faces_incident": faces,
                "description": (
                    "open boundary edge — an unsewn or missing face"
                    if faces == 1
                    else f"edge shared by {faces} faces — a self-touch/coincident face"
                ),
            }
        )
        if len(out) >= max_entries:
            break
    return out, edges


def _collect(wrapped: Any) -> list[dict[str, Any]]:
    import OCP.TopAbs as topabs
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopExp import TopExp_Explorer

    analyzer = BRepCheck_Analyzer(wrapped)
    if analyzer.IsValid():
        return []

    defects, topo_edges = _edge_topological_defects(wrapped, MAX_DEFECTS)
    for enum_name, label in _SUBSHAPE_KINDS:
        enum = getattr(topabs, f"TopAbs_{enum_name}")
        explorer = TopExp_Explorer(wrapped, enum)
        index = -1
        while explorer.More():
            index += 1
            sub = explorer.Current()
            explorer.Next()
            if analyzer.IsValid(sub):
                continue
            # the topological pass already located this edge precisely
            if enum_name == "EDGE" and any(e.IsSame(sub) for e in topo_edges):
                continue
            status = _status_names(analyzer, sub)
            defects.append(
                {
                    "kind": f"brep_invalid_{label}",
                    "shape_type": label,
                    "index": index,
                    "center": _center(sub),
                    "description": (
                        f"BRepCheck: {', '.join(status)}"
                        if status
                        else f"BRepCheck reports this {label} invalid"
                    ),
                    "status": status,
                }
            )
            if len(defects) >= MAX_DEFECTS:
                return defects[:MAX_DEFECTS]
    return defects[:MAX_DEFECTS]


def locate_defects(shape: Any) -> list[dict[str, Any]]:
    """Locate B-Rep validity defects on ``shape``, each with 3D coordinates.

    Returns ``[{kind, shape_type, index, center, description, ...}]`` — an
    empty list when the shape passes BRepCheck. ``kind`` is ``open_edge`` /
    ``nonmanifold_edge`` for topological boundary problems and
    ``brep_invalid_<face|edge|wire|shell|solid|vertex>`` for BRepCheck
    failures. ``index`` is the explorer-order index (matching
    ``shape.faces()[i]`` / edge-map order). Never raises: failures return
    ``[{"error": str(e)}]``. Capped at :data:`MAX_DEFECTS`.
    """
    try:
        wrapped = getattr(shape, "wrapped", shape)
        if wrapped is None:
            raise TypeError(f"cannot locate defects on {type(shape).__name__}")
        return _collect(wrapped)
    except Exception as e:
        return [{"error": f"defect location failed: {e}"}]


def summarize_defects(defects: list[dict[str, Any]]) -> str:
    """One-line rendering of a :func:`locate_defects` result."""
    if not defects:
        return "no B-Rep defects located"
    if all("error" in d for d in defects):
        return str(defects[0]["error"])
    counts = Counter(str(d.get("kind", "?")) for d in defects)
    kinds = ", ".join(f"{kind}x{n}" if n > 1 else kind for kind, n in sorted(counts.items()))
    locs: list[str] = []
    for d in defects:
        center = d.get("center")
        if center is None or len(locs) >= 3:
            continue
        where = f"{d.get('kind')} {d.get('shape_type', '')}[{d.get('index')}]"
        locs.append(f"{where} @ ({center[0]:.3g}, {center[1]:.3g}, {center[2]:.3g})")
    tail = " — " + "; ".join(locs) if locs else ""
    return f"{len(defects)} defect(s) [{kinds}]{tail}"


def defect_summary(shape: Any) -> str:
    """One-line defect summary of ``shape`` for logs/errors — never raises."""
    try:
        return summarize_defects(locate_defects(shape))
    except Exception as e:
        return f"defect summary failed: {e}"


def step_roundtrip_check(
    shape: Any,
    step_path: Path,
    *,
    max_faces: int = STEP_GATE_MAX_FACES,
    volume_tol: float = STEP_GATE_VOLUME_TOL,
    write: bool = True,
) -> dict[str, Any]:
    """STEP roundtrip gate: export ``shape`` to ``step_path``, re-import it,
    compare validity + volume.

    Returns a report dict: ``{"checked": True, "ok": True}`` on a clean
    roundtrip; ``{"checked": True, "ok": False, "warning": str, ...}`` when the
    re-import is invalid or the volume drifted more than ``volume_tol``;
    ``{"checked": False, "skipped": reason}`` when the shape is too large to
    gate cheaply; ``{"checked": False, "error": str}`` when the gate itself
    failed (export/import raised). Never raises.

    ``write=False`` re-imports an existing ``step_path`` instead of writing a
    fresh export (used when the pipeline already shipped a STEP this run).
    Files this function writes itself are removed again afterwards.
    """
    from build123d import export_step, import_step

    try:
        face_count = len(shape.faces())
    except Exception as e:
        return {"checked": False, "error": f"cannot inspect shape: {e}"}
    if face_count > max_faces:
        return {
            "checked": False,
            "skipped": f"{face_count} faces > {max_faces} — STEP roundtrip too slow",
        }

    report: dict[str, Any] = {"checked": True}
    if write:
        try:
            export_step(shape, str(step_path))
        except Exception as e:
            report.update({"checked": False, "error": f"STEP export failed: {e}"})
            return report
    try:
        try:
            reimported = import_step(str(step_path))
        except Exception as e:
            report.update(
                {
                    "ok": False,
                    "warning": f"STEP roundtrip: re-import of the written file failed ({e})"
                    " — the exported artifact is unreadable",
                }
            )
            return report

        original_valid = bool(shape.is_valid)
        reimported_valid = bool(reimported.is_valid)
        report["original_valid"] = original_valid
        report["reimported_valid"] = reimported_valid

        try:
            original_volume = float(shape.volume)
            reimported_volume = float(reimported.volume)
        except Exception:
            original_volume = reimported_volume = 0.0
        report["volume_mm3"] = round(original_volume, 3)
        report["reimported_volume_mm3"] = round(reimported_volume, 3)
        delta = (
            abs(reimported_volume - original_volume) / max(abs(original_volume), 1e-9)
            if original_volume
            else 0.0
        )
        report["volume_delta_pct"] = round(delta * 100, 3)

        problems: list[str] = []
        if not reimported_valid:
            problems.append("re-imported shape fails BRepCheck")
            defects = locate_defects(reimported)
            if defects and not all("error" in d for d in defects):
                report["reimported_defects"] = defects[:5]
                problems.append(summarize_defects(defects))
        if delta > volume_tol:
            problems.append(
                f"volume drifted {delta * 100:.2f}% "
                f"({original_volume:.1f} -> {reimported_volume:.1f} mm^3, "
                f"tolerance {volume_tol * 100:.1f}%)"
            )
        if problems:
            report["ok"] = False
            report["warning"] = "STEP roundtrip: " + "; ".join(problems)
        else:
            report["ok"] = True
        return report
    finally:
        if write:
            with contextlib.suppress(OSError):
                Path(step_path).unlink(missing_ok=True)
