"""Guide-builder executor tasks — run on the cad-light lane (arq).

- ``extract_ports``: STEP/BRep -> deterministic attachment ports (cylinder +
  planar faces, radius/axis/gender) + a GLB tessellation for scene
  collision checks and rendering. OCC work is executor-only.
- ``render_assembly``: multi-mesh scene -> off-screen PNG via
  ``cad.render.render_scene``.

Worker calls these through ``QueueExecutor.run_task`` (same facade as
execute_version/audit_params).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cadx.logging import get_logger

log = get_logger(__name__)

_MAX_PORTS = 24
_MIN_PLANAR_AREA_FRAC = 0.01


def _face_ports(shape) -> list[dict[str, Any]]:
    """Exact attachment ports from BRep faces (build123d/OCP).

    Cylinder faces -> {pos on axis, axis dir, radius, gender male/female};
    planar faces -> {pos center, axis normal, gender flat}. Gender by
    whether the face normal points away from the cylinder axis (boss) or
    toward it (bore).
    """
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN

    def _inside(p: np.ndarray) -> bool:
        cl = BRepClass3d_SolidClassifier(shape.wrapped, gp_Pnt(*p), 1e-6)
        return bool(cl.State() == TopAbs_IN)

    faces = list(shape.faces())
    total_area = sum(float(f.area) for f in faces) or 1.0
    ports: list[dict[str, Any]] = []
    cyl_recs: list[dict[str, Any]] = []

    for idx, face in enumerate(faces):
        try:
            ad = BRepAdaptor_Surface(face.wrapped)
            ftype = ad.GetType()
            center = face.center()
            normal = face.normal_at(face.center())
            n = np.array([normal.X, normal.Y, normal.Z], dtype=float)
            n /= max(float(np.linalg.norm(n)), 1e-9)

            if ftype == GeomAbs_SurfaceType.GeomAbs_Cylinder:
                cyl = ad.Cylinder()
                ax = cyl.Axis()
                loc, d = ax.Location(), ax.Direction()
                axis = np.array([d.X(), d.Y(), d.Z()], dtype=float)
                axis /= max(float(np.linalg.norm(axis)), 1e-9)
                c_arr = np.array([center.X, center.Y, center.Z], dtype=float)
                # outward convention: the port axis points AWAY from the part
                # body toward the OPEN end — OCC's canonical direction is
                # arbitrary and a face-center heuristic coin-flips on
                # centerline features (blind bores). Classify a sample point
                # just past each cylinder end against the solid: the buried
                # end is IN, the open end is OUT.
                # true axial extent: project the face's own bbox along the
                # axis (area/2πr underestimates partial cylinders)
                fbb = face.bounding_box()
                h = abs(
                    float(
                        np.dot(
                            np.array(
                                [
                                    fbb.max.X - fbb.min.X,
                                    fbb.max.Y - fbb.min.Y,
                                    fbb.max.Z - fbb.min.Z,
                                ]
                            ),
                            axis,
                        )
                    )
                )
                eps = max(float(cyl.Radius()) * 0.05, 0.05)
                try:
                    inside_pos = _inside(c_arr + axis * (h * 0.5 + eps))
                    inside_neg = _inside(c_arr - axis * (h * 0.5 + eps))
                except Exception:
                    inside_pos = inside_neg = False
                if inside_pos and not inside_neg:
                    axis = -axis
                origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
                # pos = face center projected onto the cylinder axis
                pos = origin + axis * float(np.dot(c_arr - origin, axis))
                radial = c_arr - pos
                rn = float(np.linalg.norm(radial))
                outward = float(np.dot(n, radial / rn)) > 0.0 if rn > 1e-9 else True
                # per-corner axial span + canonical axis line for the
                # annular-end pass below (coaxial faces share axis+origin)
                corners = np.array(
                    [
                        [x, y, z]
                        for x in (fbb.min.X, fbb.max.X)
                        for y in (fbb.min.Y, fbb.max.Y)
                        for z in (fbb.min.Z, fbb.max.Z)
                    ]
                )
                ax_c = axis.copy()
                for i in range(3):
                    if abs(float(ax_c[i])) > 1e-9:
                        if ax_c[i] < 0:
                            ax_c = -ax_c
                        break
                # spans must be measured in the CANONICAL frame — faces on
                # the same axis line can carry opposite OCC directions, so
                # corners @ axis would give mirrored extents per face and
                # the end-covering check below would see no shared ends
                svals = corners @ ax_c
                o_c = origin - ax_c * float(np.dot(origin, ax_c))
                cyl_recs.append(
                    {
                        "port_name": f"cyl_{idx}",
                        "axis_c": ax_c,
                        "origin_c": o_c,
                        "radius": float(cyl.Radius()),
                        "s_min": float(svals.min()),
                        "s_max": float(svals.max()),
                        "outward": outward,
                    }
                )
                # pos = the port's OPEN side (mouth/rim plane) for every
                # gender — the same convention rim-port extraction uses,
                # so solver frames never mix mouth-vs-seat semantics.
                # axis points at the open end; depth is the seat distance
                # inward (-axis). Male tip and female mouth both sit at
                # center + axis·h/2.
                contact = pos + axis * (h * 0.5)
                ports.append(
                    {
                        "name": f"cyl_{idx}",
                        "kind": "cylinder",
                        "pos": [round(float(v), 4) for v in contact],
                        "axis": [round(float(v), 4) for v in axis],
                        "radius": round(float(cyl.Radius()), 4),
                        "depth": round(h, 4),
                        "gender": "male" if outward else "female",
                        "area": round(float(face.area), 2),
                    }
                )
            elif ftype == GeomAbs_SurfaceType.GeomAbs_Plane:
                if float(face.area) < _MIN_PLANAR_AREA_FRAC * total_area:
                    continue
                ports.append(
                    {
                        "name": f"face_{idx}",
                        "kind": "planar",
                        "pos": [round(float(v), 4) for v in (center.X, center.Y, center.Z)],
                        "axis": [round(float(v), 4) for v in n],
                        "radius": 0.0,
                        "depth": 0.0,
                        "gender": "flat",
                        "area": round(float(face.area), 2),
                    }
                )
        except Exception as exc:  # single bad face must not kill extraction
            log.debug("guide.extract_ports.face_skipped", idx=idx, err=str(exc))
    ports.sort(key=lambda p: -p["area"])
    end_ports, consumed = _annular_end_ports(cyl_recs, _inside)
    # fused ends REPLACE the single-face ports they were built from — a
    # tube's shell+bore ARE its end_0/end_1, and keeping both aliases
    # lets a consumed end look free again under the other name. Non-
    # outermost male faces (nested spigots) keep their own port.
    ports = [p for p in ports if p["name"] not in consumed]
    # fused ends rank first — they carry the full envelope picture the
    # socket model needs; single-face ports stay as fallback detail
    return (end_ports + ports)[:_MAX_PORTS]


def _annular_end_ports(
    cyl_recs: list[dict[str, Any]], inside: Any
) -> tuple[list[dict[str, Any]], set[str]]:
    """One port per OPEN annular end of a coaxial cylinder-face group.

    Rim-extraction convention: an open annular end (tube mouth, hollow
    boss mouth) is ONE port carrying BOTH envelopes — inner_radius =
    bore, outer_radius = wall OD — so the end can plug into a socket.
    Single-face ports can't express that: a bore face has no outer
    envelope, leaving generated hollow cylinders with an end that is
    unpluggable or portless. A bore spanning the group's full extent is
    a through-hole and reports depth=500 (the rim probe's convention).
    ``cyl_recs`` carry canonical axis frames (``axis_c`` sign-normalized,
    ``origin_c`` the axis-line point nearest the origin, ``s_min``/
    ``s_max`` the face's axial span along ``axis_c``).
    """
    import numpy as np

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for rec in cyl_recs:
        gkey = (
            tuple(round(v, 2) for v in rec["origin_c"]),
            tuple(round(v, 3) for v in rec["axis_c"]),
        )
        groups.setdefault(gkey, []).append(rec)
    end_ports: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for recs in groups.values():
        if len({round(r["radius"], 3) for r in recs}) < 2:
            continue  # one radius along this axis — no annulus
        s_lo = min(r["s_min"] for r in recs)
        s_hi = max(r["s_max"] for r in recs)
        d = recs[0]["axis_c"]
        o = recs[0]["origin_c"]
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, d))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        rad = ref - d * float(np.dot(ref, d))
        rad /= max(float(np.linalg.norm(rad)), 1e-9)
        # candidate mouths: every axial boundary of a BORE face — an open
        # annular edge lives where a void boundary terminates into air.
        # This covers group extrema (tube ends) AND interior planes (a
        # counterbore mouth where the recess ends but a nested spigot or
        # the wall continues). A bore's own span direction tells which
        # way the void continues: past its boundary, away from the span.
        planes: list[tuple[float, Any]] = []
        for b in (r for r in recs if not r["outward"]):
            planes.append((b["s_min"], -d))
            planes.append((b["s_max"], d))
        planes.sort(key=lambda t: t[0])
        # merge coincident planes — two bores can share a boundary s with
        # OPPOSITE open directions (one's seat is the other's mouth); try
        # every recorded direction per position, first escape wins
        spots: list[tuple[float, list[Any]]] = []
        for s_end, out_dir in planes:
            if spots and abs(s_end - spots[-1][0]) <= 0.5:
                if not any(np.array_equal(out_dir, od) for od in spots[-1][1]):
                    spots[-1][1].append(out_dir)
            else:
                spots.append((s_end, [out_dir]))
        for s_end, out_dirs in spots:
            port, used_names = _eval_end_spot(
                recs, s_end, out_dirs, o, d, rad, s_lo, s_hi, inside,
                idx=len(end_ports),
            )
            if port is None:
                continue
            consumed.update(used_names)
            end_ports.append(port)
    return end_ports, consumed


def _eval_end_spot(
    recs: list[dict[str, Any]],
    s_end: float,
    out_dirs: list[Any],
    o: Any,
    d: Any,
    rad: Any,
    s_lo: float,
    s_hi: float,
    inside: Any,
    *,
    idx: int,
) -> tuple[dict[str, Any] | None, set[str]]:
    """Evaluate one candidate mouth plane: the covering faces' radii give
    the rim annulus (inner = void opening, outer = nearest wall), a
    mid-ring probe just past the plane must leave the solid (seat
    bottoms and buried interfaces are not mouths). Returns the port dict
    plus the names of the single-face ports it absorbed."""
    import numpy as np

    covering = [r for r in recs if r["s_min"] - 0.5 <= s_end <= r["s_max"] + 0.5]
    radii = sorted({round(r["radius"], 3) for r in covering})
    if len(radii) < 2:
        return None, set()
    bore_rs = [round(r["radius"], 3) for r in covering if not r["outward"]]
    if not bore_rs:
        return None, set()
    # rim semantics: the annulus is the MATERIAL ring — inner = the void
    # opening at this plane, outer = the nearest wall surface beyond it.
    # A nested spigot inside the pocket is a separate port, never the
    # rim's inner edge.
    r_in = max(bore_rs)
    outer = [rr for rr in radii if rr > r_in]
    if not outer:
        return None, set()  # bore is the outermost surface — no wall rim
    r_out = outer[0]
    out_dir = None
    for od in out_dirs:
        probe = o + d * s_end + od * 1.0 + rad * (0.5 * (r_in + r_out))
        try:
            if not inside(probe):
                out_dir = od
                break
        except Exception:
            continue
    if out_dir is None:
        return None, set()
    # the end port absorbs every covering bore and the outermost wall —
    # a male face that is NOT the outermost ring (a spigot nested inside
    # a counterbore) keeps its distinct port
    used_names = {
        r["port_name"]
        for r in covering
        if not r["outward"] or round(r["radius"], 3) == r_out
    }
    bores = [r for r in recs if not r["outward"]]
    smallest = min(bores, key=lambda r: r["radius"])
    through = smallest["s_min"] <= s_lo + 0.5 and smallest["s_max"] >= s_hi - 0.5
    port = {
        "name": f"end_{idx}",
        "kind": "cylinder",
        "pos": [round(float(v), 4) for v in (o + d * s_end)],
        "axis": [round(float(v), 4) for v in out_dir],
        "radius": round(r_in, 4),
        "depth": 500.0 if through else round(float(smallest["s_max"] - smallest["s_min"]), 4),
        "gender": "female",
        "inner_radius": round(r_in, 4),
        "outer_radius": round(r_out, 4),
        "area": round(float(np.pi * (r_out**2 - r_in**2)), 2),
    }
    return port, used_names


async def extract_ports(ctx: dict, payload: dict[str, Any]) -> dict[str, Any]:
    """arq task — {step_path} -> {ports, bbox, mesh_path (GLB/STL)}."""
    return await asyncio.to_thread(_extract_ports_sync, payload)


def _extract_ports_sync(payload: dict[str, Any]) -> dict[str, Any]:
    import build123d as bd

    step = Path(str(payload["step_path"]))
    if not step.exists():
        return {"ok": False, "error": f"step not found: {step}"}
    try:
        if step.suffix.lower() == ".brep":
            shape = bd.Shape.import_brep(str(step))
        else:
            shape = bd.import_step(str(step))
        if isinstance(shape, (list, tuple)):
            shape = bd.Compound(list(shape)) if len(shape) > 1 else shape[0]
        bb = shape.bounding_box()
        out_dir = step.parent
        # solver-frame mesh: the guide scene/collision/render consumers and
        # part ports all share the BRep frame (Z-up mm). bd.export_gltf
        # would write meters + a -90degX up-axis node (glTF spec) — served
        # as-is to those consumers the part lands 1000x too small and
        # rotated. .guide.glb keeps the version's own model.glb artifact
        # (glTF-correct) untouched.
        mesh_path = out_dir / f"{step.stem}.guide.glb"
        try:
            from cadx.cad.exporters import export_glb_file

            export_glb_file(shape, mesh_path)
        except Exception:
            mesh_path = out_dir / f"{step.stem}.stl"
            bd.export_stl(shape, str(mesh_path))
        return {
            "ok": True,
            "ports": _face_ports(shape),
            "bbox_min": [round(float(v), 4) for v in (bb.min.X, bb.min.Y, bb.min.Z)],
            "bbox_max": [round(float(v), 4) for v in (bb.max.X, bb.max.Y, bb.max.Z)],
            "mesh_path": str(mesh_path),
        }
    except Exception as exc:
        log.warning("guide.extract_ports.failed", step=str(step), err=str(exc))
        return {"ok": False, "error": str(exc)[:300]}


async def render_assembly(ctx: dict, payload: dict[str, Any]) -> dict[str, Any]:
    """arq task — {entries:[{mesh_path, transform, color, opacity}], view,
    out_path} -> {image_path}."""
    return await asyncio.to_thread(_render_assembly_sync, payload)


def _render_assembly_sync(payload: dict[str, Any]) -> dict[str, Any]:
    from cadx.cad.render import render_available, render_scene

    if not render_available():
        return {"ok": False, "error": "render unavailable in executor"}
    out = render_scene(
        payload.get("entries") or [],
        Path(str(payload["out_path"])),
        view=str(payload.get("view") or "iso"),
        size=int(payload.get("size") or 1024),
    )
    return {"ok": bool(out), "image_path": str(out) if out else ""}
