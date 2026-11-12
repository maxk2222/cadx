"""Guide-builder executor tasks — run on the cad-light lane (arq).

- ``extract_ports``: STEP/BRep → deterministic attachment ports (cylinder +
  planar faces, radius/axis/gender) + a GLB tessellation for scene
  collision checks and rendering. OCC work is executor-only.
- ``render_assembly``: multi-mesh scene → off-screen PNG via
  ``cad.render.render_scene``.

Worker calls these through ``QueueExecutor.run_task`` (same facade as
execute_version/audit_params).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MAX_PORTS = 24
_MIN_PLANAR_AREA_FRAC = 0.01


def _face_ports(shape) -> list[dict[str, Any]]:
    """Exact attachment ports from BRep faces (build123d/OCP).

    Cylinder faces → {pos on axis, axis dir, radius, gender male/female};
    planar faces → {pos center, axis normal, gender flat}. Gender by
    whether the face normal points away from the cylinder axis (boss) or
    toward it (bore).
    """
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType

    faces = list(shape.faces())
    total_area = sum(float(f.area) for f in faces) or 1.0
    ports: list[dict[str, Any]] = []

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
                origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
                c = np.array([center.X, center.Y, center.Z], dtype=float)
                # pos = face center projected onto the cylinder axis
                pos = origin + axis * float(np.dot(c - origin, axis))
                radial = c - pos
                rn = float(np.linalg.norm(radial))
                outward = float(np.dot(n, radial / rn)) > 0.0 if rn > 1e-9 else True
                ports.append(
                    {
                        "name": f"cyl_{idx}",
                        "kind": "cylinder",
                        "pos": [round(float(v), 4) for v in pos],
                        "axis": [round(float(v), 4) for v in axis],
                        "radius": round(float(cyl.Radius()), 4),
                        "depth": round(float(face.area / max(2 * 3.14159 * cyl.Radius(), 1e-9)), 4),
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
    return ports[:_MAX_PORTS]


def extract_ports(ctx: dict, payload: dict[str, Any]) -> dict[str, Any]:
    """arq task — {step_path} → {ports, bbox, mesh_path (GLB/STL)}."""
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
        mesh_path = out_dir / f"{step.stem}.glb"
        try:
            bd.export_gltf(shape, str(mesh_path), binary=True)
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


def render_assembly(ctx: dict, payload: dict[str, Any]) -> dict[str, Any]:
    """arq task — {entries:[{mesh_path, transform, color, opacity}], view,
    out_path} → {image_path}."""
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
