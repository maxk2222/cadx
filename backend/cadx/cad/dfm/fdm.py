"""FDM-specific DfM checks."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from build123d import GeomType, Shape

from cadx.cad.analysis import Analysis
from cadx.cad.dfm.common import wall_thickness_stats
from cadx.cad.dfm.report import DfmWarning

OVERHANG_COS = math.sin(math.radians(40))  # faces > ~50° from vertical


def overhang_ratio(mesh: trimesh.Trimesh, direction: str = "+z") -> float:
    """Fraction of surface area facing >~50° down from the build direction.

    Same metric as the FDM overhang DFM check, parameterised on the print
    direction (`+z` plate below the part, `-z` plate above) — used to score
    TopOpt candidates before the DFM report even runs."""
    ax = {"x": 0, "y": 1, "z": 2}[direction[1]]
    sgn = 1.0 if direction[0] == "+" else -1.0
    plate = float(mesh.bounds[0][ax] if sgn > 0 else mesh.bounds[1][ax])
    normals = mesh.face_normals
    centers = mesh.triangles_center
    mask = (normals[:, ax] * sgn < -OVERHANG_COS) & (sgn * (centers[:, ax] - plate) > 0.5)
    return float(mesh.area_faces[mask].sum()) / max(mesh.area, 1e-9)


def fdm_mesh_checks(
    mesh: trimesh.Trimesh | None,
    analysis: Analysis,
    *,
    min_wall_mm: float = 0.8,
    nozzle_mm: float = 0.4,
) -> list[DfmWarning]:
    """FDM checks that only need a triangle mesh (usable for mesh-only versions)."""
    warnings: list[DfmWarning] = []
    if mesh is None:
        return warnings

    # thin walls: flag only when a meaningful fraction of the surface is thin
    # (edge/corner samples produce grazing rays with false small thickness)
    stats = wall_thickness_stats(mesh, min_wall_mm)
    if stats is not None:
        p5, frac = stats
        if p5 < min_wall_mm and frac > 0.15:
            warnings.append(
                DfmWarning(
                    code="thin_wall",
                    severity="high" if frac > 0.5 else "medium",
                    message=(
                        f"{frac * 100:.0f}% of surface is thinner than {min_wall_mm} mm "
                        f"(min ~{p5:.2f} mm, ~{min_wall_mm / max(nozzle_mm, 0.01):.0f} perimeters)"
                    ),
                    value=p5,
                    threshold=min_wall_mm,
                )
            )

    # overhangs: downward-facing triangles above the build plate
    normals = mesh.face_normals
    centers = mesh.triangles_center
    mask = (normals[:, 2] < -OVERHANG_COS) & (centers[:, 2] > 0.5)
    overhang_area = float(mesh.area_faces[mask].sum())
    ratio = overhang_area / max(mesh.area, 1e-9)
    if ratio > 0.05:
        warnings.append(
            DfmWarning(
                code="overhang",
                severity="high" if ratio > 0.2 else "medium",
                message=(
                    f"{ratio * 100:.0f}% of surface is overhang > 50deg from vertical "
                    "- needs support or reorientation"
                ),
                value=round(ratio * 100, 1),
                threshold=5.0,
            )
        )

    # bridges: horizontal downward faces far above the plate
    mask2 = (normals[:, 2] < -0.9) & (centers[:, 2] > 1.0)
    if mask2.any():
        tris = mesh.triangles[mask2]
        for tri in tris[:64]:
            span = float(np.ptp(tri[:, :2], axis=0).max())
            if span > 20.0:
                warnings.append(
                    DfmWarning(
                        code="bridge",
                        severity="medium",
                        message=f"horizontal span ~{span:.0f} mm may need support/bridging",
                        value=round(span, 1),
                        threshold=20.0,
                    )
                )
                break
    return warnings


def _looks_like_small_hole(shape: Shape, face: object, r: float) -> bool:
    """A small cylindrical face counts as a hole only when its axis sits in
    void surrounded by material — inside-fraction of a ring sampled around
    the axis >= 0.7 (same discriminator as the feature extractor). Concave
    fillet arcs score lower and are skipped. Classification failure keeps
    the advisory warning rather than silently dropping it."""
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        from cadx.cad.features import _inside_fraction

        cyl = BRepAdaptor_Surface(face.wrapped).Cylinder()  # type: ignore[attr-defined]
        a = cyl.Axis().Direction()
        axis = [a.X(), a.Y(), a.Z()]
        n = math.sqrt(sum(c * c for c in axis)) or 1.0
        axis = [c / n for c in axis]
        loc = cyl.Axis().Location()
        fc = face.center()  # type: ignore[attr-defined]
        d = [fc.X - loc.X(), fc.Y - loc.Y(), fc.Z - loc.Z()]
        par = sum(d[i] * axis[i] for i in range(3))
        on_axis = [loc.X() + par * axis[0], loc.Y() + par * axis[1], loc.Z() + par * axis[2]]
        return _inside_fraction(shape, on_axis, axis, r) >= 0.7
    except Exception:
        return True


def fdm_brep_checks(
    shape: Shape,
    analysis: Analysis,
    *,
    thread_count: int = 0,
    thread_compensation_mm: float = 0.0,
) -> list[DfmWarning]:
    """FDM checks that need analytic B-Rep features (holes, threads)."""
    warnings: list[DfmWarning] = []

    # small / horizontal holes from B-Rep
    for f in shape.faces():
        try:
            if f.geom_type != GeomType.CYLINDER:
                continue
            r = float(f.radius or 0.0)
            # r==0 is a degenerate adaptor read, not a hole; and concave
            # fillet arcs (gear tooth roots, inside corners) are not holes —
            # their axis runs through mostly open space
            if 0.0 < r < 1.0 and _looks_like_small_hole(shape, f, r):
                warnings.append(
                    DfmWarning(
                        code="small_hole",
                        severity="low",
                        message=f"hole radius {r:.2f} mm < 1 mm may not print cleanly",
                        value=r,
                        threshold=1.0,
                    )
                )
        except Exception:
            continue
    for cf in analysis.cylindrical_faces:
        if abs(cf.axis[2]) < 0.3 and cf.radius > 3.0:
            warnings.append(
                DfmWarning(
                    code="horizontal_hole",
                    severity="low",
                    message=(
                        f"horizontal hole r={cf.radius:.2f} mm prints as an ellipse on top "
                        "- consider teardrop profile or reorientation"
                    ),
                )
            )
            break

    if thread_count > 0 and thread_compensation_mm == 0:
        warnings.append(
            DfmWarning(
                code="printed_thread",
                severity="low",
                message=(
                    "real threads modelled but thread_compensation_mm is 0 - "
                    "calibrate for your printer/material (typical 0.1-0.3 mm)"
                ),
            )
        )
    return warnings


def fdm_checks(
    shape: Shape,
    mesh: trimesh.Trimesh | None,
    analysis: Analysis,
    *,
    min_wall_mm: float = 0.8,
    nozzle_mm: float = 0.4,
    thread_count: int = 0,
    thread_compensation_mm: float = 0.0,
) -> list[DfmWarning]:
    return fdm_mesh_checks(
        mesh, analysis, min_wall_mm=min_wall_mm, nozzle_mm=nozzle_mm
    ) + fdm_brep_checks(
        shape,
        analysis,
        thread_count=thread_count,
        thread_compensation_mm=thread_compensation_mm,
    )
