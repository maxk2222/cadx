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
            if r < 1.0:
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
