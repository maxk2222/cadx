"""Shared DfM helpers + common checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from cadx.cad.analysis import Analysis
from cadx.cad.dfm.report import DfmWarning


def load_mesh(stl_path: Path) -> trimesh.Trimesh | None:
    try:
        m = trimesh.load(str(stl_path))
        return m if isinstance(m, trimesh.Trimesh) else None
    except Exception:
        return None


def common_checks(
    analysis: Analysis, mesh: trimesh.Trimesh | None, build_volume: list[float]
) -> list[DfmWarning]:
    warnings: list[DfmWarning] = []
    if analysis.solid_count != 1:
        warnings.append(
            DfmWarning(
                code="multiple_solids",
                severity="medium",
                message=f"part has {analysis.solid_count} disconnected solids",
            )
        )
    if analysis.watertight is False:
        warnings.append(
            DfmWarning(
                code="not_watertight",
                severity="high",
                message="mesh is not watertight - may fail slicing/toolpath generation",
            )
        )
    size = analysis.bbox_size
    for i, axis in enumerate("XYZ"):
        if size[i] > build_volume[i]:
            warnings.append(
                DfmWarning(
                    code="exceeds_build_volume",
                    severity="high",
                    message=f"part exceeds build volume on {axis}: {size[i]:.1f} > {build_volume[i]:.1f} mm",
                    value=size[i],
                    threshold=build_volume[i],
                )
            )
    if min(size) < 2.0:
        warnings.append(
            DfmWarning(
                code="tiny_part",
                severity="low",
                message=f"smallest dimension is {min(size):.2f} mm (< 2 mm)",
                value=min(size),
                threshold=2.0,
            )
        )
    return warnings


def sample_surface(mesh: trimesh.Trimesh, count: int = 2000) -> np.ndarray:
    pts: np.ndarray = trimesh.sample.sample_surface(mesh, count)[0]
    return pts


def wall_thickness_stats(mesh: trimesh.Trimesh, min_wall: float) -> tuple[float, float] | None:
    """(5th-percentile thickness, fraction of samples thinner than min_wall).

    Edge/corner samples produce grazing hits and false small values, so a
    warning should require BOTH a low percentile and a meaningful fraction.

    thickness() queries every sample against the face soup — quadratic on
    dense meshes — so the probe runs on a ~20k-face decimated copy (the
    warning-level check doesn't need full fidelity).
    """
    try:
        probe = mesh
        if len(mesh.faces) > 20_000:
            probe = mesh.simplify_quadric_decimation(face_count=20_000)
            if not probe.is_watertight:  # decimation may open the surface
                probe = mesh.copy()
                probe.update_faces(probe.nondegenerate_faces())
        pts = sample_surface(probe, 2000)
        th = trimesh.proximity.thickness(probe, pts, exterior=False)
        p5 = float(np.percentile(th, 5))
        frac = float((th < min_wall).mean())
        return p5, frac
    except Exception:
        return None
