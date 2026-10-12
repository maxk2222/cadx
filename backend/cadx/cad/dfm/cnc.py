"""3-axis CNC DfM checks."""

from __future__ import annotations

import trimesh
from build123d import GeomType, Shape

from cadx.cad.analysis import Analysis
from cadx.cad.dfm.common import wall_thickness_stats
from cadx.cad.dfm.report import DfmWarning

UNDERCUT_NZ = -0.1


def _adjacent_faces(shape: Shape, edge: object) -> list:
    wrapped = edge.wrapped  # type: ignore[attr-defined]
    return [f for f in shape.faces() if any(wrapped.IsSame(fe.wrapped) for fe in f.edges())]


def _vertical_concave_line_edges(shape: Shape) -> list:
    """LINE edges parallel to Z with internal angle > 180deg (concave).

    Concavity test (verified empirically): for the two adjacent faces with
    outward normals n1, n2 and edge tangent t, cross(n1,n2)·t > 0 means concave.
    """
    found = []
    for e in shape.edges():
        try:
            if e.geom_type != GeomType.LINE:
                continue
            t = e.tangent_at(0.5)
            if abs(t.Z) < 0.99:
                continue
            adj = _adjacent_faces(shape, e)
            if len(adj) < 2:
                continue
            mid = e.position_at(0.5)
            n1 = adj[0].normal_at(mid)
            n2 = adj[1].normal_at(mid)
            if n1 is None or n2 is None:
                continue
            if n1.cross(n2).dot(t) > 0.5:
                found.append(e)
        except Exception:
            continue
    return found


def _deep_pockets(shape: Shape, max_z: float) -> list[tuple[float, float]]:  # noqa: C901
    """Planar faces facing +Z below the top, enclosed by near-vertical faces.

    Returns (depth, min_width) for each candidate.
    """
    results = []
    for f in shape.faces():
        try:
            if f.geom_type != GeomType.PLANE:
                continue
            n = f.normal_at()
            if n is None or n.Z < 0.99:
                continue
            z = float(f.center().Z)
            if z >= max_z - 0.5:
                continue
            # enclosure: all neighbouring faces across boundary edges are near-vertical
            neighbours = set()
            for e in f.edges():
                for other in _adjacent_faces(shape, e):
                    if other is not f:
                        neighbours.add(other)
            if not neighbours:
                continue
            vertical = 0
            total = 0
            for o in neighbours:
                try:
                    on = o.normal_at()
                    if on is None:
                        total += 1
                        continue
                    total += 1
                    if abs(on.Z) < 0.3:
                        vertical += 1
                except Exception:
                    total += 1
            if total == 0 or vertical / total < 0.6:
                continue
            depth = max_z - z
            fb = f.bounding_box()
            width = min(fb.size.X, fb.size.Y)
            if width > 0:
                results.append((depth, width))
        except Exception:
            continue
    return results


def cnc_mesh_checks(
    mesh: trimesh.Trimesh | None,
    analysis: Analysis,
) -> list[DfmWarning]:
    """CNC checks that only need a triangle mesh (usable for mesh-only versions)."""
    warnings: list[DfmWarning] = []
    if mesh is None:
        return warnings

    normals = mesh.face_normals
    centers = mesh.triangles_center
    z_max = analysis.bbox_max[2]
    mask = (normals[:, 2] < UNDERCUT_NZ) & (centers[:, 2] > 0.5) & (centers[:, 2] < z_max - 0.2)
    undercut_ratio = float(mesh.area_faces[mask].sum()) / max(mesh.area, 1e-9)
    if undercut_ratio > 0.01:
        warnings.append(
            DfmWarning(
                code="undercut",
                severity="high",
                message=(
                    f"{undercut_ratio * 100:.0f}% of surface not reachable from +Z "
                    "(3-axis setup) - consider two-sided machining or redesign"
                ),
                value=round(undercut_ratio * 100, 1),
                threshold=1.0,
            )
        )

    stats = wall_thickness_stats(mesh, 1.0)
    if stats is not None:
        p5, frac = stats
        if p5 < 1.0 and frac > 0.15:
            warnings.append(
                DfmWarning(
                    code="thin_wall",
                    severity="medium",
                    message=(
                        f"thinnest regions ~{p5:.2f} mm < 1 mm - fragile under cutting forces"
                    ),
                    value=p5,
                    threshold=1.0,
                )
            )
    return warnings


def cnc_brep_checks(
    shape: Shape,
    analysis: Analysis,
    *,
    tool_radius_mm: float = 1.5,
    thread_count: int = 0,
) -> list[DfmWarning]:
    """CNC checks that need analytic B-Rep features (sharp corners, pockets)."""
    warnings: list[DfmWarning] = []

    sharp = _vertical_concave_line_edges(shape)
    if sharp:
        warnings.append(
            DfmWarning(
                code="sharp_internal_corner",
                severity="high",
                message=(
                    f"{len(sharp)} sharp internal vertical corner(s) - "
                    f"add fillet >= tool radius ({tool_radius_mm} mm)"
                ),
                value=float(len(sharp)),
                threshold=tool_radius_mm,
            )
        )

    for depth, width in _deep_pockets(shape, float(analysis.bbox_max[2])):
        ratio = depth / width
        if ratio > 4.0:
            warnings.append(
                DfmWarning(
                    code="deep_pocket",
                    severity="medium",
                    message=f"pocket depth/width ratio {ratio:.1f}:1 (> 4:1) - hard to machine",
                    value=round(ratio, 1),
                    threshold=4.0,
                )
            )
            break

    if thread_count > 0:
        warnings.append(
            DfmWarning(
                code="modelled_thread",
                severity="low",
                message=(
                    "real thread geometry modelled - machinists usually tap threads; "
                    "prefer cosmetic thread + tap note in the drawing"
                ),
            )
        )
    return warnings


def cnc_checks(
    shape: Shape,
    mesh: trimesh.Trimesh | None,
    analysis: Analysis,
    *,
    tool_radius_mm: float = 1.5,
    thread_count: int = 0,
) -> list[DfmWarning]:
    return cnc_mesh_checks(mesh, analysis) + cnc_brep_checks(
        shape, analysis, tool_radius_mm=tool_radius_mm, thread_count=thread_count
    )
