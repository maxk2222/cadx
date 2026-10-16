"""Optional PyMeshLab repair engine for the mesh-ops pipeline.

PyMeshLab (MeshLab bindings, GPL) is an *optional* executor dependency —
this module never imports it eagerly, `pm_available()` is the cheap cached
probe callers gate on, and every function raises RuntimeError when the
package is missing so a caller that skipped the probe still fails loudly.

All functions take and return trimesh.Trimesh; pymeshlab filter errors
propagate to the caller (the mesh_ops ladder catches and falls through to
the next repair stage).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from cadx.logging import get_logger

log = get_logger(__name__)

# default alpha-wrap granularity: alpha = bbox diagonal / this — equivalent
# to a ~200-samples-per-axis rebuild, so callers can compare it against the
# shared rebuild-resolution budget (_VOXEL_REPAIR_MAX_CELLS in mesh_ops)
ALPHA_WRAP_DEFAULT_DIV = 200.0

_pml: Any = None
_pml_tried = False


def _pm() -> Any:
    """Import pymeshlab once; returns the module or None (same cached-lazy
    pattern as `_rembg` in silhouette.py)."""
    global _pml, _pml_tried
    if _pml is None and not _pml_tried:
        _pml_tried = True
        try:
            import pymeshlab as pml

            _pml = pml
        except Exception:
            _pml = None
    return _pml


def pm_available() -> bool:
    """True when pymeshlab is installed and importable. Never raises."""
    return _pm() is not None


def _require() -> Any:
    pml = _pm()
    if pml is None:
        raise RuntimeError("pymeshlab is not installed — install the 'repair' extra")
    return pml


def _to_meshset(mesh: trimesh.Trimesh) -> Any:
    pml = _require()
    ms = pml.MeshSet()
    ms.add_mesh(
        pml.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
    )
    return ms


def _from_meshset(ms: Any) -> trimesh.Trimesh:
    cm = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(cm.vertex_matrix()),
        faces=np.asarray(cm.face_matrix()),
        process=False,
    )


def _require_output(out: trimesh.Trimesh, op: str) -> trimesh.Trimesh:
    if not len(out.faces) or not len(out.vertices):
        raise ValueError(f"pymeshlab {op} produced an empty mesh")
    return out


def _bbox_diag(mesh: trimesh.Trimesh) -> float:
    b = np.asarray(mesh.bounds, dtype=float)
    return float(np.linalg.norm(b[1] - b[0]))


def _pct(pml: Any, length_mm: float, diag_mm: float) -> Any:
    """Absolute length (mm) -> pymeshlab PercentageValue of the bbox diagonal."""
    return pml.PercentageValue(float(length_mm) / max(diag_mm, 1e-9) * 100.0)


def alpha_wrap(
    mesh: trimesh.Trimesh,
    alpha: float | None = None,
    offset: float | None = None,
) -> trimesh.Trimesh:
    """Watertight, 2-manifold reskin via the alpha-wrap filter.

    `alpha` is the wrap granularity in mesh units — default bbox diagonal /
    ALPHA_WRAP_DEFAULT_DIV (finer than the filter's built-in 2%, still
    bounded by the diagonal so wrap complexity is scale-relative, not
    input-face-relative). `offset` shrinks the result back onto the surface
    — default alpha/10. The output is guaranteed watertight and manifold by
    construction; multi-component inputs wrap per component (caller keeps
    the dominant body).
    """
    pml = _require()
    diag = max(_bbox_diag(mesh), 1e-9)
    a = diag / ALPHA_WRAP_DEFAULT_DIV if alpha is None else float(alpha)
    o = a / 10.0 if offset is None else float(offset)
    ms = _to_meshset(mesh)
    ms.generate_alpha_wrap(alpha=_pct(pml, a, diag), offset=_pct(pml, o, diag))
    out = _require_output(_from_meshset(ms), "alpha_wrap")
    log.debug("mesh.pml.alpha_wrap", faces_in=len(mesh.faces), faces_out=len(out.faces))
    return out


def isotropic_remesh(mesh: trimesh.Trimesh, target_len_mm: float) -> trimesh.Trimesh:
    """Uniform-edge remesh (isotropic explicit remeshing).

    `target_len_mm` is the desired edge length in mesh units — the pml
    replacement for the voxel remesh: keeps sharp detail the marching-cubes
    grid smears away and stays watertight when the input is closed.
    """
    pml = _require()
    if target_len_mm <= 0:
        raise ValueError(f"target_len_mm must be > 0, got {target_len_mm}")
    ms = _to_meshset(mesh)
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=_pct(pml, float(target_len_mm), _bbox_diag(mesh)),
        iterations=5,
        checksurfdist=True,
    )
    out = _require_output(_from_meshset(ms), "isotropic_remesh")
    log.debug("mesh.pml.remesh", faces_in=len(mesh.faces), faces_out=len(out.faces))
    return out


def quadric_decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation to ~target_faces (topology preserved).

    Higher quality than the fast_simplification/trimesh path — quadrics are
    evaluated on the full mesh and boundary/topology are respected."""
    _require()
    if target_faces <= 0:
        raise ValueError(f"target_faces must be > 0, got {target_faces}")
    if len(mesh.faces) <= target_faces:
        return mesh
    ms = _to_meshset(mesh)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=True,
        preservenormal=True,
        optimalplacement=True,
    )
    out = _require_output(_from_meshset(ms), "quadric_decimate")
    log.debug("mesh.pml.decimate", faces_in=len(mesh.faces), faces_out=len(out.faces))
    return out


def taubin_smooth(mesh: trimesh.Trimesh, steps: int = 10) -> trimesh.Trimesh:
    """Volume-preserving Taubin smoothing (lambda/mu two-step shrink+correct)."""
    _require()
    if steps <= 0:
        return mesh
    ms = _to_meshset(mesh)
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=int(steps))
    return _require_output(_from_meshset(ms), "taubin_smooth")


def close_holes(mesh: trimesh.Trimesh, max_hole_edges: int = 100) -> trimesh.Trimesh:
    """Triangulated fill for boundary loops up to `max_hole_edges` edges.

    Bigger than trimesh's fill_holes threshold — the point is closing
    generator output, not preserving intentional openings, so openings
    larger than the cap are left alone (caller decides whether the mesh
    must be watertight)."""
    _require()
    ms = _to_meshset(mesh)
    ms.meshing_close_holes(maxholesize=int(max_hole_edges))
    return _require_output(_from_meshset(ms), "close_holes")


def remove_non_manifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair non-manifold edges + vertices (edge faces removed, verts split).

    Produces a strictly 2-manifold complex — may open boundaries where
    offending geometry was removed; pair with close_holes when watertight
    output is required."""
    _require()
    ms = _to_meshset(mesh)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    return _require_output(_from_meshset(ms), "remove_non_manifold")
