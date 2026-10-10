"""Output mesh cleanup: floaters, normals, hole filling, GLB export."""

from __future__ import annotations

import numpy as np
import trimesh


def clean_mesh(mesh: trimesh.Trimesh, min_component_frac: float = 0.01) -> trimesh.Trimesh:
    try:
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            vols = np.array([max(abs(b.volume), 1e-12) for b in bodies])
            keep = vols >= vols.max() * min_component_frac
            bodies = [b for b, k in zip(bodies, keep, strict=True) if k] or [
                bodies[int(vols.argmax())]
            ]
            mesh = trimesh.util.concatenate(bodies)
    except Exception:
        pass
    for fn in (
        mesh.merge_vertices,
        mesh.remove_unreferenced_vertices,
        mesh.remove_infinite_values,
        mesh.fix_normals,
        lambda: trimesh.repair.fill_holes(mesh),
    ):
        try:
            fn()
        except Exception:
            pass
    return mesh


def export_glb(mesh: trimesh.Trimesh) -> bytes:
    """GLB bytes for the /v1/jobs/{id}/mesh download."""
    import io

    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()
