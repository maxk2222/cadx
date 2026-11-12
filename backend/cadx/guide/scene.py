"""Assembly scene — worker-side mesh loading + manifold3d collision checks.

Meshes come from the parts' GLB assets (executor-produced for CAD parts,
uploaded directly for mesh uploads). Boolean intersection uses the same
manifold3d backend as ``cad/mesh_boolean.py`` — trimesh's FCL collision
module is NOT a dependency here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cadx.logging import get_logger

log = get_logger(__name__)

_EPS_VOL_MM3 = 1.0  # collision verdict threshold


def _load_mesh(path: str | Path):
    import trimesh  # local — heavy dep, worker-side only when needed

    scene = trimesh.load(str(path))
    return scene.to_geometry() if hasattr(scene, "to_geometry") else scene


def _manifold(mesh):
    from manifold3d import Manifold, Mesh

    return Manifold(
        mesh=Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )


def _intersection_volume(a, b) -> float:
    """Boolean-intersection volume in mm3 — manifold3d has no ``*`` sugar in
    this version; batch_boolean is the supported pairwise call."""
    from manifold3d import Manifold, OpType

    inter = Manifold.batch_boolean([_manifold(a), _manifold(b)], OpType.Intersect)
    return float(inter.volume()) if inter is not None else 0.0
