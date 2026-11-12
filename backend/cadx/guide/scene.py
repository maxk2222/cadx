"""Assembly scene — worker-side mesh loading + manifold3d collision checks.

Meshes come from the parts' GLB assets (executor-produced for CAD parts,
uploaded directly for mesh uploads). Boolean intersection uses the same
manifold3d backend as ``cad/mesh_boolean.py`` — trimesh's FCL collision
module is NOT a dependency here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

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


def _to_manifold_mesh(man) -> object:
    import trimesh

    m = man.to_mesh()
    return trimesh.Trimesh(vertices=m.vert_properties[:, :3], faces=m.tri_verts)


class _PlacedMesh:
    __slots__ = ("iid", "mesh")

    def __init__(self, iid: str, mesh) -> None:
        self.iid = iid
        self.mesh = mesh


class Scene:
    """Instances → transformed trimeshes; AABB prefilter + manifold boolean."""

    def __init__(self) -> None:
        self._placed: dict[str, _PlacedMesh] = {}

    def clear(self) -> None:
        self._placed.clear()

    def add(self, iid: str, mesh_path: str | Path, transform: list[list[float]]) -> None:
        mesh = _load_mesh(mesh_path)
        mesh.apply_transform(np.asarray(transform, dtype=float).reshape(4, 4))
        self._placed[iid] = _PlacedMesh(iid, mesh)

    def remove(self, iid: str) -> None:
        self._placed.pop(iid, None)

    def collision_report(
        self,
        iid: str,
        mesh_path: str | Path,
        transform: list[list[float]],
    ) -> dict:
        """Check a candidate transform against all placed meshes.

        AABB prefilter → manifold3d boolean intersection → volume verdict.
        """
        probe = _load_mesh(mesh_path)
        probe.apply_transform(np.asarray(transform, dtype=float).reshape(4, 4))
        pmin, pmax = probe.bounds
        hits: list[dict] = []
        for other in self._placed.values():
            omin, omax = other.mesh.bounds
            if not (np.all(pmax >= omin - 1e-6) and np.all(pmin <= omax + 1e-6)):
                continue
            try:
                inter = _manifold(probe).__mul__(_manifold(other.mesh))
                vol = float(inter.volume) if inter is not None else 0.0
            except Exception as exc:  # degenerate meshes → warn, not crash
                log.warning("guide.collision.bool_failed", iid=iid, other=other.iid, err=str(exc))
                vol = -1.0
            if vol > _EPS_VOL_MM3 or vol < 0:
                hits.append(
                    {
                        "other": other.iid,
                        "volume_mm3": round(max(vol, 0.0), 4),
                        "check_failed": vol < 0,
                    }
                )
        # depth proxy: max overlap of AABBs among hits (cheap, honest label)
        return {
            "ok": not hits,
            "collisions": hits,
            "bbox_overlap_mm": round(
                max(
                    (
                        float(np.min(np.minimum(pmax, omax_h) - np.maximum(pmin, omin_h)))
                        for o in self._placed.values()
                        for omin_h, omax_h in [o.mesh.bounds]
                        if any(h["other"] == o.iid for h in hits)
                    ),
                    default=0.0,
                ),
                4,
            ),
        }

    def all_pairs_collision_report(self) -> dict:
        hits: list[dict] = []
        items = list(self._placed.values())
        for i, a in enumerate(items):
            amin, amax = a.mesh.bounds
            for b in items[i + 1 :]:
                bmin, bmax = b.mesh.bounds
                if not (np.all(amax >= bmin - 1e-6) and np.all(amin <= bmax + 1e-6)):
                    continue
                try:
                    inter = _manifold(a.mesh).__mul__(_manifold(b.mesh))
                    vol = float(inter.volume) if inter is not None else 0.0
                except Exception as exc:
                    log.warning("guide.collision.bool_failed", a=a.iid, b=b.iid, err=str(exc))
                    vol = -1.0
                if vol > _EPS_VOL_MM3 or vol < 0:
                    hits.append(
                        {
                            "a": a.iid,
                            "b": b.iid,
                            "volume_mm3": round(max(vol, 0.0), 4),
                            "check_failed": vol < 0,
                        }
                    )
        return {"ok": not hits, "collisions": hits}
