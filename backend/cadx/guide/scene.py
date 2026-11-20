"""Assembly scene — worker-side mesh loading + manifold3d collision checks.

Meshes come from the parts' GLB assets (executor-produced for CAD parts,
uploaded directly for mesh uploads). Boolean intersection uses the same
manifold3d backend as ``cad/mesh_boolean.py`` — trimesh's FCL collision
module is NOT a dependency here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cadx.logging import get_logger

log = get_logger(__name__)

_EPS_VOL_MM3 = 1.0  # collision verdict threshold
_MESH_CACHE_MAX = 32  # parsed GLBs kept per Scene — bound, LRU-evicted

# deterministic per-instance colors — shared by mission renders and the
# final emit pass so a part looks the same in both
PALETTE = ["#7aa2f7", "#9ece6a", "#e0af68", "#f7768e", "#bb9af7", "#4dc3ff", "#ff9e64"]


def _load_mesh(path: str | Path):
    import trimesh  # local — heavy dep, worker-side only when needed

    scene = trimesh.load(str(path))
    return scene.to_geometry() if hasattr(scene, "to_geometry") else scene


# Z-up CAD world -> glTF Y-up basis change (proper rotation, -90degX).
# Baked into the exported artifact so the assembly stands upright in ANY
# glTF consumer (three.js, Unity import, OS previews) — trimesh's glTF
# export writes vertex data as-is and emits no axis-convention node, so
# a raw Z-up scene would lie on its side in every viewer.
_ZUP_TO_GLTF = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def merged_glb(
    placements: list[tuple[str, str | Path, list]],
    out_path: str | Path,
    *,
    world_frame: str = "z_up",
) -> Path | None:
    """Combine placed instances into one GLB — ``placements`` is
    [(name, mesh_path, transform_4x4)] expressed in ``world_frame``
    (``GuideState.world_frame``: "z_up" for the CAD route, "y_up" for
    blender mode). Deterministic: geometry order and node names follow
    the placement order. Returns None when nothing loadable remains."""
    import trimesh

    conv = _ZUP_TO_GLTF if world_frame == "z_up" else None
    scene = trimesh.Scene()
    for name, mesh_path, transform in placements:
        p = Path(mesh_path)
        if not p.exists():
            continue
        mesh = _load_mesh(p)
        t = np.asarray(transform, dtype=float).reshape(4, 4)
        mesh.apply_transform(t if conv is None else conv @ t)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    if not scene.geometry:
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    return out


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


class _PlacedMesh:
    __slots__ = ("iid", "mesh")

    def __init__(self, iid: str, mesh) -> None:
        self.iid = iid
        self.mesh = mesh


class Scene:
    """Instances → transformed trimeshes; AABB prefilter + manifold boolean."""

    def __init__(self) -> None:
        self._placed: dict[str, _PlacedMesh] = {}
        self._mesh_cache: dict[str, tuple[float, Any]] = {}

    def clear(self) -> None:
        self._placed.clear()

    def _base_mesh(self, mesh_path: str | Path) -> Any:
        """Pristine parsed mesh for ``mesh_path`` — cached by resolved path
        and re-parsed on mtime change (a normalize/re-export pass rewrites
        the file in place). Callers must transform a ``.copy()`` so the
        cached base stays clean for the next probe."""
        p = Path(mesh_path).resolve()
        mtime = p.stat().st_mtime if p.is_file() else -1.0
        key = str(p)
        entry = self._mesh_cache.get(key)
        if entry is None or entry[0] != mtime:
            entry = (mtime, _load_mesh(p))
            self._mesh_cache[key] = entry
            while len(self._mesh_cache) > _MESH_CACHE_MAX:
                self._mesh_cache.pop(next(iter(self._mesh_cache)))
        else:
            self._mesh_cache[key] = self._mesh_cache.pop(key)  # LRU refresh
        return entry[1]

    def add(self, iid: str, mesh_path: str | Path, transform: list[list[float]]) -> None:
        mesh = self._base_mesh(mesh_path).copy()
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
        probe = self._base_mesh(mesh_path).copy()
        probe.apply_transform(np.asarray(transform, dtype=float).reshape(4, 4))
        pmin, pmax = probe.bounds
        hits: list[dict] = []
        for other in self._placed.values():
            omin, omax = other.mesh.bounds
            if not (np.all(pmax >= omin - 1e-6) and np.all(pmin <= omax + 1e-6)):
                continue
            try:
                vol = _intersection_volume(probe, other.mesh)
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
                    vol = _intersection_volume(a.mesh, b.mesh)
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

    def mesh_interior_frac(
        self,
        host_iid: str,
        probe_iid: str,
        sample_cap: int = 400,
    ) -> float | None:
        """Fraction of the probe's sampled surface verts inside the host's
        mesh interior — ray-parity ``contains`` (rtree-ray batch), so it
        also reads open shells (covers, hoods) where watertight tests
        fail. Boolean collision cannot see this: a part inside a hollow
        cavity intersects no solid.

        Returns ``None`` when the probe cannot be evaluated (missing
        instance, no verts, parity backend failure) — callers must treat
        ``None`` as "no evidence", never as a violation."""
        host = self._placed.get(host_iid)
        probe = self._placed.get(probe_iid)
        if host is None or probe is None or not len(probe.mesh.vertices):
            return None
        verts = probe.mesh.vertices
        if len(verts) > sample_cap:
            # deterministic stride sample — stable evidence across runs
            idx = np.linspace(0, len(verts) - 1, sample_cap).astype(int)
            verts = verts[idx]
        try:
            inside = host.mesh.contains(verts)
        except Exception as exc:
            log.warning(
                "guide.interior.contains_failed",
                host=host_iid,
                probe=probe_iid,
                err=str(exc),
            )
            return None
        return float(np.asarray(inside, dtype=bool).sum()) / float(len(verts))
