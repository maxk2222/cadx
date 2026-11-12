"""Part metadata extraction — the `extract_ports` tool's worker side.

Mesh uploads (GLB/STL): trimesh + heuristic ports, runs in-worker (cheap).
BRep versions/assets (STEP): dispatched to the executor ``extract_ports``
task on the cad-light lane — OCC work never runs in the worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cadx.guide.state import GuidePart, PortInfo
from cadx.logging import get_logger

log = get_logger(__name__)

_BREP_EXTS = {".step", ".stp", ".brep"}
_MESH_EXTS = {".glb", ".gltf", ".stl", ".obj", ".ply", ".3mf"}


def part_ext(part: GuidePart) -> str:
    src = (part.step_path or part.mesh_path or "").lower()
    for ext in _BREP_EXTS | _MESH_EXTS:
        if src.endswith(ext):
            return ext
    return ""


def is_brep(part: GuidePart) -> bool:
    return part_ext(part) in _BREP_EXTS


def _cylinder_ports_worker(mesh, part: GuidePart) -> list[PortInfo]:
    """Heuristic ports from a mesh: bore/boss cylinder segments via
    trimesh grouping + plane faces via surface segmentation. Best-effort —
    BRep parts get exact face ports from the executor task instead."""
    import trimesh

    ports: list[PortInfo] = []
    try:
        # group_vectors returns (unique_normals, inverse) — inverse[i] is the
        # ragged face-index list of group i, NOT a flat per-face map
        _unique, inverse = trimesh.grouping.group_vectors(mesh.face_normals, angle=0.05)
        planar_idx = 0
        total_area = float(getattr(mesh, "area", 0.0)) or 1.0
        for cluster in inverse:
            cluster = np.asarray(cluster, dtype=int).reshape(-1)
            # filter by covered surface AREA, not tri count — a coarse mesh's
            # real planar faces can be 2 tris each
            if float(np.asarray(mesh.area_faces[cluster]).sum()) / total_area < 0.02:
                continue
            fid = int(cluster[0])
            n = np.asarray(mesh.face_normals[fid], dtype=float)
            ctr = np.asarray(mesh.triangles[cluster]).reshape(-1, 3).mean(axis=0)
            ports.append(
                PortInfo(
                    name=f"face_{planar_idx}",
                    kind="planar",
                    pos=[round(float(v), 4) for v in ctr],
                    axis=[round(float(v), 4) for v in (n / max(np.linalg.norm(n), 1e-9))],
                    gender="flat",
                )
            )
            planar_idx += 1
            if planar_idx >= 12:
                break
    except Exception as exc:
        log.warning("guide.ports.mesh_failed", part=part.id, err=str(exc))
    return ports


def extract_ports_worker(part: GuidePart) -> GuidePart:
    """Mesh-source part → trimesh heuristic ports + bbox (in-worker)."""
    if part.ports and part.status == "analyzed":
        return part
    src = part.mesh_path or part.step_path
    if not src or not Path(src).exists():
        part.status = "error"
        return part
    import trimesh

    try:
        m = trimesh.load(src)
        if hasattr(m, "to_geometry"):
            m = m.to_geometry()
        part.bbox_min = [float(v) for v in m.bounds[0]]
        part.bbox_max = [float(v) for v in m.bounds[1]]
        part.ports = _cylinder_ports_worker(m, part)
        part.status = "analyzed"
    except Exception as exc:
        log.warning("guide.ports.load_failed", part=part.id, err=str(exc))
        part.status = "error"
        part.error = str(exc)[:300]
    return part


async def extract_ports_executor(part: GuidePart, exec_client: Any) -> GuidePart:
    """BRep part → executor task: exact face ports + GLB tessellation."""
    result = await exec_client.extract_ports({"step_path": part.step_path})
    if not result.get("ok"):
        part.status = "error"
        part.error = str(result.get("error") or "extract_ports failed")[:300]
        return part
    data = result.get("extract_ports") or result
    part.ports = [PortInfo.model_validate(p) for p in data.get("ports", [])]
    part.bbox_min = [float(v) for v in data.get("bbox_min", part.bbox_min)]
    part.bbox_max = [float(v) for v in data.get("bbox_max", part.bbox_max)]
    if data.get("mesh_path"):
        part.mesh_path = str(data["mesh_path"])
    part.status = "analyzed" if part.ports else "error"
    if not part.ports:
        part.error = "no ports extracted"
    return part


async def ensure_metadata(part: GuidePart, exec_client: Any = None) -> GuidePart:
    """Dispatch: BRep → executor, mesh → worker, no-source → error."""
    if part.status == "analyzed" and part.ports:
        return part
    if is_brep(part) and part.step_path:
        if exec_client is None:
            part.status = "error"
            part.error = "BRep source but no executor client"
            return part
        return await extract_ports_executor(part, exec_client)
    if part.mesh_path:
        return extract_ports_worker(part)
    part.status = "error"
    part.error = "no cad source (step_path/mesh_path missing)"
    return part


def merge_llm_ports(part: GuidePart, llm_ports: list[dict[str, Any]]) -> int:
    """LLM-named semantic ports merged onto geometric ports — names from the
    LLM, geometry stays server-side. Returns count of ports renamed."""
    renamed = 0
    for lp in llm_ports:
        want = str(lp.get("name", "")).strip()
        src = str(lp.get("from_port", lp.get("port", ""))).strip()
        if not want or not src:
            continue
        for p in part.ports:
            if p.name == src:
                p.name = want
                if lp.get("gender") in ("male", "female", "flat"):
                    p.gender = str(lp["gender"])
                renamed += 1
                break
    return renamed
