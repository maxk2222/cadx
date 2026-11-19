"""Part metadata extraction — the `extract_ports` tool's worker side.

Mesh uploads (GLB/STL): trimesh + heuristic ports, runs in-worker (cheap).
BRep versions/assets (STEP): dispatched to the executor ``extract_ports``
task on the cad-light lane — OCC work never runs in the worker.
"""

from __future__ import annotations

import hashlib
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


def normalize_mesh_units(src: Path, dst_dir: Path) -> Path:
    """glTF is spec'd in METERS; the CAD contract is mm. An uploaded .glb
    whose max extent is under ~5 units is almost certainly a meter-scale
    part (a real sub-5mm assembly part is implausible) — bake a x1000
    normalized copy into the guide dir so scene/solver/render all see mm.
    STL/OBJ stay untouched (CAD convention there is already mm)."""
    if src.suffix.lower() not in {".glb", ".gltf"}:
        return src
    import trimesh

    try:
        m = trimesh.load(src)
        if hasattr(m, "to_geometry"):
            m = m.to_geometry()
        ext = np.asarray(m.bounds[1] - m.bounds[0], dtype=float)
        if float(ext.max()) >= 5.0:
            return src
        m.apply_scale(1000.0)
        dst_dir.mkdir(parents=True, exist_ok=True)
        # stem alone collides — every codegen version exports "model.glb",
        # so two generated parts would overwrite each other's normalized
        # copy; a deterministic path hash keeps the names unique
        tag = hashlib.md5(str(src).encode()).hexdigest()[:8]
        dst = dst_dir / f"{src.stem}_{tag}_mm.glb"
        m.export(str(dst))
        log.info(
            "guide.mesh_units_normalized",
            src=src.name,
            dims_before=[round(float(v), 4) for v in ext],
        )
        return dst
    except Exception as exc:
        log.warning("guide.mesh_normalize_failed", src=src.name, err=str(exc))
        return src


def _closed_sharp_loops(mesh) -> list[np.ndarray]:
    """Vertex loops along sharp edges — the rim of a socket mouth / spigot
    tip is a closed ring of high-dihedral edges. Returns vertex-index loops
    (every vertex valence 2, >=8 verts)."""
    from collections import defaultdict

    fn = mesh.face_normals
    fa = mesh.face_adjacency
    if len(fa) == 0:
        return []
    ang = np.degrees(np.arccos(np.clip((fn[fa[:, 0]] * fn[fa[:, 1]]).sum(1), -1.0, 1.0)))
    sharp = np.unique(np.sort(mesh.face_adjacency_edges[ang > 25.0], axis=1), axis=0)
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in sharp:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    seen: set[int] = set()
    loops: list[np.ndarray] = []
    for s in list(adj):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            c = stack.pop()
            comp.append(c)
            for n in adj[c]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(comp) >= 8 and all(len(adj[v]) == 2 for v in comp):
            loops.append(np.asarray(comp, dtype=int))
    return loops


def _loop_circle(mesh, loop: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Least-squares plane fit on loop vertices → (center, normal, radius).
    Rejects non-planar/non-round loops via the third singular value."""
    pts = np.asarray(mesh.vertices[loop], dtype=float)
    ctr = pts.mean(axis=0)
    _u, sv, vh = np.linalg.svd(pts - ctr, full_matrices=False)
    if sv[2] > max(sv[0], 1e-9) * 0.15:
        return None  # not planar
    normal = vh[2]
    radii = np.linalg.norm(pts - ctr, axis=1)
    r = float(radii.mean())
    if r < 1.0 or float(radii.std()) > r * 0.2:
        return None  # not round enough or degenerate
    return ctr, normal, r


def _bore_measure(
    mesh, center: np.ndarray, axis: np.ndarray, lip_r: float
) -> tuple[float, float]:
    """Probe the bore with rings of inward rays: the closest hit across
    all rings is the SEAT depth (a nut face / socket shoulder / bore
    bottom); `inner_r` is the largest ring radius that still reaches the
    seat — the clear radius a plug may occupy. Through-bores and open
    tube ends report depth=500 (unbounded → not a socket)."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(axis, ref)
    t1 /= max(np.linalg.norm(t1), 1e-9)
    t2 = np.cross(axis, t1)
    radii = [lip_r * f for f in (0.3, 0.55, 0.75, 0.9)]
    origins = np.stack(
        [
            center + axis * 1.5 + t1 * (r * np.cos(a)) + t2 * (r * np.sin(a))
            for r in radii
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
    )
    ring_idx = np.repeat(np.arange(len(radii)), 6)
    try:
        locs, rays, _tris = mesh.ray.intersects_location(
            origins, np.tile(-axis, (len(origins), 1)), multiple_hits=True
        )
        dist = np.linalg.norm(locs - origins[rays], axis=1) - 1.5
        ring_hit = np.full(len(radii), 500.0)
        for i in range(len(radii)):
            d = dist[ring_idx[rays] == i]
            if len(d):
                ring_hit[i] = float(np.min(d))
    except Exception:
        ring_hit = np.full(len(radii), 500.0)
    seat = float(np.min(ring_hit))
    if seat >= 400.0:
        return 500.0, lip_r * 0.9
    # clear radius = largest ring that still reaches ~the seat; a solid
    # face (seat≈0) blocks every ring early → no clear bore at all
    reachable = [r for r, h in zip(radii, ring_hit, strict=True) if h >= max(seat * 0.8, 4.0)]
    inner = max(reachable) if reachable else 0.0
    return max(seat, 0.0), float(inner)


def _nested_stub(
    mesh, center: np.ndarray, axis: np.ndarray, lip_r: float, seat: float
) -> tuple[np.ndarray, float, float] | None:
    """Union-nut stub INSIDE a bounded socket: radial rays at depth
    stations inside the bore hit [stub bore wall, stub outer wall, nut
    wall] — an inner surface below the lip radius at consistent stations
    means a recessed tube the outer rim alone can't express (the real
    Außengewinde-Interface of a union fitting). The stub can start
    BEHIND the measured seat — stations scan to ~2.2x the seat.
    Returns (tip_pos, inner_r, outer_r) of the stub mouth, else None."""
    if not (4.0 <= seat < 400.0) or lip_r <= 3.0:
        return None
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(axis, ref)
    t1 /= max(np.linalg.norm(t1), 1e-9)
    t2 = np.cross(axis, t1)
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    dirs = np.stack([t1 * np.cos(a) + t2 * np.sin(a) for a in ang])
    stations = np.linspace(seat * 0.5, min(seat * 2.2, 80.0), 8)
    shell_t: list[float] = []
    inner_r = outer_r = 0.0
    for t in stations:
        o = center - axis * t
        try:
            locs, rays, _tris = mesh.ray.intersects_location(
                np.tile(o, (len(dirs), 1)), dirs, multiple_hits=True
            )
        except Exception:
            return None
        firsts: list[float] = []
        seconds: list[float] = []
        locs = np.asarray(locs).reshape(-1, 3)
        for k in range(len(dirs)):
            hits = locs[np.asarray(rays) == k]
            if len(hits) == 0:
                continue
            d = np.sort(np.linalg.norm(hits - o, axis=1))
            if d[0] < lip_r * 0.8:
                firsts.append(float(d[0]))
                if len(d) >= 2 and d[1] < lip_r * 1.15:
                    seconds.append(float(d[1]))
        if len(firsts) < 5:
            continue
        shell_t.append(float(t))
        inner_r = float(np.median(firsts))
        outer_r = float(np.median(seconds)) if seconds else inner_r
    if not shell_t:
        return None
    tip = min(shell_t)  # station closest to the mouth = stub's free end
    tip_pos = center - axis * tip
    return tip_pos, inner_r, outer_r


def _rim_ports(mesh, part: GuidePart) -> list[PortInfo]:
    """Ports from closed sharp-edge rim loops — detects socket mouths and
    spigot tips that planar-normal clustering misses on curved fittings
    (elbows, reducers). Gender: a rim with a concentric LARGER partner loop
    on the same axis bounds an annular face = socket mouth (female); a
    solo rim is a tube/spigot end (male) UNLESS a bore recedes inward —
    a recessed lip (boss opening, captive nut) is a socket too."""
    if not mesh.is_watertight:
        mesh = mesh.copy()
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if len(mesh.faces) == 0:
        return []
    try:
        loops = _closed_sharp_loops(mesh)
    except Exception as exc:
        log.warning("guide.ports.rim_failed", part=part.id, err=str(exc))
        return []
    circles: list[tuple[np.ndarray, np.ndarray, float]] = []
    for lp in loops:
        fit = _loop_circle(mesh, lp)
        if fit is not None and 1.0 <= fit[2] <= 200.0:
            circles.append(fit)
    # group concentric coplanar rims (socket inner+outer edge share a
    # plane); the INNER rim is the bore mouth — that's the port
    used = [False] * len(circles)
    ports: list[PortInfo] = []
    for i in range(len(circles)):
        if used[i]:
            continue
        used[i] = True
        partner = _concentric_partner(circles, i, used)
        if partner is not None:
            used[partner] = True
        port = _rim_port(mesh, circles, i, partner, len(ports))
        ports.append(port)
        # union nut / recessed boss: a bounded socket can hide a male
        # stub inside — without it, fittings land in the nut's void
        # instead of on the actual threaded tube
        if port.kind == "cylinder" and 4.0 <= port.depth < 400.0:
            stub = _nested_stub(
                mesh,
                np.asarray(port.pos, dtype=float),
                np.asarray(port.axis, dtype=float),
                port.radius,
                port.depth,
            )
            if stub is not None:
                _tip_pos, stub_in, stub_out = stub
                # emitted as a MALE stub — the Außengewinde site a fitting
                # wraps. The port sits at the ENCLOSING SOCKET'S mouth: the
                # nut's face lands at the mouth plane when threading on
                # (the stub continues out past it like a real pipe end),
                # not sunk inside the boss cavity at tip_pos.
                ports.append(
                    PortInfo(
                        name=f"rim_{len(ports)}",
                        kind="cylinder",
                        pos=list(port.pos),
                        axis=list(port.axis),
                        gender="male",
                        radius=round(stub_out, 3),
                        inner_radius=round(stub_in, 3),
                        outer_radius=round(stub_out, 3),
                        depth=0.0,
                        nested_in=port.name,
                    )
                )
    return ports


def _concentric_partner(
    circles: list[tuple[np.ndarray, np.ndarray, float]], i: int, used: list[bool]
) -> int | None:
    c, n, r = circles[i]
    for j in range(i + 1, len(circles)):
        if used[j]:
            continue
        c2, n2, r2 = circles[j]
        if abs(float(np.dot(n, n2))) < 0.98:
            continue
        if np.linalg.norm(c - c2) < max(r, r2) * 0.6:
            return j
    return None


def _rim_port(
    mesh,
    circles: list[tuple[np.ndarray, np.ndarray, float]],
    i: int,
    partner: int | None,
    idx: int,
) -> PortInfo:
    c, n, r = circles[i]
    outer_r = 0.0
    if partner is not None:
        outer_r = max(r, circles[partner][2])
        if circles[partner][2] < r:
            c, n, r = circles[partner]  # inner rim wins for sockets
    # outward axis: away from the part's centroid
    away = c - np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    nn = n if float(np.dot(n, away)) >= 0 else -n
    depth, inner_r = _bore_measure(mesh, c, nn, r)
    if partner is not None:
        gender = "female"
    elif 4.0 <= depth < 400.0:
        gender = "female"  # bounded seat inside = socket (boss, nut, shoulder)
    else:
        gender = "male"  # solid stub end OR open tube end (through-bore)
    return PortInfo(
        name=f"rim_{idx}",
        kind="cylinder",
        pos=[round(float(v), 4) for v in c],
        axis=[round(float(v), 4) for v in nn],
        gender=gender,
        radius=round(r, 3),
        depth=round(depth, 2),
        inner_radius=round(inner_r, 3) if inner_r else 0.0,
        outer_radius=round(outer_r, 3) if outer_r else 0.0,
    )


def _cylinder_ports_worker(mesh, part: GuidePart) -> list[PortInfo]:
    """Heuristic ports from a mesh: closed sharp-edge rim loops (sockets,
    spigots — the real mating ports) + large planar clusters (mounting
    faces). BRep parts get exact face ports from the executor instead."""
    import trimesh

    ports: list[PortInfo] = []
    try:
        ports.extend(_rim_ports(mesh, part))
        # group_vectors returns (unique_normals, inverse) — inverse[i] is the
        # ragged face-index list of group i, NOT a flat per-face map
        _unique, inverse = trimesh.grouping.group_vectors(mesh.face_normals, angle=0.05)
        planar_idx = 0
        total_area = float(getattr(mesh, "area", 0.0)) or 1.0
        for cluster in inverse:
            cluster = np.asarray(cluster, dtype=int).reshape(-1)
            # planar clusters are mounting-face hints — require a solid
            # share so tube-facet noise doesn't flood the port list
            if float(np.asarray(mesh.area_faces[cluster]).sum()) / total_area < 0.035:
                continue
            fid = int(cluster[0])
            n = np.asarray(mesh.face_normals[fid], dtype=float)
            ctr = np.asarray(mesh.triangles[cluster]).reshape(-1, 3).mean(axis=0)
            ports.append(
                PortInfo(
                    name=f"face_{planar_idx}",
                    kind="plane",
                    pos=[round(float(v), 4) for v in ctr],
                    axis=[round(float(v), 4) for v in (n / max(np.linalg.norm(n), 1e-9))],
                    gender="flat",
                )
            )
            planar_idx += 1
            if planar_idx >= 6:
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
