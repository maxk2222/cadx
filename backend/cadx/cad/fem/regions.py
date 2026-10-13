"""Map LLM-declared RegionSpecs onto node/element masks of the VoxelDomain.

Region kinds (see llm.schemas.RegionSpec):
- face:     outer element layer on one bbox side (legacy alias of surface)
- surface:  actual part surface facing `axis` — follows curved shapes
- area:     XY sub-area (polygon/x_range/y_range/center+diameter), optionally
            restricted to the surface facing `axis`
- cylinder: material within radius around an axis line (e.g. a bore/mount)
- box:      cube centred at `center` with edge `diameter_mm` (or 4*margin)
- hole:     measured anchor hole (`hole_index`, 1-based) + margin
- all:      the whole solid

An empty selection is reported as a warning (caller decides to retry via the
LLM loads step or to continue without that region).
"""

from __future__ import annotations

import numpy as np

from cadx.cad.fem.voxelize import VoxelDomain, shift_mask
from cadx.llm.schemas import LoadCase, RegionSpec

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _axis_parts(axis: str) -> tuple[int, int]:
    """('x'|'y'|'z' index, sign +1/-1) from strings like '+z'/'-x'."""
    a = (axis or "+z").strip().lower()
    sign = -1 if a.startswith("-") else 1
    letter = a.lstrip("+-")[-1] if a.lstrip("+-") else "z"
    return _AXIS_IDX.get(letter, 2), sign


def _layers(domain: VoxelDomain, margin_mm: float) -> int:
    return max(1, round(max(margin_mm, 0.1) / domain.pitch))


def _solid_node_mask(domain: VoxelDomain) -> np.ndarray:
    en = domain.element_nodes()
    solid_nodes = np.unique(en[domain.solid.ravel()])
    mask = np.zeros(domain.node_count, dtype=bool)
    mask[solid_nodes] = True
    return mask


def _nodes_of_cells(domain: VoxelDomain, cell_mask: np.ndarray) -> np.ndarray:
    """Union of corner nodes of selected cells -> flat node mask."""
    en = domain.element_nodes()
    mask = np.zeros(domain.node_count, dtype=bool)
    mask[np.unique(en[cell_mask])] = True
    return mask


def _surface_cells(domain: VoxelDomain, axis: str, margin_mm: float) -> np.ndarray:
    """Cells forming the actual surface facing `axis` (exposed face outward),
    plus `margin_mm` worth of layers behind it. Works on curved/organic
    shapes — unlike a global bbox layer it follows the silhouette."""
    idx, sign = _axis_parts(axis)
    d = np.zeros(3, dtype=int)
    d[idx] = sign
    exposed = domain.solid & ~shift_mask(domain.solid, d)
    lay = _layers(domain, margin_mm)
    out = exposed.copy()
    for k in range(1, lay):
        out |= shift_mask(exposed, -d * k) & domain.solid
    return np.asarray(out)


def _face_cells(domain: VoxelDomain, axis: str, margin_mm: float) -> np.ndarray:
    """Legacy 'face' region — alias of `_surface_cells` (real surface cells,
    not a global bbox layer, so curved tops/bottoms work)."""
    return _surface_cells(domain, axis, margin_mm)


def _area_cells(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """XY sub-area of the part: polygon / x_range / y_range / center+diameter,
    optionally restricted to the surface facing `region.axis`."""
    if region.axis:
        sel = _surface_cells(domain, region.axis, region.margin_mm)
    else:
        sel = domain.solid.copy()
    centers = domain.element_centers().reshape((*domain.dims, 3))
    cx, cy = centers[..., 0], centers[..., 1]
    inside = np.ones(domain.dims, dtype=bool)
    restricted = False
    if len(region.polygon) >= 3:
        from shapely import contains_xy
        from shapely.geometry import Polygon

        inside &= np.asarray(contains_xy(Polygon(region.polygon), cx, cy)).reshape(domain.dims)
        restricted = True
    if len(region.x_range) == 2:
        lo, hi = sorted(float(v) for v in region.x_range)
        inside &= (cx >= lo) & (cx <= hi)
        restricted = True
    if len(region.y_range) == 2:
        lo, hi = sorted(float(v) for v in region.y_range)
        inside &= (cy >= lo) & (cy <= hi)
        restricted = True
    if not restricted and len(region.center) >= 2 and region.diameter_mm:
        r = region.diameter_mm / 2 + region.margin_mm
        inside &= (cx - region.center[0]) ** 2 + (cy - region.center[1]) ** 2 <= r * r
        restricted = True
    if not restricted:
        return np.zeros(domain.dims, dtype=bool)
    return np.asarray(sel & inside)


def _hole_cells(domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None) -> np.ndarray:
    """Material around a measured hole anchor: cylinder along z at the
    anchor's (x, y) with radius = hole radius + margin."""
    if not anchors or not region.hole_index or region.hole_index < 1:
        return np.zeros(domain.dims, dtype=bool)
    if region.hole_index > len(anchors):
        return np.zeros(domain.dims, dtype=bool)
    a = anchors[region.hole_index - 1]
    radius = float(a.get("dia") or 0.0) / 2 + region.margin_mm
    return _cylinder_cells(domain, "+z", [float(a["x"]), float(a["y"])], radius)


def _cylinder_cells(
    domain: VoxelDomain, axis: str, center: list[float], radius: float
) -> np.ndarray:
    idx, _ = _axis_parts(axis)
    others = [i for i in range(3) if i != idx]
    centers = domain.element_centers().reshape((*domain.dims, 3))
    if len(center) >= 2:
        # use the two perpendicular components of the given center hint
        c = np.array([center[others[0]], center[others[1]]], dtype=float)
    else:
        lo, hi = domain.origin, domain.origin + domain.pitch * np.asarray(domain.dims)
        c = np.array([(lo[others[0]] + hi[others[0]]) / 2, (lo[others[1]] + hi[others[1]]) / 2])
    d2 = (centers[..., others[0]] - c[0]) ** 2 + (centers[..., others[1]] - c[1]) ** 2
    return np.asarray((d2 <= radius * radius) & domain.solid)


def _box_cells(domain: VoxelDomain, center: list[float], edge: float) -> np.ndarray:
    centers = domain.element_centers().reshape((*domain.dims, 3))
    if len(center) >= 3:
        c = np.asarray(center[:3], dtype=float)
    else:
        lo = domain.origin
        hi = domain.origin + domain.pitch * np.asarray(domain.dims)
        c = (lo + hi) / 2
    half = edge / 2
    inside = np.all(np.abs(centers - c[None, None, None, :]) <= half, axis=-1)
    return np.asarray(inside & domain.solid)


def _points_cells(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """Solid cells within `radius_mm` of any painted surface point (GUI brush).

    Picked points sit ON the surface, so a sphere of `radius` selects a patch
    of near-surface material — the natural target for clamps and load pads."""
    if len(region.points) == 0:
        return np.zeros(domain.dims, dtype=bool)
    radius = region.radius_mm or max(2.0 * region.margin_mm, 3.0)
    radius += domain.pitch  # centres of boundary cells sit ~pitch behind the skin
    centers = domain.element_centers().reshape((*domain.dims, 3))
    sel = np.zeros(domain.dims, dtype=bool)
    pts = np.asarray([p[:3] for p in region.points if len(p) >= 3], dtype=float)
    for p in pts[:600]:
        d2 = np.sum((centers - p[None, None, None, :]) ** 2, axis=-1)
        sel |= d2 <= radius * radius
    return np.asarray(sel & domain.solid)


def _dominant_axis(normal: list[float] | np.ndarray) -> str:
    """'+x'/'-z'/... for the dominant component of a free-form normal."""
    n = np.asarray(normal, dtype=float)
    idx = int(np.argmax(np.abs(n)))
    return f"{'-' if n[idx] < 0 else '+'}{'xyz'[idx]}"


def region_cell_mask(
    domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None = None
) -> np.ndarray:
    """Bool (nx,ny,nz) over cells: which solid cells belong to the region."""
    kind = region.kind
    if kind == "all":
        return domain.solid.copy()
    if kind == "face":
        return _face_cells(domain, region.axis or "+z", region.margin_mm)
    if kind == "surface":
        return _surface_cells(domain, region.axis or "+z", region.margin_mm)
    if kind == "area":
        return _area_cells(domain, region)
    if kind == "hole":
        return _hole_cells(domain, region, anchors)
    if kind == "cylinder":
        radius = (region.diameter_mm or 4 * region.margin_mm) / 2 + region.margin_mm
        return _cylinder_cells(domain, region.axis or "+z", region.center, radius)
    if kind == "box":
        edge = region.diameter_mm or 4 * region.margin_mm
        return _box_cells(domain, region.center, edge)
    if kind == "points":
        return _points_cells(domain, region)
    return np.zeros(domain.dims, dtype=bool)


def region_node_mask(
    domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None = None
) -> np.ndarray:
    """Bool (nnodes,) over all grid nodes belonging to the region."""
    cells = region_cell_mask(domain, region, anchors)
    if not cells.any():
        return np.zeros(domain.node_count, dtype=bool)
    return _nodes_of_cells(domain, cells.ravel())


def region_face_normal(region: RegionSpec) -> np.ndarray | None:
    """Outward unit normal for face-like regions (used for pressure loads)."""
    if region.kind == "points":
        if len(region.normal) >= 3:
            n = np.asarray(region.normal[:3], dtype=float)
            mag = float(np.linalg.norm(n))
            if mag > 1e-6:
                return n / mag
        if region.axis:
            idx, sign = _axis_parts(region.axis)
            v = np.zeros(3)
            v[idx] = float(sign)
            return v
        return None
    if region.kind not in ("face", "surface", "area"):
        return None
    idx, sign = _axis_parts(region.axis or "+z")
    v = np.zeros(3)
    v[idx] = float(sign)
    return v


def surface_area_per_node(domain: VoxelDomain, cell_mask: np.ndarray, axis: str) -> np.ndarray:
    """(nnodes,) mm² of part-surface facing `axis` shared onto corner nodes.

    Each region cell whose outward neighbour is void contributes pitch²/4 to
    each of the 4 face-corner nodes — total equals the real selected area,
    so `pressure * areas` gives a physically distributed force.
    """
    idx_a, sign = _axis_parts(axis)
    d = np.zeros(3, dtype=int)
    d[idx_a] = sign
    face_cells = cell_mask & ~shift_mask(domain.solid, d)
    face_corners = [c for c in range(8) if ((c >> idx_a) & 1) == (sign > 0)]
    areas = np.zeros(domain.node_count, dtype=float)
    sel = np.flatnonzero(face_cells.ravel())
    if len(sel):
        en = domain.element_nodes()
        nodes = en[sel][:, face_corners].ravel()
        np.add.at(areas, nodes, domain.pitch**2 / len(face_corners))
    return areas


def resolve_case(
    domain: VoxelDomain,
    case: LoadCase,
    anchors: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve a load case into (fixed_node_mask, F vector, warnings).

    F is a dense (3*nnodes,) force vector in Newton. Loads with zero matching
    nodes are skipped with a warning.
    """
    warnings: list[str] = []
    nn = domain.node_count
    fixed = np.zeros(nn, dtype=bool)
    F = np.zeros(3 * nn)

    for r in case.fixed:
        m = region_node_mask(domain, r, anchors)
        if not m.any():
            warnings.append(
                f"case '{case.name}': fixed region {r.kind} {r.axis or ''} matched no nodes"
            )
            continue
        fixed |= m

    for load in case.loads:
        m = region_node_mask(domain, load.region, anchors)
        if not m.any():
            warnings.append(
                f"case '{case.name}': load region {load.region.kind} "
                f"{load.region.axis or ''} matched no nodes"
            )
            continue
        nsel = int(m.sum())
        if load.force_n and len(load.force_n) == 3:
            f = np.asarray(load.force_n, dtype=float)
            idx = np.flatnonzero(m)
            for c in range(3):
                F[3 * idx + c] += f[c] / nsel
        elif load.pressure_mpa is not None:
            n = region_face_normal(load.region)
            if n is None:
                warnings.append(
                    f"case '{case.name}': pressure loads need a face/surface/area region "
                    f"(got {load.region.kind})"
                )
                continue
            # real selected surface area shared onto the face nodes;
            # pressure acts AGAINST the outward normal (into the part)
            cells = region_cell_mask(domain, load.region, anchors)
            axis = load.region.axis or (
                _dominant_axis(load.region.normal) if len(load.region.normal) >= 3 else "+z"
            )
            areas = surface_area_per_node(domain, cells, axis)
            idx = np.flatnonzero(areas > 0)
            if not len(idx):
                warnings.append(
                    f"case '{case.name}': pressure region has no surface facing "
                    f"{axis} — check axis/area"
                )
                continue
            f = -n * float(load.pressure_mpa)
            for c in range(3):
                F[3 * idx + c] += f[c] * areas[idx]
            warnings.append(
                f"case '{case.name}': pressure {load.pressure_mpa:g} MPa over "
                f"{areas.sum():.0f} mm² = {float(load.pressure_mpa) * areas.sum():.0f} N"
            )
        else:
            warnings.append(f"case '{case.name}': load has neither force_n nor pressure_mpa")

    if not fixed.any():
        warnings.append(f"case '{case.name}': no fixed region — case is unsolvable")
    if not np.any(F):
        warnings.append(f"case '{case.name}': all loads resolved to zero force")

    return fixed, F, warnings


def keep_element_mask(
    domain: VoxelDomain, regions: list[RegionSpec], anchors: list[dict] | None = None
) -> np.ndarray:
    """Union of cell masks for non-design (keep) regions, flat cell order."""
    mask = np.zeros(domain.cell_count, dtype=bool)
    for r in regions:
        mask |= region_cell_mask(domain, r, anchors).ravel()
    return mask


def empty_region_warnings(
    domain: VoxelDomain, regions: list[RegionSpec], anchors: list[dict] | None = None
) -> list[str]:
    """One warning per region that resolves to zero cells — such regions are
    silently ignored downstream, so make the miss explicit."""
    out = []
    for r in regions:
        if not region_cell_mask(domain, r, anchors).any():
            idx = f"#{r.hole_index}" if r.hole_index else ""
            out.append(f"region {r.kind} {r.axis or ''}{idx} matched no cells — ignored")
    return out
