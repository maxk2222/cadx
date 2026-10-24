"""Map LLM-declared RegionSpecs onto node/element masks of the VoxelDomain.

Region kinds (see llm.schemas.RegionSpec):
- face:     outer element layer on one bbox side (legacy alias of surface)
- surface:  actual part surface facing `axis` — follows curved shapes
- area:     XY sub-area (polygon/x_range/y_range/center+diameter), optionally
            restricted to the surface facing `axis`
- cylinder: material within radius around an axis line (e.g. a bore/mount)
- box:      cube centred at `center` with edge `diameter_mm` (or 4*margin)
- hole:     measured feature anchor (`hole_index`, 1-based) + margin; honours
            the anchor's `axis` (any of ±x/y/z) and 3D centre
- anchor:   alias of hole (typed feature anchors: thread/insert/bearing)
- bore:     bore WALL of a hole anchor — solid cells bordering the void
            cylinder; the natural target for bearing loads
- rim:      outer radial band of width `margin_mm` following the part
            silhouette around `axis` (gear/sprocket tooth ring); an exact
            annulus when `diameter_mm`+`inner_diameter_mm` are given
- border:   band of width `margin_mm` at the bbox outer edges along the
            axes named in `axis` (letters or signed; default "xy")
- points:   painted surface patch (GUI brush)
- all:      the whole solid

`resolve_case` keeps the legacy node-mask signature; `resolve_case_dofs`
additionally returns the per-DOF constraint mask that `fix_mode` regions
need ("all" = encastre, "normal" = pin only the axis DOF: roller/symmetry).

An empty selection is reported as a warning (caller decides to retry via the
LLM loads step or to continue without that region).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from cadx.cad.fem.voxelize import FACE_CORNERS, VoxelDomain, shift_mask
from cadx.llm.schemas import LoadCase, RegionSpec
from cadx.logging import get_logger

log = get_logger(__name__)

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_AXIS3 = {"+x": 0, "+y": 1, "+z": 2, "-x": 0, "-y": 1, "-z": 2, "x": 0, "y": 1, "z": 2}


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


def _exterior_void(domain: VoxelDomain) -> np.ndarray:
    """Void cells connected to the domain bbox boundary — the outside air.

    Enclosed cavities (sealed pockets inside shell/infill parts) are NOT
    exterior: their walls must never receive loads or fixtures — a clamp
    or press physically contacts the outside skin only.
    """
    void = ~domain.solid
    labels, _ = ndimage.label(void, structure=ndimage.generate_binary_structure(3, 1))
    border = np.concatenate(
        [
            labels[0, :, :].ravel(),
            labels[-1, :, :].ravel(),
            labels[:, 0, :].ravel(),
            labels[:, -1, :].ravel(),
            labels[:, :, 0].ravel(),
            labels[:, :, -1].ravel(),
        ]
    )
    outside_ids = set(np.unique(border).tolist()) - {0}
    return np.asarray(np.isin(labels, list(outside_ids)))


def _surface_cells(domain: VoxelDomain, axis: str, margin_mm: float) -> np.ndarray:
    """Cells forming the actual exterior surface facing `axis` (exposed face
    outward), plus `margin_mm` worth of layers behind it. Works on
    curved/organic shapes — unlike a global bbox layer it follows the
    silhouette. Only void connected to the outside counts as exposure:
    enclosed cavity walls and internal lattice faces are not fixtures or
    load faces."""
    idx, sign = _axis_parts(axis)
    d = np.zeros(3, dtype=int)
    d[idx] = sign
    outside = _exterior_void(domain)
    # a solid cell is exposed in direction `d` when its neighbour cell is
    # outside air — or it sits on the domain border itself
    edge = np.zeros(domain.dims, dtype=bool)
    sl: list[slice] = [slice(None)] * 3
    sl[idx] = slice(-1, None) if sign > 0 else slice(0, 1)
    edge[tuple(sl)] = True
    exposed = domain.solid & (shift_mask(outside, d) | edge)
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
    cx, cy, cz = centers[..., 0], centers[..., 1], centers[..., 2]
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
    if len(region.z_range) == 2:
        lo, hi = sorted(float(v) for v in region.z_range)
        inside &= (cz >= lo) & (cz <= hi)
        restricted = True
    if not restricted and len(region.center) >= 2 and region.diameter_mm:
        d2 = (cx - region.center[0]) ** 2 + (cy - region.center[1]) ** 2
        if region.inner_diameter_mm is not None:
            # exact annulus — the declared radii bound the ring, no margin
            r_out = region.diameter_mm / 2
            r_in = region.inner_diameter_mm / 2
            inside &= (d2 <= r_out * r_out) & (d2 >= r_in * r_in)
        else:
            r = region.diameter_mm / 2 + region.margin_mm
            inside &= d2 <= r * r
        restricted = True
    if not restricted:
        # `area` with an axis but no XY bounds means "the whole face" —
        # degrade gracefully to the axis-facing surface instead of nothing
        return np.asarray(sel if region.axis else np.zeros(domain.dims, dtype=bool))
    return np.asarray(sel & inside)


def _axial_wall_mid(
    domain: VoxelDomain, idx: int, center: np.ndarray, dia: float
) -> float | None:
    """Axial centre of the wall a bore pierces, estimated on the voxel grid.

    Anchors without a measured axial position (no `z`/`pos[ax]`) used to
    drop onto the bbox mid-plane — a bore through the 10 mm plate of a
    40 mm stepped part then resolved at z=20 instead of z=5. Sample the
    solid columns on a ring just outside the bore radius: per column the
    SHORTEST contiguous material run along the axis is the pierced wall
    (the plate), not bulk further along; the median of the run midpoints
    locates the wall. None when no ring column carries material."""
    others = [i for i in range(3) if i != idx]
    centers = domain.element_centers().reshape((*domain.dims, 3))
    du = centers[..., others[0]] - float(center[others[0]])
    dv = centers[..., others[1]] - float(center[others[1]])
    r = np.hypot(du, dv)
    r_ring = max(dia / 2.0 + domain.pitch, 2.0 * domain.pitch)
    band = domain.solid & (r >= r_ring - domain.pitch) & (r <= r_ring + domain.pitch)
    # walk each rim column along the axis (moved last for easy run scans)
    sol = np.moveaxis(domain.solid, idx, -1).reshape(-1, domain.dims[idx])
    rng = np.moveaxis(band, idx, -1).reshape(-1, domain.dims[idx])
    zc = domain.origin[idx] + (np.arange(domain.dims[idx], dtype=float) + 0.5) * domain.pitch
    mids: list[float] = []
    for line, in_ring in zip(sol, rng, strict=True):
        if not in_ring.any():
            continue
        padded = np.concatenate([[False], line, [False]])
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        if not len(starts):
            continue
        k = int(np.argmin(ends - starts))
        mids.append(0.5 * (zc[starts[k]] + zc[ends[k] - 1]))
    if not mids:
        return None
    return float(np.median(mids))


def _anchor_geometry(
    domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None
) -> tuple[np.ndarray, np.ndarray, float, float | None] | None:
    """(center3d mm, unit axis, bore dia, axial half-extent) for an anchor region.

    Legacy top-view anchors carry only x/y (+z implied); scanned anchors add
    `axis` and `z`. The centre coordinate along the axis defaults to the part
    mid-plane. `half` (mm) limits the axial extent of the selection when the
    anchor knows its depth; None = whole part span along the axis.
    """
    if not anchors or not region.hole_index or region.hole_index < 1:
        return None
    if region.hole_index > len(anchors):
        return None
    a = anchors[region.hole_index - 1]
    axis = str(a.get("axis") or region.axis or "+z")
    idx = _AXIS3.get(axis.lower(), 2)
    sign = -1.0 if axis.startswith("-") else 1.0
    avec = np.zeros(3)
    avec[idx] = sign
    lo, hi = domain.origin, domain.origin + domain.pitch * np.asarray(domain.dims)
    mid = (lo + hi) / 2
    center = np.array(
        [float(a.get("x") or mid[0]), float(a.get("y") or mid[1]), float(a.get("z") or mid[2])]
    )
    pos = a.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        for i in range(3):
            if pos[i] is not None:
                center[i] = float(pos[i])
    dia = float(a.get("dia") or 0.0)
    az = a.get("z")
    axc: float | None = None
    if az is not None:
        axc = float(az)
    elif isinstance(pos, (list, tuple)) and len(pos) == 3 and pos[idx] is not None:
        axc = float(pos[idx])
    if axc is None:
        # no measured axial centre — estimate the pierced wall's mid from
        # the shortest material run on a ring around the bore before
        # degrading to the bbox mid-plane
        axc = _axial_wall_mid(domain, idx, center, dia)
    if axc is None:
        axc = float(mid[idx])
        log.warning(
            "fem.anchor_axial_unknown",
            hole_index=region.hole_index,
            axis=axis,
            fallback_z=float(mid[idx]),
            reason="anchor has no z/pos axial centre and no resolvable wall run",
        )
    center[idx] = axc
    depth = a.get("depth_mm")
    half = float(depth) / 2 + max(region.margin_mm, domain.pitch) if depth else None
    return center, avec, dia, half


def _axial_slab(domain: VoxelDomain, axis_idx: int, center: float, half: float) -> np.ndarray:
    centers = domain.element_centers().reshape((*domain.dims, 3))
    return np.asarray(np.abs(centers[..., axis_idx] - center) <= half)


def _cylinder_disc_cells(
    domain: VoxelDomain, axis: str, center: list[float], radius: float
) -> np.ndarray:
    """Geometric disc mask — cells whose centre lies within `radius` of the
    axis line at `center`, NOT intersected with solid (for void-fraction
    checks, where the void cells are the signal)."""
    idx, _ = _axis_parts(axis)
    others = [i for i in range(3) if i != idx]
    centers = domain.element_centers().reshape((*domain.dims, 3))
    c = np.array([center[others[0]], center[others[1]]], dtype=float)
    d2 = (centers[..., others[0]] - c[0]) ** 2 + (centers[..., others[1]] - c[1]) ** 2
    return np.asarray(d2 <= radius * radius)


def _anchor_has_void(domain: VoxelDomain, axis: str, center: list[float], dia: float) -> bool:
    """True when the interior cylinder of the measured bore is mostly
    void — the anchor sits on a real hole. An anchor inside solid material
    (phantom mesh feature, stale hole_index) has no void to clamp a ring
    around and must resolve to nothing rather than pin a material column.
    A bore too small for the grid (no cell centre inside the interior
    radius) is inconclusive and passes — the grow loop handles it."""
    if dia <= 0:
        return True
    r_inner = max(dia / 2 - domain.pitch, domain.pitch)
    inner = _cylinder_disc_cells(domain, axis, center, r_inner)
    if not inner.any():
        return True
    solid_frac = float((inner & domain.solid).sum()) / float(inner.sum())
    return solid_frac <= 0.5


def _hole_cells(domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None) -> np.ndarray:
    """Material ring around a hole anchor (any axis). The bore itself is
    void, so the cylinder degenerates to the ring clamped by the pin.

    The measured dia can undersize the real bore (countersinks, tolerances)
    which would leave the disc fully inside the void — grow until it hits
    solid material. When the anchor position holds NO void at all (phantom
    anchor inside solid material) the mask stays empty instead — the
    region fallback then pins a declared boundary patch rather than a
    phantom material column."""
    g = _anchor_geometry(domain, region, anchors)
    if g is None:
        return np.zeros(domain.dims, dtype=bool)
    center, avec, dia, half = g
    idx = int(np.argmax(np.abs(avec)))
    axis = f"+{'xyz'[idx]}"
    ctr = [float(v) for v in center]
    if not _anchor_has_void(domain, axis, ctr, dia):
        return np.zeros(domain.dims, dtype=bool)
    # keep-ring floor: a bore used as a fixture/load point needs meat around
    # it — at least 2 mm or a fifth of the bore diameter, whichever is more
    margin = max(float(region.margin_mm), 2.0, 0.2 * dia, domain.pitch)
    radius = dia / 2 + margin
    cells = _cylinder_cells(domain, axis, ctr, radius)
    for _ in range(4):
        if cells.any():
            break
        radius += margin
        cells = _cylinder_cells(domain, axis, ctr, radius)
    if half is not None:
        cells &= _axial_slab(domain, idx, center[idx], half)
    return cells


def _bore_cells(domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None) -> np.ndarray:
    """The bore WALL: solid cells directly facing the anchor's void cylinder.

    Unlike `hole` (the material ring around the bore, for clamping), `bore`
    selects the inner wall layer — where a pin/bearing actually presses.
    Falls back to the tight `hole` ring when the bore can't be located."""
    g = _anchor_geometry(domain, region, anchors)
    if g is None:
        return np.zeros(domain.dims, dtype=bool)
    center, avec, dia, half = g
    idx = int(np.argmax(np.abs(avec)))
    axis = f"+{'xyz'[idx]}"
    ctr = [float(v) for v in center]
    # same void check as _hole_cells, BEFORE the wall ring — a phantom
    # anchor must not fall through to the hole grow loop and resurrect a
    # material column out of solid material
    if not _anchor_has_void(domain, axis, ctr, dia):
        return np.zeros(domain.dims, dtype=bool)
    # solid cells whose radial distance is within [0.5*dia, 0.5*dia + band]
    band = max(domain.pitch * 1.6, 0.35 * dia, 1.0)
    in_cyl = _cylinder_cells(domain, axis, ctr, dia / 2 + band) & ~_cylinder_cells(
        domain, axis, ctr, max(dia / 2 - domain.pitch, 0.0)
    )
    if half is not None:
        in_cyl &= _axial_slab(domain, idx, center[idx], half)
    if in_cyl.any():
        return np.asarray(in_cyl, dtype=bool)
    return np.asarray(_hole_cells(domain, region, anchors), dtype=bool)


def _cylinder_cells(
    domain: VoxelDomain,
    axis: str,
    center: list[float],
    radius: float,
    inner_radius: float = 0.0,
    z_range: list[float] | None = None,
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
    mask = (d2 <= radius * radius) & (d2 >= inner_radius * inner_radius) & domain.solid
    if z_range is not None and len(z_range) == 2:
        # the range bounds the CYLINDER axis coordinate (idx), not literal z
        lo_z, hi_z = sorted(float(v) for v in z_range)
        mask &= (centers[..., idx] >= lo_z) & (centers[..., idx] <= hi_z)
    return np.asarray(mask)


def _range_clip(domain: VoxelDomain, region: RegionSpec, sel: np.ndarray) -> np.ndarray:
    """Intersect a cell selection with the optional x/y/z_range bounds."""
    ranges = (region.x_range, region.y_range, region.z_range)
    if not any(len(rng) == 2 for rng in ranges):
        return sel
    centers = domain.element_centers().reshape((*domain.dims, 3))
    for i, rng in enumerate(ranges):
        if len(rng) == 2:
            lo, hi = sorted(float(v) for v in rng)
            sel &= (centers[..., i] >= lo) & (centers[..., i] <= hi)
    return sel


def _z_clip(domain: VoxelDomain, region: RegionSpec, sel: np.ndarray) -> np.ndarray:
    """Clip a selection to `z_range` when the region declares one (Z in mm)."""
    if len(region.z_range) != 2:
        return sel
    centers = domain.element_centers().reshape((*domain.dims, 3))
    lo, hi = sorted(float(v) for v in region.z_range)
    return np.asarray(sel & (centers[..., 2] >= lo) & (centers[..., 2] <= hi))


def _rim_cells(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """Outer radial rim of width `margin_mm` following the part silhouette
    around `axis` (e.g. a gear/sprocket tooth ring).

    Per 5° angle sector the local outer radius of the solid cells sets the
    band floor, so non-circular parts and toothed rims keep a uniform band
    inward from their own outer tips instead of a global-radius cut that
    would swallow the hub. `diameter_mm` + `inner_diameter_mm` select an
    exact annulus instead of the silhouette band."""
    axis = region.axis or "+z"
    if region.diameter_mm and region.inner_diameter_mm is not None:
        return _cylinder_cells(
            domain,
            axis,
            region.center,
            float(region.diameter_mm) / 2.0,
            inner_radius=float(region.inner_diameter_mm) / 2.0,
            z_range=region.z_range or None,
        )
    idx, _ = _axis_parts(axis)
    others = [i for i in range(3) if i != idx]
    centers = domain.element_centers().reshape((*domain.dims, 3))
    lo = domain.origin
    hi = domain.origin + domain.pitch * np.asarray(domain.dims, dtype=float)
    c0 = (
        float(region.center[others[0]])
        if len(region.center) > others[0]
        else float((lo[others[0]] + hi[others[0]]) / 2)
    )
    c1 = (
        float(region.center[others[1]])
        if len(region.center) > others[1]
        else float((lo[others[1]] + hi[others[1]]) / 2)
    )
    u = centers[..., others[0]] - c0
    v = centers[..., others[1]] - c1
    r = np.hypot(u, v)
    r_solid = r[domain.solid]
    if not r_solid.size:
        return np.zeros(domain.dims, dtype=bool)
    margin = max(float(region.margin_mm), domain.pitch)
    bins = 72
    ang = ((np.arctan2(v, u) + math.pi) / (2.0 * math.pi) * bins).astype(int) % bins
    sector_max = np.zeros(bins, dtype=float)
    np.maximum.at(sector_max, ang[domain.solid], r_solid)
    r_outer = float(r_solid.max())
    local_max = np.where(sector_max[ang] > 0, sector_max[ang], r_outer)
    # include all teeth down to min(local_max - margin, r_outer - margin)
    r_floor = np.minimum(local_max - margin, r_outer - margin)
    sel = domain.solid & (r >= r_floor)
    return np.asarray(_range_clip(domain, region, sel), dtype=bool)


def _border_cells(domain: VoxelDomain, region: RegionSpec) -> np.ndarray:
    """Solid cells within `margin_mm` of the bbox outer edges along the axes
    named in `axis`: a signed axis ("+x"/"-z") selects only that one side,
    plain letters ("x", "xy", "xyz") select both sides of each named axis.
    Default "xy" — the lateral clamp edges of a plate."""
    ax_s = (region.axis or "xy").strip().lower()
    margin = max(float(region.margin_mm), domain.pitch)
    lo = domain.origin
    hi = domain.origin + domain.pitch * np.asarray(domain.dims, dtype=float)
    centers = domain.element_centers().reshape((*domain.dims, 3))
    sel = np.zeros(domain.dims, dtype=bool)
    if ax_s.startswith(("+", "-")):
        idx, sign = _axis_parts(ax_s)
        if sign > 0:
            sel |= hi[idx] - centers[..., idx] <= margin
        else:
            sel |= centers[..., idx] - lo[idx] <= margin
    else:
        for ch in ax_s:
            i = _AXIS_IDX.get(ch)
            if i is None:
                continue
            sel |= (centers[..., i] - lo[i] <= margin) | (hi[i] - centers[..., i] <= margin)
    sel &= domain.solid
    return np.asarray(_range_clip(domain, region, sel), dtype=bool)


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
    of near-surface material — the natural target for clamps and load pads.

    Rasterised in index space: each point only visits the cells inside its
    sphere's bounding box instead of scanning the whole domain. Dense
    strokes are deduplicated on a half-pitch grid so heavy painting can't
    blow up the cost."""
    if len(region.points) == 0:
        return np.zeros(domain.dims, dtype=bool)
    radius = region.radius_mm or max(2.0 * region.margin_mm, 3.0)
    radius += domain.pitch  # centres of boundary cells sit ~pitch behind the skin
    dims = np.asarray(domain.dims)
    r2 = radius * radius
    ext_cells = radius / domain.pitch
    sel = np.zeros(domain.dims, dtype=bool)
    seen: set[tuple[int, int, int]] = set()
    half = domain.pitch * 0.5
    for p in region.points[:4000]:
        if len(p) < 3:
            continue
        pt = np.asarray(p[:3], dtype=float)
        # dedupe stroke points closer than half a pitch — same ~cell patch
        key: tuple[int, int, int] = tuple(  # type: ignore[assignment]
            math.floor(float(v) / half) for v in (pt - domain.origin)
        )
        if key in seen:
            continue
        seen.add(key)
        rel = (pt - domain.origin) / domain.pitch - 0.5
        lo = np.maximum(np.floor(rel - ext_cells).astype(int), 0)
        hi = np.minimum(np.floor(rel + ext_cells).astype(int) + 1, dims)
        if np.any(hi <= lo):
            continue
        gi, gj, gk = np.mgrid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        c0 = domain.origin[None, :] + domain.pitch * (
            np.stack([gi, gj, gk], axis=-1).astype(float) + 0.5
        )
        d2 = np.sum((c0 - pt[None, None, None, :]) ** 2, axis=-1)
        sel[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] |= d2 <= r2
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
        sel = _face_cells(domain, region.axis or "+z", region.margin_mm)
        return _z_clip(domain, region, sel)
    if kind == "surface":
        sel = _surface_cells(domain, region.axis or "+z", region.margin_mm)
        return _z_clip(domain, region, sel)
    if kind == "area":
        return _area_cells(domain, region)
    if kind == "rim":
        return _rim_cells(domain, region)
    if kind == "border":
        return _border_cells(domain, region)
    if kind in ("hole", "anchor"):
        return np.asarray(_hole_cells(domain, region, anchors), dtype=bool)
    if kind == "bore":
        return _bore_cells(domain, region, anchors)
    if kind == "cylinder":
        if region.inner_diameter_mm is not None:
            # exact annulus — the declared radii bound the ring, no margin
            r_in = float(region.inner_diameter_mm) / 2.0
            r_out = (
                float(region.diameter_mm)
                if region.diameter_mm
                else float(region.inner_diameter_mm) + 2.0 * region.margin_mm
            ) / 2.0
            return _cylinder_cells(
                domain,
                region.axis or "+z",
                region.center,
                r_out,
                inner_radius=r_in,
                z_range=region.z_range or None,
            )
        radius = (region.diameter_mm or 4 * region.margin_mm) / 2 + region.margin_mm
        return _cylinder_cells(
            domain,
            region.axis or "+z",
            region.center,
            radius,
            z_range=region.z_range or None,
        )
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


def _region_hint_point(
    domain: VoxelDomain,
    region: RegionSpec,
    anchors: list[dict] | None,
    anchor: dict | None,
) -> np.ndarray | None:
    """Best-effort 3D position hint for a region: anchor centre first, then
    `center`, then the mean of painted `points`. Missing components default
    to the domain mid-plane."""
    if anchor is not None:
        g = _anchor_geometry(domain, region, anchors)
        if g is not None:
            return g[0]
    lo = domain.origin
    hi = domain.origin + domain.pitch * np.asarray(domain.dims)
    mid = (lo + hi) / 2
    if len(region.center) >= 2:
        c = mid.copy()
        for i, v in enumerate(region.center[:3]):
            c[i] = float(v)
        return c
    pts = [p for p in region.points if len(p) >= 3]
    if pts:
        return np.asarray(pts, dtype=float).mean(axis=0)
    return None


def _hint_has_axis(region: RegionSpec, anchor: dict | None, idx: int) -> bool:
    """Whether the position hint actually carries the AXIAL coordinate.
    `_region_hint_point` fills missing components with the domain mid-plane —
    a 2-component `center` means the caller never stated an elevation, so
    the band filter must not trust that component."""
    if anchor is not None:
        pos = anchor.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) == 3 and pos[idx] is not None:
            return True
        return idx == 2 and anchor.get("z") is not None
    if len(region.center) >= 3 and region.center[idx] is not None:
        return True
    return any(len(p) >= 3 and p[idx] is not None for p in region.points)


def _region_fallback(
    domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None
) -> tuple[np.ndarray, str]:
    """Deterministic bbox-derived patch for a region that resolves to no
    cells — the real surface layer on the region's (or its anchor's) axis
    side, e.g. a declared "-x" fixture becomes the part's -x skin.

    Axis source: `region.axis`, else the anchor's `axis`, else the dominant
    painted `normal`, else the bbox face nearest a position hint (anchor
    centre / `center` / painted points).
    With a hint the patch is clipped to a disc around it (~bore dia +
    margin); a clipped patch that stays empty widens to the whole face —
    the declared axis is more reliable than the declared position.

    Returns (cell_mask, axis_label). Empty mask + "" when no side can be
    derived — the region counts as unresolved.
    """
    anchor = None
    if region.hole_index and anchors and 1 <= region.hole_index <= len(anchors):
        anchor = anchors[region.hole_index - 1]
    ax_s = (region.axis or "").strip().lower()
    if not ax_s and anchor is not None:
        ax_s = str(anchor.get("axis") or "").strip().lower()
    if ax_s in ("x", "y", "z"):
        ax_s = "+" + ax_s
    if not ax_s and len(region.normal) >= 3:
        ax_s = _dominant_axis(region.normal)
    hint = _region_hint_point(domain, region, anchors, anchor)
    if ax_s not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        if hint is None:
            return np.zeros(domain.dims, dtype=bool), ""
        # no declared direction — pin the bbox face nearest the hint
        lo = domain.origin
        hi = domain.origin + domain.pitch * np.asarray(domain.dims)
        d = np.concatenate([hint - lo, hi - hint])
        i = int(np.argmin(d))
        ax_s = f"{'+' if i >= 3 else '-'}{'xyz'[i % 3]}"
    idx, _sign = _axis_parts(ax_s)
    margin = max(float(region.margin_mm), domain.pitch)
    face = _surface_cells(domain, ax_s, margin)
    face = np.asarray(_z_clip(domain, region, face), dtype=bool)
    if hint is not None:
        centers = domain.element_centers().reshape((*domain.dims, 3))
        dia = float((anchor or {}).get("dia") or region.diameter_mm or 0.0)
        r_clip = max(dia, 2.0 * margin) + domain.pitch
        others = [i for i in range(3) if i != idx]
        d2 = sum((centers[..., o] - float(hint[o])) ** 2 for o in others)
        clipped = face & (d2 <= r_clip * r_clip)
        # the hint's AXIAL coordinate matters too: a declared boss-top surface
        # must not bind the lower same-axis face that happens to share the
        # in-plane disc (e.g. a plate shoulder under a boss). Keep only the
        # axial band nearest the hint when cells exist at that elevation —
        # but only when the hint actually carried that component: 2-part
        # centers default the axis coord to the mid-plane, which would
        # exclude every real surface cell.
        if clipped.any() and _hint_has_axis(region, anchor, idx):
            band = clipped & (
                np.abs(centers[..., idx] - float(hint[idx])) <= max(2.0 * domain.pitch, margin)
            )
            if band.any():
                clipped = band
        if clipped.any():
            return np.asarray(clipped, dtype=bool), ax_s
    return np.asarray(face, dtype=bool), ax_s


def _exterior_normal_mean(domain: VoxelDomain, cell_mask: np.ndarray) -> np.ndarray | None:
    """Mean outward unit normal of the selected cells' exposed faces.

    Each direction ±x/±y/±z votes with the count of selected cells whose
    neighbour that way is outside air (or the domain border); the
    normalised vector sum is the average skin normal — radial/organic
    regions like `rim` get their net pull direction this way. ``None``
    when nothing is exposed or symmetric faces cancel out.
    """
    outside = _exterior_void(domain)
    acc = np.zeros(3)
    for ax in range(3):
        for sgn in (1, -1):
            d = np.zeros(3, dtype=int)
            d[ax] = sgn
            edge = np.zeros(domain.dims, dtype=bool)
            sl: list[slice] = [slice(None)] * 3
            sl[ax] = slice(-1, None) if sgn > 0 else slice(0, 1)
            edge[tuple(sl)] = True
            exp = cell_mask & (shift_mask(outside, d) | edge)
            acc[ax] += sgn * float(exp.sum())
    mag = float(np.linalg.norm(acc))
    if mag < 1e-9:
        return None
    return acc / mag


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
    if region.kind == "border":
        # a signed border side ("+x") presses along that axis; unsigned
        # multi-side bands have no unique normal (callers fall back to
        # the mean exposed-face normal)
        ax_s = (region.axis or "").strip().lower()
        if not ax_s.startswith(("+", "-")):
            return None
        idx, sign = _axis_parts(ax_s)
        v = np.zeros(3)
        v[idx] = float(sign)
        return v
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
    face_corners = list(FACE_CORNERS[idx_a][int(sign > 0)])
    areas = np.zeros(domain.node_count, dtype=float)
    sel = np.flatnonzero(face_cells.ravel())
    if len(sel):
        en = domain.element_nodes()
        nodes = en[sel][:, face_corners].ravel()
        np.add.at(areas, nodes, domain.pitch**2 / len(face_corners))
    return areas


def _couple_weights(domain: VoxelDomain, cells: np.ndarray) -> np.ndarray | None:
    """(nnodes,) normalised exposed-surface share for a couple's traction.

    A moment on a patch is a tangential SURFACE traction, not a body load
    — weight each node by its share of the region's exposed skin (summed
    over all six axes, like normal_force_n does per-axis). None when the
    region is fully buried (no skin) — callers then apply a uniform
    volumetric couple. Bore-wall cells expose faces to the bore void, so
    hole/anchor torques weight onto the wall automatically.
    """
    areas = np.zeros(domain.node_count)
    for ax in ("+x", "-x", "+y", "-y", "+z", "-z"):
        areas += surface_area_per_node(domain, cells, ax)
    tot = float(areas.sum())
    return areas / tot if tot > 0 else None


def _grow_load_spec(r: RegionSpec, pitch: float) -> RegionSpec | None:
    """A copy of a load RegionSpec enlarged by ~one pitch — the patch grows
    around its declared centre (center/points stay fixed, no centroid
    shift). ``None`` for kinds that cannot grow this way (hole/bore
    anchors carry their own ring logic, `all` is already maximal)."""
    if r.kind == "box":
        edge = (r.diameter_mm or 4 * r.margin_mm) + pitch
        return r.model_copy(update={"diameter_mm": edge})
    if r.kind == "points":
        radius = (r.radius_mm or max(2.0 * r.margin_mm, 3.0)) + pitch
        return r.model_copy(update={"radius_mm": radius})
    if r.kind in ("area", "surface", "face", "cylinder", "rim", "border"):
        return r.model_copy(update={"margin_mm": r.margin_mm + pitch})
    return None


def _anchor_note(domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None) -> str:
    """`` (anchor #i at (x, y, z) — verify ...)`` suffix for unresolved
    hole/anchor/bore regions — the resolved anchor identity + position so a
    phantom anchor (stale hole_index, mesh artefact) is visible in the
    warning. Empty string when the region names no anchor."""
    idx = region.hole_index
    if not idx or not anchors or not (1 <= idx <= len(anchors)):
        return ""
    g = _anchor_geometry(domain, region, anchors)
    if g is None:
        return f" (anchor #{idx} — verify the hole exists in the geometry)"
    c = g[0]
    return (
        f" (anchor #{idx} at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) — "
        "verify the hole exists in the geometry)"
    )


def _fixed_dof_mask(domain: VoxelDomain, region: RegionSpec, node_mask: np.ndarray) -> np.ndarray:
    """(nnodes, 3) bool — which DOFs of the region's nodes are constrained.

    fix_mode "all"    -> encastre (all three translational DOFs)
    fix_mode "normal" -> pin only the axis DOF: roller on a face, sliding
                         support, or a symmetry plane. Axis source: the
                         region's `axis`, else the anchor's axis, else the
                         dominant component of `normal`.
    """
    out = np.zeros((domain.node_count, 3), dtype=bool)
    if region.fix_mode == "normal":
        axis = region.axis or (region.normal[:3] if len(region.normal) >= 3 else "") or "+z"
        idx = _AXIS3.get(str(axis).lower(), 2)
        if len(region.normal) >= 3 and not region.axis:
            idx = int(np.argmax(np.abs(np.asarray(region.normal[:3], dtype=float))))
        out[node_mask, idx] = True
    else:
        out[node_mask, :] = True
    return out


def resolve_case_dofs(  # noqa: C901
    domain: VoxelDomain,
    case: LoadCase,
    anchors: list[dict] | None = None,
    density_g_cm3: float = 1.24,
    applied_out: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve a load case into (fixed_dof_mask (nnodes,3), F vector, warnings).

    F is a dense (3*nnodes,) force vector in Newton. Loads with zero matching
    nodes are skipped with a warning. Beyond force_n/pressure_mpa this handles
    bearing_n (cosine wall distribution), moment_nm (pure couple), preload_n
    (axial bore-mouth pair), normal_force_n (total force along the region's
    inward surface normal), remote_point (force + induced moment) and g_load
    (body force over the region's mass). bearing_n/preload_n/moment_nm on a
    hole/anchor region introduce the load on the bore WALL (`_bore_cells`),
    never on the keep ring; a `kind="all"` fixed region is refused (it would
    rigid-lock the whole part).

    `applied_out`, when given a list, receives one record per load region
    that matched nodes — {"kind","axis","hole_index","nodes","cells",
    "centroid_mm","mechanisms","force_n","couple_nm","couple_direct_nm",
    "couple_induced_nm","extent_mm"} plus, when a couple was requested,
    {"couple_requested_nm","couple_deficit_nm","couple_footprint_mm"} —
    the measured resultant force and couple about the region centroid as
    actually applied to F (achieved, not requested).
    """
    from cadx.cad.fem import loads as ld

    warnings: list[str] = []
    nn = domain.node_count
    fixed3 = np.zeros((nn, 3), dtype=bool)
    F = np.zeros(3 * nn)

    def _nodes_cells(r: RegionSpec) -> tuple[np.ndarray, np.ndarray, str]:
        """(node_mask, cell_mask, fallback_axis): the region's own cells, or
        a deterministic bbox boundary patch when it resolves to nothing."""
        cells = np.asarray(region_cell_mask(domain, r, anchors), dtype=bool)
        if cells.any():
            return _nodes_of_cells(domain, cells.ravel()), cells, ""
        fb, ax = _region_fallback(domain, r, anchors)
        if fb.any():
            return _nodes_of_cells(domain, fb.ravel()), fb, ax
        return np.zeros(nn, dtype=bool), cells, ""

    for r in case.fixed:
        if r.kind == "all":
            warnings.append(
                f"case '{case.name}': fixed region 'all' is invalid "
                "(would constrain every DOF) — skipped"
            )
            continue
        m, cells, fb_ax = _nodes_cells(r)
        tag = f"#{r.hole_index}" if r.hole_index else (r.axis or "")
        if not m.any():
            warnings.append(f"case '{case.name}': fixed region {r.kind} {tag} matched no nodes")
            continue
        if fb_ax:
            warnings.append(
                f"case '{case.name}': region_fallback — fixed region {r.kind} {tag} "
                f"unresolved on the geometry; pinned {fb_ax} boundary patch "
                f"({int(cells.sum())} cells)" + _anchor_note(domain, r, anchors)
            )
        fixed3 |= _fixed_dof_mask(domain, r, m)
        if r.fix_mode == "normal":
            warnings.append(
                f"case '{case.name}': '{r.description or r.kind}' is a roller/symmetry "
                "support — the part can still slide; make sure the case is stable"
            )

    for load in case.loads:
        r = load.region
        m, cells, fb_ax = _nodes_cells(r)
        if r.kind in ("hole", "anchor") and (
            load.bearing_n or load.preload_n or load.moment_nm
        ):
            # a bolt presses on the bore WALL, not on the clamp ring around
            # the hole — re-resolve through _bore_cells (itself falling back
            # to the hole ring when no wall can be located). A couple on a
            # hole/anchor is shaft/bolt torque and transmits the same way:
            # as tangential traction on the wall.
            bore = _bore_cells(domain, r, anchors)
            if bore.any():
                cells = bore
                m = _nodes_of_cells(domain, cells.ravel())
        tag = f"#{r.hole_index}" if r.hole_index else (r.axis or "")
        if not m.any():
            warnings.append(f"case '{case.name}': load region {r.kind} {tag} matched no nodes")
            continue
        # a load patch of only a few cells concentrates a point-load
        # singularity — grow the declared region by ~one pitch at a time
        # (centre kept) until it covers a ~2-pitch block; warn when it
        # still can't (never applied to case.fixed regions). A couple
        # additionally needs a ~3-pitch footprint perpendicular to its
        # axis — smaller patches concentrate near-singular nodal forces
        # (~M/footprint); growth gets a few extra tries but stops early
        # when the footprint cannot widen (e.g. a thin plate's margin
        # only grows in-plane).
        min_load_cells = max(8, round(0.5 * (2 * domain.pitch) ** 3 / domain.pitch**3))
        want_cpl = (
            np.asarray(load.moment_nm, dtype=float)
            if load.moment_nm and len(load.moment_nm) == 3
            else np.zeros(3)
        )
        lever_pt = (
            np.asarray(load.remote_point[:3], dtype=float)
            if load.remote_point
            and len(load.remote_point) >= 3
            and load.force_n
            and len(load.force_n) == 3
            else None
        )
        force_v = (
            np.asarray(load.force_n[:3], dtype=float) if lever_pt is not None else None
        )

        def _req_couple(
            m_now: np.ndarray,
            _want: np.ndarray = want_cpl,
            _lever: np.ndarray | None = lever_pt,
            _force: np.ndarray | None = force_v,
        ) -> np.ndarray:
            """Requested couple (N·m): declared moment + remote-lever part
            about the CURRENT node set's centroid (it moves as the patch
            grows — same reference the apply code uses below). Defaults pin
            this load's vectors (B023 — the closure outlives the loop var)."""
            req = _want.copy()
            if _lever is not None and _force is not None:
                c = domain.node_coords()[np.flatnonzero(m_now)].mean(axis=0)
                req += np.cross(_lever - c, _force) / 1e3
            return req

        def _extent_perp(m_now: np.ndarray, req: np.ndarray) -> float:
            """Span (mm) of the node set perpendicular to the couple axis —
            the lever the traction field can work with."""
            u = req / np.linalg.norm(req)
            P = domain.node_coords()[np.flatnonzero(m_now)]
            rp = P - P.mean(axis=0)
            rp = rp - np.outer(rp @ u, u)
            return float(np.ptp(rp))

        couple_req = float(np.linalg.norm(_req_couple(m))) >= 1e-3
        max_tries = 6 if couple_req else 3
        tries = 0
        prev_extent = float("inf")
        stalls = 0
        moment_grew = False
        while tries < max_tries:
            too_small = bool(cells.sum() < min_load_cells)
            weak = False
            if couple_req:
                req = _req_couple(m)
                ext = _extent_perp(m, req)
                idx_now = np.flatnonzero(m)
                w_now = _couple_weights(domain, cells)
                deficit = ld.moment_deficit(
                    domain,
                    idx_now,
                    req,
                    weights=w_now[idx_now] if w_now is not None else None,
                )
                weak = ext < 3.0 * domain.pitch or float(
                    np.linalg.norm(deficit)
                ) > 0.1 * float(np.linalg.norm(req))
                # rasterisation makes extent grow in steps — one flat try is
                # normal, two consecutive mean growth cannot help (thin plate)
                if weak and too_small is False and tries > 0:
                    if ext <= prev_extent + 1e-9:
                        stalls += 1
                        if stalls >= 2:
                            break
                    else:
                        stalls = 0
                prev_extent = ext
            if not (too_small or weak):
                break
            grown = _grow_load_spec(r, domain.pitch)
            if grown is None:
                break  # kinds that cannot grow this way (hole/bore/all)
            r = grown
            m, cells, fb_ax = _nodes_cells(r)
            tries += 1
            moment_grew = moment_grew or weak
        if moment_grew:
            warnings.append(
                f"case '{case.name}': widened the {r.kind} region to "
                f"{int(cells.sum())} cells so the couple spreads over a "
                "~3-pitch footprint"
            )
        if fb_ax:
            warnings.append(
                f"case '{case.name}': region_fallback — load region {r.kind} {tag} "
                f"unresolved on the geometry; applied on {fb_ax} boundary patch "
                f"({int(cells.sum())} cells)" + _anchor_note(domain, r, anchors)
            )
        elif (
            not r.material
            and (
                load.force_n
                or load.bearing_n
                or load.moment_nm
                or load.remote_point
                or load.normal_force_n
            )
            and cells.sum() > 0.5 * domain.solid_count
        ):
            # a discrete force smeared over most of the part understates local
            # stress — legitimate for uniform pressure/g-loads, suspicious for
            # force/moment/bearing loads (e.g. an unbounded `area` on a thin
            # shell selects ~70% of the cells)
            warnings.append(
                f"case '{case.name}': load region {r.kind} {tag} covers "
                f"{100.0 * cells.sum() / max(domain.solid_count, 1):.0f}% of the "
                "part — the force is spread very thin; narrow it with "
                "x_range/y_range/center or confirm uniform loading is intended"
            )
        if cells.sum() < min_load_cells:
            warnings.append(
                f"case '{case.name}': load region {r.kind} {tag} covers only "
                f"{int(cells.sum())} cells — point-load singularity risk"
            )
        idx = np.flatnonzero(m)
        # a couple is a surface traction — weight the tangential field by
        # each node's exposed-area share (uniform fallback when buried)
        w_skin = _couple_weights(domain, cells) if couple_req else None
        w_idx = w_skin[idx] if w_skin is not None else None

        rec: dict[str, Any] | None = None
        F_before: np.ndarray | None = None
        direct_nm = np.zeros(3)
        induced_nm = np.zeros(3)
        req_nm = np.zeros(3)  # declared moment + remote-lever request
        mechs: list[str] = []
        if applied_out is not None:
            F_before = F.copy()
            rec = {
                "kind": r.kind,
                "axis": r.axis or "",
                "hole_index": r.hole_index,
                "nodes": len(idx),
                "cells": int(cells.sum()),
                "centroid_mm": [
                    round(float(v), 2)
                    for v in domain.node_coords()[idx].mean(axis=0)
                ],
                "mechanisms": mechs,
            }

        applied = False
        if load.remote_point and len(load.remote_point) >= 3 and not (
            load.force_n and len(load.force_n) == 3
        ):
            warnings.append(
                f"case '{case.name}': remote_point given without force_n — "
                "a remote point only levers a force; add force_n"
            )
        if load.direction_cone_deg and not (
            (load.force_n and len(load.force_n) == 3)
            or (load.moment_nm and len(load.moment_nm) == 3)
            or (load.bearing_dir and len(load.bearing_dir) == 3)
        ):
            warnings.append(
                f"case '{case.name}': direction_cone_deg on a load with no "
                "direction vector (force_n/moment_nm/bearing_dir) — ignored"
            )
        if load.force_n and len(load.force_n) == 3:
            f = np.asarray(load.force_n, dtype=float)
            if load.remote_point and len(load.remote_point) >= 3:
                # force applied at a remote point: same resultant + the
                # moment (p - centroid) x F the lever arm induces
                c = domain.node_coords()[idx].mean(axis=0)
                lever = np.asarray(load.remote_point[:3], dtype=float) - c
                moment = np.cross(lever, f) / 1e3  # N·mm -> N·m
                req_nm += moment
                ld.spread_force(F, idx, f)
                got_v = ld.spread_moment(domain, F, idx, moment, weights=w_idx)
                induced_nm += got_v
                want = float(np.linalg.norm(moment))
                got = float(np.linalg.norm(got_v))
                miss = float(np.linalg.norm(moment - got_v))
                if want >= 1e-3 and got < 0.9 * want:
                    warnings.append(
                        f"case '{case.name}': remote force at {load.remote_point[:3]} "
                        f"should add {want:.2f} N·m but only {got:.2f} N·m applied "
                        f"— the {_dominant_axis(moment - got_v)} component could not "
                        f"resolve ({miss:.2f} N·m short; region lacks extent normal "
                        "to it); widen the region or use a bore wall"
                    )
            else:
                ld.spread_force(F, idx, f)
            applied = True
            mechs.append("force_n")
        if load.bearing_n is not None and load.bearing_n > 0:
            g = _anchor_geometry(domain, r, anchors)
            if g is None:
                warnings.append(
                    f"case '{case.name}': bearing_n needs a hole/anchor/bore region "
                    f"(kind={r.kind}, hole_index={r.hole_index}, "
                    f"anchors={len(anchors) if anchors else 0})"
                )
            else:
                center, avec, _dia, _half = g
                dvec = (
                    np.asarray(load.bearing_dir[:3], dtype=float)
                    if len(load.bearing_dir) >= 3
                    else np.asarray(load.force_n[:3] if len(load.force_n) >= 3 else [0, -1, 0])
                )
                got = ld.bearing_load(domain, F, idx, avec, center, dvec, float(load.bearing_n))
                if got <= 0:
                    warnings.append(
                        f"case '{case.name}': bearing load on anchor "
                        f"#{r.hole_index} at ({center[0]:.1f}, {center[1]:.1f}, "
                        f"{center[2]:.1f}) Ø{_dia:g} produced no force — no bore "
                        "wall resolved there; check that hole_index maps to a "
                        "real bore on the current part"
                    )
                applied = applied or got > 0
                if got > 0:
                    mechs.append("bearing_n")
        if load.moment_nm and len(load.moment_nm) == 3:
            want_v = np.asarray(load.moment_nm, dtype=float)
            req_nm += want_v
            got_v = ld.spread_moment(domain, F, idx, want_v, weights=w_idx)
            direct_nm += got_v
            want = float(np.linalg.norm(want_v))
            got = float(np.linalg.norm(got_v))
            miss = float(np.linalg.norm(want_v - got_v))
            if want >= 1e-3 and got < 0.9 * want:
                warnings.append(
                    f"case '{case.name}': moment on {r.kind} applied "
                    f"{got:.2f} of {want:.2f} N·m — the "
                    f"{_dominant_axis(want_v - got_v)} component could not "
                    f"resolve ({miss:.2f} N·m short; region lacks extent "
                    "normal to it); widen the region or target a "
                    "bore/cylinder wall"
                )
            applied = applied or got > 0
            if got > 0:
                mechs.append("moment_nm")
        if load.preload_n is not None and load.preload_n > 0:
            g = _anchor_geometry(domain, r, anchors)
            if g is None:
                warnings.append(f"case '{case.name}': preload_n needs a hole/anchor/bore region")
            else:
                _center, avec, _dia, _h = g
                got = ld.bolt_preload(domain, F, cells.ravel(), avec, float(load.preload_n))
                if got <= 0:
                    warnings.append(
                        f"case '{case.name}': preload on anchor #{r.hole_index} at "
                        f"({_center[0]:.1f}, {_center[1]:.1f}, {_center[2]:.1f}) "
                        "found no open bore mouths (blind hole or solid region?)"
                    )
                applied = applied or got > 0
                if got > 0:
                    mechs.append("preload_n")
        if load.normal_force_n is not None and abs(float(load.normal_force_n)) > 0:
            n = region_face_normal(r)
            if n is None:
                # no declared direction — average the outward normals of
                # the selected cells' exposed faces instead
                n = _exterior_normal_mean(domain, cells)
            if n is None:
                warnings.append(
                    f"case '{case.name}': normal_force_n on {r.kind} — "
                    "no outward surface normal resolvable"
                )
            else:
                # + pushes INTO the part along the inward normal
                f_vec = -n * float(load.normal_force_n)
                # area-weighted spread on the exposed face when the region
                # presents an identifiable skin; even split otherwise
                areas = surface_area_per_node(domain, cells, _dominant_axis(n))
                tot = float(areas.sum())
                if tot > 0:
                    pidx = np.flatnonzero(areas > 0)
                    w = areas[pidx] / tot
                    for cc in range(3):
                        F[3 * pidx + cc] += f_vec[cc] * w
                else:
                    ld.spread_force(F, idx, f_vec)
                applied = True
                mechs.append("normal_force_n")
        if load.g_load and len(load.g_load) == 3:
            rho = density_g_cm3
            if r.material:
                from cadx.cad.fem.materials import resolve_material

                rho = resolve_material(r.material).density_g_cm3
            got = ld.body_force(domain, F, cells.ravel(), np.asarray(load.g_load, dtype=float), rho)
            if got <= 0:
                warnings.append(f"case '{case.name}': g_load on {r.kind} produced no force")
            else:
                warnings.append(f"case '{case.name}': gravity on {r.kind} = {got:.1f} N")
            applied = applied or got > 0
            if got > 0:
                mechs.append("g_load")
        if load.pressure_mpa is not None:
            n = region_face_normal(r)
            dirs: list[tuple[str, np.ndarray]] = []
            if n is not None:
                # real selected surface area shared onto the face nodes;
                # pressure acts AGAINST the outward normal (into the part)
                axis = r.axis or (_dominant_axis(r.normal) if len(r.normal) >= 3 else "+z")
                dirs = [(axis, n)]
            elif r.kind == "all":
                # pressure on the whole part = hydrostatic squeeze: every
                # exposed face gets -p along its own outward normal
                for ax in ("+x", "-x", "+y", "-y", "+z", "-z"):
                    ai, sgn = _axis_parts(ax)
                    v = np.zeros(3)
                    v[ai] = float(sgn)
                    dirs.append((ax, v))
            elif r.kind == "cylinder":
                # grip squeeze around a shaft-like body: pressure on the
                # lateral skin only — the four directions normal to the
                # cylinder axis, never the end caps
                cax = (r.axis or "+z").lower()
                perp = tuple(ax for ax in ("+x", "-x", "+y", "-y", "+z", "-z") if ax[-1] != cax[-1])
                for ax in perp:
                    ai, sgn = _axis_parts(ax)
                    v = np.zeros(3)
                    v[ai] = float(sgn)
                    dirs.append((ax, v))
            if not dirs:
                warnings.append(
                    f"case '{case.name}': pressure loads need a face/surface/area region "
                    f"(got {r.kind})"
                )
            else:
                total_area = 0.0
                for axis, nv in dirs:
                    areas = surface_area_per_node(domain, cells, axis)
                    pidx = np.flatnonzero(areas > 0)
                    if not len(pidx):
                        continue
                    fv = -nv * float(load.pressure_mpa)
                    for cc in range(3):
                        F[3 * pidx + cc] += fv[cc] * areas[pidx]
                    total_area += float(areas.sum())
                if total_area <= 0:
                    warnings.append(
                        f"case '{case.name}': pressure region has no exposed surface "
                        "— check axis/area"
                    )
                else:
                    warnings.append(
                        f"case '{case.name}': pressure {load.pressure_mpa:g} MPa over "
                        f"{total_area:.0f} mm² = "
                        f"{float(load.pressure_mpa) * total_area:.0f} N"
                    )
                    applied = True
                    mechs.append("pressure_mpa")
        if not applied:
            warnings.append(f"case '{case.name}': load on {r.kind} resolved to zero force")
        cpl_mag = float(np.linalg.norm(req_nm))
        cpl_extent = 0.0
        if cpl_mag >= 1e-3:
            cpl_extent = _extent_perp(m, req_nm)
            if r.kind not in ("hole", "anchor", "bore") and cpl_extent < 3.0 * domain.pitch:
                # bolt torque on a small bore is legitimately concentrated —
                # the physical interface IS that small; surface patches are not
                peak_n = cpl_mag * 1e3 / max(cpl_extent, 1e-6)
                warnings.append(
                    f"case '{case.name}': couple {cpl_mag:.2f} N·m on {r.kind} "
                    f"{tag} concentrates on a ~{cpl_extent:.0f} mm footprint — "
                    f"≈{peak_n:.0f} N nodal forces; widen the region if this "
                    "isn't the real contact patch"
                )
        if rec is not None and F_before is not None:
            dF = (F - F_before).reshape(-1, 3)
            sel = np.abs(dF).sum(axis=1) > 0
            res = dF[sel].sum(axis=0) if sel.any() else np.zeros(3)
            c = np.asarray(rec["centroid_mm"], dtype=float)
            couple = (
                np.cross(domain.node_coords()[sel] - c, dF[sel]).sum(axis=0) / 1e3
                if sel.any()
                else np.zeros(3)
            )
            rec["force_n"] = [round(float(v), 2) for v in res]
            rec["force_mag_n"] = round(float(np.linalg.norm(res)), 2)
            rec["couple_nm"] = [round(float(v), 3) for v in couple]
            rec["couple_mag_nm"] = round(float(np.linalg.norm(couple)), 3)
            rec["couple_direct_nm"] = [round(float(v), 3) for v in direct_nm]
            rec["couple_induced_nm"] = [round(float(v), 3) for v in induced_nm]
            rec["extent_mm"] = [
                round(float(v), 2)
                for v in np.ptp(domain.node_coords()[idx], axis=0)
            ]
            if cpl_mag >= 1e-3:
                rec["couple_requested_nm"] = [round(float(v), 3) for v in req_nm]
                rec["couple_deficit_nm"] = [
                    round(float(v), 3) for v in req_nm - (direct_nm + induced_nm)
                ]
                rec["couple_footprint_mm"] = round(cpl_extent, 2)
            applied_out.append(rec)

    if not fixed3.any():
        warnings.append(f"case '{case.name}': no fixed region — case is unsolvable")
    if not np.any(F):
        warnings.append(f"case '{case.name}': all loads resolved to zero force")

    return fixed3, F, warnings


def resolve_case(
    domain: VoxelDomain,
    case: LoadCase,
    anchors: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Legacy wrapper — collapses the per-DOF mask to a node-any mask."""
    fixed3, F, warnings = resolve_case_dofs(domain, case, anchors)
    return fixed3.any(axis=1), F, warnings


def case_reactions(
    domain: VoxelDomain,
    fixed3: np.ndarray,
    F: np.ndarray,
    applied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load-case sanity report: applied resultant + moment about the centroid
    of the constrained nodes, plus constrained-DOF counts. Equilibrium needs
    reactions to balance the resultants.

    `applied` (the per-load records produced by `resolve_case_dofs`) splits
    the moment into the parts the caller declared: pure couples
    (`moment_direct_nm`) and remote-lever induced moments
    (`moment_induced_nm`). The remaining share is the transport moment of
    the resultant force acting at the load centroids."""
    res = F.reshape(-1, 3).sum(axis=0)
    out: dict[str, Any] = {
        "resultant_n": [round(float(v), 2) for v in res],
        "resultant_mag_n": round(float(np.linalg.norm(res)), 2),
        "fixed_nodes": int(fixed3.any(axis=1).sum()),
        "fixed_dofs": [int(fixed3[:, c].sum()) for c in range(3)],
        "moment_nm": [0.0, 0.0, 0.0],
        "moment_mag_nm": 0.0,
    }
    nodes = np.flatnonzero(fixed3.any(axis=1))
    if len(nodes):
        c = domain.node_coords()[nodes].mean(axis=0)
        sel = np.flatnonzero(np.abs(F.reshape(-1, 3)).sum(axis=1) > 0)
        if len(sel):
            r = domain.node_coords()[sel] - c
            M = np.cross(r, F.reshape(-1, 3)[sel]).sum(axis=0) / 1e3  # N·mm -> N·m
            out["moment_nm"] = [round(float(v), 3) for v in M]
            out["moment_mag_nm"] = round(float(np.linalg.norm(M)), 3)
    if applied is not None:
        direct = np.zeros(3)
        induced = np.zeros(3)
        deficit = np.zeros(3)
        for rec in applied:
            direct += np.asarray(rec.get("couple_direct_nm") or [0, 0, 0], dtype=float)
            induced += np.asarray(rec.get("couple_induced_nm") or [0, 0, 0], dtype=float)
            deficit += np.asarray(rec.get("couple_deficit_nm") or [0, 0, 0], dtype=float)
        out["moment_direct_nm"] = [round(float(v), 3) for v in direct]
        out["moment_induced_nm"] = [round(float(v), 3) for v in induced]
        out["moment_mag_direct_nm"] = round(float(np.linalg.norm(direct)), 3)
        out["moment_mag_induced_nm"] = round(float(np.linalg.norm(induced)), 3)
        # requested minus achieved — nonzero only when a region could not
        # represent part of its couple (axis named in the case warnings)
        out["moment_deficit_nm"] = [round(float(v), 3) for v in deficit]
        out["moment_mag_deficit_nm"] = round(float(np.linalg.norm(deficit)), 3)
    return out


def material_cell_factors(
    domain: VoxelDomain,
    keep_regions: list[RegionSpec],
    anchors: list[dict] | None,
    base_E: float,
) -> np.ndarray:
    """(nelem,) Young's-modulus multiplier for insert/material zones.

    A keep-region with `material` set overrides the base stiffness inside the
    region (e.g. a brass heat-set insert in a PLA part). Unset materials leave
    the factor at 1."""
    from cadx.cad.fem.materials import resolve_material

    factors = np.ones(domain.cell_count)
    for r in keep_regions:
        if not r.material:
            continue
        cells = region_cell_mask(domain, r, anchors).ravel()
        factors[cells] = float(resolve_material(r.material).elastic_modulus_mpa) / base_E
    return factors


def material_cell_classes(
    domain: VoxelDomain,
    keep_regions: list[RegionSpec],
    anchors: list[dict] | None,
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Per-class constitutive split of the material zones.

    Companion to `material_cell_factors`: a keep-region with `material` set
    overrides E inside the region (the factor path) AND carries its own
    Poisson ratio, which the single-D recovery cannot express (a steel
    insert in PLA under-reads its von Mises by orders of magnitude with the
    base matrix). Each DISTINCT resolved material — deduped on its
    (E, nu) pair — becomes one class.

    Returns ``(zone_cls, class_Ds)``: ``zone_cls`` is (ncells,) int32, -1
    outside every material zone and 0..k-1 inside (last region wins on
    overlap); ``class_Ds`` is the list of unit-E ``_D(nu_zone)`` matrices,
    one per class. ``(None, [])`` when no region declares a material.
    """
    from cadx.cad.fem.materials import resolve_material
    from cadx.cad.fem.solver import _D

    zone_cls = np.full(domain.cell_count, -1, dtype=np.int32)
    class_Ds: list[np.ndarray] = []
    seen: dict[tuple[float, float], int] = {}
    for r in keep_regions or []:
        if not getattr(r, "material", None):
            continue
        mat = resolve_material(r.material)
        key = (float(mat.elastic_modulus_mpa), float(mat.poisson))
        idx = seen.get(key)
        if idx is None:
            idx = len(class_Ds)
            seen[key] = idx
            class_Ds.append(_D(float(mat.poisson)))
        cells = region_cell_mask(domain, r, anchors).ravel()
        zone_cls[cells] = idx
    if not class_Ds:
        return None, []
    return zone_cls, class_Ds


def keep_element_mask(
    domain: VoxelDomain, regions: list[RegionSpec], anchors: list[dict] | None = None
) -> np.ndarray:
    """Union of cell masks for non-design (keep) regions, flat cell order."""
    mask = np.zeros(domain.cell_count, dtype=bool)
    for r in regions:
        mask |= region_cell_mask(domain, r, anchors).ravel()
    return mask


def case_anchor_keep_mask(
    domain: VoxelDomain, cases: list[LoadCase], anchors: list[dict] | None
) -> np.ndarray:
    """Bool (nx,ny,nz) protective keep rings around every hole/anchor/bore
    region the load cases touch — fixed regions and load regions alike.

    A bore that carries a fixture or a load must keep its material ring in
    TopOpt even when the user forgot a matching keep_region; union this
    mask into the passive/keep mask so no loaded or clamped bore gets
    optimized away."""
    out = np.zeros(domain.dims, dtype=bool)
    for case in cases or []:
        regions: list[RegionSpec] = list(case.fixed)
        regions += [ld.region for ld in case.loads]
        for r in regions:
            if r.kind not in ("hole", "anchor", "bore"):
                continue
            keep = r.model_copy(update={"margin_mm": max(float(r.margin_mm), 2.0)})
            out |= np.asarray(_hole_cells(domain, keep, anchors), dtype=bool)
    return out


def case_region_breakdown(
    domain: VoxelDomain, case: LoadCase, anchors: list[dict] | None = None
) -> list[dict[str, Any]]:
    """Per-region resolved cell counts for reporting — one entry per fixed
    region and per load region. Lets summaries show WHICH region resolved
    how much instead of only aggregate node counts."""
    def _entry(role: str, r: RegionSpec) -> dict[str, Any]:
        cells = region_cell_mask(domain, r, anchors)
        nodes = _nodes_of_cells(domain, cells.ravel())
        idx = np.flatnonzero(nodes)
        return {
            "role": role,
            "kind": r.kind,
            "axis": r.axis or "",
            "hole_index": r.hole_index,
            "cells": int(cells.sum()),
            "nodes": len(idx),
            "centroid_mm": (
                [round(float(v), 2) for v in domain.node_coords()[idx].mean(axis=0)]
                if len(idx)
                else None
            ),
        }

    out: list[dict[str, Any]] = [
        *(_entry("fixed", r) for r in case.fixed),
        *(_entry("load", ld.region) for ld in case.loads),
    ]
    return out


def auto_keep_regions(anchors: list[dict] | None, margin_mm: float = 3.0) -> list[RegionSpec]:
    """Keep regions around every detected fastener anchor — for
    OptimizeSpec.keep_mode="auto" so mounts/bores/inserts survive TopOpt."""
    out: list[RegionSpec] = []
    for a in anchors or []:
        idx = int(a.get("index") or 0)
        if idx < 1:
            continue
        kind = str(a.get("kind") or "hole")
        label = a.get("bolt_size") or a.get("name") or kind
        out.append(
            RegionSpec(
                kind="hole",
                hole_index=idx,
                margin_mm=margin_mm,
                description=f"auto-keep {label} #{idx}",
            )
        )
    return out


def auto_void_regions(anchors: list[dict] | None) -> list[RegionSpec]:
    """Void regions INSIDE every fastener anchor's bore — for grow_mm>0 runs
    where the dilated design domain would otherwise let the optimizer fill
    anchored bores back up. The cylinder is slightly narrower than the
    measured bore so the wall cells stay designable/kept."""
    out: list[RegionSpec] = []
    for a in anchors or []:
        idx = int(a.get("index") or 0)
        dia = float(a.get("dia") or 0.0)
        if idx < 1 or dia <= 0:
            continue
        center = a.get("pos")
        if not (isinstance(center, (list, tuple)) and len(center) == 3):
            center = [a.get("x"), a.get("y"), a.get("z")]
        out.append(
            RegionSpec(
                kind="cylinder",
                axis=str(a.get("axis") or "+z"),
                center=[float(c) if c is not None else 0.0 for c in center],
                diameter_mm=dia * 0.9,
                margin_mm=0.0,
                description=f"auto-void bore #{idx}",
            )
        )
    return out


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
