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

import numpy as np

from cadx.cad.fem.voxelize import VoxelDomain, shift_mask
from cadx.llm.schemas import LoadCase, RegionSpec

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
        # `area` with an axis but no XY bounds means "the whole face" —
        # degrade gracefully to the axis-facing surface instead of nothing
        return np.asarray(sel if region.axis else np.zeros(domain.dims, dtype=bool))
    return np.asarray(sel & inside)


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
    az = a.get("z")
    if az is not None:
        center[idx] = float(az)
    elif isinstance(pos, (list, tuple)) and len(pos) == 3 and pos[idx] is not None:
        center[idx] = float(pos[idx])
    else:
        center[idx] = mid[idx]
    depth = a.get("depth_mm")
    half = float(depth) / 2 + max(region.margin_mm, domain.pitch) if depth else None
    return center, avec, float(a.get("dia") or 0.0), half


def _axial_slab(domain: VoxelDomain, axis_idx: int, center: float, half: float) -> np.ndarray:
    centers = domain.element_centers().reshape((*domain.dims, 3))
    return np.asarray(np.abs(centers[..., axis_idx] - center) <= half)


def _hole_cells(domain: VoxelDomain, region: RegionSpec, anchors: list[dict] | None) -> np.ndarray:
    """Material ring around a hole anchor (any axis). The bore itself is
    void, so the cylinder degenerates to the ring clamped by the pin.

    The measured dia can undersize the real bore (countersinks, tolerances)
    which would leave the disc fully inside the void — grow until it hits
    solid material."""
    g = _anchor_geometry(domain, region, anchors)
    if g is None:
        return np.zeros(domain.dims, dtype=bool)
    center, avec, dia, half = g
    idx = int(np.argmax(np.abs(avec)))
    margin = max(region.margin_mm, domain.pitch)
    radius = dia / 2 + margin
    axis = f"+{'xyz'[idx]}"
    ctr = [float(v) for v in center]
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
    # solid cells whose radial distance is within [0.5*dia, 0.5*dia + band]
    band = max(domain.pitch * 1.6, 0.35 * dia, 1.0)
    ctr = [float(v) for v in center]
    in_cyl = _cylinder_cells(domain, axis, ctr, dia / 2 + band) & ~_cylinder_cells(
        domain, axis, ctr, max(dia / 2 - domain.pitch, 0.0)
    )
    if half is not None:
        in_cyl &= _axial_slab(domain, idx, center[idx], half)
    if in_cyl.any():
        return np.asarray(in_cyl, dtype=bool)
    return np.asarray(_hole_cells(domain, region, anchors), dtype=bool)


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
        return _face_cells(domain, region.axis or "+z", region.margin_mm)
    if kind == "surface":
        return _surface_cells(domain, region.axis or "+z", region.margin_mm)
    if kind == "area":
        return _area_cells(domain, region)
    if kind in ("hole", "anchor"):
        return np.asarray(_hole_cells(domain, region, anchors), dtype=bool)
    if kind == "bore":
        return _bore_cells(domain, region, anchors)
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
    if hint is not None:
        centers = domain.element_centers().reshape((*domain.dims, 3))
        dia = float((anchor or {}).get("dia") or region.diameter_mm or 0.0)
        r_clip = max(dia, 2.0 * margin) + domain.pitch
        others = [i for i in range(3) if i != idx]
        d2 = sum((centers[..., o] - float(hint[o])) ** 2 for o in others)
        clipped = face & (d2 <= r_clip * r_clip)
        if clipped.any():
            return np.asarray(clipped, dtype=bool), ax_s
    return np.asarray(face, dtype=bool), ax_s


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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve a load case into (fixed_dof_mask (nnodes,3), F vector, warnings).

    F is a dense (3*nnodes,) force vector in Newton. Loads with zero matching
    nodes are skipped with a warning. Beyond force_n/pressure_mpa this handles
    bearing_n (cosine wall distribution), moment_nm (pure couple), preload_n
    (axial bore-mouth pair), remote_point (force + induced moment) and g_load
    (body force over the region's mass).
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
        m, cells, fb_ax = _nodes_cells(r)
        tag = f"#{r.hole_index}" if r.hole_index else (r.axis or "")
        if not m.any():
            warnings.append(f"case '{case.name}': fixed region {r.kind} {tag} matched no nodes")
            continue
        if fb_ax:
            warnings.append(
                f"case '{case.name}': region_fallback — fixed region {r.kind} {tag} "
                f"unresolved on the geometry; pinned {fb_ax} boundary patch "
                f"({int(cells.sum())} cells)"
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
        tag = f"#{r.hole_index}" if r.hole_index else (r.axis or "")
        if not m.any():
            warnings.append(f"case '{case.name}': load region {r.kind} {tag} matched no nodes")
            continue
        if fb_ax:
            warnings.append(
                f"case '{case.name}': region_fallback — load region {r.kind} {tag} "
                f"unresolved on the geometry; applied on {fb_ax} boundary patch "
                f"({int(cells.sum())} cells)"
            )
        idx = np.flatnonzero(m)

        applied = False
        if load.force_n and len(load.force_n) == 3:
            f = np.asarray(load.force_n, dtype=float)
            if load.remote_point and len(load.remote_point) >= 3:
                # force applied at a remote point: same resultant + the
                # moment (p - centroid) x F the lever arm induces
                c = domain.node_coords()[idx].mean(axis=0)
                lever = np.asarray(load.remote_point[:3], dtype=float) - c
                moment = np.cross(lever, f) / 1e3  # N·mm -> N·m
                ld.spread_force(F, idx, f)
                got = ld.spread_moment(domain, F, idx, moment)
                want = float(np.linalg.norm(moment))
                if want >= 1e-3 and got <= 0:
                    warnings.append(
                        f"case '{case.name}': remote force at {load.remote_point[:3]} "
                        f"should add {want:.2f} N·m but the region cannot carry a "
                        "couple (too small or degenerate) — only the force applied"
                    )
            else:
                ld.spread_force(F, idx, f)
            applied = True
        if load.bearing_n is not None and load.bearing_n > 0:
            g = _anchor_geometry(domain, r, anchors)
            if g is None:
                warnings.append(f"case '{case.name}': bearing_n needs a hole/anchor/bore region")
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
                        f"#{r.hole_index} produced no force — check bore/direction"
                    )
                applied = applied or got > 0
        if load.moment_nm and len(load.moment_nm) == 3:
            got = ld.spread_moment(domain, F, idx, np.asarray(load.moment_nm, dtype=float))
            if got <= 0:
                warnings.append(
                    f"case '{case.name}': moment on {r.kind} produced no torque "
                    "(region too small or degenerate)"
                )
            applied = applied or got > 0
        if load.preload_n is not None and load.preload_n > 0:
            g = _anchor_geometry(domain, r, anchors)
            if g is None:
                warnings.append(f"case '{case.name}': preload_n needs a hole/anchor/bore region")
            else:
                _center, avec, _dia, _h = g
                got = ld.bolt_preload(domain, F, cells.ravel(), avec, float(load.preload_n))
                if got <= 0:
                    warnings.append(
                        f"case '{case.name}': preload on anchor #{r.hole_index} found no "
                        "open bore mouths (blind hole or solid region?)"
                    )
                applied = applied or got > 0
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
        if not applied:
            warnings.append(f"case '{case.name}': load on {r.kind} resolved to zero force")

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
    domain: VoxelDomain, fixed3: np.ndarray, F: np.ndarray
) -> dict[str, list[float] | float]:
    """Load-case sanity report: applied resultant + moment about the centroid
    of the constrained nodes, plus constrained-DOF counts. Equilibrium needs
    reactions to balance the resultants."""
    res = F.reshape(-1, 3).sum(axis=0)
    out: dict[str, list[float] | float] = {
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


def keep_element_mask(
    domain: VoxelDomain, regions: list[RegionSpec], anchors: list[dict] | None = None
) -> np.ndarray:
    """Union of cell masks for non-design (keep) regions, flat cell order."""
    mask = np.zeros(domain.cell_count, dtype=bool)
    for r in regions:
        mask |= region_cell_mask(domain, r, anchors).ravel()
    return mask


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
