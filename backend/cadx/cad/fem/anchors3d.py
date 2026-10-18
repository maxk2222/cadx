"""Feature anchors scanned from the mesh on ALL axes.

`cad.mesh_features.detect_mesh_features` only scans +z (the top view). For
FEM anchors we need side holes too — a bolt bore on ±x/±y is just as much a
fixture point. This module casts ray heightfields along each axis, finds
through-tunnels (enclosed voids where rays pass through) and dimples
(countersinks / insert pockets), and emits anchor dicts compatible with
`agent.anchors.Anchor` (x, y, dia, axis, z, depth_mm, kind, source="scan").

Runs in the executor (trimesh + scipy + skimage only).
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# axis -> (axial index, in-plane u index, in-plane v index)
_AXES: dict[str, tuple[int, int, int]] = {
    "+x": (0, 1, 2),
    "+y": (1, 0, 2),
    "+z": (2, 0, 1),
}

# signed axis -> (axial index, sign of the ENTRY face: +1 = bounds high side)
_SIGNED_AXES: dict[str, tuple[int, int]] = {
    "+x": (0, +1),
    "-x": (0, -1),
    "+y": (1, +1),
    "-y": (1, -1),
    "+z": (2, +1),
    "-z": (2, -1),
}

# the two in-plane indices per axial index (same order convention as _AXES)
_IN_PLANE: dict[int, tuple[int, int]] = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

# anchor kinds a fastener/load can attach to (dimples/pockets are excluded —
# RegionSpec.hole_index counts only these, in list order)
FASTENER_KINDS = {"hole", "thread", "insert", "bearing", "bearing_seat", "tunnel", "boss"}


def is_fastener_anchor(a: dict[str, Any]) -> bool:
    return str(a.get("kind") or "hole") in FASTENER_KINDS


def _heightfield_axis(
    mesh: trimesh.Trimesh, axis: str, step_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, us, vs): max hit coord along `axis` per (u, v) cell, NaN where no hit."""
    ax, au, av = _AXES[axis]
    us = np.arange(mesh.bounds[0][au], mesh.bounds[1][au] + step_mm, step_mm)
    vs = np.arange(mesh.bounds[0][av], mesh.bounds[1][av] + step_mm, step_mm)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    origins = np.zeros((uu.size, 3))
    origins[:, au] = uu.ravel()
    origins[:, av] = vv.ravel()
    origins[:, ax] = mesh.bounds[1][ax] + 10.0
    directions = np.zeros((uu.size, 3))
    directions[:, ax] = -1.0
    locs, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    H = np.full(uu.shape, np.nan)
    for r, loc in zip(ray_ids, locs, strict=True):
        i, j = np.unravel_index(r, uu.shape)
        z = loc[ax]
        if np.isnan(H[i, j]) or z > H[i, j]:
            H[i, j] = z
    return H, us, vs


def _anchor(axis: str, u: float, v: float, dia: float, depth: float, kind: str) -> dict[str, Any]:
    """Anchor dict for a feature at in-plane (u, v) on `axis`.

    `x`/`y` keep the legacy top-view fields; `pos` carries the full 3D
    in-plane position (the axial coordinate stays None — unknown until FEM
    places it on the bore)."""
    _ax, au, av = _AXES[axis]
    p: list[float | None] = [None, None, None]
    p[au], p[av] = u, v
    return {
        "x": round(float(u if au == 0 else v if av == 0 else 0.0), 2),
        "y": round(float(u if au == 1 else v if av == 1 else 0.0), 2),
        "dia": round(float(dia), 2),
        "axis": axis,
        "z": None,
        "pos": [round(float(c), 2) if c is not None else None for c in p],
        "depth_mm": round(float(depth), 2),
        "kind": kind,
        "source": "scan",
    }


def scan_mesh_anchors(
    mesh: trimesh.Trimesh | str | Path,
    step_mm: float = 0.7,
    min_area_mm2: float = 3.0,
    dimple_depth_mm: float = 1.5,
) -> list[dict[str, Any]]:
    """Detect hole/dimple anchors on +x/+y/+z. Returns anchor dicts (unindexed)."""
    import scipy.ndimage as ndi
    from skimage.measure import regionprops

    if not isinstance(mesh, trimesh.Trimesh):
        loaded = trimesh.load(str(mesh), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            return []
        mesh = loaded
    out: list[dict[str, Any]] = []
    for axis in _AXES:
        try:
            H, us, vs = _heightfield_axis(mesh, axis, step_mm)
        except Exception:
            continue
        inside = ~np.isnan(H)
        footprint = ndi.binary_fill_holes(inside)
        rim = inside & ~ndi.binary_erosion(inside)
        ax = _AXES[axis][0]
        span = float(mesh.bounds[1][ax] - mesh.bounds[0][ax])

        # tunnels: void cells fully enclosed by material. Rays on boundary
        # cells still clip the bore wall, so the void area under-reports the
        # diameter by ~one bin — add half a step back.
        lab, n = ndi.label(footprint & ~inside)
        for i in range(1, n + 1):
            cells = lab == i
            area = float(cells.sum()) * step_mm * step_mm
            if area < min_area_mm2:
                continue
            ii, jj = np.where(cells)
            out.append(
                _anchor(
                    axis,
                    float(us[ii].mean()),
                    float(vs[jj].mean()),
                    2.0 * np.sqrt(area / np.pi) + 0.5 * step_mm,
                    span,
                    "hole",
                )
            )

        # dimples: local dips below the surrounding max (countersinks, pockets)
        Hf = np.nan_to_num(H, nan=0.0)
        win = max(round(7.0 / step_mm), 3)
        local_max = ndi.maximum_filter(Hf, size=win)
        lab, n = ndi.label(((local_max - Hf) > dimple_depth_mm) & inside)
        for prop in regionprops(lab):
            area = float(prop.area) * step_mm * step_mm
            if not (8.0 <= area <= 200.0):
                continue
            per_mm = float(prop.perimeter) * step_mm or step_mm
            if 4.0 * np.pi * (area / (per_mm * per_mm)) < 0.5:
                continue
            cells = lab == prop.label
            # a "dip" touching the silhouette rim is outline curvature, not a
            # pocket — real dimples are surrounded by material
            if (cells & rim).any():
                continue
            ii, jj = np.where(cells)
            depth = float((local_max[cells] - Hf[cells]).mean())
            # a dip whose bottom reaches the far face is a through-feature
            # (e.g. a countersunk bore whose cone swallowed the void column)
            near_face = float(mesh.bounds[0][ax]) + step_mm
            kind = "hole" if float(Hf[cells].min()) <= near_face else "dimple"
            out.append(
                _anchor(
                    axis,
                    float(us[ii].mean()),
                    float(vs[jj].mean()),
                    2.0 * np.sqrt(area / np.pi) + 0.5 * step_mm,
                    depth,
                    kind,
                )
            )
    return out


def merge_anchor_lists(
    existing: list[dict[str, Any]] | None,
    scanned: list[dict[str, Any]],
    tol_mm: float = 2.5,
) -> list[dict[str, Any]]:
    """Merge scanned anchors into the declared/measured list.

    Same-axis anchors within `tol_mm` in the in-plane coords are the same
    feature — the scanned entry enriches the existing one (real depth_mm /
    confirmed axis) instead of duplicating it. Re-indexes 1..n.
    """

    def _plane(a: dict[str, Any]) -> tuple[float, float]:
        _ax, au, av = _AXES.get(str(a.get("axis") or "+z"), (2, 0, 1))
        pos = a.get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            p = [float(c) if c is not None else 0.0 for c in pos]
        else:
            p = [
                float(a.get("x") or 0.0),
                float(a.get("y") or 0.0),
                float(a.get("z") or 0.0),
            ]
        return p[au], p[av]

    merged: list[dict[str, Any]] = [dict(a) for a in existing or []]
    for s in scanned:
        s_axis = str(s.get("axis") or "+z")
        su, sv = _plane(s)
        match = None
        for e in merged:
            if str(e.get("axis") or "+z") != s_axis:
                continue
            eu, ev = _plane(e)
            if float(np.hypot(eu - su, ev - sv)) <= tol_mm:
                match = e
                break
        if match is None:
            merged.append(dict(s))
            continue
        if not match.get("depth_mm") and s.get("depth_mm"):
            match["depth_mm"] = s["depth_mm"]
        if match.get("kind") == "hole" and s.get("kind") != "hole":
            match["kind"] = s["kind"]
        msrc = str(match.get("source") or "")
        if s.get("source") == "scan" and msrc and "scan" not in msrc:
            match["source"] = f"{msrc}+scan"
    # fastener anchors first so `index` == hole_index for mounts/bores; dimples
    # and pockets stay reachable via `anchor` regions but can't shadow a bore
    merged.sort(key=lambda a: not is_fastener_anchor(a))
    for i, a in enumerate(merged):
        a["index"] = i + 1
    return merged


# --------------------------------------------------------------------------- #
# blind-feature probe                                                           #
# --------------------------------------------------------------------------- #

_PROBE_RAYS = 8  # rays on the bore ring (plus the center ray)
_PROBE_RING_FRAC = 0.35  # ring radius = this fraction of the declared dia


def _afield(a: Any, key: str, default: Any = None) -> Any:
    """Anchor field lookup that accepts dicts and Anchor-like objects."""
    if isinstance(a, dict):
        return a.get(key, default)
    return getattr(a, key, default)


def _probe_axis(a: Any) -> str:
    """Signed axis string for an anchor; 'x'/'y'/'z' normalize to '+x'/..."""
    raw = str(_afield(a, "axis") or "+z").strip().lower()
    if raw in ("x", "y", "z"):
        raw = "+" + raw
    return raw if raw in _SIGNED_AXES else "+z"


def _probe_uv(a: Any, ax: int) -> tuple[float, float] | None:
    """In-plane (u, v) probe point for an anchor on axial index `ax`."""
    au, av = _IN_PLANE[ax]
    u = v = None
    pos = _afield(a, "pos")
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        u, v = pos[au], pos[av]
    if (u is None or v is None) and ax == 2:
        # legacy 2D anchors carry only x/y — the z-plane position
        u = _afield(a, "x") if u is None else u
        v = _afield(a, "y") if v is None else v
    if u is None or v is None:
        return None
    return float(u), float(v)


def probe_blind_features(
    mesh: trimesh.Trimesh,
    anchors: list[Any],
    *,
    log: Any = None,
) -> list[dict[str, Any]]:
    """Ray-probe each declared anchor for a real cavity — incl. BLIND bores.

    `scan_mesh_anchors` only proves through-tunnels (rays that cross the
    whole part); a blind tap/pocket never shows up there. For every anchor
    with a resolvable in-plane position we cast a centre ray plus a ring of
    parallel rays (radius ~0.35·dia) inward along -axis, starting just
    outside the entry face:

    - the entry face is the bounds side the axis sign points at ("+x" enters
      from x_max — same convention as `scan_mesh_anchors`/`_heightfield_axis`);
    - a wider reference ring estimates the local entry surface, so organic
      dips below the bbox face do not fake bore depth;
    - a ray that never hits counts as unobstructed (deep) — robust on
      non-watertight meshes;
    - the open-ray fraction over the probe disc estimates the effective bore
      diameter (measured_dia_mm ≈ dia·sqrt(fraction)).

    Returns one dict per input anchor:
    {"name", "kind", "axis", "pos", "dia_mm", "depth_mm",
     "found": any cavity detected at the anchor,
     "open": measured depth >= 80% of expected (depth_mm else axis span),
     "measured_depth_mm": float|None, "measured_dia_mm": float|None}.
    """
    out: list[dict[str, Any]] = []
    for a in anchors or []:
        axis = _probe_axis(a)
        ax, sign = _SIGNED_AXES[axis]
        au, av = _IN_PLANE[ax]
        dia = float(_afield(a, "dia") or _afield(a, "dia_mm") or 0.0)
        depth = _afield(a, "depth_mm")
        pos = _afield(a, "pos")
        pos_out: list[float | None] | None = None
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            with contextlib.suppress(TypeError, ValueError):
                pos_out = [float(c) if c is not None else None for c in pos]
        rec: dict[str, Any] = {
            "name": str(_afield(a, "name") or f"anchor {_afield(a, 'index', '?')}"),
            "kind": str(_afield(a, "kind") or "hole"),
            "axis": axis,
            "pos": pos_out,
            "dia_mm": round(dia, 2) if dia > 0 else None,
            "depth_mm": round(float(depth), 2) if depth else None,
            "found": False,
            "open": False,
            "measured_depth_mm": None,
            "measured_dia_mm": None,
        }
        out.append(rec)
        uv = _probe_uv(a, ax)
        bounds = mesh.bounds
        if uv is None or bounds is None:
            continue
        span = float(bounds[1][ax] - bounds[0][ax])
        if span <= 0:
            continue
        u, v = uv
        d_probe = dia if dia > 0 else 4.0
        standoff = max(1.0, 0.05 * span)

        # probe bundle: centre ray + ring at ~0.35·dia; the wider reference
        # ring samples the surrounding entry surface level
        offs = [(0.0, 0.0)]
        r_ring = _PROBE_RING_FRAC * d_probe
        offs += [
            (r_ring * math.cos(t), r_ring * math.sin(t))
            for t in np.linspace(0.0, 2.0 * np.pi, _PROBE_RAYS, endpoint=False)
        ]
        n_probe = len(offs)
        r_ref = max(0.8 * d_probe, r_ring + 2.0)
        offs += [
            (r_ref * math.cos(t), r_ref * math.sin(t))
            for t in np.linspace(0.0, 2.0 * np.pi, _PROBE_RAYS, endpoint=False)
        ]
        origins = np.zeros((len(offs), 3))
        origins[:, au] = u + np.asarray([o[0] for o in offs])
        origins[:, av] = v + np.asarray([o[1] for o in offs])
        origins[:, ax] = (
            float(bounds[1][ax]) + standoff if sign > 0 else float(bounds[0][ax]) - standoff
        )
        directions = np.zeros((len(offs), 3))
        directions[:, ax] = -float(sign)
        try:
            locs, ray_ids, _ = mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=directions, multiple_hits=True
            )
        except Exception as e:
            if log is not None:
                with contextlib.suppress(Exception):
                    log.debug("blind_probe_ray_failed", error=str(e))
            continue
        # first-hit distance per ray; inf = never hit = unobstructed/deep
        first = np.full(len(offs), np.inf)
        for r_id, loc in zip(ray_ids, locs, strict=True):
            d = float(np.linalg.norm(loc - origins[int(r_id)]))
            if d < first[int(r_id)]:
                first[int(r_id)] = d
        ref_hits = first[n_probe:]
        hitting = ref_hits[np.isfinite(ref_hits)]
        surface = float(hitting.min()) if len(hitting) else standoff
        clear = first[:n_probe] - surface  # free travel past the entry surface
        expected = float(depth) if depth else span
        open_min = max(1.0, 0.02 * expected)
        open_rays = clear > open_min
        if not open_rays.any():
            continue
        rec["found"] = True
        measured_depth = float(np.median(np.minimum(clear[open_rays], span)))
        rec["measured_depth_mm"] = round(measured_depth, 2)
        rec["measured_dia_mm"] = round(d_probe * math.sqrt(float(open_rays.mean())), 2)
        rec["open"] = measured_depth >= 0.8 * expected
    return out
