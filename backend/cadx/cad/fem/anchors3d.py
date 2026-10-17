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
