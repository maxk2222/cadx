"""Render fixed/load/keep region selections on the voxel domain.

One labelled PNG per load case: the selections painted onto the smooth part
skin (fixed=blue, load=red, keep=green) with force arrows landing on each
load centroid. Used by the optimize session's `preview_regions` tool so the
agent can sanity-check where forces land before spending a SIMP run.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from cadx.cad.fem.regions import (
    _region_fallback,
    case_reactions,
    region_cell_mask,
    region_face_normal,
    resolve_case_dofs,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec
from cadx.logging import get_logger

log = get_logger(__name__)

_FIXED, _LOAD, _KEEP, _VOID = 1, 2, 3, 4


def case_region_masks(
    domain: VoxelDomain,
    case: LoadCase,
    keep_regions: list[RegionSpec],
    anchors: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fixed, load, keep) boolean cell masks for one case on the domain."""
    fixed = np.zeros(domain.dims, dtype=bool)
    load = np.zeros(domain.dims, dtype=bool)
    keep = np.zeros(domain.dims, dtype=bool)
    for r in case.fixed:
        fixed |= region_cell_mask(domain, r, anchors)
    for ld in case.loads:
        load |= region_cell_mask(domain, ld.region, anchors)
    for r in keep_regions:
        keep |= region_cell_mask(domain, r, anchors)
    return fixed, load, keep


def cells_preview(
    domain: VoxelDomain,
    cases: list[LoadCase],
    keep_regions: list[RegionSpec],
    anchors: list[dict] | None = None,
    max_points: int = 4000,
) -> list[dict[str, Any]]:
    """Subsampled cell-centre coordinates per case for a 3D overlay in the
    structural editor — the actual resolved selection, not a 2D render."""
    centers = domain.element_centers().reshape((*domain.dims, 3))
    out: list[dict[str, Any]] = []
    for case in cases:
        fixed, load, keep = case_region_masks(domain, case, keep_regions, anchors)
        entry: dict[str, Any] = {"case": case.name}
        for name, mask in (("fixed", fixed), ("load", load), ("keep", keep)):
            c = centers[mask]
            if len(c) > max_points:
                c = c[np.linspace(0, len(c) - 1, max_points).astype(int)]
            entry[name] = [[round(float(v), 3) for v in row] for row in c]
        out.append(entry)
    return out


_TAG_RGB = {
    0: (184, 190, 198),  # base part — light grey
    _FIXED: (65, 105, 225),  # blue
    _LOAD: (220, 70, 60),  # red
    _KEEP: (70, 170, 90),  # green
    _VOID: (150, 90, 200),  # purple
}


def _load_vec(domain: VoxelDomain, load: LoadSpec, lmask: np.ndarray) -> np.ndarray | None:
    """Resultant vector for one load spec — direction + magnitude for the
    preview arrow. Returns None when the load kind draws no arrow."""
    vec: np.ndarray | None = None
    if load.force_n and len(load.force_n) >= 3:
        vec = np.asarray(load.force_n[:3], dtype=float)
    elif load.bearing_n:
        d = load.bearing_dir[:3] if len(load.bearing_dir) >= 3 else [0.0, -1.0, 0.0]
        vec = np.asarray(d, dtype=float) / (np.linalg.norm(d) or 1.0) * float(load.bearing_n)
    elif getattr(load, "normal_force_n", None):
        n = region_face_normal(load.region)
        if n is not None:
            vec = -n * float(load.normal_force_n)
    elif load.pressure_mpa is not None:
        n = region_face_normal(load.region)
        if n is not None:
            vec = -n * float(load.pressure_mpa) * int(lmask.sum()) * domain.pitch**2
    elif load.preload_n:
        vec = np.array([0.0, 0.0, -float(load.preload_n)])
    return vec


def _load_arrows(
    domain: VoxelDomain,
    case: LoadCase,
    anchors: list[dict] | None,
    centers3d: np.ndarray,
    arrow_len: float,
) -> tuple[list[tuple[Any, Any]], list[str], float]:
    """(arrows+markers, force legend strings, total N) — every load kind
    gets a direction arrow + centroid marker, not just point forces:
    bearing/pressure/preload are just as easy to misplace."""
    import pyvista as pv

    force_n = 0.0
    arrows: list[tuple[Any, Any]] = []
    f_strs: list[str] = []
    for load in case.loads:
        lmask = region_cell_mask(domain, load.region, anchors)
        if not lmask.any():
            # declared region missed the geometry — draw on the
            # deterministic fallback patch, not mid-air
            lmask = _region_fallback(domain, load.region, anchors)[0]
        vec = _load_vec(domain, load, lmask)
        if vec is None:
            continue
        if load.force_n and len(load.force_n) >= 3:
            f_strs.append(f"F=({vec[0]:.0f},{vec[1]:.0f},{vec[2]:.0f}) N")
        n = float(np.linalg.norm(vec))
        if n <= 0 or not lmask.any():
            continue
        force_n += n
        # arrow TIP lands on the load-region centroid
        centroid = centers3d[lmask].mean(axis=0)
        unit = vec / n
        arrows.append(
            (
                pv.Arrow(start=centroid - unit * arrow_len, direction=unit, scale=arrow_len),
                pv.Sphere(radius=domain.pitch, center=centroid),
            )
        )
    if force_n == 0.0:
        # moment/g loads draw no single arrow — fall back to the resolved
        # resultant so the caption isn't "0 N"
        F = resolve_case_dofs(domain, case, anchors)[1]
        force_n = float(np.linalg.norm(F.reshape(-1, 3).sum(axis=0)))
    return arrows, f_strs, force_n


def render_regions(
    domain: VoxelDomain,
    cases: list[LoadCase],
    keep_regions: list[RegionSpec],
    out_dir: Path,
    anchors: list[dict] | None = None,
    source_mesh: trimesh.Trimesh | None = None,
) -> list[str]:
    """Write regions_<case>.png previews (iso + top, with force arrows);
    returns the filenames written."""
    try:
        import pyvista as pv

        from cadx.cad.fem.analyze import _part_surface
    except Exception:
        log.warning("fem.preview.no_pyvista")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    lo = np.asarray(domain.origin, dtype=float)
    hi = lo + domain.pitch * np.asarray(domain.dims, dtype=float)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    centers3d = domain.element_centers().reshape((*domain.dims, 3))
    arrow_len = 0.25 * diag

    written: list[str] = []
    for case in cases:
        try:
            fixed_cells, load_cells, keep_cells = case_region_masks(
                domain, case, keep_regions, anchors
            )
            arrows, f_strs, force_n = _load_arrows(
                domain, case, anchors, centers3d, arrow_len
            )

            tag = np.zeros(domain.dims, dtype=np.uint8)
            tag[keep_cells] = _KEEP
            # loads outrank auto-keep rings so bore loads stay visible
            tag[load_cells] = _LOAD
            tag[fixed_cells] = _FIXED  # constraints outrank loads on overlap
            grid = pv.ImageData(
                dimensions=tuple(d + 1 for d in domain.dims),
                spacing=(domain.pitch,) * 3,
                origin=tuple(domain.origin),
            )
            # cell arrays are (nx,ny,nz) C-order; ImageData wants x-fastest
            grid.cell_data["solid"] = domain.solid.ravel(order="F").astype(np.uint8)
            grid.cell_data["tag"] = tag.ravel(order="F")
            body = grid.threshold(0.5, scalars="solid")

            # region tags painted onto the smooth part skin — opaque voxel
            # dumps turned unreadable on thin shells (a legitimately large
            # 'surface' selection covered most cells and rendered as
            # full-height red stripes)
            skin = _part_surface(source_mesh, body)
            with contextlib.suppress(Exception):
                for _ in range(3):
                    if getattr(skin, "n_faces", 0) >= 12000:
                        break
                    skin = skin.subdivide(1, subfilter="linear")
            rgb = np.tile(np.asarray(_TAG_RGB[0], dtype=np.uint8), (skin.n_points, 1))
            try:
                from scipy import ndimage
                from scipy.spatial import cKDTree

                # only boundary cells carry surface paint — nearest-ANY-cell
                # lookups hit untagged interior cells and speckle the overlay
                boundary = domain.solid & ~ndimage.binary_erosion(domain.solid)
                flat = centers3d.reshape(-1, 3)[boundary.ravel()]
                nn = cKDTree(flat).query(np.asarray(skin.points))[1]
                ptag = tag.ravel()[np.flatnonzero(boundary.ravel())[nn]]
                for value, color in _TAG_RGB.items():
                    rgb[ptag == value] = color
            except Exception:
                pass

            caption = (
                f"{case.name}: fixed {int(fixed_cells.sum())} | load "
                f"{int(load_cells.sum())} ({force_n:.0f} N) | keep "
                f"{int(keep_cells.sum())} cells"
            )

            pl: Any = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1600, 700])
            pl.set_background("white")

            def _add_regions(
                pl: Any = pl, skin: Any = skin, rgb: Any = rgb, arrows: Any = arrows
            ) -> None:
                # no voxel body ghost — the skin carries the part shape; a
                # translucent sparse-voxel halo reintroduces the floating-
                # debris look on thin shells
                pl.add_mesh(skin, scalars=rgb, rgb=True, opacity=1.0, smooth_shading=True)
                for arrow, marker in arrows:
                    pl.add_mesh(arrow, color="darkorange", opacity=1.0)
                    pl.add_mesh(marker, color="orange", opacity=1.0)

            pl.subplot(0, 0)
            _add_regions()
            pl.camera_position = "iso"
            pl.reset_camera()
            pl.add_text(caption, position="upper_edge", font_size=9, color="black")
            legend = "blue=fix red=load green=keep"
            if f_strs:
                legend += " | " + "; ".join(f_strs)
            pl.add_text(legend, position="lower_edge", font_size=9, color="black")
            pl.add_text("iso", position="upper_left", font_size=9, color="black")

            pl.subplot(0, 1)
            _add_regions()
            pl.camera.position = (center[0], center[1], center[2] + diag)
            pl.camera.focal_point = tuple(center)
            pl.camera.up = (0.0, 1.0, 0.0)
            pl.reset_camera()
            pl.camera.parallel_projection = True
            pl.add_text("top", position="upper_left", font_size=9, color="black")

            png = out_dir / f"regions_{case.name}.png"
            pl.screenshot(str(png))
            pl.close()
            written.append(png.name)
        except Exception as e:
            log.warning("fem.preview.render_failed", case=case.name, error=str(e)[:200])
    return written


def resolve_case_summary(
    domain: VoxelDomain, case: LoadCase, anchors: list[dict] | None = None
) -> dict[str, Any]:
    """Compact numbers for one case — region hit counts, resultant force and
    moment about the constraint centroid (sanity check for the caller)."""
    fixed3, F, warnings = resolve_case_dofs(domain, case, anchors)
    forces = F.reshape(-1, 3)
    loaded = np.linalg.norm(forces, axis=1) > 0
    return {
        "case": case.name,
        "fixed_nodes": int(fixed3.any(axis=1).sum()),
        "loaded_nodes": int(loaded.sum()),
        "total_force_n": round(float(np.linalg.norm(forces[loaded].sum(axis=0))), 2),
        "reactions": case_reactions(domain, fixed3, F),
        "warnings": warnings,
    }
