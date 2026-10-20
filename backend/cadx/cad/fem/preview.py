"""Render fixed/load/keep region selections on the voxel domain.

One labelled PNG per load case: the part semi-transparent grey, fixed cells
blue, load cells red, keep cells green. Used by the optimize session's
`preview_regions` tool so the agent can sanity-check where forces land
before spending a SIMP run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cadx.cad.fem.regions import case_reactions, region_cell_mask, resolve_case_dofs
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import LoadCase, RegionSpec
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


def render_regions(
    domain: VoxelDomain,
    cases: list[LoadCase],
    keep_regions: list[RegionSpec],
    out_dir: Path,
    anchors: list[dict] | None = None,
) -> list[str]:
    """Write regions_<case>.png previews (iso + top, with force arrows);
    returns the filenames written."""
    try:
        import pyvista as pv
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
            force_n = 0.0
            arrows: list[tuple[Any, Any]] = []  # (arrow, centroid marker)
            f_strs: list[str] = []
            for load in case.loads:
                if load.force_n and len(load.force_n) >= 3:
                    vec = np.asarray(load.force_n[:3], dtype=float)
                    n = float(np.linalg.norm(vec))
                    force_n += n
                    f_strs.append(f"F=({vec[0]:.0f},{vec[1]:.0f},{vec[2]:.0f}) N")
                    if n <= 0:
                        continue
                    lmask = region_cell_mask(domain, load.region, anchors)
                    if not lmask.any():
                        continue
                    # arrow TIP lands on the load-region centroid
                    centroid = centers3d[lmask].mean(axis=0)
                    unit = vec / n
                    arrows.append(
                        (
                            pv.Arrow(
                                start=centroid - unit * arrow_len,
                                direction=unit,
                                scale=arrow_len,
                            ),
                            pv.Sphere(radius=domain.pitch, center=centroid),
                        )
                    )
                elif load.pressure_mpa is not None:
                    force_n += float(load.pressure_mpa) * int(load_cells.sum()) * domain.pitch**2

            tag = np.zeros(domain.dims, dtype=np.uint8)
            tag[fixed_cells] = _FIXED
            tag[load_cells] = _LOAD
            tag[keep_cells] = _KEEP
            grid = pv.ImageData(
                dimensions=tuple(d + 1 for d in domain.dims),
                spacing=(domain.pitch,) * 3,
                origin=tuple(domain.origin),
            )
            grid.cell_data["solid"] = domain.solid.ravel().astype(np.uint8)
            grid.cell_data["tag"] = tag.ravel()
            body = grid.threshold(0.5, scalars="solid")

            caption = (
                f"{case.name}: fixed {int(fixed_cells.sum())} | load "
                f"{int(load_cells.sum())} ({force_n:.0f} N) | keep "
                f"{int(keep_cells.sum())} cells — blue=fix red=load green=keep"
            )
            if f_strs:
                caption += " | " + "; ".join(f_strs)

            pl: Any = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1600, 700])
            pl.set_background("white")

            def _add_regions(
                pl: Any = pl, body: Any = body, grid: Any = grid, arrows: Any = arrows
            ) -> None:
                pl.add_mesh(body, color="#b8bec6", opacity=0.25, smooth_shading=True)
                for value, color in ((_FIXED, "blue"), (_LOAD, "red"), (_KEEP, "green")):
                    sub = grid.threshold([value - 0.5, value + 0.5], scalars="tag")
                    if sub.n_cells:
                        pl.add_mesh(sub, color=color, opacity=0.9)
                pl.add_mesh(body.outline(), color="#555555", line_width=1.0)
                for arrow, marker in arrows:
                    pl.add_mesh(arrow, color="darkorange", opacity=1.0)
                    pl.add_mesh(marker, color="orange", opacity=1.0)

            pl.subplot(0, 0)
            _add_regions()
            pl.camera_position = "iso"
            pl.reset_camera()
            pl.add_text(caption, position="upper_edge", font_size=9, color="black")
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
