"""Single-solve FEM verification: mesh -> voxel grid -> per-case stress report.

Writes fem/report.json plus pyvista renders (von Mises clip + exaggerated
deformation) into <out_dir>/fem/. Pure function — used by the fem_analyze
task and by the TopOpt post-verification.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pydantic import BaseModel, Field, computed_field

from cadx.cad.fem.materials import MaterialProps, resolve_material
from cadx.cad.fem.regions import case_reactions, material_cell_factors, resolve_case_dofs
from cadx.cad.fem.solver import (
    CaseSolve,
    FemSolution,
    _fixed_dofs,
    assemble,
    reduce_case,
    solve_reduced,
)
from cadx.cad.fem.voxelize import VoxelDomain, volume_of_solid, voxelize
from cadx.llm.schemas import LoadCase
from cadx.logging import get_logger

log = get_logger(__name__)


class FemCaseResult(BaseModel):
    name: str
    max_vm_mpa: float = 0.0
    max_disp_mm: float = 0.0
    safety_factor: float | None = None
    compliance_j: float = 0.0
    converged: bool = True
    solver: str = ""
    warnings: list[str] = Field(default_factory=list)
    # applied resultant force / moment about the constraint centroid —
    # sanity check the resolved loads did what was requested
    reactions: dict = Field(default_factory=dict)
    vonmises_png: str | None = None
    deformed_png: str | None = None


class FemReport(BaseModel):
    material: dict = Field(default_factory=dict)
    grid: dict = Field(default_factory=dict)  # pitch, dims, elements, dofs
    cases: list[FemCaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    note: str = (
        "voxel-FEM estimate (linear elasticity, ~10-20% accuracy) — "
        "design feedback, not certification"
    )

    @computed_field  # serialized into report.json / API payloads
    @property
    def worst_safety_factor(self) -> float | None:
        sfs = [c.safety_factor for c in self.cases if c.safety_factor is not None]
        return min(sfs) if sfs else None


def _grid_image(domain: VoxelDomain, elem_scalar: np.ndarray, node_vec: np.ndarray):
    """pyvista ImageData with per-cell scalar + per-node vector arrays."""
    import pyvista as pv

    grid = pv.ImageData(
        dimensions=tuple(d + 1 for d in domain.dims),
        spacing=(domain.pitch,) * 3,
        origin=tuple(domain.origin),
    )
    grid.cell_data["vm"] = elem_scalar
    grid.cell_data["solid"] = domain.solid.ravel().astype(np.uint8)
    grid.point_data["u"] = node_vec
    return grid


def _part_surface(mesh: trimesh.Trimesh | None, body: Any) -> Any:
    """Smooth part skin for overlay — the real trimesh surface when the run
    has one, else the voxel body's extracted surface (still blocky, but
    confined to the part)."""
    import pyvista as pv

    faces = getattr(mesh, "faces", None)
    if mesh is not None and faces is not None and len(faces):
        try:
            faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
            return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)
        except Exception:
            pass
    return body.extract_surface(algorithm="dataset_surface")


def _render_case(
    domain: VoxelDomain,
    sol: FemSolution,
    case_name: str,
    out_dir: Path,
    suffix: str = "",
    source_mesh: trimesh.Trimesh | None = None,
) -> tuple[str | None, str | None]:
    """Render a 2x2 von-Mises sheet + exaggerated deformed shape. Returns filenames.

    The sheet shows iso / top / front views of the per-cell stress field
    projected onto the smooth part surface, plus a y-mid section view that
    clips the voxel body so interior hotspots stay visible."""
    try:
        import pyvista as pv
    except Exception:
        return None, None

    vm_png = f"vonmises{suffix}_{case_name}.png"
    def_png = f"deformed{suffix}_{case_name}.png"
    try:
        grid = _grid_image(domain, sol.elem_vm, sol.node_disp)
        body = grid.threshold(0.5, scalars="solid")
        vmax = max(float(sol.elem_vm.max()), 1e-9)

        lo = np.asarray(domain.origin, dtype=float)
        hi = lo + domain.pitch * np.asarray(domain.dims, dtype=float)
        center = (lo + hi) / 2.0
        diag = float(np.linalg.norm(hi - lo)) or 1.0

        skin = _part_surface(source_mesh, body)
        if getattr(skin, "n_faces", 0) < 8000:
            # smooth coarse analytic surfaces so the overlay isn't faceted
            with contextlib.suppress(Exception):
                skin = skin.subdivide(1, subfilter="linear")
        # paint the per-cell field onto the smooth skin — nearest element
        # centre gives a continuous overlay instead of blocky voxel faces
        try:
            from scipy.spatial import cKDTree

            solid_mask = np.asarray(domain.solid).ravel()
            centers = domain.element_centers()[solid_mask]
            vals = np.asarray(sol.elem_vm, dtype=float)[solid_mask]
            nn = cKDTree(centers).query(np.asarray(skin.points))[1]
            skin.point_data["vm"] = vals[nn]
        except Exception:
            pass
        has_vm = "vm" in skin.point_data
        skin_kw: dict[str, Any] = (
            {"scalars": "vm", "cmap": "turbo", "clim": [0, vmax]}
            if has_vm
            else {"color": "#c8ccd4"}
        )
        try:
            edges = skin.extract_feature_edges(30)
        except Exception:
            edges = None

        pl: Any = pv.Plotter(off_screen=True, shape=(2, 2), window_size=[1400, 1000])
        pl.set_background("white")

        def _add_stress_view(show_bar: bool) -> None:
            # outer views: smooth skin painted with the projected field —
            # hotspot VOXELS belong to the section view; on thin curved
            # shells boundary cells poke past the skin and read as floating
            # debris (observed on the bail handle)
            pl.add_mesh(
                skin,
                opacity=0.92,
                smooth_shading=True,
                show_scalar_bar=show_bar and has_vm,
                scalar_bar_args={"title": "von Mises [MPa]"} if has_vm else None,
                **skin_kw,
            )
            if edges is not None and edges.n_cells:
                pl.add_mesh(
                    edges,
                    color="#222222",
                    line_width=1.0,
                    opacity=0.5,
                    show_scalar_bar=False,
                )

        pl.subplot(0, 0)
        _add_stress_view(True)
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.add_text("iso", position="upper_left", font_size=10, color="black")
        pl.add_text(
            f"{case_name}: max {sol.max_vm_mpa:.1f} MPa, max disp {sol.max_disp_mm:.2f} mm",
            position="upper_edge",
            font_size=10,
            color="black",
        )

        pl.subplot(0, 1)
        _add_stress_view(False)
        pl.camera.position = (center[0], center[1], center[2] + diag)
        pl.camera.focal_point = tuple(center)
        pl.camera.up = (0.0, 1.0, 0.0)
        pl.reset_camera()
        pl.camera.parallel_projection = True
        pl.add_text("top", position="upper_left", font_size=10, color="black")

        pl.subplot(1, 0)
        _add_stress_view(False)
        pl.camera.position = (center[0], center[1] + diag, center[2])
        pl.camera.focal_point = tuple(center)
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.reset_camera()
        pl.camera.parallel_projection = True
        pl.add_text("front", position="upper_left", font_size=10, color="black")

        pl.subplot(1, 1)
        # keep the -Y half: from a +Y camera the cut face exposes the interior
        section = body.clip(normal="y", origin=tuple(center))
        pl.add_mesh(section, scalars="vm", cmap="turbo", clim=[0, vmax], show_scalar_bar=False)
        pl.add_mesh(skin, color="#c8ccd4", opacity=0.28, show_scalar_bar=False)
        pl.camera.position = (center[0], center[1] + diag, center[2])
        pl.camera.focal_point = tuple(center)
        pl.camera.up = (0.0, 0.0, 1.0)
        pl.reset_camera()
        pl.camera.parallel_projection = True
        pl.add_text("section y-mid", position="upper_left", font_size=10, color="black")

        pl.screenshot(str(out_dir / vm_png))
        pl.close()

        scale = domain.pitch * min(domain.dims) * 0.15 / max(sol.max_disp_mm, 1e-6)
        factor = min(scale, domain.pitch * min(domain.dims))
        # deformed smooth skin: displace the real surface by the nearest
        # voxel node's solution vector — smooth shape, field-driven motion
        u_nodes = np.asarray(sol.node_disp, dtype=float).reshape(-1, 3)
        pl = pv.Plotter(off_screen=True, window_size=[900, 700])
        pl.set_background("white")
        pl.add_mesh(skin, color="#b0b8c4", opacity=0.22, show_scalar_bar=False)
        try:
            from scipy.spatial import cKDTree

            nn = cKDTree(domain.node_coords()).query(np.asarray(skin.points))[1]
            skin_w = skin.copy()
            skin_w.points = skin_w.points + u_nodes[nn] * factor
            pl.add_mesh(
                skin_w,
                opacity=0.85,
                smooth_shading=True,
                scalar_bar_args={"title": "von Mises [MPa]"} if has_vm else None,
                **skin_kw,
            )
        except Exception:
            warped = body.warp_by_vector("u", factor=factor)
            pl.add_mesh(warped, scalars="vm", cmap="turbo", clim=[0, vmax])
        pl.add_mesh(body.outline(), color="#888888", line_width=1.0)
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.add_text(
            f"{case_name}: deformed x{factor:.0f}",
            position="upper_edge",
            font_size=10,
            color="black",
        )
        pl.screenshot(str(out_dir / def_png))
        pl.close()
    except Exception as e:
        log.warning("fem.render_failed", case=case_name, error=str(e)[:200])
        return None, None
    return vm_png, def_png


def _report_skeleton(mat: MaterialProps) -> FemReport:
    return FemReport(
        material={
            "name": mat.name,
            "elastic_modulus_mpa": mat.elastic_modulus_mpa,
            "poisson": mat.poisson,
            "yield_mpa": mat.yield_mpa,
            "density_g_cm3": mat.density_g_cm3,
        }
    )


def run_fem(
    mesh: trimesh.Trimesh,
    load_cases: list[LoadCase],
    material_name: str = "",
    *,
    material: MaterialProps | None = None,
    voxel_mm: float = 1.5,
    max_elements: int = 150_000,
    out_dir: Path | None = None,
    anchors: list[dict] | None = None,
    keep_regions: list | None = None,
    spsolve_dofs: int = 120_000,
    mem_budget_mb: float | None = None,
    suffix: str = "",
    cancelled: Callable[[], bool] | None = None,
) -> FemReport:
    """Solve each load case on the voxelized mesh. Always returns a report —
    per-case failures are captured in case.warnings."""
    mat = material or resolve_material(material_name)
    report = _report_skeleton(mat)
    try:
        domain = voxelize(mesh, voxel_mm, max_elements)
    except Exception as e:
        report.warnings.append(f"voxelization failed: {e}")
        return report
    if mem_budget_mb:
        from cadx.cad.fem.voxelize import estimate_fem_memory_mb

        direct = 3 * domain.node_count <= spsolve_dofs
        est = estimate_fem_memory_mb(domain, len(load_cases), direct)
        if est > mem_budget_mb * 0.75:
            report.warnings.append(
                f"predicted FEM memory ~{est:.0f} MB exceeds the executor budget "
                f"({mem_budget_mb:.0f} MB) — solve skipped; raise fem_voxel_mm "
                "or exec_mem_limit_mb"
            )
            return report
    _solve_cases(
        domain,
        mat,
        load_cases,
        report,
        out_dir,
        anchors,
        suffix,
        keep_regions,
        spsolve_dofs,
        source_mesh=mesh,
        cancelled=cancelled,
    )
    return report


def run_fem_domain(
    domain: VoxelDomain,
    solid_mask: np.ndarray,
    load_cases: list[LoadCase],
    material_name: str = "",
    *,
    material: MaterialProps | None = None,
    out_dir: Path | None = None,
    anchors: list[dict] | None = None,
    keep_regions: list | None = None,
    spsolve_dofs: int = 120_000,
    suffix: str = "",
    source_mesh: trimesh.Trimesh | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FemReport:
    """Verify a structure defined directly on an existing grid — used for
    TopOpt post-verification where re-voxelizing the smoothed export mesh
    would drop thin members."""
    mat = material or resolve_material(material_name)
    report = _report_skeleton(mat)
    sub = VoxelDomain(
        origin=domain.origin,
        pitch=domain.pitch,
        dims=domain.dims,
        solid=np.asarray(solid_mask, dtype=bool).reshape(domain.dims),
        pitch_requested=domain.pitch,
    )
    _solve_cases(
        sub,
        mat,
        load_cases,
        report,
        out_dir,
        anchors,
        suffix,
        keep_regions,
        spsolve_dofs,
        source_mesh=source_mesh,
        cancelled=cancelled,
    )
    return report


def _solve_cases(
    domain: VoxelDomain,
    mat: MaterialProps,
    load_cases: list[LoadCase],
    report: FemReport,
    out_dir: Path | None,
    anchors: list[dict] | None = None,
    suffix: str = "",
    keep_regions: list | None = None,
    spsolve_dofs: int = 120_000,
    source_mesh: trimesh.Trimesh | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    report.grid = {
        "pitch_mm": round(domain.pitch, 4),
        "dims": list(domain.dims),
        "solid_elements": domain.solid_count,
        "volume_mm3": round(volume_of_solid(domain), 1),
    }

    if not load_cases:
        report.warnings.append("no load cases provided")
        return

    elem_E = np.where(domain.solid.ravel(), mat.elastic_modulus_mpa, 0.0)
    if keep_regions:
        # insert/material zones: keep-regions with `material` override E
        factors = material_cell_factors(domain, keep_regions, anchors, mat.elastic_modulus_mpa)
        elem_E = elem_E * factors

    fem_dir = (out_dir / "fem") if out_dir else None
    if fem_dir:
        fem_dir.mkdir(parents=True, exist_ok=True)

    # K is identical across load cases (elem_E is fixed — only the BCs and
    # the load vector differ): assemble once (lazily, on the first solvable
    # case) and share one reduced system (Kf extraction + factorization)
    # between cases whose fixed-DOF signature is identical.
    asm: tuple[Any, np.ndarray, np.ndarray] | None = None  # (K, edof, active)
    ctx_cache: dict[bytes, CaseSolve] = {}

    for i, case in enumerate(load_cases):
        if cancelled is not None and cancelled():
            report.warnings.append("cancelled — remaining load cases skipped")
            break
        res = FemCaseResult(name=case.name or f"case{i + 1}")
        fixed3, F, warnings = resolve_case_dofs(
            domain, case, anchors, density_g_cm3=mat.density_g_cm3
        )
        res.warnings += warnings
        res.reactions = case_reactions(domain, fixed3, F)
        if not fixed3.any() or not np.any(F):
            report.cases.append(res)
            continue
        if asm is None:
            asm = assemble(domain, elem_E, mat.poisson)
        K, edof, active = asm
        sig = np.unique(_fixed_dofs(fixed3, domain.node_count)).tobytes()
        ctx = ctx_cache.get(sig)
        if ctx is None:
            ctx = reduce_case(domain, K, edof, fixed3, spsolve_dofs=spsolve_dofs)
            ctx_cache[sig] = ctx
        sol = solve_reduced(
            domain,
            ctx,
            edof,
            active,
            F,
            elem_E,
            mat.poisson,
            E_ref=mat.elastic_modulus_mpa,
        )
        res.warnings += sol.warnings
        res.max_vm_mpa = round(sol.max_vm_mpa, 3)
        res.max_disp_mm = round(sol.max_disp_mm, 4)
        res.compliance_j = round(sol.compliance_j / 1000.0, 6)  # N*mm -> J
        res.converged = sol.converged
        res.solver = sol.solver
        res.safety_factor = round(mat.yield_mpa / sol.max_vm_mpa, 2) if sol.max_vm_mpa > 0 else None
        if res.safety_factor is not None and res.safety_factor < 1.0:
            res.warnings.append(f"safety factor {res.safety_factor} < 1 — likely failure")
        report.grid["dofs"] = sol.dofs
        if fem_dir:
            vm_png, def_png = _render_case(
                domain, sol, res.name, fem_dir, suffix, source_mesh=source_mesh
            )
            res.vonmises_png = vm_png
            res.deformed_png = def_png
        report.cases.append(res)

    report.warnings.append(
        "boundary resolution is voxel-limited; stresses near load/fix regions are approximate"
    )


def report_to_json(report: FemReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def default_load_case() -> LoadCase:
    """Placeholder when no loads were declared: fixed bottom, 100 N on top."""
    from cadx.llm.schemas import LoadSpec, RegionSpec

    return LoadCase(
        name="default",
        fixed=[RegionSpec(kind="face", axis="-z", description="bottom face")],
        loads=[
            LoadSpec(
                region=RegionSpec(kind="face", axis="+z", description="top face"),
                force_n=[0.0, 0.0, -100.0],
            )
        ],
    )


def fem_feedback_lines(report: FemReport, sf_min: float) -> list[tuple[str, str]]:
    """(severity, text) engineering feedback for the agent's issue list.

    ``sf < 1``/``sf < sf_min`` are majors — the part fails the declared load.
    An over-built part (``sf > 3*sf_min``) is only a minor advisory: being too
    strong is never a defect, and on form-locked tasks (mesh reuse, "change
    nothing else") the advice is unactionable anyway."""
    lines: list[tuple[str, str]] = []
    for c in report.cases:
        sf = c.safety_factor
        if sf is None:
            # unsolvable case (empty regions / zero resultant) — without this
            # line the failure is invisible to the improve loop and the model
            # keeps iterating blind
            why = "; ".join(dict.fromkeys(c.warnings)) if c.warnings else "no result"
            lines.append(
                (
                    "major",
                    f"fem[{c.name}]: no solvable result — {why}; check that the "
                    "region's hole_index/area actually matches material on the "
                    "current part",
                )
            )
            continue
        if sf < 1.0:
            lines.append(
                (
                    "major",
                    f"fem[{c.name}]: safety factor {sf:.2f} < 1 — part fails under the "
                    f"declared load (max stress {c.max_vm_mpa:.1f} MPa); add material "
                    f"near the load/fix regions",
                )
            )
        elif sf < sf_min:
            lines.append(
                (
                    "major",
                    f"fem[{c.name}]: safety factor {sf:.2f} below target {sf_min:g} — "
                    f"strengthen the load path (walls, ribs, fillets near mounts)",
                )
            )
        elif sf > 3.0 * sf_min:
            lines.append(
                (
                    "minor",
                    f"fem[{c.name}]: safety factor {sf:.2f} well above target {sf_min:g} — "
                    f"part could be lightened (thinner walls or topology optimization)",
                )
            )
        if not math.isfinite(c.max_disp_mm) or c.max_disp_mm <= 0:
            continue
    return lines
