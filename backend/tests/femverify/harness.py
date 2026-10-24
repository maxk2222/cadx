"""Shared harness for the FEM verification suite.

Loads declarative case JSONs (evalkit style), runs the voxel FEM in-process
(voxelize → resolve_case_dofs → assemble/reduce_case/solve_reduced +
recover_elem_stress for stress-component probes) and evaluates probes
against published/analytical references.

Probe tolerance model:
- `tol_rel` — relative error band around `ref` (|got-ref| <= tol_rel·|ref|)
- `tol_abs` — absolute band, used when ref ≈ 0 or as an additional bound
- `min`/`max` — one-sided bounds instead of a reference point
- `kind: trend` — the case sweeps `pitches_mm`; the last (finest) error must
  be inside `tol_rel` AND no worse than the coarsest error (converges toward
  the reference instead of away), allowing a 10% noise slack per step
- `kind: convergence` — same sweep, but every step must stay within the
  previous error × (1 + slack) — monotone approach, final inside tol
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadx.cad.fem.solver import (
    FemSolution,
    assemble,
    recover_elem_stress,
    reduce_case,
    solve_reduced,
)
from cadx.cad.fem.voxelize import VoxelDomain, voxelize
from cadx.llm.schemas import LoadCase

from . import geometry

CASE_DIR = Path(__file__).parent / "cases"
MAX_ELEMENTS = 400_000

# Voigt order produced by solver._B_matrix / recover_elem_stress
_STRESS_COMPS = {"sx": 0, "sy": 1, "sz": 2, "txy": 3, "tyz": 4, "tzx": 5}
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}

# session report collector — conftest flushes it when CADX_FEMVERIFY_REPORT=1
REPORT: list[dict[str, Any]] = []


@dataclass
class SolvedCase:
    name: str
    fixed3: np.ndarray
    F: np.ndarray
    sol: FemSolution
    warnings: list[str]


@dataclass
class RunResult:
    spec: dict[str, Any]
    domain: VoxelDomain
    cases: list[SolvedCase]
    sig_full: np.ndarray  # (ncells, 6) stress at cell centres, 0 inactive
    pitch: float = field(default=0.0)


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    spec.setdefault("_file", path.name)
    return spec


def case_files(tier: str | None = None) -> list[Path]:
    out = []
    for p in sorted(CASE_DIR.glob("*.json")):
        spec = json.loads(p.read_text())
        if tier is None or spec.get("tier", "fast") == tier:
            out.append(p)
    return out


def _solved_case(
    spec: dict[str, Any], case_name: str | None = None
) -> LoadCase:
    cases = [LoadCase.model_validate(c) for c in spec["cases"]]
    if case_name is None:
        return cases[0]
    for c in cases:
        if c.name == case_name:
            return c
    raise KeyError(f"case '{case_name}' not in {spec['_file']}")


def run_spec(spec: dict[str, Any], pitch_mm: float | None = None) -> RunResult:
    """Voxelize + solve every declared load case; return the full result."""
    mesh = geometry.build(spec["geometry"]["builder"], spec["geometry"].get("args"))
    pitch = pitch_mm or float(spec.get("pitch_mm") or 2.0)
    dom = voxelize(mesh, pitch, MAX_ELEMENTS)
    mat = spec.get("material") or {}
    E = float(mat.get("E", 3500.0))
    nu = float(mat.get("nu", 0.36))
    anchors = spec.get("anchors") or None

    solid = dom.solid.ravel()
    elem_E = np.full(dom.cell_count, E)
    K, edof, active = assemble(dom, elem_E, nu, solid)
    solved: list[SolvedCase] = []
    from cadx.cad.fem.regions import resolve_case_dofs

    for lc in spec["cases"]:
        case = LoadCase.model_validate(lc)
        fixed3, F, warns = resolve_case_dofs(dom, case, anchors)
        apply_manual_pins(dom, fixed3, lc.get("manual_pins") or spec.get("manual_pins") or [])
        ctx = reduce_case(dom, K, edof, fixed3)
        sol = solve_reduced(dom, ctx, edof, active, F, elem_E, nu)
        solved.append(SolvedCase(case.name, fixed3, F, sol, warns))

    _, sig = recover_elem_stress(dom, solved[0].sol.u, edof, active, elem_E, nu)
    sig_full = np.zeros((dom.cell_count, 6))
    sig_full[active] = sig
    return RunResult(spec, dom, solved, sig_full, pitch=dom.pitch)


def sig_for(rr: RunResult, idx: int) -> np.ndarray:
    """Recover the (ncells, 6) stress field for `cases[idx]` — sig_full is
    for case 0 only."""
    if idx == 0:
        return rr.sig_full
    solid = rr.domain.solid.ravel()
    mat = rr.spec.get("material") or {}
    E = float(mat.get("E", 3500.0))
    nu = float(mat.get("nu", 0.36))
    elem_E = np.full(rr.domain.cell_count, E)
    K, edof, active = assemble(rr.domain, elem_E, nu, solid)
    _, sig = recover_elem_stress(
        rr.domain, rr.cases[idx].sol.u, edof, active, elem_E, nu
    )
    out = np.zeros((rr.domain.cell_count, 6))
    out[active] = sig
    return out


def select_nodes(dom: VoxelDomain, sel: dict[str, Any]) -> np.ndarray:
    """Node indices matching coordinate predicates — shared by
    ``manual_pins`` and node-set probes. ``"min"``/``"max"`` select the
    outermost node plane of the solid (within half a pitch), a float
    selects nodes within half a pitch of that coordinate; all selectors
    AND together."""
    en = dom.element_nodes()
    used = np.unique(en[dom.solid.ravel()])
    nc = dom.node_coords()[used]
    mask = np.ones(len(used), dtype=bool)
    tol = dom.pitch * 0.51 + 1e-9
    for ax_key, v in sel.items():
        c = nc[:, _AXIS_IDX[ax_key]]
        if v == "min":
            mask &= c <= c.min() + tol
        elif v == "max":
            mask &= c >= c.max() - tol
        else:
            mask &= np.abs(c - float(v)) <= tol
    return used[mask]


def apply_manual_pins(dom: VoxelDomain, fixed3: np.ndarray, pins: list[dict[str, Any]]) -> None:
    """Textbook boundary conditions: pin a DOF on nodes selected by
    coordinate predicates — knife-edge lines and point supports that
    region kinds cannot express (a cell always spans ≥2 node planes).

    Pin spec: {"dof": "x|y|z", "sel": {"x": "min"|"max"|float, ...}} —
    one DOF per pin entry; see ``select_nodes`` for sel syntax.
    """
    for pin in pins:
        fixed3[select_nodes(dom, pin["sel"]), _AXIS_IDX[pin["dof"]]] = True


def _nearest_solid_cell(dom: VoxelDomain, xyz: list[float]) -> int:
    centers = dom.element_centers()
    solid = dom.solid.ravel()
    idx = np.flatnonzero(solid)
    d = np.linalg.norm(centers[idx] - np.asarray(xyz, dtype=float), axis=1)
    return int(idx[int(d.argmin())])


def _nearest_used_node(dom: VoxelDomain, xyz: list[float]) -> int:
    en = dom.element_nodes()
    used = np.unique(en[dom.solid.ravel()])
    coords = dom.node_coords()[used]
    n = int(used[int(np.linalg.norm(coords - np.asarray(xyz, dtype=float), axis=1).argmin())])
    return n


def probe_value(rr: RunResult, probe: dict[str, Any]) -> float:
    """Evaluate one probe declaration on a solved run."""
    kind = probe["probe"]
    ci = int(probe.get("case", 0))
    sc = rr.cases[ci]
    sol = sc.sol
    if kind == "max_disp_mm":
        return sol.max_disp_mm
    if kind == "max_vm_mpa":
        return sol.max_vm_mpa
    if kind == "compliance_j":
        return sol.compliance_j
    if kind == "work_disp":
        # work-equivalent displacement: 2U/|F| — exact for spread loads
        # where single-node probes see Saint-Venant end effects
        ftot = float(np.abs(sc.F.reshape(-1, 3).sum(axis=0)).sum())
        return sol.compliance_j / ftot if ftot > 1e-12 else float("nan")
    if kind == "converged":
        return float(sol.converged)
    if kind == "disp_at":
        n = _nearest_used_node(rr.domain, probe["xyz"])
        ax = probe.get("axis", "mag")
        v = sol.node_disp[n]
        if ax == "mag":
            return float(np.linalg.norm(v))
        return float(v[_AXIS_IDX[ax]])
    if kind == "disp_nodes":
        nodes = select_nodes(rr.domain, probe["sel"])
        ax = probe.get("axis", "mag")
        v = sol.node_disp[nodes]
        if ax == "mag":
            return float(np.linalg.norm(v, axis=1).mean())
        return float(v[:, _AXIS_IDX[ax]].mean())
    if kind == "disp_diff":
        # displacement difference between two node sets — cancels the
        # boundary-layer offset near supports/loaded faces, leaving the
        # clean interior strain field
        ax = _AXIS_IDX[probe.get("axis", "x")]
        v1 = sol.node_disp[select_nodes(rr.domain, probe["sel1"]), ax].mean()
        v2 = sol.node_disp[select_nodes(rr.domain, probe["sel2"]), ax].mean()
        return float(v1 - v2)
    if kind == "vm_at":
        return float(sol.elem_vm[_nearest_solid_cell(rr.domain, probe["xyz"])])
    if kind == "stress_at":
        sig = sig_for(rr, ci)
        return float(sig[_nearest_solid_cell(rr.domain, probe["xyz"]), _STRESS_COMPS[probe["comp"]]])
    if kind == "resultant_n":
        return float(sc.F.reshape(-1, 3).sum(axis=0)[_AXIS_IDX[probe.get("axis", "z")]])
    raise ValueError(f"unknown probe kind '{kind}'")


@dataclass
class ProbeResult:
    case: str
    probe: str
    ref: float | None
    got: float
    rel_err: float | None
    ok: bool
    note: str = ""


def check_probe(spec: dict[str, Any], expect: dict[str, Any], rr: RunResult) -> ProbeResult:
    got = probe_value(rr, expect)
    name = f"{spec['name']}:{expect['probe']}"
    if "min" in expect or "max" in expect:
        ok = got >= expect.get("min", -np.inf) and got <= expect.get("max", np.inf)
        return ProbeResult(name, expect["probe"], None, got, None, ok,
                           f"bounds [{expect.get('min')},{expect.get('max')}]")
    ref = float(expect["ref"])
    tol_rel = float(expect.get("tol_rel", 0.10))
    tol_abs = expect.get("tol_abs")
    err = got - ref
    rel = abs(err) / abs(ref) if abs(ref) > 1e-12 else abs(err)
    ok = rel <= tol_rel if tol_abs is None else (abs(err) <= max(float(tol_abs), tol_rel * abs(ref)))
    return ProbeResult(name, expect["probe"], ref, got, rel, ok)


def evaluate(spec: dict[str, Any]) -> list[ProbeResult]:
    """Run a single-pitch case and check every expectation."""
    rr = run_spec(spec)
    out = [check_probe(spec, e, rr) for e in spec.get("expect", [])]
    record(spec, out, rr)
    return out


def evaluate_trend(spec: dict[str, Any]) -> list[ProbeResult]:
    """Sweep pitches; assert the probe approaches its reference."""
    trend = spec["trend"]
    probe = trend["probe"]
    ref = float(trend["ref"])
    tol_rel = float(trend.get("tol_rel", 0.15))
    monotone = spec.get("kind") == "convergence"
    errs: list[float] = []
    vals: list[float] = []
    for p in spec["pitches_mm"]:
        rr = run_spec(spec, pitch_mm=float(p))
        v = probe_value(rr, probe)
        vals.append(v)
        errs.append(abs(v - ref) / max(abs(ref), 1e-12))
    ok = errs[-1] <= tol_rel and errs[-1] <= errs[0] * 1.10
    if monotone:
        ok = ok and all(e <= errs[i] * 1.15 + 0.01 for i, e in enumerate(errs[1:], 1))
    name = f"{spec['name']}:{probe['probe']}"
    res = ProbeResult(
        name, probe["probe"], ref, vals[-1], errs[-1], ok,
        f"pitches {spec['pitches_mm']} → vals {[round(v,4) for v in vals]}",
    )
    record(spec, [res], None)
    return [res]


def record(spec: dict[str, Any], results: list[ProbeResult], rr: RunResult | None) -> None:
    for r in results:
        REPORT.append(
            {
                "case": spec["name"],
                "file": spec.get("_file"),
                "probe": r.probe,
                "ref": r.ref,
                "got": round(r.got, 6),
                "rel_err": None if r.rel_err is None else round(r.rel_err, 5),
                "ok": r.ok,
                "note": r.note,
                "pitch": None if rr is None else rr.pitch,
            }
        )


def flush_report() -> Path | None:
    """Write the session report if CADX_FEMVERIFY_REPORT is set."""
    if not REPORT or os.environ.get("CADX_FEMVERIFY_REPORT") != "1":
        return None
    import datetime

    out_dir = Path(__file__).parents[3] / "evals" / "results" / "femverify"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"report-{ts}.json"
    path.write_text(json.dumps(REPORT, indent=2))
    return path


def format_results(results: list[ProbeResult]) -> str:
    lines = []
    for r in results:
        ref = "-" if r.ref is None else f"{r.ref:.4g}"
        err = "-" if r.rel_err is None else f"{r.rel_err * 100:.1f}%"
        flag = "OK " if r.ok else "FAIL"
        lines.append(f"  [{flag}] {r.case}: got {r.got:.4g} vs ref {ref} ({err}) {r.note}")
    return "\n".join(lines)


def apply_body_field(dom: VoxelDomain, F: np.ndarray, b_fn) -> None:
    """Lumped body force for manufactured-solution tests.

    b_fn(x, y, z) -> (bx, by, bz) in N/mm³ is evaluated at each solid cell
    centre and distributed as b·pitch³/8 onto the cell's 8 corner nodes —
    the same lumping body_force() uses for uniform g-loads.
    """
    centers = dom.element_centers()
    en = dom.element_nodes()
    solid = np.flatnonzero(dom.solid.ravel())
    b = np.asarray([b_fn(*c) for c in centers[solid]], dtype=float)
    per_node = b * (dom.pitch**3) / 8.0
    nodes = en[solid]
    for a in range(8):
        np.add.at(F, 3 * nodes[:, a, None] + np.arange(3)[None, :], per_node)
