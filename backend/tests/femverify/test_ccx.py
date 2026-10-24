"""Cross-solver verification vs CalculiX — opt-in, needs `ccx` on PATH or
CCX_PATH. The writer test itself runs unconditionally."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cadx.cad.fem.solver import assemble, reduce_case, solve_reduced
from cadx.cad.fem.voxelize import voxelize

from .xverify import ccx as ccx_mod

E_MPA = 3500.0
NU = 0.36


def _cantilever_case():
    """20x8x4 beam @ pitch 1 — 160 elements, -x face pinned, +x face -z tip load."""
    mesh = trimesh.creation.box(extents=[20.0, 8.0, 4.0])
    mesh.apply_translation([10.0, 0.0, 2.0])
    dom = voxelize(mesh, 1.0, 100_000)
    coords = dom.node_coords()
    en = dom.element_nodes()
    used = np.unique(en[dom.solid.ravel()])
    u = coords[used]
    fixed3 = np.zeros((dom.node_count, 3), dtype=bool)
    fixed3[used[u[:, 0] < 1e-9]] = True
    F = np.zeros(dom.node_count * 3)
    tip = used[u[:, 0] > 20.0 - 1e-9]
    F[3 * tip + 2] = -10.0 / len(tip)
    return dom, fixed3, F, used


def test_inp_writer_structure(tmp_path):
    """The exported deck carries identical physics: same node/element counts,
    same total force, same fixed DOFs."""
    dom, fixed3, F, used = _cantilever_case()
    nelem = ccx_mod.write_inp(tmp_path / "beam.inp", dom, fixed3, F, E_MPA, NU)
    text = (tmp_path / "beam.inp").read_text()
    assert nelem == int(dom.solid.sum())
    # node lines: between *NODE and *ELEMENT
    node_block = text.split("*NODE")[1].split("*ELEMENT")[0].strip().splitlines()
    assert len(node_block) == len(used)
    # CLOAD sums to the applied force
    cload = text.split("*CLOAD")[1].split("*END STEP")[0].strip().splitlines()
    total_fz = sum(float(ln.split(",")[2]) for ln in cload if ", 3," in ln)
    assert total_fz == pytest.approx(-10.0)
    n_fixed = int(fixed3[used].sum())
    n_bnd = len(text.split("*BOUNDARY")[1].split("*STEP")[0].strip().splitlines())
    assert n_bnd == n_fixed


def test_ccx_same_mesh_matches(tmp_path):
    """Identical mesh + BCs + loads solved by ccx: nodal displacements must
    match our solver to ~1e-4 rel (solver-precision agreement)."""
    ccx_bin = ccx_mod.find_ccx()
    if ccx_bin is None:
        pytest.skip("no ccx binary (set CCX_PATH or install calculix)")
    dom, fixed3, F, used = _cantilever_case()

    solid = dom.solid.ravel()
    elem_E = np.full(dom.cell_count, E_MPA)
    K, edof, active = assemble(dom, elem_E, NU, solid)
    ctx = reduce_case(dom, K, edof, fixed3)
    sol = solve_reduced(dom, ctx, edof, active, F, elem_E, NU)
    ours = sol.node_disp[used]

    inp = tmp_path / "beam.inp"
    ccx_mod.write_inp(inp, dom, fixed3, F, E_MPA, NU)
    frd = ccx_mod.run_ccx(ccx_bin, inp)
    ccx_u = ccx_mod.read_frd_displacements(frd)
    theirs = np.stack([ccx_u[i + 1] for i in range(len(used))])

    ref = np.abs(theirs).max()
    assert ref > 0
    err = np.abs(ours - theirs).max() / ref
    assert err < 1e-4, f"same-mesh divergence: max rel err {err:.3g}"
