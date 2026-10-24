"""Optional cross-solver verification against CalculiX (ccx).

Same-mesh comparison: the voxel domain IS a Hex8 mesh — every solid cell is
one C3D8 element on the regular node grid, and our node ordering matches the
Abaqus convention exactly. Exporting that mesh lets an independent solver
check our assembly + solve on identical discretization: displacements must
agree to solver precision (~1e-4 rel), not to a benchmark's percent-level
tolerance.

Opt-in: needs a `ccx` binary — set CCX_PATH or put it on PATH. Everything
here skips cleanly without it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from cadx.cad.fem.voxelize import VoxelDomain


def find_ccx() -> str | None:
    env = os.environ.get("CCX_PATH")
    if env and Path(env).is_file():
        return env
    return shutil.which("ccx")


def write_inp(
    path: Path,
    dom: VoxelDomain,
    fixed3: np.ndarray,
    F: np.ndarray,
    E: float,
    nu: float,
) -> int:
    """Serialize the domain as a C3D8 deck. Returns the element count.

    Only used nodes are written (renumbered); void-only nodes would dangle.
    Loads go in as *CLOAD on the exact nodes our solver sees — no region
    re-resolution, so both sides solve byte-identical physics inputs."""
    en = dom.element_nodes()
    solid = np.flatnonzero(dom.solid.ravel())
    used = np.unique(en[solid])
    remap = -np.ones(dom.node_count, dtype=np.int64)
    remap[used] = np.arange(len(used))
    coords = dom.node_coords()[used]
    F3 = F.reshape(-1, 3)

    lines = ["*HEADING", "cadx femverify same-mesh cross-check", "*NODE"]
    lines += [f"{i + 1}, {c[0]:.10g}, {c[1]:.10g}, {c[2]:.10g}" for i, c in enumerate(coords)]
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALL")
    for e_no, e in enumerate(solid, start=1):
        nodes = ", ".join(str(int(remap[n]) + 1) for n in en[e])
        lines.append(f"{e_no}, {nodes}")
    lines.append(f"*MATERIAL, NAME=M\n*ELASTIC\n{E:.10g}, {nu:.6g}")
    lines.append("*SOLID SECTION, ELSET=ALL, MATERIAL=M\n,")

    fixed_used = fixed3[used]
    lines.append("*BOUNDARY")
    for i in np.flatnonzero(fixed_used.any(axis=1)):
        for d in range(3):
            if fixed_used[i, d]:
                lines.append(f"{i + 1}, {d + 1}, {d + 1}, 0.0")

    lines.append("*STEP\n*STATIC")
    loaded = np.abs(F3[used]).sum(axis=1) > 0
    lines.append("*CLOAD")
    for i in np.flatnonzero(loaded):
        for d in range(3):
            v = F3[used][i, d]
            if v != 0.0:
                lines.append(f"{i + 1}, {d + 1}, {v:.10g}")
    lines.append("*NODE FILE\nU\n*EL FILE\nS\n*END STEP")
    path.write_text("\n".join(lines) + "\n")
    return len(solid)


def run_ccx(ccx: str, inp: Path) -> Path:
    """Run `ccx job` in the inp's directory; returns the .frd path."""
    job = inp.with_suffix("")
    subprocess.run(
        [ccx, job.name],
        cwd=inp.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    frd = job.with_suffix(".frd")
    if not frd.is_file():
        raise RuntimeError(f"ccx produced no .frd for {inp.name}")
    return frd


_NODE_RE = re.compile(r"^\s*2\s+(\d+)\s")
_DISP_RE = re.compile(r"^\s*-1(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")


def read_frd_displacements(frd: Path) -> dict[int, np.ndarray]:
    """{node_id: (ux,uy,uz)} from the -4 DISP result block."""
    out: dict[int, np.ndarray] = {}
    in_block = False
    for line in frd.read_text().splitlines():
        if line.strip().startswith("-4") and "DISP" in line:
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("-3"):
                break
            m = _DISP_RE.match(line)
            if m:
                out[int(m.group(1))] = np.array(
                    [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                )
    return out
