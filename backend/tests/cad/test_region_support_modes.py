"""Regression coverage for support-mode honesty additions.

- `fix_mode="bearing"` pins the two DOFs normal to the bore axis (collar),
  leaving only axial slide — never the axial pin `normal` produced.
- `lint_support_modes` names the free rigid modes a spec's fixed set leaves.
- `cavity(seal_openings_mm)` resolves the inner skin of an open vessel whose
  openings are narrower than the seal threshold.
"""

from __future__ import annotations

import numpy as np

from cadx.cad.fem import regions
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import Dim, LoadCase, RegionSpec


def _dim(v: float) -> Dim:
    return Dim(value=v, unit="mm", source="stated")


def _dom() -> VoxelDomain:
    solid = np.ones((12, 12, 12), dtype=bool)
    return VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=solid.shape, solid=solid)


def test_bearing_pins_normal_plane_dofs() -> None:
    dom = _dom()
    r = RegionSpec(kind="hole", axis="+z", fix_mode="bearing")
    nodes = np.zeros(dom.node_count, dtype=bool)
    nodes[:10] = True
    mask = regions._fixed_dof_mask(dom, r, nodes)
    # z-bore: x and y pinned, z stays free (axial slide)
    assert mask[:10, 0].all() and mask[:10, 1].all()
    assert not mask[:10, 2].any()


def test_bearing_only_on_bore_kinds() -> None:
    dom = _dom()
    r = RegionSpec(kind="face", axis="+z", fix_mode="bearing")
    nodes = np.zeros(dom.node_count, dtype=bool)
    nodes[:10] = True
    mask = regions._fixed_dof_mask(dom, r, nodes)
    # non-bore kinds fall back to the generic fix_mode path (all pinned)
    assert mask[:10].all()


def test_lint_names_free_modes() -> None:
    from cadx.agent.steps.intake import lint_support_modes
    from cadx.llm.schemas import PartSpec

    spec = PartSpec(
        name="arm", summary="", length_x=_dim(100.0), width_y=_dim(80.0), height_z=_dim(40.0)
    )
    spec.load_cases = [
        LoadCase(
            name="roller seat",
            fixed=[RegionSpec(kind="hole", axis="+z", fix_mode="normal", hole_index=1)],
            loads=[],
        )
    ]
    notes = lint_support_modes(spec)
    assert len(notes) == 1
    # normal on a z-hole pins only tz -> 5 modes free
    assert "tx" in notes[0] and "rx" in notes[0]
    assert "unsolvable" in notes[0]


def test_lint_bearing_leaves_axial_only() -> None:
    from cadx.agent.steps.intake import lint_support_modes
    from cadx.llm.schemas import PartSpec

    spec = PartSpec(
        name="arm", summary="", length_x=_dim(100.0), width_y=_dim(80.0), height_z=_dim(40.0)
    )
    spec.load_cases = [
        LoadCase(
            name="collar",
            fixed=[RegionSpec(kind="hole", axis="+z", fix_mode="bearing", hole_index=1)],
            loads=[],
        )
    ]
    notes = lint_support_modes(spec)
    assert len(notes) == 1
    assert "tz" in notes[0]
    assert "tx" not in notes[0]


def test_cavity_seal_openings() -> None:
    # open vessel: shell with a small neck — plain cavity floods empty,
    # sealed resolves the inner skin
    solid = np.zeros((20, 20, 20), dtype=bool)
    solid[2:18, 2:18, 2:18] = True
    solid[5:15, 5:15, 5:15] = False  # hollow interior
    solid[9:11, 9:11, 15:] = False  # narrow opening at the top
    dom = VoxelDomain(origin=np.zeros(3), pitch=1.0, dims=solid.shape, solid=solid)

    open_case = regions.region_cell_mask(dom, RegionSpec(kind="cavity"))
    assert not open_case.any()  # open neck floods — honest empty

    sealed_case = regions.region_cell_mask(
        dom, RegionSpec(kind="cavity", seal_openings_mm=4.0)
    )
    assert sealed_case.any()  # sealed: inner skin resolves


def test_lint_cavity_without_seal_flagged() -> None:
    from cadx.agent.steps.intake import lint_region_semantics
    from cadx.llm.schemas import LoadSpec, PartSpec

    spec = PartSpec(
        name="tank", summary="", length_x=_dim(80.0), width_y=_dim(80.0), height_z=_dim(120.0)
    )
    spec.load_cases = [
        LoadCase(
            name="hydro test",
            fixed=[RegionSpec(kind="face", axis="-z", fix_mode="all")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="cavity", description="inner pressure"),
                    pressure_mpa=0.9,
                )
            ],
        )
    ]
    notes = lint_region_semantics(spec)
    assert len(notes) == 1
    assert "seal_openings_mm" in notes[0]
    assert "open nozzle floods" in notes[0]


def test_lint_cavity_with_seal_quiet() -> None:
    from cadx.agent.steps.intake import lint_region_semantics
    from cadx.llm.schemas import LoadSpec, PartSpec

    spec = PartSpec(
        name="tank", summary="", length_x=_dim(80.0), width_y=_dim(80.0), height_z=_dim(120.0)
    )
    spec.load_cases = [
        LoadCase(
            name="hydro test",
            fixed=[RegionSpec(kind="face", axis="-z", fix_mode="all")],
            loads=[
                LoadSpec(
                    region=RegionSpec(
                        kind="cavity", description="inner", seal_openings_mm=24.0
                    ),
                    pressure_mpa=0.9,
                )
            ],
        )
    ]
    assert lint_region_semantics(spec) == []
