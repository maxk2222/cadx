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
from cadx.llm.schemas import Dim, LoadCase, PartSpec, RegionSpec


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

    sealed_case = regions.region_cell_mask(dom, RegionSpec(kind="cavity", seal_openings_mm=4.0))
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
                    region=RegionSpec(kind="cavity", description="inner", seal_openings_mm=24.0),
                    pressure_mpa=0.9,
                )
            ],
        )
    ]
    assert lint_region_semantics(spec) == []


def test_drop_dead_anchor_refs(monkeypatch) -> None:
    from cadx.agent.steps import intake
    from cadx.llm.schemas import LoadSpec, PartSpec

    # 3 spec anchors exist; hole_index 6 is a dead reference the rebinder
    # could not rescue — the region must be dropped, not left to poison the
    # case with a silent matched-no-nodes at FEM time
    monkeypatch.setattr("cadx.agent.anchors._spec_anchors", lambda s: [1, 2, 3])
    spec = PartSpec(
        name="tank", summary="", length_x=_dim(80.0), width_y=_dim(80.0), height_z=_dim(120.0)
    )
    spec.load_cases = [
        LoadCase(
            name="hydro",
            fixed=[
                RegionSpec(kind="hole", hole_index=1, fix_mode="all"),
                RegionSpec(kind="hole", hole_index=6, fix_mode="all"),
            ],
            loads=[LoadSpec(region=RegionSpec(kind="surface", axis="+z"), pressure_mpa=0.9)],
        )
    ]
    notes = intake.drop_dead_anchor_refs(spec)
    assert len(notes) == 1 and "hole_index 6" in notes[0]
    assert [r.hole_index for r in spec.load_cases[0].fixed] == [1]


def test_drop_dead_anchor_refs_empty_table_keeps() -> None:
    from cadx.agent.steps import intake
    from cadx.llm.schemas import PartSpec

    # empty anchor table means indices can't be validated — a hole_index
    # might still resolve by axis/center hint at FEM time, so keep it
    spec = PartSpec(
        name="tank", summary="", length_x=_dim(80.0), width_y=_dim(80.0), height_z=_dim(120.0)
    )
    spec.load_cases = [
        LoadCase(
            name="hydro",
            fixed=[RegionSpec(kind="hole", hole_index=6, fix_mode="all")],
            loads=[],
        )
    ]
    assert intake.drop_dead_anchor_refs(spec) == []
    assert len(spec.load_cases[0].fixed) == 1


def test_cavity_thin_shell_mesh_verified() -> None:
    """A watertight thin-shell vessel leaks through the voxel wall (cell-center
    containment drops cells in a ~pitch-thick curved wall), so plain
    binary_fill_holes floods the cavity to outside. The mesh-verified gap-seal
    must recover the inner skin — while a genuinely OPEN vessel stays empty."""
    import trimesh

    from cadx.cad.fem.voxelize import voxelize

    # hollow sealed box: 3 mm walls — thin enough to leak at coarse pitch
    outer = trimesh.creation.box(extents=[40.0, 40.0, 40.0])
    inner = trimesh.creation.box(extents=[34.0, 34.0, 34.0])
    shell = outer.difference(inner, engine="manifold")
    dom = voxelize(shell, pitch_mm=2.0, max_elements=200_000)

    sealed = regions.region_cell_mask(dom, RegionSpec(kind="cavity"))
    assert sealed.any(), "sealed thin-shell cavity should resolve via mesh verify"

    # same shell but with a real opening drilled through the top wall
    hole = trimesh.creation.cylinder(radius=4.0, height=12.0, sections=24)
    hole.apply_translation([0.0, 0.0, 24.0])
    open_shell = shell.difference(hole, engine="manifold")
    dom_open = voxelize(open_shell, pitch_mm=2.0, max_elements=200_000)
    open_mask = regions.region_cell_mask(dom_open, RegionSpec(kind="cavity"))
    assert not open_mask.any(), "a real Ø8 opening must stay honest-empty"


# --------------------------------------------------------------------------- #
# normalize_support_modes — 'normal' on a bore is never a support              #
# --------------------------------------------------------------------------- #


def _spec_with_cases(cases: list[LoadCase]) -> PartSpec:
    spec = PartSpec(
        name="arm", summary="", length_x=_dim(100.0), width_y=_dim(80.0), height_z=_dim(40.0)
    )
    spec.load_cases = cases
    return spec


def test_normalize_upgrades_coaxial_bores_to_all() -> None:
    """Robot-arm pattern: N coaxial mounting bores declared fix_mode 'normal'
    pin only the shared axial DOF — a collar ('bearing') would still leave tz
    free, so the upgrade must escalate to 'all' (bolted seat)."""
    from cadx.agent.steps.intake import sanitize_load_cases

    spec = _spec_with_cases(
        [
            LoadCase(
                name="arm_weight_shear",
                fixed=[
                    RegionSpec(kind="bore", axis="+z", fix_mode="normal", hole_index=i)
                    for i in (2, 3, 4, 5)
                ],
                loads=[],
            )
        ]
    )
    sanitize_load_cases(spec)
    assert [r.fix_mode for r in spec.load_cases[0].fixed] == ["all"] * 4
    # the rescue is recorded in assumptions, and the unsolvable lint is gone
    assert any("upgraded to 'all'" in a for a in spec.assumptions)
    assert not any("unsolvable" in a for a in spec.assumptions)


def test_normalize_prefers_bearing_when_sufficient() -> None:
    """Two bores on orthogonal axes: upgrading both to 'bearing' already
    covers all 6 rigid modes — the minimal support wins over encastre."""
    from cadx.agent.steps.intake import sanitize_load_cases

    spec = _spec_with_cases(
        [
            LoadCase(
                name="pinned pair",
                fixed=[
                    RegionSpec(kind="hole", axis="+x", fix_mode="normal", hole_index=1),
                    RegionSpec(kind="hole", axis="+y", fix_mode="normal", hole_index=2),
                ],
                loads=[],
            )
        ]
    )
    sanitize_load_cases(spec)
    assert [r.fix_mode for r in spec.load_cases[0].fixed] == ["bearing"] * 2
    assert any("upgraded to 'bearing'" in a for a in spec.assumptions)


def test_normalize_leaves_plane_normal_untouched() -> None:
    """'normal' on a plane is a legitimate roller/symmetry support — never
    rewritten, even when the case stays under-constrained."""
    from cadx.agent.steps.intake import sanitize_load_cases

    spec = _spec_with_cases(
        [
            LoadCase(
                name="roller rest",
                fixed=[RegionSpec(kind="plane", axis="-z", fix_mode="normal")],
                loads=[],
            )
        ]
    )
    sanitize_load_cases(spec)
    assert spec.load_cases[0].fixed[0].fix_mode == "normal"
    # under-constraint still surfaces via the honest lint note
    assert any("unsolvable" in a for a in spec.assumptions)


def test_normalize_leaves_solvable_case_untouched() -> None:
    """A case already covering all 6 modes keeps a stray 'normal' bore as
    declared — the upgrade only fires when it rescues a singular case."""
    from cadx.agent.steps.intake import sanitize_load_cases

    spec = _spec_with_cases(
        [
            LoadCase(
                name="clamped",
                fixed=[
                    RegionSpec(kind="face", axis="-z", fix_mode="all"),
                    RegionSpec(kind="hole", axis="+z", fix_mode="normal", hole_index=1),
                ],
                loads=[],
            )
        ]
    )
    sanitize_load_cases(spec)
    modes = [r.fix_mode for r in spec.load_cases[0].fixed]
    assert modes == ["all", "normal"]
    assert not any("upgraded" in a for a in spec.assumptions)


def test_normalize_mixed_supports_minimal_upgrade() -> None:
    """Mixed supports: a 'normal' hole upgrades to 'bearing' when the plane
    roller already supplies the missing axis pin — the case ends fully
    constrained and the lint stays silent."""
    from cadx.agent.steps.intake import lint_support_modes, sanitize_load_cases

    spec = _spec_with_cases(
        [
            LoadCase(
                name="half fixed",
                fixed=[
                    RegionSpec(kind="plane", axis="-z", fix_mode="normal"),
                    RegionSpec(kind="hole", axis="+z", fix_mode="normal", hole_index=1),
                ],
                loads=[],
            )
        ]
    )
    sanitize_load_cases(spec)
    modes = {r.kind: r.fix_mode for r in spec.load_cases[0].fixed}
    # the plane's roller pin (tz) plus a hole collar covers all six modes —
    # 'bearing' suffices, so the minimal upgrade wins over 'all'
    assert modes == {"plane": "normal", "hole": "bearing"}
    spec.assumptions = []
    notes = lint_support_modes(spec)
    assert not notes  # hole 'bearing' + plane 'normal' = fully constrained
