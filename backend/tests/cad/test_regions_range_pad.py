"""Sub-voxel range bands — autonomous robot-arm-mount regression.

The agent declared ``z_range=[25,25]`` ("the seat layer at z=25") on a
~1.9 mm pitch grid. Exact clipping selected ZERO cell centres → the
400 N seat load resolved to 0 N → the strict `fem_loadcases` gate
failed the case. A band thinner than one pitch means a coordinate
*layer*, not an empty interval — clip sites expand it to the nearest
cell layer and the resolve warns about the snap. Ranges >= one pitch
must clip exactly as before.
"""

from __future__ import annotations

import numpy as np

from cadx.cad.fem.regions import (
    _cylinder_cells,
    _pad_range_band,
    _range_clip,
    _z_clip,
    region_cell_mask,
    resolve_case_dofs,
)
from cadx.cad.fem.voxelize import VoxelDomain
from cadx.llm.schemas import LoadCase, LoadSpec, RegionSpec

PITCH = 2.0
DIMS = (10, 10, 10)  # cell centres at odd mm: 1,3,...,19


def _dom() -> VoxelDomain:
    return VoxelDomain(
        origin=np.zeros(3),
        pitch=PITCH,
        dims=DIMS,
        solid=np.ones(DIMS, dtype=bool),
    )


class TestPadRangeBand:
    def test_zero_width_expands_symmetric(self):
        lo, hi = _pad_range_band(25.0, 25.0, PITCH)
        assert (lo, hi) == (24.0, 26.0)

    def test_sub_pitch_expands_around_centre(self):
        lo, hi = _pad_range_band(25.0, 25.4, PITCH)
        assert lo < 25.0 and hi > 25.4
        assert (lo + hi) / 2 == 25.2  # centre preserved

    def test_wide_band_untouched(self):
        assert _pad_range_band(20.0, 30.0, PITCH) == (20.0, 30.0)

    def test_exactly_one_pitch_untouched(self):
        assert _pad_range_band(24.0, 26.0, PITCH) == (24.0, 26.0)


class TestRangeClipPadding:
    def test_zero_width_z_range_selects_layer(self):
        """[25,25] on a pitch-2 grid must select the z=25-adjacent layer,
        not nothing (the actual agent declaration that failed the eval)."""
        dom = _dom()
        sel = np.ones(DIMS, dtype=bool)
        out = _z_clip(dom, RegionSpec(kind="area", z_range=[9.0, 9.0]), sel)
        assert out.sum() > 0
        z_centers = dom.element_centers().reshape((*DIMS, 3))[..., 2]
        assert z_centers[out].min() == 9.0
        assert z_centers[out].max() == 9.0

    def test_sub_pitch_range_selects(self):
        sel = np.ones(DIMS, dtype=bool)
        out = _range_clip(_dom(), RegionSpec(kind="area", z_range=[9.0, 9.4]), sel)
        assert out.sum() > 0

    def test_wide_range_unchanged(self):
        dom = _dom()
        sel = np.ones(DIMS, dtype=bool)
        out = _range_clip(dom, RegionSpec(kind="area", z_range=[5.0, 13.0]), sel)
        z_centers = dom.element_centers().reshape((*DIMS, 3))[..., 2]
        assert (z_centers[out] >= 5.0).all()
        assert (z_centers[out] <= 13.0).all()

    def test_out_of_domain_band_is_empty_not_crash(self):
        """A snapped band entirely outside the part still selects nothing —
        padding widens, it does not invent off-domain cells."""
        dom = _dom()
        sel = np.ones(DIMS, dtype=bool)
        out = _z_clip(dom, RegionSpec(kind="area", z_range=[40.0, 40.0]), sel)
        assert out.sum() == 0

    def test_cylinder_axis_band_padded(self):
        """z_range on a cylinder bounds the AXIS coordinate — a zero-width
        band must still select the cylinder's layer."""
        dom = _dom()
        out = _cylinder_cells(dom, "z", [5.0, 5.0], 3.0, z_range=[9.0, 9.0])
        assert out.sum() > 0
        z_centers = dom.element_centers().reshape((*DIMS, 3))[..., 2]
        assert (z_centers[out] == 9.0).all()

    def test_full_region_resolve_padded(self):
        """End-to-end: an `area` region with z_range=[9,9] resolves cells
        through region_cell_mask (the path resolve_case takes)."""
        dom = _dom()
        cells = region_cell_mask(dom, RegionSpec(kind="area", z_range=[9.0, 9.0]))
        assert cells.sum() > 0


class TestSnapWarning:
    def _case(self, z_range: list[float]) -> LoadCase:
        return LoadCase(
            name="seat",
            fixed=[RegionSpec(kind="plane", axis="-z")],
            loads=[
                LoadSpec(
                    region=RegionSpec(kind="area", z_range=z_range),
                    force_n=[0.0, 0.0, -400.0],
                )
            ],
        )

    def test_sub_pitch_band_warns(self):
        _, F, warnings = resolve_case_dofs(_dom(), self._case([9.0, 9.0]))
        assert any("thinner than one voxel" in w for w in warnings)
        assert float(np.abs(F).sum()) > 0  # the load actually lands now

    def test_wide_band_no_snap_warning(self):
        _, F, warnings = resolve_case_dofs(_dom(), self._case([5.0, 13.0]))
        assert not any("thinner than one voxel" in w for w in warnings)
        assert float(np.abs(F).sum()) > 0
