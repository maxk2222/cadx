"""mesh_score — deterministic candidate scoring + pre-gate thresholds."""

from __future__ import annotations

from typing import ClassVar

from cadx.agent.mesh_score import score_mesh_candidate, support_score


class _Dim:
    def __init__(self, v):
        self.value = v


class _Spec:
    length_x = _Dim(120.0)
    width_y = _Dim(38.0)
    height_z = _Dim(8.0)
    features: ClassVar[list] = []


def _res(bbox, outline=None, holes=None):
    return {
        "analysis": {"bbox_size": bbox},
        "mesh": {
            "features": {
                "outline_mm": outline or [],
                "holes": holes or [],
            }
        },
    }


RECT = [[-60, -19], [60, -19], [60, 19], [-60, 19], [-60, -19]]


def test_good_candidate_scores_high():
    res = _res([119.0, 38.0, 8.0], outline=RECT, holes=[])
    assert score_mesh_candidate(res, _Spec(), RECT) > 0.8


def test_blob_candidate_scores_low():
    res = _res(
        [143.0, 123.0, 40.0], outline=[[-70, -60], [70, -60], [70, 60], [-70, 60], [-70, -60]]
    )
    assert score_mesh_candidate(res, _Spec(), RECT) < 0.35


def test_no_outline_falls_back_to_aspect():
    res = _res([119.0, 38.0, 8.0])
    assert score_mesh_candidate(res, _Spec(), None) > 0.9
    res = _res([60.0, 60.0, 30.0])
    assert score_mesh_candidate(res, _Spec(), None) < 0.5


def test_banana_candidate_killed_by_height():
    """A standing volumetric mesh from a plan-view input scores hopeless."""
    res = _res([119.0, 113.0, 111.0])
    assert score_mesh_candidate(res, _Spec(), RECT) < 0.3
    res = _res([119.0, 113.0, 111.0])
    assert score_mesh_candidate(res, _Spec(), None) < 0.25


def _knob(lying: bool):
    """Skirt + dome knob (~40 dia, 45 tall); `lying` tips it onto its side."""
    import numpy as np
    import trimesh

    skirt = trimesh.creation.cylinder(radius=18.0, height=8.0, sections=48)
    skirt.apply_translation([0, 0, 4.0])
    dome = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    dome.apply_scale([1, 1, 0.6])
    dome.apply_translation([0, 0, 12.0])  # spans z 0..24, base flush with skirt
    m = trimesh.util.concatenate([skirt, dome])
    if lying:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, 0, -float(m.bounds[0][2])])
    return m


def test_support_score_standing_beats_lying():
    """The orientation signal dims cannot give: resting on the skirt base
    covers most of the footprint; lying on the side contacts on a strip."""
    assert support_score(_knob(lying=False)) > 0.6
    assert support_score(_knob(lying=True)) < 0.4


def test_orientation_decisive_margin():
    from cadx.agent.orchestrator import _orientation_decisive

    # the knob regression: equal dim violations, Δscore 0.01 — NOT decisive
    assert _orientation_decisive((0, -0.98), (0, -0.99)) is False
    # clear margin is decisive
    assert _orientation_decisive((0, -0.80), (0, -0.99)) is True
    # fewer dim violations is decisive regardless of score
    assert _orientation_decisive((1, -0.50), (0, -0.40)) is True
