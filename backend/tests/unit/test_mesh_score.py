"""mesh_score — deterministic candidate scoring + pre-gate thresholds."""

from __future__ import annotations

from typing import ClassVar

from cadx.agent.mesh_score import score_mesh_candidate


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
