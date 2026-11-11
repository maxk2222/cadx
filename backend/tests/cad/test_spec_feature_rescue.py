"""spec_feature_rescue — deterministic rebuild of spec-declared mount
features the generated code dropped or malformed (the executor safety-net
behind helper-adoption instability)."""

from __future__ import annotations

import pytest
from build123d import Box

from cadx.cad import cadlib, mesh_boolean
from cadx.cad import spec_feature_rescue as sfr


def _tank_spec() -> dict:
    return {
        "name": "domed tank",
        "features": [
            {
                "type": "other",
                "name": "foot lug",
                "count": 3,
                "axis": "z",
                "dims": {"width": 18.0, "thickness": 6.0, "depth_total": 20.0},
            },
            {
                "type": "hole",
                "name": "foot bolt hole",
                "count": 3,
                "axis": "-z",
                "dims": {"diameter": 6.5},
            },
        ],
    }


def _vessel(*, feet: int = 0) -> object:
    return cadlib.sealed_tank(
        outer_d=80,
        height=120,
        wall=3,
        dome_h=8,
        feet=feet,
        foot_w=18,
        foot_t=6,
        foot_h=20,
        foot_hole_d=6.5,
    )


def _tank_spec_per_lug() -> dict:
    """The per-instance layout real specs produce: the host shell is itself
    a feature (mentioning the mounting plane in its position text) followed
    by one feature per lug and per bolt hole."""

    def lug(n: int) -> dict:
        return {
            "type": "other",
            "name": f"foot lug {n}",
            "count": 1,
            "axis": "z",
            "dims": {"width": 18.0, "thickness": 6.0, "head_diameter": 20.0},
            "position": "at base rim, equally spaced 120 deg",
        }

    def hole(n: int) -> dict:
        return {
            "type": "hole",
            "name": f"foot bolt hole {n}",
            "count": 1,
            "axis": "z",
            "dims": {"diameter": 6.5},
            "position": f"through foot lug {n}, centered in lug",
        }

    return {
        "name": "domed tank",
        "features": [
            {
                "type": "other",
                "name": "cylindrical shell with top and bottom dome caps",
                "count": 1,
                "axis": "z",
                "dims": {"diameter": 80.0, "height": 120.0, "wall_thickness": 3.0},
                "position": "centered on Z axis, from z=0 (foot base) to z=120",
            },
            *[lug(i) for i in range(1, 4)],
            *[hole(i) for i in range(1, 4)],
        ],
    }


def test_bare_vessel_gets_lugs_and_holes():
    """The recurring failure: model calls sealed_tank(feet=0) or drops
    disjoint hand-built lugs → bare vessel with no feet/holes. The rescue
    must rebuild the declared mount ring deterministically."""
    out, applied = sfr.rescue_spec_features(_vessel(feet=0), _tank_spec(), mesh_boolean)
    assert applied
    bb = out.bounding_box()
    # same envelope the reviewed helper produces for the same spec values
    assert bb.size.X <= 85.0 and bb.size.Y <= 85.0
    assert len(out.solids()) == 1


def test_rescued_feet_reach_the_base_and_roundtrip(tmp_path):
    """A convex-domed host can't take grounded feet cleanly — a lug crossing
    the dome equator fuse-seam serialises as unsewn shells and an elevated lug
    misses the FEM -z support band. The rescue must rebuild the bottom as a
    skirt + recessed head so the feet sit at the base plane, and the part must
    survive the STEP roundtrip gate."""
    from build123d import export_step, import_step

    out, applied = sfr.rescue_spec_features(_vessel(feet=0), _tank_spec(), mesh_boolean)
    assert applied
    bb = out.bounding_box()
    assert pytest.approx(0.0, abs=0.1) == bb.min.Z
    # a grounded bolt-hole is present: empty at the hole axis near the base
    assert not out.is_inside((40.9, 0.0, 8.0))
    # interior stays sealed — hollow axis but solid caps and solid head
    assert not out.is_inside((0.0, 0.0, 60.0))
    assert out.is_inside((0.0, 0.0, 118.5))
    step = tmp_path / "rescued.step"
    export_step(out, str(step))
    q = import_step(str(step))
    assert q.is_valid
    assert abs(q.volume - out.volume) / out.volume < 0.02


def test_realized_ring_is_not_rescued():
    """When the model DID adopt the helper correctly, the rescue must not
    double-add lugs — the ring already holds clean bolt holes."""
    _out, applied = sfr.rescue_spec_features(_vessel(feet=3), _tank_spec(), mesh_boolean)
    assert applied == []


def test_rotated_ring_is_not_rescued():
    """The ring may be realized on ANY azimuth — generated code laying the
    lugs out on a rotated offset must not earn a duplicate rescue ring on
    the canonical positions (observed: probing 0/120/240 only double-added
    a second ring onto a rotated-but-realized one)."""
    from build123d import Rot

    _out, applied = sfr.rescue_spec_features(
        Rot(0, 0, 60) * _vessel(feet=3), _tank_spec(), mesh_boolean
    )
    assert applied == []


def test_host_shell_is_not_a_mount():
    """Regression: the spec's own host-shell feature mentions the mounting
    plane in its position text ("from z=0 (foot base)"). Matching mount
    words against the full feature text counted the shell as a fourth lug
    and read the ring height from the host envelope — the rescue then
    stamped 4 full-height fins onto every correct part."""
    _out, applied = sfr.rescue_spec_features(_vessel(feet=3), _tank_spec_per_lug(), mesh_boolean)
    assert applied == []


def test_host_shell_does_not_poison_rescued_ring():
    """Same spec on a bare vessel: the rescue fires, but must rebuild the
    declared 3 lugs at lug height — not 4 full-height fins read off the
    host dims."""
    out, applied = sfr.rescue_spec_features(_vessel(feet=0), _tank_spec_per_lug(), mesh_boolean)
    assert applied
    # a lug stands at the 0-degree declared position, at lug height —
    # probed tangentially beside the Ø6.5 bolt hole axis at r≈40.9
    assert out.is_inside((40.9, 4.5, 12.0))
    # nothing protrudes at 90 deg (the bogus 4th lug) or at mid-height
    assert not out.is_inside((0.0, 41.0, 60.0))
    assert not out.is_inside((41.0, 4.5, 60.0))


def test_dropped_feature_is_not_resurrected():
    """A feature the model explicitly dropped stays dropped — the rescue
    honours feature_drops rather than fighting the audit trail."""
    spec = _tank_spec()
    spec["feature_drops"] = [
        {
            "feature_index": 0,
            "feature": spec["features"][0],
            "reason": "model dropped the foot lugs",
            "iteration": 1,
            "session": "codegen",
            "recorded_at": "x",
        }
    ]
    _o, applied = sfr.rescue_spec_features(_vessel(feet=0), spec, mesh_boolean)
    assert applied == []


def test_non_round_host_skips_honestly():
    """A prismatic host has no base ring — the lug rescue must skip with no
    change instead of fusing lugs onto a box corner."""
    _o, applied = sfr.rescue_spec_features(Box(50, 30, 20), _tank_spec(), mesh_boolean)
    assert applied == []
    assert len(_o.solids()) == 1


def test_no_mount_feature_is_untouched():
    """A spec with only a standalone hole and no mount feature does not
    trigger the additive lug rescue."""
    spec = {
        "name": "plate",
        "features": [
            {
                "type": "hole",
                "name": "mounting hole",
                "count": 1,
                "axis": "z",
                "dims": {"diameter": 5.0},
            },
        ],
    }
    _o, applied = sfr.rescue_spec_features(_vessel(feet=0), spec, mesh_boolean)
    assert applied == []


def test_base_ring_reports_wall_and_cavity():
    """_base_ring reads the outer wall radius and inner cavity skin — the
    measurements the lug reach is derived from."""
    ring = sfr._base_ring(_vessel(feet=0), 12.0, mesh_boolean)
    assert ring is not None
    host_r, cavity_r = ring
    assert host_r == pytest.approx(40.0, abs=2.0)
    assert cavity_r == pytest.approx(37.0, abs=2.0)


def _hub(*, keyway: bool = False) -> object:
    """Ø40x28 hub, Ø10 through-bore at (0,0), optional 3x1.8 keyway on +y."""
    from build123d import Cylinder, Pos

    hub = Pos(0, 0, 14) * Cylinder(20, 28) - Pos(0, 0, 14) * Cylinder(5, 28.4)
    if keyway:
        hub -= Pos(0, 5.75, 14) * Box(3.0, 2.1, 28)
    return hub


def _prop_spec() -> dict:
    return {
        "name": "propeller",
        "features": [
            {
                "type": "through_bore",
                "name": "shaft bore",
                "count": 1,
                "axis": "",
                "dims": {"diameter": 10.0, "depth": 28.0},
                "position": "concentric on Z axis, full height",
            },
            {
                "type": "keyway_slot",
                "name": "keyway seat",
                "count": 1,
                "axis": "",
                "dims": {"width": 3.0, "depth_partial": 1.8, "slot_length": 28.0},
                "position": "in bore wall on +y side, full height z=0..28",
            },
        ],
    }


def test_keyway_rescue_cuts_declared_slot():
    """marine_propeller r21: the model never managed a real 3x1.8 DIN 6885
    keyway in the Ø10 bore wall — deterministic rescue cuts it on the
    declared +y side for the wall's full axial span."""
    from OCP.gp import gp_Pnt

    hub = _hub(keyway=False)
    out, applied = sfr.rescue_keyway_slots(hub, _prop_spec()["features"], mesh_boolean)
    assert applied and "keyway" in applied[0]
    assert not out.is_inside(gp_Pnt(0, 5.9, 14))  # slot interior void
    assert out.is_inside(gp_Pnt(0, -5.9, 14))  # opposite wall untouched
    assert out.is_inside(gp_Pnt(5.9, 0, 14))  # tangentially beside the slot


def test_keyway_rescue_idempotent_and_side_aware():
    from OCP.gp import gp_Pnt

    hub = _hub(keyway=True)
    out, applied = sfr.rescue_keyway_slots(hub, _prop_spec()["features"], mesh_boolean)
    assert applied == []  # realized slot — probe reads void, no cut
    feats = _prop_spec()["features"]
    feats[1] = {**feats[1], "position": "in bore wall on -x side, full height"}
    out2, applied2 = sfr.rescue_keyway_slots(hub, feats, mesh_boolean)
    assert applied2  # declared side moved — rescue cuts the -x slot too
    assert not out2.is_inside(gp_Pnt(-5.9, 0, 14))


def test_keyway_rescue_skips_when_bore_missing():
    hub = _hub(keyway=False)
    feats = [f for f in _prop_spec()["features"] if f["type"] == "keyway_slot"]
    out, applied = sfr.rescue_keyway_slots(hub, feats, mesh_boolean)
    assert applied == []  # no host bore — nothing to hang the keyway on
