"""cadlib.machine — semantic face targeting + canonical machining cutters.

The M8-sphere failure mode: the agent placed `Pos(0,0,60)*tap` freehand and
the thread landed off-axis. machine_on_face resolves the face, projects the
pick point onto the skin and builds a normal frame — identical semantics on
planar, sloped, cylindrical and spherical faces."""

import math

import pytest
from build123d import Box, Pos, Rot, Sphere

from cadx.cad.cadlib import machine
from cadx.cad.cadlib.ops import set_detail_mode


@pytest.fixture(autouse=True)
def _preview_mode():
    set_detail_mode("preview")
    yield
    set_detail_mode("preview")


def _bbox(part) -> list[float]:
    bb = part.bounding_box()
    return [bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z]


def test_describe_faces_orders_by_area_and_reports_kind_normal() -> None:
    box = Box(60, 40, 20)
    faces = machine.describe_faces(box)
    assert len(faces) == 6
    assert [f["kind"] for f in faces] == ["planar"] * 6
    # largest first: 60x40 before 40x20
    assert faces[0]["area_mm2"] == pytest.approx(2400.0)
    assert faces[-1]["area_mm2"] == pytest.approx(800.0)
    normals = {tuple(f["normal"]) for f in faces}
    assert len(normals) == 6


def test_resolve_face_axis_word_index_and_pick() -> None:
    box = Box(60, 40, 20)
    for word, want in (("top", [0, 0, 1]), ("+z", [0, 0, 1]), ("left", [-1, 0, 0]),
                       ("front", [0, 1, 0])):
        f = machine.resolve_face(box, word)
        n = f.normal_at(f.center())
        assert [round(v, 2) for v in (n.X, n.Y, n.Z)] == want, word
    f0 = machine.resolve_face(box, 0)
    assert f0.area == pytest.approx(2400.0)
    # a pick point near the front wall resolves that wall
    f_pick = machine.resolve_face(box, (0, 25, 0))
    n = f_pick.normal_at(f_pick.center())
    assert pytest.approx(1.0) == n.Y


def test_face_frame_origin_on_surface_and_outward_normal() -> None:
    ball = Sphere(radius=30)
    face = next(iter(ball.faces()))
    frame = machine.face_frame(face, point=(0, 0, 40))
    o = frame.location.position
    n = frame.z_dir
    assert pytest.approx((0, 0, 30), abs=1e-6) == (o.X, o.Y, o.Z)
    assert pytest.approx((0, 0, 1), abs=1e-6) == (n.X, n.Y, n.Z)


def test_machine_tap_on_sphere_pole_cuts_into_body() -> None:
    ball = Sphere(radius=30)
    out, rec = machine.machine_on_face(
        ball, face=(0, 0, 40), op="tap", major_d=8, pitch=1.25, depth=25
    )
    assert rec["kind"] == "thread"
    assert rec["external"] is False
    assert rec["major_diameter"] == pytest.approx(8.0)
    assert rec["position_mm"] == pytest.approx([0, 0, 30], abs=1e-3)
    assert rec["normal"] == pytest.approx([0, 0, 1], abs=1e-3)
    assert ball.volume - out.volume > 500  # bore + chamfer removed material
    assert out.is_valid


def test_machine_tap_on_side_wall_is_normal_not_vertical() -> None:
    """The regression: a wall tap must follow the face normal (-x), not the
    global -z the agent would get from `Pos(x,y,z)*tap` guesswork."""
    box = Box(60, 40, 20)
    out, rec = machine.machine_on_face(
        box, face="left", op="tap", major_d=8, pitch=1.25, depth=15
    )
    assert rec["normal"] == pytest.approx([-1, 0, 0], abs=1e-3)
    assert rec["position_mm"] == pytest.approx([-30, 0, 0], abs=1e-3)
    assert out.is_valid
    # the tap drilled INTO the wall along -x: material gone near the wall
    assert box.volume - out.volume > 400


def test_machine_drill_through_on_top_face_with_uv_offset() -> None:
    box = Box(60, 40, 20)
    out, rec = machine.machine_on_face(
        box, face="top", op="drill", dia=5, at=(10, 5)
    )
    assert rec["position_mm"] == pytest.approx([10, 5, 10], abs=1e-3)
    assert out.is_valid
    # through hole: both the top and bottom face gained a rim — volume loss
    assert box.volume - out.volume == pytest.approx(math.pi * 2.5**2 * 20, rel=0.05)


def test_machine_countersink_and_counterbore() -> None:
    box = Box(60, 40, 20)
    out, rec = machine.machine_on_face(
        box, face="top", op="cbore", dia=5, cbore_dia=9, cbore_depth=3, at=(-15, 0)
    )
    assert rec["kind"] == "counterbore"
    assert out.is_valid
    out2, rec2 = machine.machine_on_face(
        box, face="top", op="csk", dia=5, csk_dia=9, csk_angle=90, at=(15, 0)
    )
    assert rec2["kind"] == "countersink"
    assert out2.is_valid


def test_machine_on_sloped_face_follows_the_surface_normal() -> None:
    box = Box(60, 40, 20)
    sloped = box - Pos(0, 0, 20) * Rot(0, 30, 0) * Box(80, 50, 20)
    # the ramp face: pick near the middle of the wedge
    ramp = machine.resolve_face(sloped, (0, 0, 12))
    frame = machine.face_frame(ramp, point=(0, 0, 12))
    n = frame.z_dir
    # the slope normal tilts off +z — a fixed "+z" guess would miss this
    assert pytest.approx(math.cos(math.radians(30)), abs=0.1) == n.Z
    assert abs(n.X) > 0.4
    out, rec = machine.machine_on_face(
        sloped, face=(0, 0, 12), op="drill", dia=4, depth=8
    )
    assert out.is_valid
    assert rec["normal"] == pytest.approx([n.X, n.Y, n.Z], abs=1e-3)


def test_machine_cutter_missing_body_raises_instead_of_silently_passing(
    monkeypatch,
) -> None:
    box = Box(60, 40, 20)

    def distant_cutter(op, **kwargs):
        return Pos(0, 0, 1000) * Box(1, 1, 1)

    monkeypatch.setattr(machine, "cutter_for", distant_cutter)
    with pytest.raises(ValueError, match="removed no material"):
        machine.machine_on_face(box, face="top", op="drill", dia=4, depth=8)


def test_machine_boss_adds_material() -> None:
    box = Box(60, 40, 20)
    out, rec = machine.machine_on_face(box, face="top", op="boss", dia=12, depth=6)
    assert rec["kind"] == "boss"
    assert out.volume > box.volume
    assert out.is_valid


def test_tap_export_mode_builds_real_helix() -> None:
    set_detail_mode("export")
    box = Box(60, 40, 20)
    out, rec = machine.machine_on_face(
        box, face="top", op="tap", major_d=8, pitch=1.25, depth=15, at=(15, 0)
    )
    assert rec["kind"] == "thread"
    assert rec["detail"] == "export"
    assert out.is_valid
    # the real groove must remove MORE material than the smooth preview bore
    set_detail_mode("preview")
    out_prev, _ = machine.machine_on_face(
        box, face="top", op="tap", major_d=8, pitch=1.25, depth=15, at=(15, 0)
    )
    assert out.volume < out_prev.volume


def test_resolve_face_rejects_unknown_descriptor() -> None:
    box = Box(60, 40, 20)
    with pytest.raises(ValueError, match="unknown face descriptor"):
        machine.resolve_face(box, "diagonal-left-upper")
    with pytest.raises(ValueError, match="out of range"):
        machine.resolve_face(box, 99)
