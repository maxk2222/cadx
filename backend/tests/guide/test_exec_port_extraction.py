"""Executor BRep port extraction — annular end fusion.

Generated parts arrive as STEP/BRep: every cylindrical FACE is a separate
surface, so a hollow tube produced only a male shell port + a bare female
bore port — neither carrying the rim envelope (inner+outer radii) that
mesh-extracted ``rim_*`` ports provide. The result was a generated pipe
whose ends could never plug into a socket (``_can_plug`` needs an outer
envelope) and posed as bogus receivers. The annular-end pass fuses
coaxial face groups into one port per OPEN end carrying both envelopes.
"""

import build123d as bd
import pytest

from cadx.cad.guide_exec import _face_ports


def _by_name(ports: list[dict], name: str) -> dict | None:
    return next((p for p in ports if p["name"] == name), None)


def test_through_tube_gets_two_envelope_end_ports() -> None:
    """A hollow DN22 tube (OD 22, bore 20) must emit one fused annular-end
    port per open end with BOTH radii — the same semantics rim extraction
    gives uploaded mesh pipes. depth=500 marks the through-bore."""
    tube = bd.Cylinder(radius=11.0, height=70.0) - bd.Cylinder(radius=10.0, height=70.05)
    ports = _face_ports(tube)
    ends = [p for p in ports if p["name"].startswith("end_")]
    assert len(ends) == 2
    for end in ends:
        assert end["gender"] == "female"
        assert end["inner_radius"] == pytest.approx(10.0)
        assert end["outer_radius"] == pytest.approx(11.0)
        assert end["depth"] == pytest.approx(500.0)  # through-bore marker
    # the two ends sit on opposite faces of the tube
    zs = sorted(end["pos"][2] for end in ends)
    assert zs == pytest.approx([-35.0, 35.0])
    # the fused faces are CONSUMED — shell+bore ARE the ends, and keeping
    # a second alias per physical end would let a consumed joint look
    # free under the other name (the eval failure's retry loop)
    assert not any(p["name"].startswith("cyl_") for p in ports)


def test_solid_cylinder_gets_no_annular_port() -> None:
    """A solid rod has no open bore — no annulus, no end port. Its male
    shell port remains the only interface."""
    rod = bd.Cylinder(radius=5.0, height=40.0)
    ports = _face_ports(rod)
    assert not any(p["name"].startswith("end_") for p in ports)
    assert any(p["gender"] == "male" for p in ports)


def test_blind_bore_boss_reports_bounded_end() -> None:
    """A hollow boss open on one side only: the bore doesn't span the
    group extent, so the end port carries the measured seat depth, not
    the through-bore marker."""
    boss = bd.Cylinder(radius=12.0, height=20.0) - bd.Cylinder(radius=9.0, height=15.0).locate(
        bd.Pos(0, 0, 2.5)
    )  # bore from z=-10 to z=+5, bottom at +5..+10 solid
    ports = _face_ports(boss)
    ends = [p for p in ports if p["name"].startswith("end_")]
    assert len(ends) == 1
    assert ends[0]["inner_radius"] == pytest.approx(9.0)
    assert ends[0]["outer_radius"] == pytest.approx(12.0)
    assert ends[0]["depth"] == pytest.approx(15.0)  # bounded seat, not 500


def test_nested_spigot_keeps_its_own_port() -> None:
    """A male spigot nested inside a counterbore is NOT the pocket's
    outer wall — it must keep its standalone port while the recess
    rim becomes the annular end (gap ring inner..outer)."""
    part = bd.Cylinder(radius=12.0, height=20.0)  # base block, z -10..10
    part += bd.Cylinder(radius=6.0, height=10.0).locate(bd.Pos(0, 0, 10))  # spigot on top
    part -= bd.Cylinder(radius=9.0, height=8.0).locate(
        bd.Pos(0, 0, 7)
    )  # counterbore around it, depth to z=3
    ports = _face_ports(part)
    ends = [p for p in ports if p["name"].startswith("end_")]
    # the counterbore mouth is one annular end — rim semantics mean the
    # MATERIAL ring: inner = the recess opening (r=9), outer = the wall
    # edge (r=12). The spigot inside the pocket is not part of the rim.
    assert len(ends) == 1
    assert ends[0]["inner_radius"] == pytest.approx(9.0)
    assert ends[0]["outer_radius"] == pytest.approx(12.0)
    assert ends[0]["depth"] == pytest.approx(7.0)  # seat at z=3
    # ...and the nested spigot keeps its own pluggable port
    males = [p for p in ports if p["gender"] == "male"]
    assert any(p["radius"] == pytest.approx(6.0) for p in males)


def test_off_axis_hollow_bosses_stay_separate_groups() -> None:
    """Two parallel hollow bosses at different positions are separate
    coaxial groups — each reports its own end port where it actually is.
    (A plain drilled hole through a plate has NO coaxial outer wall and
    correctly gets none — a hole is a socket, not a pluggable end.)"""
    part = bd.Box(60, 20, 10)
    for x in (-15.0, 15.0):
        boss = bd.Cylinder(radius=8.0, height=24.0).locate(bd.Pos(x, 0, 7))
        boss -= bd.Cylinder(radius=6.0, height=20.0).locate(bd.Pos(x, 0, 9))
        part += boss
    ports = _face_ports(part)
    ends = [p for p in ports if p["name"].startswith("end_")]
    assert len(ends) == 2
    xs = sorted({round(e["pos"][0]) for e in ends})
    assert xs == [-15.0, 15.0]
    for e in ends:
        assert e["inner_radius"] == pytest.approx(6.0)
        assert e["outer_radius"] == pytest.approx(8.0)


def test_extracted_mesh_is_solver_frame(tmp_path) -> None:
    """The mesh extract_ports serves must share the port/bbox frame (BRep
    Z-up mm). bd.export_gltf emits meters + a -90degX up-axis node per the
    glTF spec — consumers that apply solver transforms to the served mesh
    (renders, merged_glb, GuideScene, collision) then saw the part 1000x
    too small and tipped over. It also must NOT clobber the version's own
    model.glb artifact (that file stays glTF-correct)."""
    import numpy as np
    import trimesh

    from cadx.cad.guide_exec import _extract_ports_sync

    tube = bd.Cylinder(radius=11.0, height=70.0)
    step = tmp_path / "model.step"
    bd.export_step(tube, str(step))
    res = _extract_ports_sync({"step_path": str(step)})
    assert res["ok"]
    mp = tmp_path / "model.guide.glb"
    assert res["mesh_path"] == str(mp) and mp.exists()
    assert not (tmp_path / "model.glb").exists()  # artifact name untouched
    g = trimesh.load(mp)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    # Z-up mm: the 70mm tube stands along Z, not meters along Y
    assert g.extents[2] == pytest.approx(70.0, abs=0.5)
    # and bounds match the reported BRep bbox exactly
    assert np.allclose(
        g.bounds, [res["bbox_min"], res["bbox_max"]], atol=0.5
    )
