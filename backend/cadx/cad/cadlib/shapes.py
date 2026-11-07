"""cadlib shapes — pure factories returning a ``Part`` you position yourself.

    from cadlib import boss, mounting_ear
    with BuildPart() as p:
        Box(80, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        Pos(30, 0, 8) * boss(diameter=12, height=6, hole_d=4.3)

Factories do NOT need a BuildPart context (algebra mode). Unless documented
otherwise the returned Part rests on the XY plane (min Z = 0) and is centred
at the origin, so `Pos(x, y, z) * helper(...)` drops it into place.
"""

import contextlib
import math

from build123d import (
    Align,
    Axis,
    Box,
    BuildLine,
    BuildSketch,
    Cone,
    Cylinder,
    Ellipse,
    EllipticalCenterArc,
    Line,
    Plane,
    Polygon,
    Pos,
    Rot,
    Text,
    ThreePointArc,
    Vector,
    extrude,
    fillet,
    loft,
    make_face,
    revolve,
)
from build123d.topology import Part

_CZ = (Align.CENTER, Align.CENTER, Align.MIN)


def boss(
    diameter: float,
    height: float,
    hole_d: float | None = None,
    hole_depth: float | None = None,
    draft_deg: float = 0.0,
) -> Part:
    """Vertical cylindrical boss (screw dome) on XY, centred. Optional bore and
    draft angle for moulding/FDM strength."""
    r = diameter / 2
    if draft_deg:
        top_r = max(0.2, r - math.tan(math.radians(draft_deg)) * height)
        body = Cone(r, top_r, height, align=_CZ)
    else:
        body = Cylinder(r, height, align=_CZ)
    if hole_d:
        depth = hole_depth if hole_depth is not None else height
        body -= Pos(0, 0, height - depth - 0.5) * Cylinder(hole_d / 2, depth + 0.5, align=_CZ)
    return body


def tube(outer_d: float, inner_d: float, length: float) -> Part:
    """Open hollow cylinder along Z, resting on XY, centred — bore clean
    through both ends (bushings, spacers, ferrule/tube holders). Lay it flat
    with `Rot(0, 90, 0) * tube(...)` (axis → X) or `Rot(90, 0, 0)` (→ Y)."""
    return Cylinder(outer_d / 2, length, align=_CZ) - Pos(0, 0, -0.5) * Cylinder(
        inner_d / 2, length + 1.0, align=_CZ
    )


def _dome_cap(radius: float, height: float) -> Part:
    """Half-ellipsoid solid cap — equator at z=0, crown bulging up to
    z=``height``. Built by revolving a quarter-ellipse profile, which keeps
    the cap watertight by construction (no loft/fillet fragile ops)."""
    with BuildSketch(Plane.XZ) as sk:
        with BuildLine():
            EllipticalCenterArc((0.0, 0.0), radius, height, arc_size=90.0)
            Line((0.0, height), (0.0, 0.0))
            Line((0.0, 0.0), (radius, 0.0))
        make_face()
    return revolve(sk.sketch, axis=Axis.Z)


def _vessel_body(radius: float, height: float, dome_h: float) -> Part:
    """Solid vessel shape z=0..``height``: straight cylindrical shell plus
    domed (or flat, ``dome_h`` = 0) caps on both ends."""
    if dome_h <= 0:
        return Cylinder(radius, height, align=_CZ)
    shell_h = height - 2 * dome_h
    body = Pos(0, 0, dome_h) * Cylinder(radius, shell_h, align=_CZ)
    body += Pos(0, 0, height - dome_h) * _dome_cap(radius, dome_h)
    body += Pos(0, 0, dome_h) * Rot(180, 0, 0) * _dome_cap(radius, dome_h)
    return body


def sealed_tank(
    outer_d: float,
    height: float,
    wall: float = 3.0,
    dome_h: float = 0.0,
    feet: int = 0,
    foot_w: float = 18.0,
    foot_t: float = 6.0,
    foot_h: float = 20.0,
    foot_hole_d: float = 0.0,
) -> Part:
    """Closed cylindrical tank — shell + sealed end caps, no openings.

    Rests on XY (min Z = 0), centred. ``wall`` applies to shell AND caps; the
    interior is one fully enclosed cavity — no bore/cut bridges it.
    ``dome_h`` > 0 gives both ends a half-ellipsoid cap of that height
    (``height`` stays the total outer height); pass 0 for flat welded end
    plates. ``feet`` welds that many radial lug feet around the base circle,
    each ``foot_w`` wide (tangential) x ``foot_h`` tall (Z), protruding
    ``foot_t`` beyond the wall; ``foot_hole_d`` > 0 cuts a bolt-hole through
    Z centred in each lug's protruding part. Pass 0 feet for a plain sealed
    vessel."""
    if wall * 2 >= outer_d or wall * 2 >= height:
        raise ValueError("sealed_tank: wall too thick for the given size")
    # a dome can never be deeper than half the body minus the end walls
    dome_h = min(dome_h, (height - 2 * wall) / 2)
    body = _vessel_body(outer_d / 2, height, dome_h)
    # the cavity is the same vessel shrunk by ~wall on every side — nested
    # smaller vessel keeps the sealed interior watertight by construction
    cavity = Pos(0, 0, wall) * _vessel_body(
        outer_d / 2 - wall, height - 2 * wall, max(0.0, dome_h - wall)
    )
    tank = body - cavity
    # radial lug feet around the base circle: X=protrusion (points radially
    # after the Z-rotation), Y=width (tangential), Z=height. The lug inner
    # edge embeds ~2 mm into the wall; it protrudes enough that an optional
    # Ø foot_hole_d bolt-hole through Z sits in solid material clear of the
    # wall — the hole lands at the protruding part's centre.
    embed = 2.0
    protrude = max(foot_t, foot_hole_d + 4.0)
    lug_r = outer_d / 2 - embed + protrude / 2
    for i in range(feet):
        a = i * 2 * math.pi / feet
        lug = Rot(0, 0, math.degrees(a)) * Box(protrude, foot_w, foot_h, align=_CZ)
        tank += Pos(lug_r * math.cos(a), lug_r * math.sin(a), 0) * lug
        if foot_hole_d > 0:
            hole_r = outer_d / 2 + protrude / 2 - embed
            tank -= Pos(hole_r * math.cos(a), hole_r * math.sin(a), -0.5) * Cylinder(
                foot_hole_d / 2, foot_h + 1.0, align=_CZ
            )
    return tank


def strut(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    width: float,
    thickness: float,
    *,
    embed: float = 1.0,
) -> Part:
    """Straight rectangular bar from p0 to p1 (world mm). ``width`` runs along
    the axis perpendicular to p0->p1 closest to world Y; ``thickness`` lies in
    the bar plane. Both ends extend ``embed`` mm past p0/p1 so the strut fuses
    into the bodies it connects. NOT a filled gusset — use rib() for that."""
    d = Vector(*p1) - Vector(*p0)
    length = d.length + 2 * embed
    x = d.normalized()
    y = Vector(0, 1, 0) - x * x.dot(Vector(0, 1, 0))
    if y.length < 1e-6:
        y = Vector(1, 0, 0)
    y = y.normalized()
    z = x.cross(y)
    mid = (Vector(*p0) + Vector(*p1)) * 0.5
    plane = Plane(origin=mid, x_dir=x, z_dir=z)
    return plane * Box(length, width, thickness)


def arch_bridge(
    span_x: float,
    apex_z: float,
    width_y: float,
    thickness: float,
    *,
    foot_z: float = 0.0,
    embed: float = 1.0,
    n_sections: int = 15,
) -> Part:
    """Arched bar/handle between two feet at ±``span_x``/2, arch rising in the
    XZ plane so the top outer surface reaches ``apex_z``. The feet sit
    ``embed`` mm below ``foot_z`` so a fuse overlaps — place the part by its
    feet, not its apex.

    USE THIS for arched handles/bridges — the hand-rolled version collapses
    to a flat bar because ``Plane(normal=...)`` is not valid build123d (the
    kwarg is ``z_dir``) and the loft then never leaves the XY plane. Here the
    section planes are built from the arc's own frame: ``x_dir`` is the
    in-plane (XZ) radial direction — ``(tg.Z, 0, -tg.X)`` rotated 90° off the
    tangent — and ``z_dir`` is the tangent, so each ellipse stands
    perpendicular to the arch curve.

    Cross-section: ``thickness`` (radial, in the XZ arch plane) by ``width_y``
    (global Y). Example:
    ``part += arch_bridge(span_x=90, apex_z=25, width_y=8, thickness=6)``.
    """
    # the arc rides the cross-section CENTRE: the apex control point sits at
    # apex_z - thickness/2 so the outer (+radial) surface tops out at apex_z
    arc = ThreePointArc(
        Vector(-span_x / 2, 0, foot_z - embed),
        Vector(0, 0, apex_z - thickness / 2),
        Vector(span_x / 2, 0, foot_z - embed),
    ).edges()[0]
    n = max(2, n_sections)
    sections = []
    for i in range(n):
        u = i / (n - 1)
        pt = arc.position_at(u)
        tg = arc.tangent_at(u)
        nm = Vector(tg.Z, 0, -tg.X).normalized()  # in-plane radial direction
        with BuildSketch(Plane(origin=pt, x_dir=nm, z_dir=tg)) as sk:
            Ellipse(x_radius=thickness / 2, y_radius=width_y / 2)
        sections.append(sk.sketch)
    try:
        return loft(sections, ruled=False)
    except TypeError:
        return loft(sections)


def rib(length: float, height: float, thickness: float) -> Part:
    """Triangular gusset rib: base along +X on XY, apex at X=0 rising to
    ``height``, ``thickness`` centred on Y=0. Fuse to a wall: ``p.part += Pos(...) * rib(...)``."""
    profile = Rot(90, 0, 0) * Polygon(
        (0, 0), (length, 0), (0, height), align=(Align.MIN, Align.MIN)
    )
    return Pos(0, thickness / 2, 0) * extrude(profile, amount=thickness)


def mounting_ear(
    length: float,
    width: float,
    thickness: float,
    hole_d: float | None = None,
    hole_offset: float = 0.0,
    tip_radius: float | None = None,
) -> Part:
    """Flat mounting ear on XY, centred. Optional Ø ``hole_d`` through Z at
    ``hole_offset`` along X; ``tip_radius`` adds a rounded end at +X."""
    ear = Box(length, width, thickness, align=_CZ)
    if tip_radius:
        ear += Pos(length / 2, 0, 0) * Cylinder(tip_radius, thickness, align=_CZ)
    if hole_d:
        ear -= Pos(hole_offset, 0, -0.5) * Cylinder(hole_d / 2, thickness + 1, align=_CZ)
    return ear


def snap_fit_cantilever(
    length: float,
    width: float,
    thickness: float,
    hook_height: float | None = None,
) -> Part:
    """Cantilever snap-fit clip: beam on XY along X, hook wedge at the +X tip
    pointing up. Print so the beam flexes in-plane (FDM)."""
    hook_h = hook_height if hook_height is not None else thickness
    hook_len = min(2.0 * thickness, length * 0.25)
    beam = Box(length, width, thickness, align=_CZ)
    hook_profile = Rot(90, 0, 0) * Polygon(
        (0, 0), (hook_len, 0), (0, hook_h), align=(Align.MIN, Align.MIN)
    )
    hook = Pos(length / 2 - hook_len, width / 2, thickness) * extrude(hook_profile, amount=width)
    return beam + hook


def living_hinge(length: float, width: float, thickness: float = 0.3) -> Part:
    """Thin membrane hinge segment on XY, centred — join two rigid shells."""
    return Box(length, width, thickness, align=_CZ)


def knurl_ring(diameter: float, height: float, pitch: float = 2.0, depth: float = 0.8) -> Part:
    """Ring of axial grip ridges for a cylindrical surface of ``diameter``:
    fuse onto a knob/shaft — ``result += knurl_ring(PARAMS["knob_d"], h)``."""
    n = max(8, round(math.pi * diameter / pitch))
    ridge_w = pitch * 0.5
    ridge = Box(depth, ridge_w, height, align=_CZ)
    ring = None
    for i in range(n):
        ang = i * 360.0 / n
        r_mid = diameter / 2 + depth / 2 - 0.15  # slight sink into the surface
        placed = Rot(0, 0, ang) * Pos(r_mid, 0, 0) * ridge
        ring = placed if ring is None else ring + placed
    assert ring is not None
    return ring


def text_label(text: str, size: float = 6.0, depth: float = 0.4, font: str = "Arial") -> Part:
    """Text raised ``depth`` mm above XY, centred — union for embossing,
    subtract for engraving."""
    return extrude(Text(text, font_size=size, font=font), amount=depth)


def filleted_box(length: float, width: float, height: float, radius: float) -> Part:
    """Box with all edges filleted at ``radius`` (as much as fits)."""
    body = Box(length, width, height, align=_CZ)
    r = min(radius, min(length, width, height) / 2 - 0.01)
    return fillet(body.edges(), radius=max(0.05, r))


def l_bracket(
    leg_x: float, leg_z: float, width: float, thickness: float, fillet_r: float = 0.0
) -> Part:
    """L-bracket: leg_x along +X on XY, leg_z up Z at X=0, ``width`` centred on Y."""
    hor = Pos(leg_x / 2, 0, 0) * Box(leg_x, width, thickness, align=_CZ)
    ver = Pos(thickness / 2, 0, leg_z / 2) * Box(
        thickness, width, leg_z, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    b = hor + ver
    if fillet_r:
        with contextlib.suppress(Exception):
            b = fillet(b.edges().filter_by(Plane.YZ), radius=fillet_r)
    return b
