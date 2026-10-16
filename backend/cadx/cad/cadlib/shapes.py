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
    Box,
    Cone,
    Cylinder,
    Plane,
    Polygon,
    Pos,
    Rot,
    Text,
    extrude,
    fillet,
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
