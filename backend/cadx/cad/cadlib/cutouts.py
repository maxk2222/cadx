"""cadlib cutouts — features that cut into the active ``BuildPart`` context.

Call them inside ``with BuildPart() as p:``. Positioning follows the enclosing
``with Locations(...)`` scope, or pass ``on=<Face|Plane|Location>`` to apply the
feature at that workplane directly.

    with BuildPart() as p:
        Box(80, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
        top = p.faces().sort_by(Axis.Z)[-1]
        hole_grid(screw, nx=2, ny=2, pitch_x=60, pitch_y=24, on=top)
"""

import math
from typing import Literal

from build123d import (
    Align,
    Box,
    BuildPart,
    Cone,
    Cylinder,
    GridLocations,
    Locations,
    Mode,
    Polygon,
    Pos,
    Rot,
    extrude,
)

Fit = Literal["Close", "Normal", "Loose"]


def _require_context(name: str) -> BuildPart:
    ctx = BuildPart._get_context(f"cadlib.{name}")
    if ctx is None:
        raise RuntimeError(f"cadlib.{name}() must run inside `with BuildPart() as p:`")
    return ctx


def _in_scope(on, make) -> None:
    """Run make() either at `on` (a Face/Plane/Location) or in the current scope."""
    if on is None:
        make()
    else:
        with Locations(on):
            make()


def _hole_ctor(screw, kind, fit, depth, counter_sunk, material, compensation):
    from bd_warehouse.fastener import (
        ClearanceHole,
        InsertHole,
        TapHole,
        ThreadedHole,
    )

    if kind == "clearance":
        return ClearanceHole(fastener=screw, fit=fit, depth=depth, counter_sunk=bool(counter_sunk))
    if kind in ("countersink", "counterbore"):
        # counter_sunk=True produces a conical seat for countersunk screws and a
        # cylindrical counterbore for cap/button-head screws — pick the screw
        # class accordingly (iso10642 head vs iso4762 head).
        return ClearanceHole(fastener=screw, fit=fit, depth=depth, counter_sunk=True)
    if kind == "tap":
        return TapHole(fastener=screw, material=material, fit=fit, depth=depth)
    if kind == "threaded":
        return ThreadedHole(
            fastener=screw,
            material=material,
            fit=fit,
            depth=depth,
            counter_sunk=bool(counter_sunk),
            simple=True,
        )
    if kind == "insert":
        return InsertHole(
            fastener=screw,
            fit=fit,
            depth=depth,
            manufacturing_compensation=compensation,
        )
    raise ValueError(
        f"unknown hole kind '{kind}' (clearance|countersink|counterbore|tap|threaded|insert)"
    )


def hole_grid(
    screw,
    nx: int = 2,
    ny: int = 2,
    pitch_x: float = 20.0,
    pitch_y: float = 20.0,
    kind: str = "clearance",
    *,
    fit: str = "Normal",
    depth: float | None = None,
    counter_sunk: bool | None = None,
    material: str = "Soft",
    compensation: float = 0.1,
    on=None,
) -> None:
    """Cut an nx-by-ny hole pattern for a bd_warehouse fastener/nut/insert.

    kind: "clearance" | "countersink" | "counterbore" | "tap" | "threaded" |
    "insert". For countersink use a CounterSunkScrew, for counterbore a
    SocketHeadCapScrew. Positions are recorded on ``screw.hole_locations``.
    """
    _require_context("hole_grid")

    def make() -> None:
        if nx * ny > 1:
            with GridLocations(pitch_x, pitch_y, nx, ny):
                _hole_ctor(screw, kind, fit, depth, counter_sunk, material, compensation)
        else:
            _hole_ctor(screw, kind, fit, depth, counter_sunk, material, compensation)

    _in_scope(on, make)


def through_hole(
    diameter: float,
    *,
    depth: float | None = None,
    csk_dia: float | None = None,
    csk_angle: float = 90.0,
    cbore_dia: float | None = None,
    cbore_depth: float | None = None,
    on=None,
) -> None:
    """Ø``diameter`` hole along the current workplane normal — through by default.

    The cutter is taller than the part diagonal, so it always cuts completely
    through regardless of where along the axis the location sits — this avoids
    the classic ``Pos(x, y, -1) * Cylinder`` mistake that leaves a membrane
    sealing the hole. ``depth`` instead makes a blind hole ``depth`` mm deep
    below the workplane. ``csk_dia`` adds a countersink cone (``csk_angle``
    degrees, opening at the workplane); ``cbore_dia`` + ``cbore_depth`` a
    cylindrical counterbore — both measured down from the workplane.

        with Locations(Pos(-15, 0, 0)):            # anywhere along the axis
            through_hole(5.0)                    # guaranteed through
        through_hole(5.0, csk_dia=8.4, on=top_face)   # countersunk, 90°
    """
    ctx = _require_context("through_hole")
    r = diameter / 2
    if depth is None:
        bb = ctx.part.bounding_box()
        s = bb.size
        height = math.sqrt(s.X * s.X + s.Y * s.Y + s.Z * s.Z) + 10.0
        z_mid = 0.0
    else:
        height = depth + 1.0  # 1 mm overshoot above the workplane
        z_mid = (1.0 - depth) / 2

    def make() -> None:
        with Locations(Pos(0, 0, z_mid)):
            Cylinder(radius=r, height=height, mode=Mode.SUBTRACT)
        if csk_dia and csk_dia > diameter:
            csk_depth = (csk_dia - diameter) / (2 * math.tan(math.radians(csk_angle) / 2))
            with Locations(Pos(0, 0, -csk_depth / 2)):
                Cone(
                    bottom_radius=r,
                    top_radius=csk_dia / 2,
                    height=csk_depth,
                    mode=Mode.SUBTRACT,
                )
        if cbore_dia and cbore_dia > diameter:
            cd = cbore_depth if cbore_depth else (cbore_dia - diameter) / 2
            with Locations(Pos(0, 0, -cd / 2)):
                Cylinder(radius=cbore_dia / 2, height=cd, mode=Mode.SUBTRACT)

    _in_scope(on, make)


def insert_pocket(
    insert,
    *,
    depth: float | None = None,
    compensation: float = 0.1,
    fit: Fit = "Normal",
    on=None,
) -> None:
    """Blind pocket for a HeatSetNut; depth defaults to the insert's own depth."""
    _require_context("insert_pocket")
    from bd_warehouse.fastener import InsertHole

    if depth is None:
        depth = getattr(insert, "depth", None)

    def make() -> None:
        InsertHole(
            fastener=insert,
            fit=fit,
            depth=depth,
            manufacturing_compensation=compensation,
        )

    _in_scope(on, make)


def bearing_seat(
    bearing,
    *,
    interference: float = 0.0,
    fit: Fit = "Normal",
    depth: float | None = None,
    retain: str = "pressfit",
    groove_offset: float = 2.0,
    on=None,
) -> None:
    """Bearing bore via PressFitHole; ``retain="snapring"`` additionally cuts an
    internal snap-ring groove ``groove_offset`` mm below the current workplane.

    retain: "pressfit" | "through" | "snapring". "through" ignores interference.
    """
    _require_context("bearing_seat")
    from bd_warehouse.bearing import PressFitHole
    from bd_warehouse.retaining_ring import InternalSnapRing, RetainingRingGroove

    if retain == "through":
        interference = 0.0
        fit = "Loose"

    def make() -> None:
        PressFitHole(bearing=bearing, interference=interference, fit=fit, depth=depth)
        if retain == "snapring":
            od = getattr(bearing, "outer_diameter", None)
            if od is None:
                raise ValueError("bearing exposes no outer_diameter for snapring")
            ring = InternalSnapRing(size=str(round(od)))
            with Locations(Pos(0, 0, -groove_offset)):
                RetainingRingGroove(ring=ring)

    _in_scope(on, make)


def keyhole_slot(
    big_d: float,
    small_d: float,
    spacing: float,
    thickness: float,
    *,
    direction: str = "+x",
    on=None,
) -> None:
    """Keyhole cutout: a Øbig_d opening merged with a Øsmall_d slot ``spacing``
    mm away along `direction` (+x|-x|+y|-y). Cuts ``thickness`` deep from the
    current workplane (usually the top face)."""
    _require_context("keyhole_slot")
    dx, dy = {
        "+x": (spacing, 0),
        "-x": (-spacing, 0),
        "+y": (0, spacing),
        "-y": (0, -spacing),
    }.get(direction, (spacing, 0))

    def make() -> None:
        Cylinder(
            radius=big_d / 2,
            height=thickness + 2,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
            mode=Mode.SUBTRACT,
        )
        with Locations(Pos(dx / 2, dy / 2, 0)):
            Cylinder(
                radius=small_d / 2,
                height=thickness + 2,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )
            # bridge the two openings into one slot
            Box(
                abs(dx) + small_d,
                abs(dy) + small_d,
                thickness + 2,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )

    _in_scope(on, make)


def oblong_slot(
    width: float,
    length: float,
    *,
    depth: float | None = None,
    direction: str = "+x",
    on=None,
) -> None:
    """Oblong (stadium) through-slot: ``length`` tip-to-tip along ``direction``
    (+x|-x|+y|-y), ``width`` across. Through by default — the cutter is taller
    than the part diagonal so no membrane can seal the slot; ``depth`` makes a
    blind pocket instead. USE THIS for elongated mounting slots (belt clips,
    adjustment slots) — the most commonly missed feature.

        with Locations(Pos(10, 0, 0)):
            oblong_slot(6.0, 20.0)             # 6x20 stadium through, along X
        oblong_slot(5.0, 12.0, direction="+y") # along Y
    """
    ctx = _require_context("oblong_slot")
    dx, dy = {
        "+x": (1.0, 0.0),
        "-x": (-1.0, 0.0),
        "+y": (0.0, 1.0),
        "-y": (0.0, -1.0),
    }.get(direction, (1.0, 0.0))
    r = width / 2
    half = max(0.0, length / 2 - r)  # end-centre offset along the slot axis
    if depth is None:
        bb = ctx.part.bounding_box()
        s = bb.size
        height = math.sqrt(s.X * s.X + s.Y * s.Y + s.Z * s.Z) + 10.0
        z_mid = 0.0
    else:
        height = depth + 1.0
        z_mid = (1.0 - depth) / 2

    def make() -> None:
        with Locations(Pos(0, 0, z_mid)):
            for sgn in (-1.0, 1.0):
                with Locations(Pos(sgn * half * dx, sgn * half * dy, 0)):
                    Cylinder(radius=r, height=height, mode=Mode.SUBTRACT)
            if half > 0:
                Box(
                    (2 * half) * abs(dx) + width * abs(dy),
                    (2 * half) * abs(dy) + width * abs(dx),
                    height,
                    mode=Mode.SUBTRACT,
                )

    _in_scope(on, make)


def t_slot(
    bolt_d: float,
    head_w: float,
    head_h: float,
    slot_w: float,
    length: float,
    *,
    on=None,
) -> None:
    """T-slot groove along ±X through ``length`` mm of material: a ``head_w`` wide
    head channel ``head_h`` deep under the workplane, opening into a ``slot_w``
    wide neck ``bolt_d`` deep. Place `on` the surface where the slot opens."""
    _require_context("t_slot")

    def make() -> None:
        with Locations(Pos(0, 0, -head_h / 2 - 0.01)):
            Box(
                length,
                head_w,
                head_h,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )
        with Locations(Pos(0, 0, -head_h - bolt_d / 2)):
            Box(
                length,
                slot_w,
                bolt_d,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )

    _in_scope(on, make)


def dovetail_slot(
    width_top: float,
    width_bottom: float,
    height: float,
    length: float,
) -> "object":
    """Dovetail groove cutter — returns a Part to subtract yourself:

    ``p.part -= Pos(x, y, z) * dovetail_slot(6, 10, 4, 60)``

    The cutter runs along ±X, centred in Y; its top edge sits at Z=0 going down,
    wider at the bottom (``width_bottom``) than at the opening (``width_top``).
    """
    wt = width_top / 2
    wb = width_bottom / 2
    profile = Rot(90, 0, 0) * Polygon(
        (-wt, 0),
        (wt, 0),
        (wb, -height),
        (-wb, -height),
        align=None,
    )
    # extrude goes +Y; centre, then rotate so the groove runs along ±X
    cutter = Pos(0, -length / 2, 0) * extrude(profile, amount=length)
    return Rot(0, 0, 90) * cutter
