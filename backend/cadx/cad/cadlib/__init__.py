"""cadlib — reviewed parametric helpers for generated build123d code.

Importable inside the executor sandbox (``import cadlib``). Two families:

- cutouts (``cadlib.cutouts``): run INSIDE ``with BuildPart() as p:`` and cut
  into the current part, usually scoped by ``with Locations(...)``.
- shapes (``cadlib.shapes``): pure factories returning a ``Part`` you position
  yourself, e.g. ``ear = Pos(20, 0, 8) * mounting_ear(...)``; ``p.part += ear``.

Constraint (enforced by the sandbox import guard): cadlib itself may only
import build123d / bd_warehouse.* / math / numpy / typing / dataclasses.
"""

from .cutouts import (
    bearing_seat,
    dovetail_slot,
    hole_grid,
    insert_pocket,
    keyhole_slot,
    t_slot,
    through_hole,
)
from .profile import profile_body
from .shapes import (
    boss,
    filleted_box,
    knurl_ring,
    l_bracket,
    living_hinge,
    mounting_ear,
    rib,
    snap_fit_cantilever,
    text_label,
)

__all__ = [
    "bearing_seat",
    "boss",
    "dovetail_slot",
    "filleted_box",
    "hole_grid",
    "insert_pocket",
    "keyhole_slot",
    "knurl_ring",
    "l_bracket",
    "living_hinge",
    "mounting_ear",
    "profile_body",
    "rib",
    "snap_fit_cantilever",
    "t_slot",
    "text_label",
    "through_hole",
]
