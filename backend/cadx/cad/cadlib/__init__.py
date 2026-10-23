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
    oblong_slot,
    slot_cutter,
    t_slot,
    through_hole,
)
from .ops import (
    bbox_dims,
    clip_to_box,
    collapse_solids,
    concave_edges,
    convex_edges,
    edges_at,
    finish_edges,
    hole_rim_edges,
    largest_solid,
    mesh_cut,
    mesh_fuse,
    mesh_intersect,
    safe_chamfer,
    safe_chamfer_shape,
    safe_fillet,
    safe_fillet_shape,
    trim_below,
)
from .profile import closed_polyline, profile_body
from .shapes import (
    arch_bridge,
    boss,
    filleted_box,
    knurl_ring,
    l_bracket,
    living_hinge,
    mounting_ear,
    rib,
    snap_fit_cantilever,
    strut,
    text_label,
    tube,
)

__all__ = [
    "arch_bridge",
    "bbox_dims",
    "bearing_seat",
    "boss",
    "clip_to_box",
    "closed_polyline",
    "collapse_solids",
    "concave_edges",
    "convex_edges",
    "dovetail_slot",
    "edges_at",
    "filleted_box",
    "finish_edges",
    "hole_grid",
    "hole_rim_edges",
    "insert_pocket",
    "keyhole_slot",
    "knurl_ring",
    "l_bracket",
    "largest_solid",
    "living_hinge",
    "mesh_cut",
    "mesh_fuse",
    "mesh_intersect",
    "mounting_ear",
    "oblong_slot",
    "profile_body",
    "rib",
    "safe_chamfer",
    "safe_chamfer_shape",
    "safe_fillet",
    "safe_fillet_shape",
    "slot_cutter",
    "snap_fit_cantilever",
    "strut",
    "t_slot",
    "text_label",
    "through_hole",
    "trim_below",
    "tube",
]
