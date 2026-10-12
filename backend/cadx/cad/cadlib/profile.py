"""cadlib profile — prismatic body from a measured planar outline.

from cadlib import profile_body
body = profile_body(load_image_profile(), PARAMS["thickness"], crown_mm=2,
                    edge_fillet=1.5)
"""

from build123d import Axis, Pos, extrude, fillet, loft, offset
from build123d.topology import Face, Part


def profile_body(
    face: Face, thickness: float, *, crown_mm: float = 0.0, edge_fillet: float = 0.0
) -> Part:
    """Prismatic body from a planar outline face.

    Extrudes `face` by `thickness`; optional crowned top (loft to an inset copy
    raised by `crown_mm`) and a top-edge fillet. The result rests on the XY
    plane and is centred exactly like the input outline.
    """
    body = extrude(face, amount=thickness)
    if crown_mm > 0:
        top = offset(face, amount=-min(crown_mm * 2, 4.0))
        cap = loft(
            [
                face.moved(Pos(0, 0, thickness)),
                top.moved(Pos(0, 0, thickness + crown_mm)),
            ]
        )
        body = body + cap
    if edge_fillet > 0:
        try:
            top_edges = body.edges().group_by(Axis.Z)[-1]
            body = fillet(top_edges, radius=edge_fillet)
        except Exception:
            pass
    return body
