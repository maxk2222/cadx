"""cadlib profile — prismatic body from a measured planar outline.

from cadlib import profile_body
body = profile_body(load_image_profile(), PARAMS["thickness"], crown_mm=2,
                    edge_fillet=1.5)
"""

from build123d import Axis, BuildLine, Polyline, Pos, extrude, fillet, loft, make_face, offset
from build123d.topology import Face, Part


def closed_polyline(
    pts: list[tuple[float, ...]],
    *,
    tol: float = 1e-3,
) -> Face:
    """Planar face from a dense sampled outline, robust to bad samples.

    Hand-sampled outlines (S-curves, splines, image traces) routinely produce
    consecutive duplicates, a repeated closing point or stray collinear
    spikes — all of which crash ``Polyline`` with a cryptic OCP error. This
    helper drops degenerate points first, then returns the filled face.
    Points are projected to Z=0 when given as (x, y, z).
    """
    cleaned: list[tuple[float, float]] = []
    for p in pts:
        xy = (float(p[0]), float(p[1]))
        if not cleaned or abs(xy[0] - cleaned[-1][0]) > tol or abs(xy[1] - cleaned[-1][1]) > tol:
            cleaned.append(xy)
    while len(cleaned) > 1 and (
        abs(cleaned[-1][0] - cleaned[0][0]) <= tol and abs(cleaned[-1][1] - cleaned[0][1]) <= tol
    ):
        cleaned.pop()
    if len(cleaned) < 3:
        raise ValueError(f"closed_polyline needs ≥3 distinct points, got {len(cleaned)}")
    with BuildLine() as bl:
        Polyline(*cleaned, close=True)
    return make_face(edges=bl.line).face()


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
