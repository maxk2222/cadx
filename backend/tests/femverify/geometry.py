"""Mesh builders for FEM verification cases.

All builders return trimesh meshes in the CADX frame contract: mm units,
part rests on z >= 0 unless stated otherwise. Benchmark geometries whose
reference values are normalized/unit-free (Cook, NAFEMS LE1/LE10) are built
in mm at the published proportions — linear-elastic normalized quantities
are scale-free, per-case notes document the mapping.
"""

from __future__ import annotations

import math
from typing import Any

import trimesh


def box(x: float, y: float, z: float, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    """Axis-aligned box: x∈[dx-x/2, dx+x/2], y∈[dy-y/2, dy+y/2], z∈[dz, dz+z]."""
    m = trimesh.creation.box(extents=[x, y, z])
    m.apply_translation([dx, dy, dz + z / 2])
    return m


def plate_hole(w: float, h: float, t: float, dia: float, cx: float = 0.0, cy: float = 0.0):
    """Plate [-w/2,w/2]x[-h/2,h/2]x[0,t] with a through-hole at (cx,cy)."""
    from shapely.geometry import Point, Polygon

    plate = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    hole = Point(cx, cy).buffer(dia / 2, resolution=128)
    return trimesh.creation.extrude_polygon(plate.difference(hole), t)


def cook_panel(t: float = 2.0):
    """Cook's membrane panel extruded in z: quad (0,0)(48,44)(48,60)(0,44) x t.

    Canonical geometry (Cook 1974, Simo & Rifai 1990): left edge x=0,
    y∈[0,44] clamped; right edge x=48, y∈[44,60] shear-loaded upward.
    Published normalized tip deflection u_y(C)·E·t/P = 23.91 (plane
    stress); C = top corner of the loaded edge = (48, 60).
    """
    from shapely.geometry import Polygon

    quad = Polygon([(0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)])
    return trimesh.creation.extrude_polygon(quad, t)


def lame_tube(ri: float, ro: float, h: float):
    """Thick-walled tube, axis along z, z ∈ [0,h], centered on the z axis."""
    return trimesh.creation.annulus(r_min=ri, r_max=ro, height=h)


def elliptic_annulus(ai: float, bi: float, ao: float, bo: float, t: float):
    """NAFEMS LE1/LE10 planform: quarter (x≥0, y≥0) of an elliptic annulus.

    Inner ellipse semi-axes (ai, bi), outer (ao, bo), extruded to thickness t
    (z ∈ [0,t]). Quarter symmetry: the cut edges lie on the x- and y-axes.
    """
    from shapely.geometry import Polygon
    from shapely.geometry import box as sbox

    def _ellipse(a: float, b: float, n: int = 512):
        return Polygon(
            [
                (a * math.cos(2 * math.pi * s / n), b * math.sin(2 * math.pi * s / n))
                for s in range(n)
            ]
        )

    outer = _ellipse(ao, bo)
    inner = _ellipse(ai, bi)
    first_quadrant = sbox(0.0, 0.0, ao + 1.0, bo + 1.0)
    quarter = outer.intersection(first_quadrant).difference(inner)
    return trimesh.creation.extrude_polygon(quarter, t)


def bearing_lug(w: float, h: float, t: float, dia: float, cx: float = 0.0, cy: float = 0.0):
    """Alias of plate_hole — the clevis/lug geometry used for bearing_load
    cross-checks (cosine wall pressure on a through-bore)."""
    return plate_hole(w, h, t, dia, cx=cx, cy=cy)


def build(builder: str, args: dict[str, Any] | None = None):
    """Dispatch a case-spec geometry declaration to a mesh builder.

    `trimesh.creation.extrude_polygon` emits non-watertight meshes; processing
    them here keeps voxelize on the exact cell-center `contains` path instead
    of the surface-distance fallback.
    """
    fn = _BUILDERS.get(builder)
    if fn is None:
        raise ValueError(f"unknown geometry builder '{builder}' — have {sorted(_BUILDERS)}")
    m = fn(**(args or {}))
    if not m.is_watertight:
        m = m.copy()
        m.process(validate=True)
    return m


_BUILDERS = {
    "box": box,
    "plate_hole": plate_hole,
    "cook_panel": cook_panel,
    "lame_tube": lame_tube,
    "elliptic_annulus": elliptic_annulus,
    "bearing_lug": bearing_lug,
}
