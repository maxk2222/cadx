"""Emit contract-conformant build123d source for reconstructed parts.

Everything emitted here follows the CADX sandbox contract (see
``cadx/agent/steps/codegen.py`` and ``cadx/cad/allowlist.py``): a top-level
``PARAMS`` dict literal, allowlist imports only (``build123d``, ``cadlib``)
and a final ``result`` assignment. Coordinate contract: X = length,
Y = width (+Y front), Z = height; the part rests on the XY plane, centred
at the origin, units mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AxisName = Literal["x", "y", "z"]


@dataclass
class ProfileVertex:
    """One half-profile vertex of a revolve.

    ``frac`` is the axial position as a fraction of ``PARAMS["length"]``
    (0 at the axis-min end); ``param`` is the PARAMS key holding the radius.
    """

    frac: float
    param: str


@dataclass
class ProfileSeg:
    """One emitted profile segment between vertices ``i0`` and ``i1``."""

    kind: Literal["line", "arc"]
    i0: int
    i1: int
    # arcs only: PARAMS key holding the (positive) fitted radius and the
    # RadiusArc side flag — RadiusArc puts the centre RIGHT of the travel
    # direction for positive radius (verified against build123d 0.13)
    radius_param: str = ""
    radius_sign: int = 1
    major: bool = False  # arc spans > 180° (needs short_sagitta=False)


def _f(v: float) -> str:
    """Compact numeric literal for generated source (always a float)."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    if "." not in s:
        s += ".0"
    return s


def _params_block(params: dict[str, float]) -> str:
    body = ",\n".join(f'    "{k}": {_f(v)}' for k, v in params.items())
    return "PARAMS = {\n" + body + "\n}"


def _axial_expr(frac: float) -> str:
    if frac <= 0.0:
        return "0.0"
    if frac >= 1.0:
        return 'PARAMS["length"]'
    return f'PARAMS["length"] * {_f(frac)}'


def emit_revolve_code(
    axis: AxisName,
    verts: list[ProfileVertex],
    segs: list[ProfileSeg],
    offset: tuple[float, float, float],
    params: dict[str, float],
    *,
    bore_param: str | None = None,
) -> str:
    """Emit a lathe part: half-profile in (axial, radial) coords → revolve.

    The profile is drawn in a workplane that contains the canonical world
    axis (XY for Axis.X, YZ for Axis.Y, XZ for Axis.Z), revolved 360°, then
    shifted by ``offset`` so the reconstructed axis line sits where the mesh
    measured it (part stays XY-centred / rests on the plane).
    """

    def pt(i: int) -> str:
        v = verts[i]
        a, r = _axial_expr(v.frac), f'PARAMS["{v.param}"]'
        # x/y axes: local (u=axial, v=radial); z axis on Plane.XZ: (u=radial, v=axial)
        return f"({a}, {r})" if axis != "z" else f"({r}, {a})"

    def axis_pt(frac: float) -> str:
        a = _axial_expr(frac)
        return f"({a}, 0.0)" if axis != "z" else f"(0.0, {a})"

    seg_lines: list[str] = []
    for s in segs:
        a, b = pt(s.i0), pt(s.i1)
        if s.kind == "arc":
            sign = "" if s.radius_sign > 0 else "-"
            sag = ", short_sagitta=False" if s.major else ""
            seg_lines.append(
                f'    RadiusArc({a}, {b}, radius={sign}PARAMS["{s.radius_param}"]{sag})'
            )
        else:
            seg_lines.append(f"    Line({a}, {b})")
    # close the half-profile: a through-bore along the axis closes through
    # the inner radius (annular section -> hollow revolve); otherwise down
    # to the rotation axis and back to the start
    r_start = float(params.get(verts[0].param, 0.0))
    r_end = float(params.get(verts[-1].param, 0.0))
    last = pt(len(verts) - 1)
    if bore_param:
        a0, a1 = _axial_expr(0.0), _axial_expr(1.0)
        bp = f'PARAMS["{bore_param}"]'
        if axis == "z":
            end_pt, start_pt, first = f"({bp}, {a1})", f"({bp}, {a0})", pt(0)
        else:
            end_pt, start_pt, first = f"({a1}, {bp})", f"({a0}, {bp})", pt(0)
        seg_lines.append(f"    Line({last}, {end_pt})")
        seg_lines.append(f"    Line({end_pt}, {start_pt})")
        seg_lines.append(f"    Line({start_pt}, {first})")
    elif r_end > 0.0:
        seg_lines.append(f"    Line({last}, {axis_pt(1.0)})")
        seg_lines.append(f"    Line({axis_pt(1.0)}, {axis_pt(0.0)})")
    else:
        seg_lines.append(f"    Line({last}, {axis_pt(0.0)})")
    if r_start > 0.0 and not bore_param:
        seg_lines.append(f"    Line({axis_pt(0.0)}, {pt(0)})")
    # when r_start == 0 the chain already ends at the start point -> closed

    opener = {
        "x": "with BuildLine() as _bl:",
        "y": "with BuildLine(Plane.YZ) as _bl:",
        "z": "with BuildLine(Plane.XZ) as _bl:",
    }[axis]
    rev_axis = {"x": "Axis.X", "y": "Axis.Y", "z": "Axis.Z"}[axis]
    ox, oy, oz = (_f(c) for c in offset)
    body = "\n".join(seg_lines)
    return f'''"""Reconstructed revolve part — emitted deterministically by cadx.cad.reconstruct."""

from build123d import *

{_params_block(params)}

{opener}
{body}
_face = make_face(edges=_bl.line)
result = Pos({ox}, {oy}, {oz}) * revolve(_face, axis={rev_axis})
'''


def emit_prism_code(
    axis: AxisName,
    outline: list[tuple[float, float]],
    a_min: float,
    params: dict[str, float],
) -> str:
    """Emit a prismatic part: measured cross-section polygon → extrude.

    ``outline`` carries the polygon in the workplane's local (u, v) coords
    — the caller remaps world radial coords per axis (for "x" the plane is
    (u=+Z, v=-Y), for "y" it is (u=+Z, v=+X); "z" is the plain XY plane so
    the reviewed ``cadlib.closed_polyline`` helper is used there).
    """
    pts = ", ".join(f"({_f(u)}, {_f(v)})" for u, v in outline)
    if axis == "z":
        pos = f"Pos(0.0, 0.0, {_f(a_min)}) * " if abs(a_min) > 1e-6 else ""
        return f'''"""Reconstructed prism part — emitted deterministically by cadx.cad.reconstruct."""

import cadlib
from build123d import *

{_params_block(params)}

_outline = [{pts}]
_face = cadlib.closed_polyline(_outline)
result = {pos}cadlib.profile_body(_face, PARAMS["thickness"])
'''
    origin = {
        "x": f"({_f(a_min)}, 0.0, 0.0)",
        "y": f"(0.0, {_f(a_min)}, 0.0)",
    }[axis]
    normal = {"x": "(1.0, 0.0, 0.0)", "y": "(0.0, 1.0, 0.0)"}[axis]
    return f'''"""Reconstructed prism part — emitted deterministically by cadx.cad.reconstruct."""

from build123d import *

{_params_block(params)}

_outline = [{pts}]
with BuildLine(Plane(origin={origin}, x_dir=(0.0, 0.0, 1.0), z_dir={normal})) as _bl:
    Polyline(*_outline, close=True)
_face = make_face(edges=_bl.line)
result = extrude(_face, amount=PARAMS["thickness"])
'''


# --------------------------------------------------------------------------- #
# composite (primitive-union reconstruction)                                   #
# --------------------------------------------------------------------------- #


@dataclass
class CompositePrim:
    """One fitted primitive of a composite part.

    ``op="add"`` unions the solid into the result, ``op="cut"`` subtracts
    it (a fitted void — e.g. an interior cylinder = bore — or a bounding
    half-space emitted as an oversized box on the far side of a plane).

    ``shape``: ``"box"`` | ``"cylinder"`` | ``"sphere"`` | ``"trim"``
    (trim is a literal-dimension box, always a cut). ``center``/``rot``
    place the canonical solid (centred at origin, main axis +Z); ``rot``
    is an (axis, angle_deg) pair or None for identity.

    ``dim_params`` maps ctor kwargs to PARAMS keys (parametric dims);
    ``dim_literals`` bakes derived dims straight into the call (cutters,
    trim boxes, clip boxes — they are measured, not user-facing).

    ``clip_*`` optionally intersects the primitive with a second placed
    box (a sphere cap clipped to the fitted surface extent, so a dome
    does not poke a full ball out through the part).
    """

    op: Literal["add", "cut"]
    shape: Literal["box", "cylinder", "sphere", "trim"]
    center: tuple[float, float, float]
    rot: tuple[tuple[float, float, float], float] | None = None
    dim_params: dict[str, str] = field(default_factory=dict)
    dim_literals: dict[str, float] = field(default_factory=dict)
    clip_center: tuple[float, float, float] | None = None
    clip_rot: tuple[tuple[float, float, float], float] | None = None
    clip_dims: tuple[float, float, float] | None = None

    @property
    def clipped(self) -> bool:
        return self.clip_center is not None and self.clip_dims is not None


def _v3(v: tuple[float, float, float]) -> str:
    return f"({_f(v[0])}, {_f(v[1])}, {_f(v[2])})"


def _place(
    center: tuple[float, float, float], rot: tuple[tuple[float, float, float], float] | None
) -> str:
    """``Pos(...) *`` or ``Location(...) *`` prefix for a placed solid."""
    if rot is None:
        return f"Pos{_v3(center)} * "
    (ax, deg) = rot
    return f"Location({_v3(center)}, {_v3(ax)}, {_f(deg)}) * "


def _prim_ctor(p: CompositePrim) -> str:
    def arg(kw: str) -> str:
        if kw in p.dim_params:
            return f'{kw}=PARAMS["{p.dim_params[kw]}"]'
        return f"{kw}={_f(p.dim_literals[kw])}"

    if p.shape == "cylinder":
        return f"Cylinder({arg('radius')}, {arg('height')})"
    if p.shape == "sphere":
        return f"Sphere({arg('radius')})"
    # box / trim: positional (length, width, height)
    out = []
    for kw in ("length", "width", "height"):
        if kw in p.dim_params:
            out.append(f'PARAMS["{p.dim_params[kw]}"]')
        else:
            out.append(_f(p.dim_literals[kw]))
    return f"Box({', '.join(out)})"


def emit_composite_code(prims: list[CompositePrim], params: dict[str, float]) -> str:
    """Emit a union/cut of placed primitives for a composite part.

    ``prims`` arrive in fit order; the emitted code unions the ``add``
    primitives first, then applies ``cut`` primitives and ``trim``
    half-space boxes. At least one ``add`` primitive is required.
    """
    lines: list[str] = []
    add_terms: list[str] = []
    cut_terms: list[str] = []
    for i, p in enumerate(prims):
        expr = f"{_place(p.center, p.rot)}{_prim_ctor(p)}"
        if p.clipped:
            assert p.clip_center is not None and p.clip_dims is not None
            lines.append(
                f"_c{i} = {_place(p.clip_center, p.clip_rot)}"
                f"Box({_f(p.clip_dims[0])}, {_f(p.clip_dims[1])}, {_f(p.clip_dims[2])})"
            )
            expr = f"({expr}) & _c{i}"
        lines.append(f"_p{i} = {expr}")
        (add_terms if p.op == "add" else cut_terms).append(f"_p{i}")
    if not add_terms:
        raise ValueError("composite emission needs at least one 'add' primitive")
    result = " + ".join(add_terms)
    if cut_terms:
        result = f"({result}) - {' - '.join(cut_terms)}"
    body = "\n".join(lines)
    return f'''"""Reconstructed composite part — emitted deterministically by cadx.cad.reconstruct."""

from build123d import *

{_params_block(params)}

{body}
result = {result}
'''
