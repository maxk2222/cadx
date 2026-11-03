"""cadlib blades — radial propeller/impeller blades lofted from station sections.

Hand-rolling a propeller blade is the single most failure-prone transform in
generated code: the section must be pitched around the *radial* axis through
its own centre (rotate about global Z instead and the blade's radius shrinks
by cos(pitch) — observed as tips reaching r=57 on a declared r=60), then
re-rotated into the blade's azimuth (applying the azimuth first double-uses
the transform and scatters the sections off the stacking line).

``radial_blade`` encodes the whole compound transform in one reviewed helper:
each station centre sits at ``(r·cos φ, r·sin φ, z)`` with
``φ = blade_azimuth + skew(r)``, and its section plane is built with
``x_dir = cos(pitch)·t̂ + sin(pitch)·ẑ`` (tangential unit vector tilted up by
the pitch angle inside the plane perpendicular to the radius) — the pitch
rotation therefore never moves the station centre, so the tip radius is
exactly ``tip_r`` by construction. ``skew`` enters as a plain azimuth offset
of the station centre, never as a plane rotation.

Sections are ellipses (``section="ellipse"`` — thick lens, the marine-prop
default) or symmetric NACA 00xx airfoils (``section="naca"`` — wing blades,
``naca_t`` sets the thickness fraction). Both are centred on the station
point, so ``chord`` is the full fore-aft extent and ``thickness`` the full
top-bottom extent.

Frame: propeller axis = global Z. The blade stack lies in the XY plane at
``z`` (mid-plane), blades radiate outward. The returned Part is centred on
the axis — place it with ``Pos(0, 0, hub_mid_z) * radial_blade(...)`` and
fuse the root stations INTO the hub (``root_r`` inside the hub radius gives
the overlap the boolean needs).

Example — 5-blade Ø120 marine propeller on a r=14 hub::

    hub = Pos(0, 0, 10) * Cylinder(14, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    blades = Pos(0, 0, 14) * radial_blade(
        n_blades=5, root_r=10.0, tip_r=60.0,
        chord=(24.0, 15.0), thickness=(4.0, 2.5),
        pitch_root_deg=42.0, pitch_tip_deg=24.0, skew_tip_deg=18.0,
    )
    part += hub + blades  # root stations buried in the hub wall
"""

import itertools
import math
from typing import Any

from build123d import BuildSketch, Ellipse, Plane, Polygon, Vector, loft
from build123d.topology import Part


def naca4(t: float, n_pts: int = 41) -> list[tuple[float, float]]:
    """Symmetric NACA 00xx profile, closed outline on the local chord axis.

    Returns (x, y) points with x in [0, 1] (leading→trailing edge) and y the
    half-thickness — scale by chord length. ``t`` is the thickness fraction
    (0.12 = 12% chord). Half-cosine x-spacing clusters points at the leading
    edge radius; the outline walks the upper surface aft then the lower
    surface forward so the closed wire is single-direction."""
    n = max(11, int(n_pts))
    pts: list[tuple[float, float]] = []
    for i in range(n):
        beta = math.pi * i / (n - 1)
        x = 0.5 * (1.0 - math.cos(beta))
        yt = (
            5.0
            * t
            * (
                0.2969 * math.sqrt(max(x, 1e-12))
                - 0.1260 * x
                - 0.3516 * x * x
                + 0.2843 * x * x * x
                - 0.1015 * x * x * x * x
            )
        )
        pts.append((x, yt))
    for i in range(n - 2, -1, -1):
        x, yt = pts[i]
        pts.append((x, -yt))
    return pts


def _station_interp(spec: dict[str, Any], s: float) -> dict[str, float]:
    """Linear-in-radius interpolation for an implicit (scalar/tuple) station."""

    def _pair(v: Any, default: float) -> tuple[float, float]:
        if v is None:
            return default, default
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return float(v[0]), float(v[1])
        return float(v), float(v)

    c0, c1 = _pair(spec.get("chord"), 20.0)
    t0, t1 = _pair(spec.get("thickness"), 4.0)
    z0, z1 = _pair(spec.get("z"), 0.0)
    return {
        "chord": c0 + (c1 - c0) * s,
        "thick": t0 + (t1 - t0) * s,
        "pitch": float(spec.get("pitch_root_deg", 45.0))
        + (float(spec.get("pitch_tip_deg", 28.0)) - float(spec.get("pitch_root_deg", 45.0))) * s,
        "skew": float(spec.get("skew_tip_deg", 15.0)) * s,
        "z": z0 + (z1 - z0) * s,
    }


def radial_blade(
    n_blades: int = 4,
    root_r: float = 8.0,
    tip_r: float = 60.0,
    *,
    chord: float | tuple[float, float] = (22.0, 14.0),
    thickness: float | tuple[float, float] = (4.0, 2.5),
    pitch_root_deg: float = 45.0,
    pitch_tip_deg: float = 28.0,
    skew_tip_deg: float = 15.0,
    n_stations: int = 7,
    section: str = "ellipse",
    naca_t: float = 0.12,
    handed: int = 1,
    z: float = 0.0,
    stations: list[dict[str, Any]] | None = None,
) -> Part:
    """`n_blades` identical blades lofted from root_r to tip_r, centred on Z.

    chord/thickness accept a scalar (constant) or ``(root, tip)`` pair for a
    linear taper; ``pitch_root_deg``/``pitch_tip_deg`` sweep the blade-angle
    schedule (pitch = angle between the chord line and the rotation plane);
    ``skew_tip_deg`` sweeps the tip azimuth back along the rotation.
    ``handed=-1`` mirrors pitch+skew for a left-handed prop.

    ``stations`` overrides the whole schedule — list of dicts sorted by r:
    ``{"r", "chord", "thick", "pitch_deg", "skew_deg", "z"}`` (thick/z also
    read as "thickness"/"z_section"; missing keys fall back to the
    interpolated defaults)."""
    n = max(1, int(n_blades))
    if tip_r <= root_r:
        raise ValueError(f"tip_r {tip_r} must exceed root_r {root_r}")
    sign = 1.0 if handed >= 0 else -1.0
    spec = {
        "chord": chord,
        "thickness": thickness,
        "pitch_root_deg": pitch_root_deg,
        "pitch_tip_deg": pitch_tip_deg,
        "skew_tip_deg": skew_tip_deg,
        "z": z,
    }

    stations_l: list[dict[str, float]] = []
    if stations:
        for st in sorted(stations, key=lambda d: float(d["r"])):
            s = (float(st["r"]) - root_r) / (tip_r - root_r)
            base = _station_interp(spec, min(max(s, 0.0), 1.0))
            base.update(
                {
                    "chord": float(st.get("chord", base["chord"])),
                    "thick": float(st.get("thick", st.get("thickness", base["thick"]))),
                    "pitch": float(st.get("pitch_deg", st.get("pitch", base["pitch"]))),
                    "skew": float(st.get("skew_deg", st.get("skew", base["skew"]))),
                    "z": float(st.get("z", st.get("z_section", base["z"]))),
                }
            )
            base["r"] = float(st["r"])
            stations_l.append(base)
    else:
        ns = max(3, int(n_stations))
        for i in range(ns):
            s = i / (ns - 1)
            base = _station_interp(spec, s)
            base["r"] = root_r + (tip_r - root_r) * s
            stations_l.append(base)
    if len(stations_l) < 2:
        raise ValueError("radial_blade needs >=2 stations to loft")

    # densify: OCC's loft tracking fails when consecutive sections rotate too
    # far relative to each other (a 3-station 40→20° pitch + 15° skew sweep
    # sews "command not done"; the same schedule at ~5-8 mm / ~8° steps lofts
    # cleanly). Subdivide every interval past the empirical caps — user
    # stations stay exact anchor points.
    dense: list[dict[str, float]] = []
    for a, b in itertools.pairwise(stations_l):
        n_seg = max(
            1,
            math.ceil(abs(b["r"] - a["r"]) / 8.0),
            math.ceil(abs(b["pitch"] - a["pitch"]) / 10.0),
            math.ceil(abs(b["skew"] - a["skew"]) / 8.0),
        )
        for j in range(n_seg):
            t = j / n_seg
            dense.append(
                {k: a[k] + (b[k] - a[k]) * t for k in ("r", "chord", "thick", "pitch", "skew", "z")}
            )
    dense.append(stations_l[-1])

    blades: list[Part] = []
    for b in range(n):
        phi0 = math.radians(360.0 * b / n)
        sketches = []
        for st in dense:
            phi = phi0 + sign * math.radians(st["skew"])
            cx = st["r"] * math.cos(phi)
            cy = st["r"] * math.sin(phi)
            cz = st["z"]
            r_hat = Vector(math.cos(phi), math.sin(phi), 0.0)
            t_hat = Vector(-math.sin(phi), math.cos(phi), 0.0)
            z_hat = Vector(0.0, 0.0, 1.0)
            # pitch = tilt of the chord inside the section plane, measured
            # from the rotation plane — rotate the sketch x_dir by pitch
            # toward +z (never the station centre: radius stays exact)
            p = sign * math.radians(st["pitch"])
            x_dir = (t_hat * math.cos(p) + z_hat * math.sin(p)).normalized()
            pln = Plane(origin=Vector(cx, cy, cz), x_dir=x_dir, z_dir=r_hat)
            with BuildSketch(pln) as sk:
                if section == "naca":
                    # naca4's yt max is 0.5*t of chord — rescale so the
                    # declared `thick` is the real full thickness
                    # (half-thickness 0.5·naca_t → thick/2 ⇒ scale thick/naca_t)
                    outline = [
                        (
                            (x - 0.5) * st["chord"],
                            y * st["thick"] / max(naca_t, 1e-9),
                        )
                        for x, y in naca4(naca_t)
                    ]
                    Polygon(*outline)
                else:
                    Ellipse(x_radius=st["chord"] / 2.0, y_radius=st["thick"] / 2.0)
            sketches.append(sk.sketch)
        # Multi-section ThruSections lofts flip cap orientation
        # non-deterministically on twisted section sequences (observed: the
        # tip or root cap sewn inward → the solid passes is_valid but every
        # downstream boolean treats it as a void; the eval propeller fused
        # to a bare hub). Two-section lofts orient consistently, so build
        # the blade as a chain of segment lofts fused on the shared station
        # faces.
        segs = [loft([sa, sb]) for sa, sb in itertools.pairwise(sketches)]
        blade_l = segs[0]
        for sgmt in segs[1:]:
            blade_l = blade_l + sgmt
        blades.append(blade_l)
    out = blades[0]
    for bl in blades[1:]:
        out += bl
    return out
