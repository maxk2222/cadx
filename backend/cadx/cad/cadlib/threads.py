"""cadlib threads — helical ISO thread ridges from few multi-turn lofts.

bd_warehouse builds a thread as one lofted solid PER turn (~76 rings for an
M10x115 screw) and leaves them to the caller as a Compound — fusing that into
the part costs a chunked O(n) boolean chain (~25-35 s in the executor) that
has segfaulted OCC outright.

``helical_thread`` builds the same ridge as a handful of multi-turn loft
pieces instead (≤4 turns each → ~20 solids for that screw, ~5 s total fuse).
Profile geometry, radii, interference overlap and the butt-jointed piece
boundaries are bd_warehouse-identical — each piece is an open spiral segment
lofted from the axial V-section, exactly what IsoThread lofts per turn, so
the pieces merge through the mating core the same way the rings do.

A single continuous pipe (``sweep()`` over the whole helix) was tried first:
it is fast to build, but its one-giant-helical-face degenerates the boolean
— 21 turns fused fine, 76 turns silently returned an EMPTY result, and
sub-span segments produced inside-out unions. Multi-turn lofts keep the
bd-proven boundary structure; beyond ~8 turns per piece the section curves
get long enough to break OCC again, so 4 turns is the safe chunk size.

Span semantics mirror bd's "raw" ends: the ridge nominally overshoots each
helix end by up to ~half a profile width — for an external screw ridge the
caller lifts the compound by 0.375·pitch so the bottom face lands flush on
the thread-tip plane while the top overshoot buries into the shank. For
cutter use (``tapped_hole`` groove) the overshoot is free margin.

``end_finishes`` is accepted for IsoThread signature parity; "square"/"raw"/
"fade" all currently map to the same raw-end geometry (see above).
"""

from typing import Literal

from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    Compound,
    Helix,
    Plane,
    Pos,
    Solid,
    Vector,
    insert,
    loft,
)

Finish = Literal["raw", "square", "fade"]

# turns per loft piece: >8-turn spiral segments produce boolean-invalid
# fragment unions in OCC (measured on M10x114.5); 4 stays comfortably inside
_MAX_TURNS_PER_PIECE = 4
# sections per turn inside one loft piece — bd uses 11, 8 keeps the flank
# smooth enough for export while staying cheap
_SECTIONS_PER_TURN = 8


def _profile_and_radii(
    major_diameter: float,
    pitch: float,
    length: float,
    hand: str,
    external: bool,
    interference: float,
):
    """``(thread_profile, root_radius)`` exactly as IsoThread computes them."""
    from bd_warehouse.thread import IsoThread, Thread

    iso = IsoThread(
        major_diameter=major_diameter,
        pitch=pitch,
        length=length,
        external=external,
        hand=hand,
        interference=interference,
        simple=True,
    )
    # Thread() rebuilds the same axial V-section IsoThread would pass to its
    # per-turn engine — simple=True gives the profile without sweeping.
    inner = Thread(
        apex_radius=iso.apex_radius,
        apex_width=pitch / 8 if external else pitch / 4,
        root_radius=iso.root_radius,
        root_width=3 * pitch / 4 if external else 7 * pitch / 8,
        pitch=pitch,
        length=length,
        interference=interference,
        hand=hand,
        simple=True,
    )
    return inner.thread_profile, iso.root_radius


def _loft_piece(profile, root_radius: float, pitch: float, z0: float, z1: float, left: bool):
    """Loft the axial V-section along the helix span [z0, z1] — one open
    spiral solid, bd-ring-identical construction at coarser granularity."""
    with BuildPart() as part:
        with BuildLine():
            path = Helix(pitch=pitch, height=z1 - z0, radius=root_radius, lefthand=left)
        n = max(2, round((z1 - z0) / pitch * _SECTIONS_PER_TURN) + 1)
        for i in range(n):
            u = i / (n - 1)
            nz = path % u
            z_dir = Vector(nz.X, nz.Y, 0).normalized()
            # local X = world +Z: profile's axial coord maps to Z, radial
            # coord to the radial direction
            with BuildSketch(Plane(path @ u, x_dir=(0, 0, 1), z_dir=z_dir)):
                insert(profile)
        loft()
    solid = part.solids()[0]
    return solid.move(Pos(0, 0, z0)) if z0 else solid


def helical_thread(
    major_diameter: float,
    pitch: float,
    length: float,
    *,
    hand: str = "right",
    external: bool = True,
    interference: float = 0.2,
    end_finishes: tuple[Finish, Finish] = ("raw", "raw"),
) -> Compound | Solid:
    """Helical ISO thread ridge as a few multi-turn loft pieces.

    Args mirror ``bd_warehouse.thread.IsoThread`` so callers can swap it in:
    ``external`` builds a bolt ridge (apex at the major diameter), ``False``
    an internal nut ridge; ``interference`` sinks the root into the mating
    core so the part fuse is robust. Returns ONE solid for short threads or
    a Compound of ≤4-turn pieces for long ones — fuse it whole onto the core
    (``_cadx_fuse_solids`` handles the grouping) or use it as a cutter.
    """
    if hand not in {"right", "left"}:
        raise ValueError(f'hand must be "right" or "left", not {hand!r}')
    for fin in end_finishes:
        if fin not in {"raw", "square", "fade"}:
            raise ValueError(f'end_finishes entries must be raw/square/fade, not {fin!r}')
    length = float(length)
    if length <= 0:
        raise ValueError(f"length must be positive, not {length}")
    pitch = float(pitch)
    if pitch <= 0:
        raise ValueError(f"pitch must be positive, not {pitch}")

    profile, root_radius = _profile_and_radii(
        float(major_diameter), pitch, length, hand, external, float(interference)
    )
    left = hand == "left"
    span = _MAX_TURNS_PER_PIECE * pitch

    pieces: list[Solid] = []
    z = 0.0
    while z < length:
        pieces.append(_loft_piece(profile, root_radius, pitch, z, min(z + span, length), left))
        z += span
    result = pieces[0] if len(pieces) == 1 else Compound(children=pieces)
    # same bookkeeping tag tapped_hole uses — the runner's thread_count
    # scan picks it up wherever the pieces land in the recipe namespace
    result._cadx_thread = {  # type: ignore[attr-defined]
        "kind": "thread",
        "major_diameter": float(major_diameter),
        "pitch": float(pitch),
        "length": float(length),
        "external": bool(external),
        "hand": hand,
    }
    return result
