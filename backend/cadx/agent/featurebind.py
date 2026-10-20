"""Deterministic feature -> cadlib binder.

The E2E post-mortem on l_bracket_cad showed the model re-derives axes from
prose even when the issue text contains the exact call — it picked
``direction="+y", normal="+z"`` for a slot that only reads correctly as
``direction="+z", normal="+x"``. This module removes that translation step:
``bind_features(spec)`` maps every spec feature it understands to a
ready-made cadlib call with resolved axes, dims and placement recipe; the
rendered block is injected into codegen/improve prompts as REQUIRED
SNIPPETS the model only has to assemble.

Every binding degrades safely: unresolvable positions become ``<...>``
placeholders with a comment, unresolvable geometry marks the whole
binding ``resolved=False`` and the model keeps full freedom there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cadx.agent.spechost import host_body, host_span
from cadx.cad.features import _feature_axis_hint
from cadx.cad.poshint import parse_coord_points
from cadx.llm.schemas import Feature, PartSpec

_MAX_BINDINGS = 12

# --------------------------------------------------------------------------- #
# small parsing helpers


def _dim(dims: dict[str, float], *keys: str) -> float | None:
    for k in keys:
        v = dims.get(k)
        if isinstance(v, int | float) and v > 0:
            return float(v)
    return None


_AXIS_NORM_RE = re.compile(r"^[+-]?[xyz]$", re.I)

_AXIS_LETTER_RE = re.compile(
    r"(?:along|entlang(?:\s+(?:der|die|den))?|längs|um)\s+([xyz])(?:[\s-]*(?:axis|achse))?"
    r"|\b([xyz])[\s-]*(?:axis|achse)\b",
    re.I,
)


def _norm_axis(s: str | None) -> str | None:
    """'x'|'y'|'z'|'+x'… → '+x' style; anything else → None."""
    s = (s or "").strip().lower()
    if not _AXIS_NORM_RE.match(s):
        return None
    return s if s[0] in "+-" else f"+{s}"


def _axis_letter(axis: str | None) -> str | None:
    return axis[1] if axis and axis[0] in "+-" else (axis or None)


def _axis_in_text(text: str) -> str | None:
    """'+x'-style axis mentioned in free text ('along X', 'x-Achse', 'längs Y')."""
    m = _AXIS_LETTER_RE.search(text.lower())
    if not m:
        return None
    return f"+{next(g for g in m.groups() if g)}"


# Locations-frame rotation that makes the local +Z of through_hole point
# along the desired world axis (through_hole always cuts along local Z).
_FRAME_ROT = {
    "+z": "",
    "-z": " * Rot(0, 180, 0)",
    "+x": " * Rot(0, 90, 0)",
    "-x": " * Rot(0, -90, 0)",
    "+y": " * Rot(-90, 0, 0)",
    "-y": " * Rot(90, 0, 0)",
}

# canonical dim keys -> axis, used to find a body's THIN axis by elimination:
# a leg {height:50, width:40, thickness:4} claims z+y, so thickness runs on x.
_DIM_AXIS = {"length": "x", "width": "y", "height": "z"}


def _thin_axis(f: Feature) -> str | None:
    """Axis along which this feature's `thickness`/`depth` dim runs."""
    claimed = {_DIM_AXIS[k] for k in _DIM_AXIS if k in (f.dims or {})}
    if len(claimed) == 2 and _dim(f.dims or {}, "thickness"):
        return next(iter({"x", "y", "z"} - claimed))
    # a single thickness on a horizontal plate means z; on a wall, look at
    # position text for the face it belongs to
    if _dim(f.dims or {}, "thickness") and not claimed:
        pos = f"{f.position or ''} {f.description or ''}".lower()
        if any(w in pos for w in ("vertical", "vertikal", "wall", "wand", "seite", "side", "end")):
            ax = _axis_in_text(pos)
            if ax:
                return _axis_letter(ax)
            return "x"  # contract: vertical walls sit at an X end by default
        return "z"
    return None


_ENV_HALF = {"x": "length_x", "y": "width_y", "z": "height_z"}


def _feature_centre(spec: PartSpec, f: Feature) -> dict[str, float | str]:
    """Resolve a feature's centre from its position text + the spec envelope.

    'z=4..54' → z=29; '-X end' on a 4 mm thin leg → x=-(L/2-t/2)=-28.
    Unresolvable axes keep '<axis>' placeholders so the emitted Pos stays
    visibly incomplete instead of silently centred."""
    text = f"{f.position or ''} {f.description or ''}".lower()
    env = {a: float(getattr(spec, _ENV_HALF[a]).value) / 2.0 for a in "xyz"}
    thin = _thin_axis(f)
    thick = _dim(f.dims or {}, "thickness")
    out: dict[str, float | str] = {}
    ranges = {g[0]: (float(g[1]) + float(g[2])) / 2.0 for g in _RANGE_RE.findall(text)}
    coords = {g[0]: float(g[1]) for g in _COORD_RE.findall(text)}
    for letter in "xyz":
        v: float | str = f"<{letter}>"
        if letter in ranges:
            v = ranges[letter]
        elif letter in coords:
            v = coords[letter]
        if letter in ranges:
            v = ranges[letter]
        elif letter in coords:
            v = coords[letter]
        if v == f"<{letter}>":
            em = _END_RE.search(text)
            end = None
            if em:
                end = (em.group(1), em.group(2)) if em.group(1) else (em.group(3), em.group(4))
            if end and end[1] == letter and letter == thin and thick:
                # '-X end' on a t-thin wall: centre sits t/2 inside the envelope face
                v = (env[letter] - thick / 2.0) * (-1 if end[0] == "-" else 1)
            elif letter == "z" and ("base" in text or "bottom" in text or "boden" in text):
                v = (thick or 0.0) / 2.0
            elif letter == "z" and ("top" in text or "oben" in text) and thick:
                v = env["z"] * 2.0 - thick / 2.0
            else:
                v = 0.0  # contract default: centred on the origin
        out[letter] = v
    return out


def _pos_str(coords: dict[str, float | str]) -> str:
    def c(a: str) -> str:
        v = coords[a]
        return f"{v:g}" if isinstance(v, int | float) else str(v)

    return f"Pos({c('x')}, {c('y')}, {c('z')})"


_COORD_RE = re.compile(r"\b([xyz])\s*[=:]\s*([+-]?\d+(?:\.\d+)?)", re.I)
_RANGE_RE = re.compile(
    r"\b([xyz])\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*\.\.\s*([+-]?\d+(?:\.\d+)?)", re.I
)
_END_RE = re.compile(r"([+-])\s*([xyz])\s*end\b|end\s*([+-])\s*([xyz])", re.I)
_APART_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:apart|abstand|spacing|pitch|entfernt)"
    r"|(?:apart|abstand|spacing|pitch)\s*(?:of|von)?\s*(\d+(?:\.\d+)?)\s*mm",
    re.I,
)
_PCD_RE = re.compile(
    r"(?:pcd|pitch\s*circle(?:\s*diameter)?|lochkreis|bolt\s*circle)\s*[Ød=]*\s*(\d+(?:\.\d+)?)",
    re.I,
)
_CENTERED_RE = re.compile(r"center|centr|mittig|zentriert|middle", re.I)


def _hole_positions(
    spec: PartSpec, f: Feature, count: int, axis: str
) -> tuple[list[tuple[float, float]], str]:
    """Per-hole positions on the plane perpendicular to the hole axis.

    Returns (points, note) — points are the two off-axis coords; the note
    documents what was resolved for the comment line."""
    text = f"{f.position or ''} {f.description or ''}".lower()
    dims = f.dims or {}
    off_letters = {a for a in "xyz" if a != _axis_letter(axis)}

    # structured Feature.centers — one row per declared instance beats any
    # text re-parse (a null axis centres on 0.0, same as the coord contract)
    centers = getattr(f, "centers", None) or []
    off = [a for a in "xyz" if a != _axis_letter(axis)]
    if len(centers) == count and centers:
        pts = [
            (
                float(getattr(c, off[0], None) or 0.0),
                float(getattr(c, off[1], None) or 0.0),
            )
            for c in centers
        ]
        return pts, "explicit positions from spec centers"

    # PCD / bolt circle → polar ring
    m = _PCD_RE.search(text)
    pcd = m.group(1) if m else _dim(dims, "pcd", "diameter_pcd", "bolt_circle")
    if pcd:
        r = float(pcd) / 2.0
        import math

        pts = [
            (
                round(r * math.cos(2 * math.pi * i / count), 3),
                round(r * math.sin(2 * math.pi * i / count), 3),
            )
            for i in range(count)
        ]
        return pts, f"{count}x on PCD {float(pcd):g} mm"

    # explicit coords "x=-15, x=+15" / "at x=10, y=20" — only off-axis letters
    # count as positions (a z= coord names the entry plane, not the pattern)
    coord_lists: dict[str, list[float]] = {}
    for letter, val in _COORD_RE.findall(text):
        if letter in off_letters:
            coord_lists.setdefault(letter, []).append(float(val))
    if coord_lists:
        n = max(len(v) for v in coord_lists.values())
        ax_letters = sorted(coord_lists)
        pts: list[tuple[float, float]] = []
        for i in range(n):
            vals = {
                letter: (
                    coord_lists[letter][i]
                    if i < len(coord_lists[letter])
                    else coord_lists[letter][-1]
                )
                for letter in ax_letters
            }
            # map each parsed letter onto its canonical off-axis slot —
            # "y=10, y=20" alone must land on the y slot, not x
            pts.append((vals.get(off[0], 0.0), vals.get(off[1], 0.0)))
        return pts, f"explicit positions from spec ({'/'.join(ax_letters)})"

    # "30 mm apart in X" with count → evenly spaced, centred, on that axis
    m = _APART_RE.search(text)
    if m and count > 1:
        spacing = float(m.group(1) or m.group(2))
        span_axis = _axis_in_text(text) or f"+{off[0]}"
        slot = off.index(_axis_letter(span_axis)) if _axis_letter(span_axis) in off else 0
        half = spacing * (count - 1) / 2.0
        pts = []
        for i in range(count):
            p = [0.0, 0.0]
            p[slot] = round(-half + i * spacing, 3)
            pts.append((p[0], p[1]))
        return pts, f"{count}x {spacing:g} mm apart on {span_axis}, centred"

    if _CENTERED_RE.search(text):
        return [(0.0, 0.0)] * count, "centred"
    return [], ""


def _pos_calls(pts: list[tuple[float, float]], axis: str, rot: str, axial: float = 0.0) -> str:
    """Locations(...) scope line(s) for resolved off-axis points; ``axial``
    sets the workplane coordinate along the bore axis (host-bound holes sit
    on the host's hi face so the cut runs lo-ward)."""
    letter = _axis_letter(axis) or "z"
    off = [a for a in "xyz" if a != letter]

    def xyz(p: tuple[float, float]) -> str:
        c = {off[0]: p[0], off[1]: p[1], letter: axial}
        return f"Pos({c['x']}, {c['y']}, {c['z']}){rot}"

    if len(pts) == 1:
        return f"Locations({xyz(pts[0])})"
    return "Locations(" + ", ".join(xyz(p) for p in pts) + ")"


# --------------------------------------------------------------------------- #
# per-type builders — return BoundFeature or None


@dataclass
class BoundFeature:
    name: str
    kind: str
    snippet: str  # full rendered block incl. comments
    resolved: bool = True


# cadlib helper names the binder can emit as bare calls — bindings_block
# scans snippets for these and emits the matching `from cadlib import …`.
CADLIB_HELPERS = frozenset(
    {
        "through_hole",
        "hole_grid",
        "oblong_slot",
        "slot_cutter",
        "keyhole_slot",
        "insert_pocket",
        "bearing_seat",
        "safe_fillet",
        "rib",
        "boss",
        "tube",
        "text_label",
        "strut",
    }
)

_STRUT_RE = re.compile(r"strut|brace|strebe|diagonal", re.I)
_NOT_STRUT_RE = re.compile(r"gusset|rib|rippe|knotenblech", re.I)


def _is_strut(f: Feature) -> bool:
    """Diagonal strut/brace — a straight bar between two points, not a
    filled triangular gusset (that is `rib`). A gusset/rib NAME or TYPE
    marks the filled triangle even when 'diagonal' appears nearby; the
    description may still call a brace a 'gusset' colloquially."""
    t = (f.type or "").lower().strip()
    if t in {"strut", "brace", "strebe"}:
        return True
    if t in _BINDERS:  # a declared hole/slot/rib/… type always wins
        return False
    name = f"{f.type or ''} {f.name or ''}".lower()
    if _NOT_STRUT_RE.search(name):
        return False
    return bool(_STRUT_RE.search(f"{name} {f.description or ''}"))


def _host_bound(
    spec: PartSpec, f: Feature, letter: str, depth: float | None
) -> tuple[Feature, float, float] | None:
    """Host-bounded through-cut target ``(host, lo, hi)`` — None for a
    full-depth bore.

    A bore through ONE declared plate of a multi-body part must not keep
    drilling down the whole axis line (a strut/rib/opposite wall on the
    same line gets pierced too — the bracket E2E cross-channels). Bound
    the cut to the host's span and sit the workplane on the host's hi
    face: through_hole(depth) cuts toward -local Z, i.e. lo-ward under
    every _FRAME_ROT frame. Applies only when the host resolves
    unambiguously and is thin (<50% of the envelope on the bore axis). A
    declared depth clearly SHALLOWER than the host span stays a blind hole;
    a depth matching the span is the plate thickness itself (intake writes
    depth=5 for a through-bore in a 5mm plate) — still bound."""
    host = host_body(spec, f)
    hr = host_span(spec, f, letter)
    if host is None or hr is None:
        return None
    lo, hi = hr
    span = hi - lo
    if (
        depth
        and "through" not in f"{f.type} {f.description}".lower()
        and abs(depth - span) > max(0.6, 0.15 * span)
    ):
        return None
    try:
        env = float(getattr(spec, _ENV_HALF[letter]).value)
    except (AttributeError, TypeError, ValueError):
        return None
    if env <= 0 or span >= 0.5 * env:
        return None
    return host, lo, hi


def _bind_hole(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    dia = _dim(dims, "diameter", "bore_diameter")
    if not dia and f.standard:
        m = re.search(r"\bM\s*(\d+(?:\.\d+)?)", f.standard)
        if m:
            try:
                from cadx.cad.catalog import nominal_hole_diameter

                kind = "tap" if re.search(r"tap|gewinde|thread", f.standard, re.I) else "clearance"
                dia = nominal_hole_diameter(int(float(m.group(1))), kind)
            except Exception:
                dia = None
    if not dia:
        return BoundFeature(
            f.name,
            "hole",
            f"# {f.name}: hole — diameter missing in spec, fill from the spec table",
            False,
        )
    axis = _norm_axis(f.axis) or _feature_axis_hint(f) or "+z"
    rot = _FRAME_ROT[axis]
    args = [f"{dia:g}"]
    depth = _dim(dims, "depth", "depth_partial", "depth_total")
    letter = _axis_letter(axis) or "z"
    bound = _host_bound(spec, f, letter, depth)
    bound_note = ""
    if bound is not None:
        host, lo, hi = bound
        args.append(f"depth={hi - lo + 0.5:g}")
        bound_note = (
            f"# through the host plate only ({host.name} "
            f"{letter}={lo:g}..{hi:g}) — other bodies on this line stay intact"
        )
    elif depth and "through" not in f"{f.type} {f.description}".lower():
        args.append(f"depth={depth:g}")
    csk = _dim(dims, "countersink_diameter", "csk_dia")
    if csk:
        args.append(f"csk_dia={csk:g}")
        if a := _dim(dims, "countersink_angle"):
            args.append(f"csk_angle={a:g}")
    cbore = _dim(dims, "counterbore_diameter", "cbore_dia")
    if cbore:
        args.append(f"cbore_dia={cbore:g}")
        if d := _dim(dims, "counterbore_depth"):
            args.append(f"cbore_depth={d:g}")
    count = max(1, int(f.count or 1))
    pts, note = _hole_positions(spec, f, count, axis)
    call = f"through_hole({', '.join(args)})"
    if pts:
        scope = _pos_calls(pts, axis, rot, axial=bound[2] if bound else 0.0)
    else:
        pos = (f.position or "").strip() or "stated location"
        scope = f"Locations(Pos(<{pos}>){rot})"
    body = (
        "with BuildPart() as _hb:\n"
        "    add(part)\n"
        f"    with {scope}:\n"
        f"        {call}\n"
        "    part = _hb.part"
    )
    rot_note = f" — local z rotated onto {axis}, the hole pierces along {axis}" if rot else ""
    std = f" ({f.standard})" if f.standard else ""
    snip = f"## {f.name} — {count}x hole Ø{dia:g}{std}, axis {axis}{rot_note}\n{body}"
    if bound_note:
        snip += f"\n{bound_note}"
    if note:
        snip += f"\n# {note}"
    return BoundFeature(f.name, "hole", snip)


def _slot_direction(f: Feature, normal: str) -> str:
    """Elongation axis of the slot — 'along Z'/'vertical'; defaults to +z for
    horizontal wall normals, +x for a +z normal (horizontal slot on a plate)."""
    text = f"{f.position or ''} {f.description or ''}".lower()
    ax = _axis_in_text(text)
    if ax and _axis_letter(ax) != _axis_letter(normal):
        return ax
    if any(k in text for k in ("vertical", "vertikal", "upright", "hochkant")):
        return "+z"
    return "+x" if _axis_letter(normal) == "z" else "+z"


def _bind_slot(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    w = _dim(dims, "slot_width", "width", "thickness")
    length = _dim(dims, "slot_length", "length")
    if not w or not length:
        return BoundFeature(
            f.name, "slot", f"# {f.name}: slot — width/length missing in spec", False
        )
    normal = _norm_axis(f.axis)
    if not normal:
        parent = host_body(spec, f)
        if parent is not None:
            ta = _thin_axis(parent)
            if ta:
                normal = f"+{ta}"
    direction = _slot_direction(f, normal) if normal else None
    desc = (f.description or "").lower()
    helper = "oblong_slot"
    if "keyhole" in desc or (f.type or "").lower() == "keyhole":
        helper = "keyhole_slot"
    if not normal:
        return BoundFeature(
            f.name,
            "slot",
            f"## {f.name} — slot {w:g}x{length:g}\n"
            f"with BuildPart() as _sb:\n"
            f"    add(part)\n"
            f"    with Locations(Pos(<wall centre>)):\n        "
            f'{helper}({w:g}, {length:g}, direction="{direction or "+z"}", normal="<wall normal>")\n'
            f"    part = _sb.part\n"
            "# normal = the axis the slot pierces THROUGH (the thin axis of the wall it sits in)",
            False,
        )
    call = f'{helper}({w:g}, {length:g}, direction="{direction}", normal="{normal}")'
    pos = (f.position or "").strip() or "wall centre"
    # "centred in <parent>" → the parent's own centre resolves the Pos
    place = f"Pos(<{pos}>)"
    parent = host_body(spec, f)
    if parent is not None and re.search(
        r"center|centr|mittig|zentriert", f"{f.position} {f.description}".lower()
    ):
        place = _pos_str(_feature_centre(spec, parent))
    perp = [a for a in "xyz" if a != _axis_letter(normal)]
    snip = (
        f"## {f.name} — slot {w:g}x{length:g}, through-axis {normal}, runs {direction}\n"
        f"with BuildPart() as _sb:\n"
        f"    add(part)\n"
        f"    with Locations({place}):\n"
        f"        {call}\n"
        f"    part = _sb.part\n"
        f"# place on the {normal}-normal wall; the {perp[0]}/{perp[1]} coords centre it"
    )
    return BoundFeature(f.name, "slot", snip)


def _bind_strut(spec: PartSpec, f: Feature) -> BoundFeature | None:
    """Straight diagonal bar between two declared endpoints via cadlib.strut —
    NOT a filled triangular gusset (the bracket E2E built one and the bore
    line then pierced it). Needs exactly two coordinate groups plus
    width/thickness dims."""
    dims = f.dims or {}
    width = _dim(dims, "width")
    thick = _dim(dims, "thickness")
    text = f"{f.position or ''} {f.description or ''}"
    pts = parse_coord_points(text)
    w_txt = f"{width:g}" if width else "<width>"
    t_txt = f"{thick:g}" if thick else "<thickness>"
    if len(pts) == 2 and width and thick:
        p0 = tuple(float(pts[0].get(a, 0.0)) for a in "xyz")
        p1 = tuple(float(pts[1].get(a, 0.0)) for a in "xyz")
        snip = (
            f"## {f.name} — straight strut between two points (NOT a filled gusset)\n"
            f"part = part + strut({p0}, {p1}, width={w_txt}, thickness={t_txt})"
        )
        return BoundFeature(f.name, "strut", snip)
    snip = (
        f"## {f.name} — straight strut between two points (NOT a filled gusset)\n"
        f"part = part + strut(<p0>, <p1>, width={w_txt}, thickness={t_txt})\n"
        "# needs exactly two endpoint coords (e.g. (x=-55, z=35) to (x=+15, z=95))"
        " and width/thickness dims — resolve the contact points from the spec"
    )
    return BoundFeature(f.name, "strut", snip, resolved=False)


def _bind_rib(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    t = _dim(dims, "thickness")
    if not t:
        return BoundFeature(f.name, "rib", f"# {f.name}: rib — thickness missing in spec", False)
    text = f"{f.position or ''} {f.description or ''}".lower()
    w2 = float(spec.width_y.value) / 2.0
    l2 = float(spec.length_x.value) / 2.0
    sink = 0.5
    # which envelope face do the ribs sit on? "outer Y sides" is the default;
    # "X ends"/"front/back" flip to X
    # explicit "+Y face"/"-X end" mentions pin down the side; "outer sides" /
    # "beidseitig" without a letter means both
    face = re.search(r"([+-])\s*([xy])\s*(?:\s*(?:face|end|side|seite))?\b", text)
    on_x = bool(face and face.group(2) == "x") or (
        bool(re.search(r"front|back|vorne|hinten|längsseite|\bx\b", text))
        and not re.search(r"\by\b|side|seite", text)
    )
    env = l2 if on_x else w2
    axis_letter = "x" if on_x else "y"
    free = "y" if on_x else "x"  # coordinate along the junction the rib bridges
    center = round(env - t / 2.0 - sink, 3)
    if face:
        signs = ("+",) if face.group(1) == "+" else ("-",)
    elif (f.count or 1) >= 2 or re.search(r"outer|both|beid", text):
        signs = ("-", "+")
    else:
        signs = ("+",)
    lines = []
    for s in signs:
        c = -center if s == "-" else center
        coords = {"x": "<junction x>", "y": "<junction y>", "z": "<base z>"}
        coords[axis_letter] = f"{c:g}"
        lines.append(
            f"part += Pos({coords['x']}, {coords['y']}, {coords['z']}) "
            f"* rib(<base>, <height>, {t:g})"
        )
    if (f.count or 1) > len(signs):
        lines.append(f"# count={f.count}: distribute the extra rib(s) along {free}")
    size = _dim(dims, "length", "height") or "≈ min(leg height, foot span)"
    face_desc = f"{signs[0]}{axis_letter}" if len(signs) == 1 else f"±{axis_letter}"
    snip = (
        f"## {f.name} — rib(s) {t:g} mm thick on the {face_desc} face(s)\n"
        + "\n".join(lines)
        + "\n# identical orientation on BOTH sides — rib() centres its thickness\n"
        "# on the Pos axis, so -Y gets NO extra rotation (Rot would flip it\n"
        "# under the part). Sink the placement ~0.5 mm into BOTH contact faces\n"
        "# (junction edge into the leg face, base into the foot top) — a rib\n"
        "# that only touches fuses non-manifold or stays detached."
        + f"\n# envelope rule: outer rib face ends at {axis_letter}=±{env:g} — "
        f"the centre offset {center:g} = {env:g} - {t:g}/2 - {sink} sinks the rib "
        f"0.5 mm into the side face so it fuses; base/height {size}"
    )
    return BoundFeature(f.name, "rib", snip)


def _bind_fillet(spec: PartSpec, f: Feature) -> BoundFeature | None:
    r = _dim(f.dims or {}, "fillet_radius", "radius")
    if not r:
        return BoundFeature(f.name, "fillet", f"# {f.name}: fillet — radius missing in spec", False)
    text = f"{f.position or ''} {f.description or ''}".lower()
    if "inner" in text or "innen" in text or "junction" in text:
        sel = "_fb.edges().filter_by(Axis.Y).sort_by(Axis.X)[0]"
        hint = "# the inner corner edge runs along Y at the leg/foot junction — pick the SHORTEST Y-edge"
    elif "rim" in text or "rand" in text or "outer" in text:
        sel = "_fb.edges() >> Axis.Z"
        hint = "# rim/top perimeter edges"
    else:
        sel = "<edges>"
        hint = f"# select the edge(s) described as: {f.position or 'see spec'}"
    snip = (
        f"## {f.name} — fillet R{r:g} (self-contained: works wherever you paste it;\n"
        f"## replace `part` if your body variable has a different name)\n"
        f"with BuildPart() as _fb:\n"
        f"    add(part)\n"
        f"    safe_fillet({sel}, {r:g})\n"
        f"    part = _fb.part\n"
        f"{hint}"
    )
    return BoundFeature(f.name, "fillet", snip)


def _bind_chamfer(spec: PartSpec, f: Feature) -> BoundFeature | None:
    d = _dim(f.dims or {}, "chamfer", "chamfer_distance", "distance")
    if not d:
        return BoundFeature(
            f.name, "chamfer", f"# {f.name}: chamfer — distance missing in spec", False
        )
    text = f"{f.position or ''} {f.description or ''}".lower()
    if "rim" in text or "outer" in text or "rand" in text:
        sel = "_cb.edges() >> Axis.Z"
    else:
        sel = "<edges>"
    snip = (
        f"## {f.name} — chamfer {d:g} mm (45°) (self-contained: works wherever you paste it;\n"
        f"## replace `part` if your body variable has a different name)\n"
        f"with BuildPart() as _cb:\n"
        f"    add(part)\n"
        f"    chamfer({sel}, length={d:g})\n"
        f"    part = _cb.part\n"
        f"# select the edge(s) described as: {f.position or 'see spec'}"
    )
    return BoundFeature(f.name, "chamfer", snip)


def _bind_boss(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    d = _dim(dims, "diameter", "bore_diameter")
    h = _dim(dims, "height", "length")
    if not d or not h:
        return BoundFeature(f.name, "boss", f"# {f.name}: boss — diameter/height missing", False)
    hole = _dim(dims, "hole_diameter", "inner_diameter", "bore_diameter")
    args = f"diameter={d:g}, height={h:g}" + (f", hole_d={hole:g}" if hole else "")
    pos = (f.position or "").strip() or "spec position"
    return BoundFeature(
        f.name,
        "boss",
        f"## {f.name} — boss Ø{d:g}x{h:g}\npart += Pos(<{pos}>) * boss({args})",
    )


def _bind_tube(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    od = _dim(dims, "diameter", "outer_diameter")
    idn = _dim(dims, "inner_diameter", "bore_diameter")
    length = _dim(dims, "length", "height")
    if not od or not idn or not length:
        return BoundFeature(
            f.name, "tube", f"# {f.name}: tube — outer/inner dia or length missing", False
        )
    axis = _norm_axis(f.axis) or _feature_axis_hint(f) or "+z"
    rot = _FRAME_ROT[axis] or ""
    pos = (f.position or "").strip() or "centre"
    return BoundFeature(
        f.name,
        "tube",
        f"## {f.name} — tube Ø{od:g}/Ø{idn:g}x{length:g}, axis {axis}\n"
        f"part += Pos(<{pos}>){rot} * tube({od:g}, {idn:g}, {length:g})",
    )


def _bind_insert(spec: PartSpec, f: Feature) -> BoundFeature | None:
    std = (f.standard or "").strip()
    if not std:
        return BoundFeature(
            f.name, "insert", f"# {f.name}: insert — no standard in spec (e.g. 'M4')", False
        )
    depth = _dim(f.dims or {}, "depth")
    dep = f", depth={depth:g}" if depth else ""
    pos = (f.position or "").strip() or "face"
    return BoundFeature(
        f.name,
        "insert",
        f"## {f.name} — heat-set insert {std}\n"
        f"with BuildPart() as _ib:\n"
        f"    add(part)\n"
        f"    with Locations(Pos(<{pos}>)):\n"
        f'        insert_pocket(HeatSetNut("{std}"){dep})\n'
        f"    part = _ib.part\n"
        "# verify the size designation against the bd_warehouse catalog",
    )


def _bind_bearing(spec: PartSpec, f: Feature) -> BoundFeature | None:
    std = (f.standard or "").strip()
    dia = _dim(f.dims or {}, "diameter", "bore_diameter")
    if not std and not dia:
        return BoundFeature(
            f.name, "bearing_seat", f"# {f.name}: bearing seat — no standard/dia", False
        )
    call = (
        f'bearing_seat(SingleRowDeepGrooveBallBearing("{std}"))'
        if std
        else f"bearing_seat(<bearing for Ø{dia:g} bore>)"
    )
    pos = (f.position or "").strip() or "face"
    return BoundFeature(
        f.name,
        "bearing_seat",
        f"## {f.name} — bearing seat {std or dia}\n"
        f"with BuildPart() as _bb:\n"
        f"    add(part)\n"
        f"    with Locations(Pos(<{pos}>)):\n        {call}\n"
        f"    part = _bb.part",
    )


def _bind_thread(spec: PartSpec, f: Feature) -> BoundFeature | None:
    size = f.thread_spec or (re.search(r"\bM\s*\d+(?:\.\d+)?", f.standard or "") or [None])[0]
    m = re.search(r"M\s*(\d+(?:\.\d+)?)", str(size or f.standard or ""))
    if not m:
        return BoundFeature(f.name, "thread", f"# {f.name}: thread — no M-size in spec", False)
    dia = _dim(f.dims or {}, "diameter")
    if not dia:
        try:
            from cadx.cad.catalog import nominal_hole_diameter

            dia = nominal_hole_diameter(int(float(m.group(1))), "tap")
        except Exception:
            dia = None
    pos = (f.position or "").strip() or "spec position"
    call = f"through_hole({dia:g})" if dia else "through_hole(<tap drill Ø>)"
    return BoundFeature(
        f.name,
        "thread",
        f"## {f.name} — tapped M{m.group(1)} hole\n"
        f"with BuildPart() as _tb:\n"
        f"    add(part)\n"
        f"    with Locations(Pos(<{pos}>)):\n        {call}\n"
        f"    part = _tb.part\n"
        f"# tap drill for M{m.group(1)}; mark thread_spec M{m.group(1)} for post-cut",
    )


def _bind_pocket(spec: PartSpec, f: Feature) -> BoundFeature | None:
    dims = f.dims or {}
    w = _dim(dims, "width", "slot_width")
    length = _dim(dims, "length")
    depth = _dim(dims, "depth", "depth_partial")
    if not w or not length or not depth:
        return BoundFeature(
            f.name, "pocket", f"# {f.name}: pocket — width/length/depth missing", False
        )
    pos = (f.position or "").strip() or "face centre"
    return BoundFeature(
        f.name,
        "pocket",
        f"## {f.name} — pocket {w:g}x{length:g}, {depth:g} deep\n"
        f"part -= Pos(<{pos}>) * Box({w:g}, {length:g}, {depth + 0.5:g}, "
        "align=(Align.CENTER, Align.CENTER, Align.MAX))",
    )


def _bind_text(spec: PartSpec, f: Feature) -> BoundFeature | None:
    txt = f.name or f.description or "label"
    size = _dim(f.dims or {}, "height", "size") or 6.0
    depth = _dim(f.dims or {}, "depth") or 0.4
    pos = (f.position or "").strip() or "face centre"
    return BoundFeature(
        f.name,
        "text",
        f'## {f.name} — embossed text\npart += Pos(<{pos}>) * text_label("{txt}", size={size:g}, depth={depth:g})',
    )


_BINDERS = {
    "hole": _bind_hole,
    "through_hole": _bind_hole,
    "through-hole": _bind_hole,
    "bore": _bind_hole,
    "countersink": _bind_hole,
    "counterbore": _bind_hole,
    "slot": _bind_slot,
    "keyhole": _bind_slot,
    "rib": _bind_rib,
    "gusset": _bind_rib,
    "fillet": _bind_fillet,
    "round": _bind_fillet,
    "chamfer": _bind_chamfer,
    "boss": _bind_boss,
    "tube": _bind_tube,
    "insert": _bind_insert,
    "heat_set": _bind_insert,
    "bearing_seat": _bind_bearing,
    "bearing": _bind_bearing,
    "thread": _bind_thread,
    "tap": _bind_thread,
    "pocket": _bind_pocket,
    "cutout": _bind_pocket,
    "text": _bind_text,
    "label": _bind_text,
}


def bind_features(spec: PartSpec | None) -> list[BoundFeature]:
    """Resolve spec features into ready-made cadlib snippets (deterministic).

    At most ``_MAX_BINDINGS`` bindings are returned — beyond that the prompt
    block would crowd out the reference docs and later features fall back to
    the model's own wording anyway.
    """
    if spec is None or not spec.features:
        return []
    out: list[BoundFeature] = []
    for f in spec.features:
        # a strut/brace binds before the type dispatch — intake calls it
        # "box"/"rib"/"gusset" but it is always the straight bar
        binder = _bind_strut if _is_strut(f) else _BINDERS.get((f.type or "").lower().strip())
        if binder is None:
            continue
        try:
            bf = binder(spec, f)
        except Exception:
            bf = None  # a binder bug must never break codegen — degrade silently
        if bf is not None:
            out.append(bf)
        if len(out) >= _MAX_BINDINGS:
            break
    return out
