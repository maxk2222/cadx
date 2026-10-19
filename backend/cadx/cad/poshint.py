"""Shared parser for machine-readable position hints in spec free text.

Single source of truth for ``x=12``-style coordinate offsets,
``75mm from -x end``-style end/edge-relative offsets, ``centred in y``
mid-plane hints and ``+y``/``front`` face words — owned here so
``cad.verifyspec`` (conformance checks) and ``agent.anchors`` (spec-anchor
synthesis) read identical hints from the same spec text.

Pure text parsing — no geometry imports — so the executor sandbox and the
worker call the same code path.
"""

from __future__ import annotations

import re
from typing import Any

# machine-readable position hints: "x=12" offsets, "75mm from -x end"
# end-relative offsets, "centred in y" mid-plane hints, "+y"/"front" face words
_POS_COORD_RE = re.compile(r"\b([xyz])\s*=\s*([+-]?\d+(?:\.\d+)?)", re.I)
_POS_END_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm)?", re.I)
# "75mm from -x end", "20 and 75mm from the +y edge" — the sign is mandatory
# ("from x end" is ambiguous which end) and binds the bbox lo/hi side
_POS_END_RE = re.compile(
    r"(?<![=\w.\-])((?:\d+(?:\.\d+)?\s*(?:mm|cm)?\s*(?:(?:,|\band\b|&|/)\s*)?)+?)"
    r"\s*(?:(?:in|back|over)\s+)?from\s+(?:the\s+)?([+-])([xyz])"
    r"(?:[\s-]*(?:ends?|edges?|sides?|faces?))?\b",
    re.I,
)
# "between 20 and 30mm"/"20 to 30mm" are ranges, not point offsets — the
# whole hint stays unverifiable rather than checking a wrong single value
_POS_END_RANGE_RE = re.compile(r"(?:\bbetween|\bto)\s*$", re.I)
# "near the +x end", "close to -y end", "toward the +x end", "am +x Ende",
# "at the -x end" — end-proximity hints with NO numeric offset (a number
# makes it a _POS_END_RE offset instead). The sign is mandatory: "near the
# x end" is ambiguous which end.
_POS_NEAR_END_RE = re.compile(
    r"\b(?:near|close\s+to|towards?|at|am|an|nahe)\s+"
    r"(?:the|a|dem|der|das|den|des)?\s*"
    r"([+-])([xyz])[\s-]*(?:ends?|edges?|sides?|ende)\b",
    re.I,
)
# a number immediately before a near-end phrase ("5mm near the +x end") is
# an offset, not a proximity hint — leave it to _POS_END_RE / unverifiable
_POS_NEAR_NUM_RE = re.compile(r"\d\s*(?:mm|cm)?\s*$", re.I)
# "centred in y", "centered about the x axis", "mid y", "middle of z" -> 0.0
_POS_MID_RE = re.compile(
    r"\bcent(?:er|re)e?d?\s+(?:in|on|about|across|along)\s+(?:the\s+)?"
    r"([xyz])(?:[\s-]*axis)?\b"
    r"|\bmid(?:dle)?[\s-]*(?:of\s+(?:the\s+)?)?([xyz])(?:[\s-]*axis)?\b",
    re.I,
)
_POS_FACE_TOKEN_RE = re.compile(r"(?<![\w])([+-][xyz])(?![\w])")
_POS_FACE_WORDS = {
    "top": "+z",
    "upper": "+z",
    "deck": "+z",
    "bottom": "-z",
    "lower": "-z",
    "underside": "-z",
    "front": "+y",
    "back": "-y",
    "rear": "-y",
    "left": "-x",
    "right": "+x",
}


def parse_position_hints(
    text: str,
) -> tuple[dict[str, float], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse machine-readable position hints out of free text.

    Returns ``(coords, faces, ends, nears)``:
    - ``coords``: ``x=12``-style absolute offsets plus ``centred in y``
      mid-plane hints folded in as ``0.0`` — ``{"x": 12.0, "y": 0.0}``.
    - ``faces``: sorted signed-axis face hints (``"+y"`` from ``+y``/``front``).
      Both signs of one axis ('front and back') cancel — not a side constraint.
    - ``ends``: ``75mm from -x end``-style end/edge offsets as dicts
      ``{"axis": "x", "end": "-x", "offset_mm": 75.0}`` — the consumer
      resolves them against a bbox/envelope via :func:`resolve_end_hint`.
    - ``nears``: ``near the +x end``-style proximity hints as dicts
      ``{"axis": "x", "end": "+x"}`` — no offset; the consumer maps them
      to an end-band region check or an estimated inset position.
    Parsed spans are masked so a token is never read as two hint kinds.
    """
    low = text.lower()
    masked = list(low)

    def _mask(span: tuple[int, int]) -> None:
        for p in range(*span):
            masked[p] = " "

    coords: dict[str, float] = {}
    for m in _POS_COORD_RE.finditer(low):
        coords[m.group(1).lower()] = float(m.group(2))
        _mask(m.span())
    ends: list[dict[str, Any]] = []
    work = "".join(masked)
    for m in _POS_END_RE.finditer(work):
        if _POS_END_RANGE_RE.search(work[: m.start(1)]):
            continue  # unmasked on purpose — the ±axis still reads as a face hint
        gtext = m.group(1)
        nums = list(_POS_END_NUM_RE.finditer(gtext))
        for n in nums:
            # bare number in a list: trust it only when conjoined by
            # "and"/"&"/"/" — "count 2, 30mm from -x" drops the 2
            if (
                not n.group(2)
                and len(nums) > 1
                and not re.search(r"(?:\band\b|&|/)\s*$", gtext[: n.start()])
                and not re.match(r"\s*(?:\band\b|&|/)", gtext[n.end() :])
            ):
                continue
            unit = (n.group(2) or "mm").lower()
            ax = m.group(3).lower()
            ends.append(
                {
                    "axis": ax,
                    "end": f"{m.group(2)}{ax}",
                    "offset_mm": float(n.group(1)) * (10.0 if unit == "cm" else 1.0),
                }
            )
        _mask(m.span())
    nears: list[dict[str, Any]] = []
    work = "".join(masked)
    for m in _POS_NEAR_END_RE.finditer(work):
        if _POS_NEAR_NUM_RE.search(work[: m.start()]):
            continue  # unmasked on purpose — the ±axis still reads as a face hint
        ax = m.group(2).lower()
        nears.append({"axis": ax, "end": f"{m.group(1)}{ax}"})
        _mask(m.span())
    for m in _POS_MID_RE.finditer("".join(masked)):
        ax = next(g for g in m.groups() if g).lower()
        coords.setdefault(ax, 0.0)
        _mask(m.span())
    work = "".join(masked)
    faces = {m.group(1) for m in _POS_FACE_TOKEN_RE.finditer(work)}
    for word, ax in _POS_FACE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", work):
            faces.add(ax)
    for letter in "xyz":
        if f"+{letter}" in faces and f"-{letter}" in faces:
            faces.discard(f"+{letter}")
            faces.discard(f"-{letter}")
    return coords, sorted(faces), ends, nears


# " bis ", " to ", "..", en/em dash as the WHOLE text between two values
# of the SAME axis mean a span declaration, not a second point ("x=-60 bis
# x=-55"). Wordy connectors ("on the leg to (x=..)", "and") still separate.
_RANGE_LINK_RE = re.compile(r"^\s*,?\s*(?:bis|to|through|thru|\.\.|\u2013|\u2014)\s*$", re.I)


def parse_coord_points(text: str) -> list[dict[str, float]]:
    """'x=-25, y=0 and x=+45, y=0' -> [{'x': -25, 'y': 0}, {'x': 45, 'y': 0}].

    Multi-point counterpart of the ``coords`` dict in
    :func:`parse_position_hints`: a spec feature often declares several
    instances on one line, and the flat dict silently keeps only the LAST
    value per axis. Every ``x=12``-style hit of ``_POS_COORD_RE`` is grouped
    in text order — a new group starts as soon as an axis letter repeats
    inside the current group, UNLESS the text between the two values is a
    range connector ("x=-60 bis x=-55") — a span is one declaration, not
    two points; the group's value becomes the span's midpoint (the precise
    lo/hi stay available via ``parse_ranges``). Empty groups are dropped;
    ``[]`` when the text carries no coordinate at all.
    """
    groups: list[dict[str, float]] = []
    cur: dict[str, float] = {}
    prev_end = 0
    for m in _POS_COORD_RE.finditer(text or ""):
        ax = m.group(1).lower()
        if ax in cur:
            between = text[prev_end : m.start()]
            if _RANGE_LINK_RE.search(between):
                cur[ax] = (cur[ax] + float(m.group(2))) / 2.0
            else:
                groups.append(cur)
                cur = {}
        if ax not in cur:
            cur[ax] = float(m.group(2))
        prev_end = m.end()
    if cur:
        groups.append(cur)
    return groups


def resolve_end_hint(axis: str, end: str, dist_mm: float, lo: float, hi: float) -> float:
    """Resolve an end-relative hint to an absolute coordinate on ``[lo, hi]``.

    ``-x``-style ends measure the offset up from ``lo`` (``lo + d``),
    ``+x``-style ends measure it down from ``hi`` (``hi - d``). ``axis`` is
    the axis letter (informational — the sign on ``end`` decides the side).
    """
    return lo + dist_mm if str(end).strip().startswith("-") else hi - dist_mm
