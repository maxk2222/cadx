"""Host-body resolution: which declared solid a spec feature lives in.

Multi-body parts (brackets, housings) name the plate a hole/slot pierces in
free text ("2x Ø5.5 in the arm", "through the vertical leg"). These helpers
resolve that host feature deterministically so downstream consumers can
bound a cut to the host's extent instead of drilling the whole envelope:

- :func:`parse_ranges` reads axis spans out of position text
  ("x from -60 to -55", "x=-60..-55", "x: -60 bis -55",
  "x zwischen -60 und -55").
- :func:`host_body` matches preposition-words ("in the ARM", "through the
  LEG") against other features' name/description — solid-ish types and
  features whose position states ranges on >= 2 axes are preferred hosts.
- :func:`host_range` is the convenience pair: host body -> its declared
  span on one axis.
"""

from __future__ import annotations

import re

from cadx.llm.schemas import Feature, PartSpec

_NUM = r"([+-]?\d+(?:\.\d+)?)"

# axis range spellings — lo/hi are sorted, order in text is irrelevant
_RANGE_RES = (
    re.compile(rf"\b([xyz])\s+(?:from|von)\s+{_NUM}\s+(?:to|bis)\s+{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s*[=:]\s*{_NUM}\s*\.\.\s*{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s*[=:]\s*{_NUM}\s+bis\s+{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s+zwischen\s+{_NUM}\s+und\s+{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s+{_NUM}\s*\.\.\s*{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s+{_NUM}\s+(?:to|bis)\s+{_NUM}", re.I),
)


def parse_ranges(text: str) -> dict[str, tuple[float, float]]:
    """Axis spans declared in free text -> {letter: (lo, hi)} sorted lo<hi.

    Recognises "x from -60 to -55", "x=-60..-55", "x: -60 bis -55" and
    "x zwischen -60 und -55" (plus the "von..bis" German mix).
    """
    out: dict[str, tuple[float, float]] = {}
    for rx in _RANGE_RES:
        for m in rx.finditer(text or ""):
            a, b = float(m.group(2)), float(m.group(3))
            out[m.group(1).lower()] = (min(a, b), max(a, b))
    return out


# word right after in/im/through/durch ("in the ARM", "durch den Schenkel")
_IN_RE = re.compile(
    r"\b(?:in the|in a|in der|im|in|through the|through|durch(?: die| den| das)?)\s+(\w+)",
    re.I,
)

# feature types that can host (i.e. contain) another feature
_SOLID_TYPES = {
    "box",
    "body",
    "plate",
    "base",
    "leg",
    "foot",
    "wall",
    "solid",
    "prism",
    "block",
    "boss",
    "tube",
}

# spec authors name host plates loosely — a bore "in the vertical" means
# the leg, one "in the top" means the arm/deck plate
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"arm", "top", "oben", "oberer", "obere", "oberes"}),
    frozenset({"leg", "schenkel", "vertical", "senkrecht"}),
)


def _expand(words: list[str]) -> set[str]:
    """Each matched word plus its synonyms (arm<->top/oben, leg<->schenkel…)."""
    out = set(words)
    for w in words:
        for g in _SYNONYM_GROUPS:
            if w in g:
                out |= g
    return out


def host_body(spec: PartSpec, f: Feature) -> Feature | None:
    """The feature this feature lives IN ('slot centered in leg' -> the leg).

    Matches a word after in/through/durch in the feature's own
    position+description against other features' name + description,
    preferring solid-ish types over plain or range-only candidates. Only
    features with a solid type or a position declaring axis ranges on >= 2
    axes qualify as hosts at all.
    """
    hay = f"{f.position or ''} {f.description or ''}".lower()
    words = [w for w in _IN_RE.findall(hay) if len(w) > 2 and w != "the"]
    if not words:
        return None
    terms = _expand(words)
    solids: list[Feature] = []
    ranged: list[Feature] = []
    for g in spec.features or []:
        if g is f:
            continue
        ghay = f"{g.name or ''} {g.description or ''}".lower()
        if not any(w in ghay for w in terms):
            continue
        if (g.type or "").lower() in _SOLID_TYPES:
            solids.append(g)
        elif len(parse_ranges(g.position or "")) >= 2:
            ranged.append(g)
    return (solids or ranged or [None])[0]


def host_range(spec: PartSpec, f: Feature, axis: str) -> tuple[float, float] | None:
    """Declared span of the host body on ``axis`` ('x'|'+x' -> 'x') — None
    when the host cannot be resolved or its position names no range there."""
    host = host_body(spec, f)
    if host is None:
        return None
    letter = str(axis or "").strip().lower().lstrip("+-")
    return parse_ranges(host.position or "").get(letter)
