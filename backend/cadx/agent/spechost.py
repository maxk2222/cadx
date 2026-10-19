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
    re.compile(
        rf"(?:from|von)\s+([xyz])\s*[=:]?\s*{_NUM}\s+(?:to|bis)\s+(?:[xyz]\s*[=:]?\s*)?{_NUM}",
        re.I,
    ),
    re.compile(rf"\b([xyz])\s*[=:]\s*{_NUM}\s*\.\.\s*{_NUM}", re.I),
    re.compile(rf"\b([xyz])\s*[=:]\s*{_NUM}\s+bis\s+(?:[xyz]\s*[=:]?\s*)?{_NUM}", re.I),
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

# tokens that never identify a host when they appear in a feature name
# ("Schenkel-Bohrung" -> "schenkel" matches the plate, "bohrung" is noise)
_NAME_STOP = {
    "bohrung",
    "bore",
    "hole",
    "durchgangsbohrung",
    "gewinde",
    "thread",
    "senkung",
    "slot",
    "nut",
    "tasche",
    "pocket",
    "senkbohrung",
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
    axes qualify as hosts at all. When no preposition is present the
    feature's own NAME is matched too ("Schenkel-Bohrung" -> the feature
    named "senkrechter Schenkel") — intake splits multi-hole features into
    name-tagged singles that carry no "im <host>" text.
    """
    hay = f"{f.position or ''} {f.description or ''}".lower()
    words = [w for w in _IN_RE.findall(hay) if len(w) > 2 and w != "the"]
    solids: list[Feature] = []
    ranged: list[Feature] = []
    if words:
        terms = _expand(words)
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
        if solids or ranged:
            return (solids or ranged)[0]
    fname = f.name or ""
    if not fname:
        return None
    ftokens = {t for t in re.findall(r"[a-zäöü]+", fname.lower()) if len(t) > 2}
    for g in spec.features or []:
        if g is f:
            continue
        gtokens = {
            t
            for t in re.findall(r"[a-zäöü]+", (g.name or "").lower())
            if len(t) > 2 and t not in _NAME_STOP
        }
        if not gtokens or not (gtokens & ftokens):
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
    return parse_ranges(f"{host.position or ''} {host.description or ''}").get(letter)


_ENV_ATTR = {"x": "length_x", "y": "width_y", "z": "height_z"}


def _env_bounds(spec: PartSpec, letter: str) -> tuple[float, float] | None:
    """Envelope span on one axis (z rests on XY -> [0, h]; x/y centred)."""
    try:
        v = float(getattr(spec, _ENV_ATTR[letter]).value)
    except (AttributeError, TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return (0.0, v) if letter == "z" else (-v / 2, v / 2)


_END_RE = {
    "-": re.compile(r"-\s*([xyz])\s*-?(?:end|ende|seite|side)\b", re.I),
    "+": re.compile(r"\+\s*([xyz])\s*-?(?:end|ende|seite|side)\b", re.I),
}
_Z_END_WORDS = {
    "-": re.compile(r"\b(?:unterkante|bottom|unten|lower)\b", re.I),
    "+": re.compile(r"\b(?:oberkante|top|oben|upper)\b", re.I),
}


def _host_end_span(host: Feature, spec: PartSpec, letter: str) -> tuple[float, float] | None:
    """Derive an end-anchored host's span from 'am -X-Ende'/'an der
    Oberkante' + its declared thickness.

    Intake often writes a plate as "<pos>-end + thickness" instead of an
    explicit axis range — "senkrechter Schenkel am -X-Ende, thickness 5"
    means x ∈ [env_lo, env_lo+5], "Arm an der Oberkante, thickness 5" means
    z ∈ [env_hi-5, env_hi]. Ambiguous (both ends named) or thickness-less
    hosts resolve to None — a wrong bound is worse than a full-depth cut.
    """
    dims = host.dims or {}
    thick = dims.get("thickness") or dims.get("thickness_mm") or dims.get("wandstaerke")
    try:
        t = float(thick)
    except (TypeError, ValueError):
        return None
    if t <= 0:
        return None
    hay = f"{host.position or ''} {host.description or ''}".lower()
    m_neg, m_pos = _END_RE["-"].search(hay), _END_RE["+"].search(hay)
    neg = m_neg is not None and m_neg.group(1) == letter
    pos = m_pos is not None and m_pos.group(1) == letter
    if letter == "z":
        neg = neg or bool(_Z_END_WORDS["-"].search(hay))
        pos = pos or bool(_Z_END_WORDS["+"].search(hay))
    if pos == neg:
        return None
    env = _env_bounds(spec, letter)
    if env is None:
        return None
    lo, hi = env
    return (hi - t, hi) if pos else (lo, lo + t)


def host_span(spec: PartSpec, f: Feature, axis: str) -> tuple[float, float] | None:
    """Host's span on ``axis`` — the declared range when present, else the
    end-anchored span derived from 'am <±axis>-Ende'/'an der Oberkante' plus
    the host's declared thickness."""
    host = host_body(spec, f)
    if host is None:
        return None
    letter = str(axis or "").strip().lower().lstrip("+-")
    declared = parse_ranges(f"{host.position or ''} {host.description or ''}").get(letter)
    if declared is not None:
        return declared
    return _host_end_span(host, spec, letter)
