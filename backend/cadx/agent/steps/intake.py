"""INTAKE — extract a PartSpec from images + text (vision model)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from cadx.agent.context import (
    images_block,
    parent_block,
    references_block,
    system_prompt,
    user_block,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import IMAGE_INPUT_KINDS, AgentCtx
from cadx.llm.schemas import Feature, LoadCase, OptimizeSpec, PartSpec
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "intake"


class _LoadCasesOut(BaseModel):
    load_cases: list[LoadCase] = Field(default_factory=list)
    optimize: OptimizeSpec | None = None


def sanitize_load_cases(spec: PartSpec) -> None:
    """Drop degenerate cases (no supports and no loads) a weak model emits as
    `{"name":"","fixed":[],"loads":[]}` — truthy lists of those would skip the
    extraction fallback and produce unsolvable FEM runs. Also clamps bogus
    negative case weights back to neutral."""
    kept: list[LoadCase] = []
    for c in spec.load_cases:
        if not c.fixed and not c.loads:
            continue
        if c.weight <= 0:
            c.weight = 1.0
        kept.append(c)
    spec.load_cases = kept


_MATERIAL_RE = re.compile(
    r"\b(petg|pet-g|pla|abs|asa|nylon|pa6|pa12|\bpa\b|pc|resin|"
    r"aluminium|aluminum|alu|steel|stahl|s235)\b",
    re.IGNORECASE,
)


def sanitize_material(spec: PartSpec, prompt: str) -> None:
    """Fill `spec.material` when the prompt names a material the model dropped.

    FEM/topopt fall back to the default (PLA) on an empty field — a declared
    "PETG" or "aluminium" then silently verifies against wrong E/yield. Scans
    the prompt and the generated part name for the material table's known
    tokens; leaves an explicitly set value untouched."""
    if (spec.material or "").strip():
        return
    for m in _MATERIAL_RE.finditer(f"{prompt or ''} {spec.name or ''}"):
        token = m.group(1).lower()
        try:
            from cadx.cad.fem.materials import resolve_material

            # keep the literal token (alias-aware) — display names like
            # "PA (nylon)" do not re-resolve downstream
            resolve_material(token)
            spec.material = token
            return
        except Exception:
            continue


_MM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\b", re.IGNORECASE)
_DIMSET_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*(\d+(?:[.,]\d+)?)\s*mm",
    re.IGNORECASE,
)


def _prompt_mm_values(prompt: str) -> list[float]:
    """Explicit millimetre values mentioned in the user prompt, largest first."""
    vals: set[float] = set()
    for m in _MM_RE.finditer(prompt):
        vals.add(float(m.group(1).replace(",", ".")))
    for m in _DIMSET_RE.finditer(prompt):
        vals.update(float(g.replace(",", ".")) for g in m.groups())
    return sorted(vals, reverse=True)


def sanitize_dims(spec: PartSpec, prompt: str) -> list[str]:
    """Recover an overall dimension the model dropped or misread.

    Weak models routinely lose an explicitly stated size ("approx 130 mm
    long") or substitute a feature size — downstream, scale_to_spec then fits
    the generated mesh to the wrong magnitude (a 130 mm grip becomes a 40 mm
    nugget). When the largest explicit mm value in the prompt exceeds every
    stated dim by a wide margin and length_x is not itself stated, re-inject
    it as the stated length."""
    notes: list[str] = []
    vals = _prompt_mm_values(prompt)
    if not vals:
        return notes
    lx = spec.length_x
    if lx.source == "stated" and lx.value > 0:
        return notes  # the model stated an overall length — trust it
    stated_max = max(
        (
            float(d.value)
            for d in (spec.length_x, spec.width_y, spec.height_z)
            if d is not None and d.source == "stated" and d.value > 0
        ),
        default=0.0,
    )
    if stated_max >= vals[0] * 0.5:
        return notes  # stated dims already cover the part's overall size
    notes.append(
        f"prompt states ~{vals[0]:g} mm but intake had length_x="
        f"{lx.value:g} {lx.source} — re-stated"
    )
    spec.length_x = lx.model_copy(update={"value": vals[0], "source": "stated"})
    return notes


_AXIS_COORD_RE = re.compile(r"\b([xyz])\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_AXIS_RANGE_RE = re.compile(
    r"\b([xyz])\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(?:bis|to|\u2013|\u2014|\.\.)\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_APEX_RE = re.compile(
    r"\b(?:scheitel|apex|peak|oberkante|top)\s*(?:bei|at)?\s*([xyz])?\s*=?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _feature_axis_extent(spec: PartSpec, axis: str) -> float:
    """Largest |coordinate| the features claim along `axis` — from position
    ranges ("x=-60 bis -30"), single coords and apex notes ("Scheitel z=30").
    For z (0-based frame) the extent IS the max coordinate; for x/y the
    caller compares against half the declared span."""
    extent = 0.0
    for f in spec.features:
        pos = str(f.position or "")
        for m in _AXIS_RANGE_RE.finditer(pos):
            if m.group(1).lower() == axis:
                extent = max(extent, abs(float(m.group(2))), abs(float(m.group(3))))
        for m in _AXIS_COORD_RE.finditer(pos):
            if m.group(1).lower() == axis:
                extent = max(extent, abs(float(m.group(2))))
        m = _APEX_RE.search(pos)
        if m and (m.group(1) or "z").lower() == axis:
            extent = max(extent, float(m.group(2)))
    return extent


def sanitize_dim_echo(spec: PartSpec, prompt: str) -> list[str]:
    """Recover a declared bbox dim that echoes another axis's value while the
    spec's own feature positions contradict it (observed: '120 x 40 x 30'
    mis-parsed as height_z=120 duplicating length_x, while every feature
    said z<=30). Only fires on the echo signature + a large contradiction —
    features legitimately do not always span the full envelope, so a mere
    mismatch is not enough. The replacement must appear in the prompt text.
    """
    notes: list[str] = []
    dims = {"x": spec.length_x, "y": spec.width_y, "z": spec.height_z}
    stated = {
        ax: float(d.value)
        for ax, d in dims.items()
        if d is not None and d.source == "stated" and float(d.value or 0) > 0
    }
    if len(stated) < 2:
        return notes
    prompt_vals = set(_prompt_mm_values(prompt or ""))
    for ax, v in stated.items():
        echoes = any(v == w for other, w in stated.items() if other != ax)
        if not echoes:
            continue
        extent = _feature_axis_extent(spec, ax)
        if not extent:
            continue
        # z positions are 0-based extents; x/y are centred half-extents
        expected = extent if ax == "z" else extent * 2.0
        if v <= expected * 1.5:
            continue
        # prompt values no stated dim claims — the strongest ground truth for
        # what the echoing dim was meant to be (e.g. "120 x 40 x 90" parsed
        # as z=120: 90 is the orphaned truth)
        unmatched = [p for p in prompt_vals if not any(abs(p - w) < 1.0 for w in stated.values())]
        if unmatched:
            new_v = min(unmatched, key=lambda p: abs(p - expected))
        elif not prompt_vals or any(abs(expected - p) < 1.0 for p in prompt_vals):
            new_v = expected  # feature extent has footing or nothing to check against
        else:
            continue  # prompt states dims and the smaller extent is not among them
        notes.append(
            f"{ax} dim {v:g} duplicates another axis but features only reach "
            f"{expected:g} — re-stated {ax}={new_v:g}"
        )
        d = dims[ax]
        setattr(
            spec,
            {"x": "length_x", "y": "width_y", "z": "height_z"}[ax],
            d.model_copy(update={"value": new_v}),
        )
    return notes


def normalize_feature_positions(spec: PartSpec) -> list[str]:
    """Rewrite model-computed coords in ``feature.position`` when they
    contradict the stated end offset — intake likes to append its own
    arithmetic ("75 mm from -x end (x=20, y=0)") and gets it wrong
    (75-45=30, not 20). The end hint carries the user's stated semantics,
    so the coord token is rewritten to the end-resolved value, leaving one
    consistent position for codegen/verify/anchors to read. Multi-offset
    hints ("20 and 75mm from -x end") are left untouched — a single coord
    cannot alias two positions. Returns change notes."""
    from cadx.cad.poshint import parse_position_hints, resolve_end_hint

    notes: list[str] = []
    try:
        bounds = {
            "x": (-float(spec.length_x.value) / 2, float(spec.length_x.value) / 2),
            "y": (-float(spec.width_y.value) / 2, float(spec.width_y.value) / 2),
        }
    except (AttributeError, TypeError, ValueError):
        return notes  # no stated envelope — end hints unresolvable
    for f in spec.features:
        pos = str(f.position or "").strip()
        if not pos:
            continue
        coords, _faces, ends, _nears = parse_position_hints(pos)
        if not coords or not ends:
            continue
        new = pos
        for m in _AXIS_COORD_RE.finditer(pos):
            ax = m.group(1).lower()
            if ax not in bounds:
                continue
            end_vals = {
                resolve_end_hint(ax, str(e["end"]), float(e["offset_mm"]), *bounds[ax])
                for e in ends
                if str(e["axis"]) == ax
            }
            if len(end_vals) != 1:
                continue  # multi-offset or none — no single truth to rewrite to
            v = next(iter(end_vals))
            if abs(float(m.group(2)) - v) <= 1.0:
                continue  # coord agrees with the end offset — consistent already
            new = new.replace(m.group(0), f"{ax}={v:g}", 1)
            notes.append(
                f"position of '{f.name or f.type}' rewritten — {m.group(0)} "
                f"contradicted the stated end offset (now {ax}={v:g})"
            )
        if new != pos:
            f.position = new
    return notes


# --------------------------------------------------------------------------- #
# degenerate-spec gate                                                          #
# --------------------------------------------------------------------------- #

_TOOL_NAMES = {
    "search_catalog",
    "lookup_part",
    "cadlib_lookup",
    "search_library",
    "library_part",
    "attach_library_part",
    "view_image",
    "generate_concept_image",
}
_FEATURE_WORDS = re.compile(
    r"\b(loch|löcher|loecher|bohrung|bohrungen|hole|holes|bore|bores|langloch|"
    r"slot|slots|nut|groove|tasche|pocket|rippe|rippen|rib|ribs|steg|web|"
    r"radius|radien|fillet|verrund|verrundung|fase|chamfer|senk|senkung|"
    r"countersunk|countersink|counter|gewinde|thread|insert|boss|zapfen|"
    r"ausschnitt|cutout|aussparung|recess)\w*",
    re.I,
)


_HOST_TYPES = ("pad", "boss", "tab", "lug", "plate", "mount", "flange", "base")
# quantifiers that promise a feature PER host/side — "je eine Bohrung",
# "one hole in each lug", "an beiden Enden" — making a missing mirror-side
# feature a real spec gap instead of an asymmetric design choice
_PER_HOST_QUANT_RE = re.compile(
    r"\b(je|jeweils|jeder|jede|each|per|both|beiden|beidseitig|either)\w*", re.I
)
# hole words as SUBSTRINGS — German compounds ("Durchgangsbohrung") break
# \b-based matching, and a false positive only costs one cheap re-ask gate
_HOLE_WORD_RE = re.compile(r"loch|bohr|hole|bore|gewinde|thread", re.I)


# P1 feature-coverage gate: every feature-word CLASS the prompt names must
# map to >= 1 spec feature whose name/type/description carries a class
# keyword. The empty-features gap only fires when the model dropped
# EVERYTHING — a spec that kept plates+holes but silently lost the named
# slot/rib/fillet still reached codegen (observed: "Langloch … verrunden …
# Verstärkungsrippen" -> spec with foot, leg and 2 holes only).
_FEATURE_CLASSES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    # (class label, prompt-word triggers, required keywords in feature text)
    (
        "hole",
        frozenset({"hole", "bore", "loch", "bohrung", "durchgang"}),
        frozenset({"hole", "bore", "loch", "bohrung", "durchgang"}),
    ),
    (
        "slot",
        frozenset({"slot", "langloch", "nut"}),
        frozenset({"slot", "langloch", "nut"}),
    ),
    (
        "rib",
        frozenset({"rib", "rippe", "ribs", "rippen", "steg", "web", "gusset"}),
        frozenset({"rib", "rippe", "gusset", "steg", "web"}),
    ),
    (
        "fillet",
        frozenset({"fillet", "radius", "radien", "verrund"}),
        frozenset({"fillet", "radius", "rund", "verrund"}),
    ),
    (
        "chamfer",
        frozenset({"chamfer", "fase"}),
        frozenset({"chamfer", "fase"}),
    ),
    (
        "pocket",
        frozenset({"pocket", "tasche", "aussparung", "recess"}),
        frozenset({"pocket", "tasche", "ausspar", "recess"}),
    ),
    (
        "thread",
        frozenset({"thread", "gewinde"}),
        frozenset({"thread", "gewinde"}),
    ),
    (
        "boss",
        frozenset({"boss", "zapfen"}),
        frozenset({"boss", "zapfen"}),
    ),
    (
        "countersink",
        frozenset({"countersink", "senk", "countersunk", "counterbore"}),
        frozenset({"counter", "senk"}),
    ),
)

# bare-substring triggers for the raw-prompt scan — catches German compounds
# the boundary-anchored _FEATURE_WORDS structurally misses
# ("Verstärkungsrippen" glues the noun inside). Only unambiguous stems are
# scanned bare: "senk" would false-trip on "senkrecht", "fase" on "Faser",
# "nut" on "Minuten"-like words — those stay word-matched only.
_CLASS_TEXT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "hole": ("bohrung", "durchgang", "loch", "hole", "bore"),
    "slot": ("langloch", "slot"),
    "rib": ("rippen", "rippe", "gusset", "ribs", "steg"),
    "fillet": ("verrund", "fillet", "radius", "radien"),
    "chamfer": ("chamfer",),
    "pocket": ("aussparung", "tasche", "pocket", "recess"),
    "thread": ("gewinde", "thread"),
    "boss": ("zapfen", "boss"),
    "countersink": ("countersink", "counterbore", "countersunk", "senkbohrung", "senkung"),
}

_UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "Ä": "a", "Ö": "o", "Ü": "u", "ß": "ss"})

# _FEATURE_WORDS swallows whole words on stem matches ("senk" + \w* eats
# "senkrechter") — these inflected false friends would fire phantom
# classes (a "senkrechter Schenkel" is the vertical leg, no countersink;
# "Nutzen"/"nützlich" is no Nut, "Faser" no Fase)
_WORD_FALSE_FRIENDS = ("senkrecht", "faser", "nutz")


def _fold_feature_text(text: str) -> str:
    """Fold umlauts and their digraph spellings so "Löcher"/"Loecher" hit
    the same class triggers as "Loch"."""
    t = text.lower().translate(_UMLAUTS)
    return t.replace("ae", "a").replace("oe", "o").replace("ue", "u")


def _prompt_feature_classes(prompt: str, words: list[str]) -> dict[str, str]:
    """Feature classes the prompt names -> {class: evidence word}.

    `words` are the _FEATURE_WORDS matches — each joins a class when it
    starts with a trigger (inflection tails ride along) or is one inside a
    longer trigger ("counter" inside "countersink"). The raw-text scan
    additionally catches compounds the regex cannot see; its triggers are
    the long, unambiguous stems only. The scan claims longest-first:
    a shorter stem inside a claimed trigger ("loch" inside "langloch",
    "bohrung" inside "senkbohrung") belongs to the more specific class —
    a Langloch is a slot, not an extra hole."""
    found: dict[str, str] = {}
    for w in words:
        fw = _fold_feature_text(w)
        if any(fw.startswith(bad) for bad in _WORD_FALSE_FRIENDS):
            continue
        for cls, triggers, _required in _FEATURE_CLASSES:
            if cls in found:
                continue
            if any(fw.startswith(t) or (len(fw) >= 5 and fw in t) for t in triggers):
                found[cls] = w
    hay = _fold_feature_text(prompt or "")
    trig_cls = [
        (t, cls) for cls, _trig, _req in _FEATURE_CLASSES for t in _CLASS_TEXT_TRIGGERS[cls]
    ]
    for t, cls in sorted(trig_cls, key=lambda p: -len(p[0])):
        if t not in hay:
            continue
        if cls not in found:
            found[cls] = t
        # the trigger's text is claimed either way — a shorter stem inside
        # it must not fire a second class afterwards
        hay = hay.replace(t, " ")
    return found


def _feature_class_gaps(spec: PartSpec, words: list[str], prompt: str) -> list[str]:
    """Prompt-named feature classes with NO matching spec feature (P1).

    Conservative: a class only reports when it is completely absent — one
    feature whose name/type/description carries any class keyword covers
    it (counts and variants are not audited). A fired gap re-triggers the
    extract_features re-ask like the empty-features case."""
    if not spec.features:
        return []
    present = _prompt_feature_classes(prompt, words)
    if not present:
        return []
    have = [
        _fold_feature_text(f"{f.name or ''} {f.type or ''} {f.description or ''}")
        for f in spec.features
    ]
    gaps: list[str] = []
    for cls, _trig, required in _FEATURE_CLASSES:
        word = present.get(cls)
        if word is None:
            continue
        if any(any(k in h for k in required) for h in have):
            continue
        gaps.append(f"prompt names {cls} ({word}) but spec.features has none")
    return gaps


def _axis_side(text: str, axis: str) -> int | None:
    """Sign (-1/+1) a feature position claims on `axis`, from absolute
    coords, ranges or face hints. None = unresolved."""
    from cadx.agent.spechost import parse_ranges
    from cadx.cad.poshint import parse_position_hints

    coords, faces, _ends, _nears = parse_position_hints(text)
    v = coords.get(axis)
    if v is None:
        rng = parse_ranges(text).get(axis)
        if rng is not None:
            v = (rng[0] + rng[1]) / 2.0
    if v is None:
        for fc in faces:
            if len(fc) == 2 and fc[1] == axis:
                v = 1.0 if fc[0] == "+" else -1.0
                break
    if v is None or abs(v) < 1e-6:
        return None
    return 1 if v > 0 else -1


def _mirror_host_gaps(spec: PartSpec) -> list[str]:
    """Hole declared for only one side of a mirrored host pair (observed:
    'je eine Durchgangsbohrung in den Laschen' -> intake emitted pads at
    x=±45 but only the -x through-hole). Host = pad/boss/tab/lug-type
    feature; binding = sign of the resolved axis position."""
    from cadx.agent.anchors import _hole_like

    gaps: list[str] = []
    for axis in ("x", "y"):
        host_sides: set[int] = set()
        hole_sides: set[int] = set()
        unresolved_holes = 0
        for f in spec.features:
            text = f"{f.position or ''} {f.description or ''}".strip()
            if not text:
                continue
            side = _axis_side(text, axis)
            if _hole_like(f):
                if side is None:
                    unresolved_holes += 1
                else:
                    hole_sides.add(side)
            elif str(f.type or "").lower() in _HOST_TYPES and side is not None:
                host_sides.add(side)
        if host_sides == {-1, 1} and hole_sides and len(hole_sides) < len(host_sides):
            present = f"+{axis}" if 1 in hole_sides else f"-{axis}"
            missing = f"-{axis}" if 1 in hole_sides else f"+{axis}"
            gaps.append(
                f"hole-like features resolve to the {present} side only while "
                f"host features span both ±{axis} — the {missing} hole appears missing"
            )
    return gaps


def spec_gaps(spec: PartSpec, prompt: str) -> list[str]:
    """Deterministic plausibility check. Returns reasons (empty = ok)."""
    gaps: list[str] = []
    if spec.name.strip().lower() in _TOOL_NAMES or "catalog search" in spec.summary.lower():
        gaps.append("spec name/summary is a tool echo")
    words = sorted({m.group(0).lower() for m in _FEATURE_WORDS.finditer(prompt or "")})
    if words and not spec.features:
        gaps.append(f"prompt names features ({', '.join(words[:8])}) but spec.features is empty")
    elif spec.features:
        # per-class coverage: a named class with NO matching feature is a
        # silent drop, not a design choice (P1)
        gaps.extend(_feature_class_gaps(spec, words, prompt or ""))
    if (
        spec.features
        and _HOLE_WORD_RE.search(prompt or "")
        and _PER_HOST_QUANT_RE.search(prompt or "")
    ):
        gaps.extend(_mirror_host_gaps(spec))
    return gaps


class _FeaturesOut(BaseModel):
    """Focused re-ask payload: every feature the prompt names, plus identity."""

    name: str = ""
    summary: str = ""
    features: list[Feature] = Field(default_factory=list)


async def extract_features(ctx: AgentCtx, spec: PartSpec) -> _FeaturesOut:
    """Second-chance extraction when the intake spec dropped named features.

    Small one-shot call (prompt + current spec as JSON + images) — cheaper
    than re-running the full intake, same pattern as extract_load_cases."""
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "current_spec": spec.model_dump(
                include={"name", "summary", "length_x", "width_y", "height_z", "features"}
            ),
        },
        ensure_ascii=False,
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_features.md")},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_FeaturesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _FeaturesOut) else _FeaturesOut()
        await ctx.trace.record(
            step="intake_features",
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed
    except Exception:
        log.warning("intake.features_retry_failed", exc_info=True)
        return _FeaturesOut()


async def extract_load_cases(
    ctx: AgentCtx, spec: PartSpec
) -> tuple[list[LoadCase], OptimizeSpec | None]:
    """Focused second-chance extraction: the main intake emitted no usable
    `load_cases` although the user described forces/optimization. Small
    one-shot call on prompt+spec — cheaper than re-running full intake.
    Also recovers the `optimize` block when intake left it empty."""
    user_text = json.dumps(
        {"prompt": ctx.prompt, "spec": spec.model_dump(exclude={"load_cases", "optimize"})},
        ensure_ascii=False,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt("intake_loads.md")},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    try:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=_LoadCasesOut,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        parsed = resp.parsed if isinstance(resp.parsed, _LoadCasesOut) else _LoadCasesOut()
        await ctx.trace.record(
            step=STEP,
            iteration=0,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=parsed.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )
        return parsed.load_cases, parsed.optimize
    except Exception:
        log.warning("intake.load_cases_retry_failed", exc_info=True)
        return [], None


def _repair_image_roles(spec: PartSpec, ctx: AgentCtx) -> None:
    """Bind blank image_roles entries to assets positionally.

    Local vision models regularly emit one entry per image but leave
    `asset_id` (and sometimes `role`) blank. Entries are one-per-image in
    image order, so a count match lets us recover the ids the model meant.
    """
    imgs = [a for a in ctx.assets if a.kind in IMAGE_INPUT_KINDS]
    shown = imgs[: ctx.runtime_settings.agent_max_input_images]
    if not shown or len(spec.image_roles) != len(shown):
        return
    if all(not r.asset_id for r in spec.image_roles):
        for r, a in zip(spec.image_roles, shown, strict=False):
            r.asset_id = a.asset_id


async def run(ctx: AgentCtx) -> PartSpec:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_block(ctx.prompt, ctx.settings, ctx.mode)}
    ]
    ref = references_block(ctx.assets)
    if ref:
        user_content.append({"type": "text", "text": ref})
    if ctx.mode == "edit" and (ctx.parent.spec or ctx.parent.code):
        user_content += parent_block(ctx.parent.code, ctx.parent.spec, ctx.parent.contact_sheet)
    user_content += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)

    use_tools = tools_enabled(ctx, "vision")
    system = system_prompt("intake_system.md")
    if use_tools:
        system += "\n\n" + system_prompt("tools_info.md")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="info")
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=PartSpec,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, PartSpec):
            from cadx.llm.structured import LLMFormatError

            raise LLMFormatError("intake session produced no spec")
        spec = res.response.parsed
        _repair_image_roles(spec, ctx)
        sanitize_load_cases(spec)
        sanitize_dims(spec, ctx.prompt or "")
        sanitize_material(spec, ctx.prompt or "")
        for note in sanitize_dim_echo(spec, ctx.prompt or ""):
            log.info("intake.dim_echo_fixed", note=note)
        if not spec.process:
            spec.process = ctx.settings.process
        ctx.spec = spec
        return spec
    resp = await ctx.llm.chat(
        "vision",
        messages,
        json_schema=PartSpec,
        temperature=0.0,
        extra_body=thinking_body(ctx, STEP),
    )
    assert isinstance(resp.parsed, PartSpec)
    await ctx.trace.record(
        step=STEP,
        iteration=0,
        model=resp.model,
        messages=messages,
        response_text=resp.text,
        parsed=resp.parsed.model_dump(),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )
    spec = resp.parsed
    _repair_image_roles(spec, ctx)
    sanitize_load_cases(spec)
    sanitize_dims(spec, ctx.prompt or "")
    sanitize_material(spec, ctx.prompt or "")
    for note in sanitize_dim_echo(spec, ctx.prompt or ""):
        log.info("intake.dim_echo_fixed", note=note)
    if not spec.process:
        spec.process = ctx.settings.process
    ctx.spec = spec
    return spec
