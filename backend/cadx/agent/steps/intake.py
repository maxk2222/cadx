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
from cadx.llm.schemas import (
    Dim,
    Feature,
    LoadCase,
    OpenQuestion,
    OptimizeSpec,
    PartSpec,
    StandardPartSpec,
)
from cadx.logging import get_logger

log = get_logger(__name__)

STEP = "intake"


class _LoadCasesOut(BaseModel):
    load_cases: list[LoadCase] = Field(default_factory=list)
    optimize: OptimizeSpec | None = None


_ANCHOR_INTENT_RE = re.compile(
    r"\bm\d+\b|flange|corner|boss|seat|insert|tapped?|thread|gewinde|"
    r"mount|plate|bracket|\barm\b|joint|rib|rim|lug|\bear\b",
    re.IGNORECASE,
)


def _anchor_intent_tokens(text: str) -> set[str]:
    """Distinctive tokens a case name / region description carries — bolt
    sizes normalise to 'm<number>' to match anchor bolt_size/name."""
    low = (text or "").lower()
    out = {f"m{m}" for m in re.findall(r"\bm(\d+)\b", low)}
    out |= {t.group(0).lower() for t in _ANCHOR_INTENT_RE.finditer(low)}
    return out


class _Rebinder:
    """State for one `hole_index` rebind pass over a spec's load cases."""

    def __init__(self, anchors: list[Any]) -> None:
        self.anchors = anchors
        self.n = len(anchors)
        self.a_toks = [
            _anchor_intent_tokens(f"{a.name or ''} {a.bolt_size or ''}") for a in anchors
        ]
        self.notes: list[str] = []

    def _best_matches(self, tokens: set[str]) -> list[int]:
        """Indices (1-based) of anchors sharing the most intent tokens."""
        best = 0
        out: list[int] = []
        for i, t in enumerate(self.a_toks):
            s = len(tokens & t)
            if s > best:
                best, out = s, [i + 1]
            elif s == best and s:
                out.append(i + 1)
        return out

    def _rebind(self, r: Any, where: str, *, is_fixed: bool) -> Any:
        if r is None or getattr(r, "hole_index", None) is None:
            return r
        if str(getattr(r, "kind", "")) not in ("hole", "anchor", "bore"):
            return r
        idx = int(r.hole_index)
        in_range = 1 <= idx <= self.n
        # a case name describes the LOAD side ("arm load on flange holes") —
        # fixtures named by it are coincidence: only out-of-range fixed
        # refs get re-pointed; an in-range fixture stays where it is.
        if is_fixed and in_range:
            return r
        cur = self.a_toks[idx - 1] if in_range else set()
        tokens = _anchor_intent_tokens(f"{where} {getattr(r, 'description', '') or ''}")
        if not tokens:
            return r  # nothing to disambiguate by — trust the index
        cur_score = len(tokens & cur)
        matches = self._best_matches(tokens)
        best_score = len(tokens & self.a_toks[matches[0] - 1]) if matches else 0
        if best_score <= cur_score:
            return r
        names = ", ".join(f"#{i} '{self.anchors[i - 1].name}'" for i in matches[:4])
        self.notes.append(
            f"{where}: hole_index {idx} "
            + ("out of range " if not in_range else "pointed at the wrong anchor ")
            + f"(max {self.n}) — rebound to {names}"
        )
        return r.model_copy(update={"hole_index": matches[0]})

    def _split(self, ld: Any, r: Any, where: str, matches: list[int]) -> list[Any]:
        self.notes.append(
            f"{where}: load split over anchors "
            + ", ".join(f"#{m} '{self.anchors[m - 1].name}'" for m in matches[:4])
        )
        return [
            ld.model_copy(update={"region": r.model_copy(update={"hole_index": m})})
            for m in matches[:4]
        ]

    def _rebind_load(self, ld: Any, r: Any, where: str, case_name: str) -> list[Any]:
        """One load entry → one-or-more rewritten entries (plural intent
        splits across every anchor the text names)."""
        tokens = _anchor_intent_tokens(f"{case_name} {r.description or ''}")
        rebound = self._rebind(r, where, is_fixed=False)
        # plural intent ("holes", "both", "all") + several matching anchors
        # → one load entry per anchor — a single bore region can't span two
        plural = bool(
            re.search(
                r"holes|bohrungen|both|all\b|beide",
                f"{case_name} {r.description or ''}",
                re.I,
            )
        )
        anchor_ref = getattr(r, "hole_index", None) is not None and str(getattr(r, "kind", "")) in (
            "hole",
            "anchor",
            "bore",
        )
        if rebound is r and anchor_ref:
            matches = self._best_matches(tokens)
            if plural and len(matches) > 1:
                return self._split(ld, r, where, matches)
        elif rebound is not r and plural:
            extra = [m for m in self._best_matches(tokens) if m != rebound.hole_index]
            if extra:
                out = [ld.model_copy(update={"region": rebound})]
                out += [
                    ld.model_copy(update={"region": r.model_copy(update={"hole_index": m})})
                    for m in extra[:3]
                ]
                return out
        return [ld.model_copy(update={"region": rebound}) if rebound is not r else ld]

    def run(self, spec: PartSpec) -> list[str]:
        for case in spec.load_cases:
            where = f"case '{case.name}'"
            case.fixed = [self._rebind(r, where, is_fixed=True) for r in (case.fixed or [])]
            new_loads: list[Any] = []
            for ld in case.loads or []:
                r = getattr(ld, "region", None)
                if r is None:
                    new_loads.append(ld)
                    continue
                new_loads += self._rebind_load(ld, r, where, case.name or "")
            if new_loads:
                case.loads = new_loads
        if spec.optimize is not None:
            spec.optimize.keep_regions = [
                self._rebind(r, "keep_regions", is_fixed=False)
                for r in (spec.optimize.keep_regions or [])
            ]
        return self.notes


def rebind_case_anchors(spec: PartSpec) -> list[str]:
    """Re-point load-case `hole_index` refs at the anchors they were meant for.

    `hole_index` counts the spec's hole-like features top to bottom — a weak
    model routinely miscounts (observed: `bore #8` on a 7-anchor part — dead
    case — and `bore #4` for the M12 seat — a 15 N·m torque applied to a
    corner mounting hole). The spec-order anchor table is authoritative:
    when the case text names an anchor the index does NOT point at (e.g.
    "M12"/"flange" in the case name vs the indexed anchor's feature name),
    rewire it; a load naming several anchors splits into one entry each.
    Out-of-range refs with no name match stay — the FEM warning is honest.
    """
    from cadx.agent.anchors import _spec_anchors

    anchors = _spec_anchors(spec)
    if not anchors:
        return []
    return _Rebinder(anchors).run(spec)


def sanitize_load_cases(spec: PartSpec) -> None:
    """Drop degenerate cases (no supports and no loads) a weak model emits as
    `{"name":"","fixed":[],"loads":[]}` — truthy lists of those would skip the
    extraction fallback and produce unsolvable FEM runs. Also clamps bogus
    negative case weights back to neutral, and rebinds `hole_index` region
    refs onto the anchors the case text actually names."""
    for note in rebind_case_anchors(spec):
        log.info("intake.case_anchor_rebound", note=note)
        spec.assumptions = (spec.assumptions or []) + [note]
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


_FULL_THREAD_RE = re.compile(
    r"vollgewinde|voll(?:es|en)?\s*gewinde|ganz(?:es|en)?\s*gewinde|"
    r"full[\s_-]*thread|entire(?:ly)?\s*thread|thread(?:ed)?\s+(?:over\s+)?"
    r"(?:the\s+)?(?:full|entire|whole)\s+length|"
    r"über\s+(?:die\s+)?(?:ganze|gesamte)\s+länge\s*(?:gewinde|thread|ver(?:sehen|telt)|gängig)?|"
    r"gewinde\s+über\s+(?:die\s+)?(?:ganze|gesamte)\s+länge",
    re.I,
)


def resolve_thread_intents(spec: PartSpec) -> list[str]:
    """Map free-text thread-reach wording onto StandardPartSpec knobs.

    Intake may write "Vollgewinde"/"full thread" only inside a covered
    feature's description — bd_warehouse would then emit the norm's partial
    thread (e.g. ISO 888 b=2d+6 = 32 mm on an M10x100) while the user asked
    for the full shank. Detect the wording (and a covered feature's numeric
    thread_length/length dim before dims get stripped) and set the recipe
    knobs the code emitter honours."""
    if not spec.standard_parts:
        return []
    from cadx.cad import catalog
    from cadx.cad.standard_parts import (
        _is_intrinsic_fastener_feature,
        covered_feature_indices,
    )

    covered = covered_feature_indices(spec)
    # an intrinsically-covered thread feature is only attributable when a
    # single screw-class part exists — otherwise explicit covers_features
    # has to disambiguate which part's reach it means
    screw_parts = [
        part
        for part in spec.standard_parts
        if (catalog.find_entry(part.class_name) or {}).get("category") == "screws"
    ]
    notes: list[str] = []
    for part in spec.standard_parts:
        if (
            part.full_thread is not None
            or part.thread_length_mm is not None
            or (part.parameters or {}).get("thread_length") is not None
        ):
            continue
        covers = set(part.covers_features or [])
        want_full = False
        want_len: float | None = None
        for i, feature in enumerate(spec.features):
            if covered and i not in covered:
                continue
            owned = bool(feature.name and feature.name in covers) or (
                len(screw_parts) == 1
                and part is screw_parts[0]
                and _is_intrinsic_fastener_feature(feature)
            )
            if not owned:
                continue
            if not ("thread" in str(feature.type or "").lower() or feature.thread_spec):
                continue
            text = f"{feature.name or ''} {feature.description or ''}"
            if _FULL_THREAD_RE.search(text):
                want_full = True
            for key in ("thread_length", "length"):
                raw = (feature.dims or {}).get(key)
                if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
                    want_len = max(want_len or 0.0, float(raw))
        if not (want_full or want_len):
            continue
        if want_len:
            part.thread_length_mm = want_len
            notes.append(
                f"{part.class_name}: covered feature asks thread length "
                f"{want_len:g} mm — recipe emits a thread-length override"
            )
        if want_full:
            part.full_thread = True
            notes.append(
                f"{part.class_name}: 'Vollgewinde/full thread' wording detected "
                "— recipe emits full shank thread"
            )
    return notes


def sanitize_standard_part_dims(spec: PartSpec) -> list[str]:
    if len(spec.standard_parts) != 1 or spec.standard_parts[0].quantity != 1:
        return []
    from cadx.cad.standard_parts import standard_part_bbox

    part = spec.standard_parts[0]
    try:
        expected = standard_part_bbox(part)
    except Exception as e:
        question = OpenQuestion(
            question=f"Complete the catalog parameters for {part.class_name}: {str(e)[:180]}",
            why_it_matters="The registered standard component cannot be generated without these values.",
            default_assumption="Use only the exact catalog class/type/size and required parameters.",
            blocking=True,
        )
        if not any(q.question == question.question for q in spec.open_questions):
            spec.open_questions.append(question)
        return [f"catalog dimensions unavailable for {part.class_name}"]
    notes: list[str] = []
    for field, value, axis in (
        ("length_x", expected[0], "X"),
        ("width_y", expected[1], "Y"),
        ("height_z", expected[2], "Z"),
    ):
        dim = getattr(spec, field)
        tolerance = max(0.25, abs(value) * 0.02)
        if dim.source == "stated" and abs(float(dim.value) - value) > tolerance:
            question_text = (
                f"The {part.class_name} catalog dimensions give {axis}={value:g} mm, "
                f"but the request states {axis}={dim.value:g} mm. Which should govern?"
            )
            existing = next((q for q in spec.open_questions if q.question == question_text), None)
            if existing is not None and not existing.blocking:
                # resolved with the default ("keep catalog dims") — apply it,
                # otherwise verify keeps flagging stated-vs-catalog forever
                setattr(
                    spec,
                    field,
                    Dim(value=value, unit="mm", source="assumed"),
                )
                notes.append(
                    f"{part.class_name} catalog {axis}={value:g} mm governs (question resolved)"
                )
                continue
            question = OpenQuestion(
                question=question_text,
                why_it_matters="A catalog part cannot keep both dimensions and the stated norm.",
                default_assumption="Keep the exact catalog dimensions.",
                blocking=True,
            )
            if existing is None:
                spec.open_questions.append(question)
            notes.append(f"{part.class_name} catalog {axis} dimension conflicts with stated size")
            continue
        if dim.source != "stated" and abs(float(dim.value) - value) > tolerance:
            setattr(
                spec,
                field,
                Dim(value=value, unit="mm", source="inferred"),
            )
            notes.append(f"{part.class_name} catalog dimensions set {axis}={value:g} mm")
    if notes:
        assumption = f"Nominal dimensions for {part.class_name} come from the installed catalog."
        if assumption not in spec.assumptions:
            spec.assumptions.append(assumption)
    notes.extend(_sanitize_covered_features(spec, part.class_name))
    return notes


def _sanitize_covered_features(spec: PartSpec, class_name: str) -> list[str]:
    """Hallucinated dims on catalog-covered features (a stated "head_diameter
    10" vs the real dk=18.8) contradict the norm table forever — the catalog
    governs every intrinsic dimension, so strip them instead of letting
    critique/improve fight the standard's geometry. Thread-pitch conflicts
    are still caught by is_standard_only via the free-text thread_spec."""
    from cadx.cad.standard_parts import covered_feature_indices

    covered = covered_feature_indices(spec)
    notes: list[str] = []
    if not covered:
        return notes
    for i, feature in enumerate(spec.features):
        if i in covered and feature.dims:
            feature.dims = {}
            notes.append(f"catalog {class_name} governs '{feature.name}' dimensions")
    # verification questions that quote a covered feature's hallucinated
    # numbers ("is the head diameter <= 10 mm?") can never pass — the
    # catalog answer is norm-defined. Keep questions about the part as a
    # whole (seating, orientation, presence).
    covered_words = {
        w
        for i, feature in enumerate(spec.features)
        if i in covered
        for w in re.findall(r"[a-zA-Z]{4,}", f"{feature.name or ''} {feature.type or ''}")
    }
    if not covered_words:
        return notes

    def _about_covered(question_text: str) -> bool:
        q_words = set(re.findall(r"[a-zA-Z]{4,}", question_text.lower()))
        return any(qw[:5] == cw.lower()[:5] for qw in q_words for cw in covered_words)

    kept = [vq for vq in spec.verification_questions if not _about_covered(str(vq))]
    for vq in spec.verification_questions:
        if _about_covered(str(vq)):
            notes.append(f"dropped catalog-internal check: {str(vq)[:80]}")
    if len(kept) != len(spec.verification_questions):
        spec.verification_questions = kept
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
    "create_standard_part",
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
_STANDARD_PART_RE = re.compile(
    r"\bM\d+(?:[x-]\d+(?:\.\d+)?)?\s*(?:schraube|screw|bolt|mutter|nut)\b|"
    r"\b(?:heat[- ]set insert|gewindeeinsatz|heatsetnut|heat-set-nut)\b",
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
    if _STANDARD_PART_RE.search(prompt or "") and not spec.standard_parts:
        gaps.append("prompt names a normed component but spec.standard_parts is empty")
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
    standard_parts: list[StandardPartSpec] = Field(default_factory=list, max_length=24)


async def extract_features(ctx: AgentCtx, spec: PartSpec) -> _FeaturesOut:
    """Second-chance extraction when the intake spec dropped named features.

    Small one-shot call (prompt + current spec as JSON + images) — cheaper
    than re-running the full intake, same pattern as extract_load_cases."""
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "current_spec": spec.model_dump(
                include={
                    "name",
                    "summary",
                    "length_x",
                    "width_y",
                    "height_z",
                    "features",
                    "standard_parts",
                }
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
    from cadx.agent.anchors import _spec_anchors

    # the model cannot see the mesh — hand it the resolved anchor table so
    # `hole_index` refs land on real anchors instead of a miscount
    anchors = _spec_anchors(spec)
    anchor_table = [
        {
            "hole_index": i + 1,
            "name": a.name,
            "kind": a.kind,
            "axis": a.axis,
            "dia": a.dia,
            "pad_dia": a.pad_dia,
            "bolt_size": a.bolt_size,
            "pos": [round(p, 2) if p is not None else None for p in (a.pos or [a.x, a.y, a.z])],
        }
        for i, a in enumerate(anchors)
    ]
    user_text = json.dumps(
        {
            "prompt": ctx.prompt,
            "spec": spec.model_dump(exclude={"load_cases", "optimize"}),
            "anchor_table": anchor_table,
        },
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
