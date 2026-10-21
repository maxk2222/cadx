"""CRITIQUE — compare generated model with input images + spec (vision model)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cadx.agent.context import (
    anchors_block,
    dfm_block,
    images_block,
    measured_features_block,
    measurements_block,
    parent_block,
    renders_block,
    request_block,
    spec_block,
    system_prompt,
)
from cadx.agent.thinking import thinking_body
from cadx.agent.toolloop import run_session, tools_enabled
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx
from cadx.llm.schemas import Critique, Discrepancy, Feature, PartSpec

STEP = "critique"


def _parts(
    ctx: AgentCtx,
    spec: PartSpec,
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    features: dict[str, Any] | None,
    dfm: dict[str, Any] | None,
    prev_issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    numeric = (
        "Numeric dimension check: " + "; ".join(dim_issues)
        if dim_issues
        else "Numeric dimension check: all stated dimensions match within tolerance."
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": "USER INPUT IMAGES (for reference):"}]
    parts += images_block(ctx.assets, ctx.runtime_settings.agent_max_input_images)
    if ctx.parent.contact_sheet is not None and (
        ctx.mode == "edit" or ctx.parent.mesh_solid is not None
    ):
        parts += parent_block(None, None, ctx.parent.contact_sheet)
        parts.append(
            {
                "type": "text",
                "text": (
                    "Only the requested changes should differ from the previous version."
                    if ctx.mode == "edit"
                    else "REFERENCE SOLID — the stored mesh this job reuses: the organic "
                    "form must be identical; only the requested cuts/features may differ."
                ),
            }
        )
    if ctx.resolved_mode in ("mesh", "hybrid"):
        parts.append(
            {
                "type": "text",
                "text": (
                    f"GENERATION MODE: {ctx.resolved_mode} — the shape comes from a "
                    "generative mesh model. Judge silhouette, proportions and the requested "
                    "features; ignore faceting/shading artifacts inherent to triangle meshes."
                ),
            }
        )
    rb = request_block(ctx.prompt)
    if rb:
        parts.append({"type": "text", "text": rb})
    parts.append({"type": "text", "text": spec_block(spec)})
    # spec-derived yes/no checks (M6, CADCodeVerify): job-stable like the
    # spec itself, so they stay inside the shared APC prefix — a "no" maps
    # to a deterministic issue the overall score cannot paper over
    if spec.verification_questions:
        q_lines = [
            "SPEC VERIFICATION QUESTIONS (answer EVERY one in "
            "verification_answers — judge the renders only):",
            *(f"{i + 1}. {q}" for i, q in enumerate(spec.verification_questions)),
        ]
        parts.append({"type": "text", "text": "\n".join(q_lines)})
    # APC ordering: input measurements/anchors are job-stable — they precede
    # the per-iteration tail (renders, analysis, findings) so critique calls
    # share one prefix across iterations
    mm = measurements_block(ctx.measurements)
    if mm:
        parts.append({"type": "text", "text": mm})
    ab = anchors_block(ctx.anchors)
    if ab:
        parts.append({"type": "text", "text": ab})
    parts += renders_block(contact_sheet, analysis)
    parts.append({"type": "text", "text": numeric})
    if prev_issues:
        pi_lines = [
            "PREVIOUS ITERATION ISSUES (flagged last iteration — verify "
            "against the renders/measurements and mark each in "
            "feature_verdicts as resolved, refuted, or still-present):"
        ]
        for it in prev_issues[:8]:
            what = str((it or {}).get("what") or "")[:160]
            if what:
                pi_lines.append(f"{len(pi_lines)}. {what}")
        if len(pi_lines) > 1:
            parts.append({"type": "text", "text": "\n".join(pi_lines)})
    mf = measured_features_block(features)
    if mf:
        parts.append({"type": "text", "text": mf})
    if feature_findings:
        lines = ["GEOMETRIC FEATURE CHECK (measured, may be approximate):"]
        for f in feature_findings:
            lines.append(
                f"- [{f.get('severity', 'minor')}] {f.get('feature')}: expected "
                f"{f.get('expected')}, actual {f.get('actual')}"
            )
        parts.append({"type": "text", "text": "\n".join(lines)})
    dfm_txt = dfm_block(dfm)
    if dfm_txt:
        parts.append({"type": "text", "text": dfm_txt})
    return parts


async def run(
    ctx: AgentCtx,
    spec: PartSpec,
    contact_sheet: Path | None,
    analysis: dict[str, Any] | None,
    dim_issues: list[str],
    dfm: dict[str, Any] | None,
    *,
    iteration: int,
    feature_findings: list[dict[str, Any]] | None = None,
    features: dict[str, Any] | None = None,
    code: str = "",
    prev_issues: list[dict[str, Any]] | None = None,
) -> Critique:
    parts = _parts(
        ctx,
        spec,
        contact_sheet,
        analysis,
        dim_issues,
        feature_findings or [],
        features,
        dfm,
        prev_issues,
    )

    system = system_prompt("critique_system.md")
    use_tools = tools_enabled(ctx, "vision")
    if use_tools:
        if code:
            system += "\n\n" + system_prompt("tools_inspect.md")
        else:
            system += (
                "\n\nNo source code is available in this generation mode — "
                "judge the renders and measured facts only. Catalog lookups "
                "via tools are still allowed."
            )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": parts},
    ]
    if use_tools:
        session = ToolSession(ctx, kind="inspect", code=code)
        res = await run_session(
            ctx,
            "vision",
            messages,
            session=session,
            json_schema=Critique,
            max_calls=ctx.runtime_settings.agent_tool_max_calls_info,
            step=STEP,
            iteration=iteration,
            temperature=0.0,
        )
        if res.response is None or not isinstance(res.response.parsed, Critique):
            from cadx.llm.structured import LLMFormatError

            raise LLMFormatError("critique session produced no verdict")
        critique = res.response.parsed
        resp = res.response
    else:
        resp = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=Critique,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        assert isinstance(resp.parsed, Critique)
        critique = resp.parsed
        # the tool session already wrote its own step=STEP record(s) — only
        # the one-shot path writes here, else token stats double-count
        await ctx.trace.record(
            step=STEP,
            iteration=iteration,
            model=resp.model,
            messages=messages,
            response_text=resp.text,
            parsed=critique.model_dump(),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_ms=resp.latency_ms,
        )

    # empty-discrepancy guard: a failing score with no issues listed gives the
    # loop nothing to fix — re-ask once for concrete problems.
    if (
        not critique.passed
        and not critique.discrepancies
        and ctx.runtime_settings.agent_critique_reask
    ):
        messages.append({"role": "assistant", "content": resp.text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"You scored {critique.score}/100 with passed=false but listed "
                    "NO discrepancies — that gives nothing to fix. List every "
                    "concrete problem now (what is wrong, where, and a fix_hint "
                    "with a target value). Respond with the complete Critique JSON."
                ),
            }
        )
        resp2 = await ctx.llm.chat(
            "vision",
            messages,
            json_schema=Critique,
            temperature=0.0,
            extra_body=thinking_body(ctx, STEP),
        )
        assert isinstance(resp2.parsed, Critique)
        await ctx.trace.record(
            step=f"{STEP}_reask",
            iteration=iteration,
            model=resp2.model,
            messages=messages,
            response_text=resp2.text,
            parsed=resp2.parsed.model_dump(),
            prompt_tokens=resp2.prompt_tokens,
            completion_tokens=resp2.completion_tokens,
            latency_ms=resp2.latency_ms,
        )
        if resp2.parsed.discrepancies or resp2.parsed.score < critique.score:
            critique = resp2.parsed

    # calibration transparency: the LLM's raw verdict vs. the deterministic
    # post-processing (synthesized discrepancies, hard rule, ceiling/floor) —
    # score jumps are otherwise unexplainable in trace.json
    raw_score, raw_passed = critique.score, critique.passed
    # the floor only applies to the LLM's own discrepancy list — capture it
    # before _finalize_critique mutates the object in place
    raw_had_discrepancies = bool(critique.discrepancies)
    # merge analysis scalars the verify passes read (envelope refute /
    # volume sanity) into a features copy — the measured-features payload
    # may not carry them yet
    feats = dict(features or {})
    if analysis:
        feats.setdefault("bbox_size", analysis.get("bbox_size"))
        feats.setdefault("volume", analysis.get("volume"))
    final = _finalize_critique(
        critique, ctx, spec, dim_issues, feature_findings or [], feats, prev_issues
    )
    await ctx.trace.record(
        step=f"{STEP}_calibrated",
        iteration=iteration,
        model="deterministic",
        messages=[],
        response_text="",
        parsed={
            "llm_score": raw_score,
            "llm_passed": raw_passed,
            "final_score": final.score,
            "final_passed": final.passed,
            "fact_ceiling": fact_ceiling(dim_issues, feature_findings or []),
            "fact_floor": (
                fact_floor(final, dim_issues, feature_findings or [])
                if raw_had_discrepancies
                else 0
            ),
            "discrepancies": len(final.discrepancies),
            "majors": sum(1 for d in final.discrepancies if d.severity == "major"),
        },
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0,
    )
    return final


_ISSUE_DIGITS_RE = re.compile(r"\d+")


def _norm_issue(text: str) -> str:
    """Issue-text normalization for repeat detection: lowercase + digit
    runs folded so 'leg is 5 mm short' matches 'leg is 4 mm short'."""
    return _ISSUE_DIGITS_RE.sub("#", (text or "").lower()).strip()


def _issue_words(text: str) -> set[str]:
    return {w for w in _NAME_WORD_RE.findall((text or "").lower()) if len(w) > 2}


def _resolved_evidence(critique: Critique, d: Discrepancy) -> bool:
    """The new critique suggests a verbatim-repeated issue is resolved:
    a verdict confirms an overlapping feature (present/resolved/refuted),
    or nothing else in the response re-describes it — a parroted line adds
    no information. An overlapping verdict still asserting the problem
    (missing/wrong) vetoes resolution outright."""
    words = _issue_words(d.what)
    if not words:
        return False
    overlapping = [fv for fv in critique.feature_verdicts if _issue_words(fv.name) & words]
    if any(fv.status in ("present", "resolved", "refuted") for fv in overlapping):
        return True
    if any(fv.status in ("missing", "wrong") for fv in overlapping):
        return False
    return not any(_issue_words(x.what) & words for x in critique.discrepancies if x is not d)


def _drop_prev_issue_repeats(critique: Critique, prev_issues: list[dict[str, Any]] | None) -> None:
    """Drop discrepancies verbatim-repeating a previous-iteration issue the
    new critique shows as resolved (P10) — re-emitting the same `what`
    every round burned fix cycles on phantoms (observed: 'the vertical leg
    is missing entirely' on a correct L-bracket, re-flagged each run).

    Conservative: only an IDENTICAL normalized `what` with resolution
    evidence is dropped; verify_question discrepancies (deterministic spec
    checks) and anything uncertain stay — a wrong drop is worse than a
    repeat."""
    prev = {
        _norm_issue(str(p.get("what") or ""))
        for p in prev_issues or []
        if isinstance(p, dict) and str(p.get("what") or "").strip()
    }
    if not prev or not critique.discrepancies:
        return
    critique.discrepancies = [
        d
        for d in critique.discrepancies
        if d.source == "verify_question"
        or _norm_issue(d.what) not in prev
        or not _resolved_evidence(critique, d)
    ]


def _downgrade_image_majors_on_silhouette(critique: Critique, ctx: AgentCtx) -> None:
    """Mesh-mode shape calibration: when the measured silhouette IoU says the
    outline matches the reference (>=0.8), majors sourced purely from image
    judgement are usually generative-texture noise or a spec-named feature
    misread as a defect (observed: knob skirt->dome step scored as a flaw) —
    downgrade to minor. Spec/prompt/dfm-sourced majors stand."""
    if getattr(ctx, "resolved_mode", "") not in ("mesh", "hybrid"):
        return
    sil = (getattr(ctx, "mesh_info", None) or {}).get("_silhouette_iou")
    if not (isinstance(sil, int | float) and sil >= 0.8):
        return
    for d in critique.discrepancies:
        if d.severity == "major" and d.source in ("image", "other"):
            d.severity = "minor"
            d.what += " (measured silhouette matches the reference)"


def _finalize_critique(
    critique: Critique,
    ctx: AgentCtx,
    spec: PartSpec,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
    features: dict[str, Any] | None,
    prev_issues: list[dict[str, Any]] | None = None,
) -> Critique:
    """Post-LLM calibration: synthesize missing discrepancies, enforce the
    feature hard rule, then bound the score by measured facts."""
    # P10: verbatim repeats of previous-iteration issues the new critique
    # shows resolved are dropped before anything else — they must not count
    # as "the model listed complaints" for the floor or burn a fix cycle
    _drop_prev_issue_repeats(critique, prev_issues)
    # the floor below protects against UNSOURCED complaints — it only applies
    # when the model itself listed them; a bare failed verdict (discrepancies
    # synthesized here) is the model's own call and must stay low
    had_llm_discrepancies = bool(critique.discrepancies)
    # measured-fact verification: 'missing' verdicts on feature kinds the
    # measurement CAN see (holes, slots, fillets, chamfers, csk/cbore) are
    # downgraded when the geometry shows the feature present — vision
    # misjudgement must not hard-fail a part the numbers say is there.
    # Runs before synthesis so a refuted verdict synthesizes as 'wrong'
    # (minor), not 'missing' (major).
    _verify_missing_verdicts(critique, spec, features or {})
    _verify_wrong_verdicts(critique, spec, features or {})
    _drop_facet_dia_noise(critique, spec, features or {})
    _downgrade_image_majors_on_silhouette(critique, ctx)
    # runs before the empty-discrepancy synthesis: a failed spec check on an
    # otherwise clean pass inserts its own discrepancy, skipping synthesis
    _apply_verification_answers(critique, spec)
    # synthesize discrepancies from measured findings if still empty
    if not critique.passed and not critique.discrepancies:
        for i in dim_issues:
            critique.discrepancies.append(
                Discrepancy(what=f"dimension mismatch — {i}", severity="major", source="spec")
            )
        for f in feature_findings:
            critique.discrepancies.append(
                Discrepancy(
                    what=f"{f.get('feature')}: expected {f.get('expected')}, "
                    f"actual {f.get('actual')}",
                    severity=f.get("severity", "minor"),
                    fix_hint=f.get("fix_hint", ""),
                    source="spec",
                )
            )
        # the model's own feature verdicts are actionable too — a 'missing' or
        # 'wrong' verdict without a discrepancy is a self-contradiction
        for fv in critique.feature_verdicts:
            if fv.status in ("missing", "wrong"):
                critique.discrepancies.append(
                    Discrepancy(
                        what=f"feature {fv.status}: {fv.name}"
                        + (f" — {fv.note}" if fv.note else ""),
                        severity="major" if fv.status == "missing" else "minor",
                        fix_hint=f"make the '{fv.name}' feature match the spec"
                        + (f" ({fv.note})" if fv.note else ""),
                        source="spec",
                    )
                )
        # last resort: a failing verdict with literally nothing actionable —
        # give the improve step the honest generic hint instead of silence
        if not critique.discrepancies:
            critique.discrepancies.append(
                Discrepancy(
                    what="overall shape does not match the reference images",
                    severity="major",
                    fix_hint="compare silhouette and proportions against the "
                    "reference views and reshape the body",
                )
            )

    # hard rule: the request names features (holes/slots/ribs/fillets/...) but
    # the measured geometry has none and the spec lists none → the model must
    # not pass (observed: bracket prompt -> plain box scored 95)
    from cadx.agent.steps.intake import _FEATURE_WORDS

    measured_n = (
        int((features or {}).get("total_holes") or 0)
        + int((features or {}).get("slots") or 0)
        + len((features or {}).get("rounds") or [])
        + int((features or {}).get("chamfer_faces") or 0)
        + int((features or {}).get("countersinks") or 0)
        + int((features or {}).get("counterbores") or 0)
    )
    if _FEATURE_WORDS.search(ctx.prompt or "") and measured_n == 0 and not spec.features:
        critique.discrepancies.append(
            Discrepancy(
                what="the request names features (holes/slots/ribs/fillets) but "
                "the geometry has none",
                where="whole part",
                severity="major",
                source="prompt",
                fix_hint="implement every feature named in the USER REQUEST",
            )
        )
        critique.passed = False
        critique.score = min(critique.score, 40)

    # fact-based calibration: measured facts bound the LLM's score so a
    # confident 95 on a broken part — or a paranoid 40 on a clean one — both
    # get pulled toward reality (T3.1). Runs AFTER normalize_score so its
    # empty-discrepancy floor cannot undo the measured ceiling.
    critique = normalize_score(critique)
    ceiling = fact_ceiling(dim_issues, feature_findings)
    if critique.score > ceiling:
        critique.score = ceiling
    floor = fact_floor(critique, dim_issues, feature_findings) if had_llm_discrepancies else 0
    if floor:
        for d in critique.discrepancies:
            if d.source in ("other", "dfm") and d.severity == "major":
                d.severity = "minor"
        critique.score = max(critique.score, floor)
    return critique


# verdict 'missing' claims on these spec-feature kinds can be checked against
# the measured geometry — presence refutes 'absent' (position/fit may still
# be wrong, which is a different verdict)
_MEASURABLE_KINDS: tuple[tuple[str, str], ...] = (
    # (keyword in feature type/name, measured-features key)
    ("hole", "total_holes"),
    ("bore", "total_holes"),
    ("slot", "slots"),
    ("fillet", "rounds"),
    ("round", "rounds"),
    ("chamfer", "chamfer_faces"),
    ("countersink", "countersinks"),
    ("counterbore", "counterbores"),
    # plate-like structure — presence is an ok plate_check entry naming
    # the feature (footprint measured inside material), not a scalar count
    ("plate", "plates"),
    ("wall", "plates"),
    ("leg", "plates"),
    ("deck", "plates"),
    ("flange", "plates"),
    ("base", "plates"),
)


_NAME_WORD_RE = re.compile(r"[a-zäöü]+")
# descriptors that locate rather than identify — "vertical" must not bridge
# a name match between "vertical leg" and "vertical foot"
_NAME_SKIP = frozenset(
    {
        "the",
        "a",
        "an",
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "einen",
        "und",
        "and",
        "mit",
        "with",
        "von",
        "vom",
        "fuer",
        "für",
        "vertical",
        "horizontal",
        "senkrecht",
        "senkrechte",
        "senkrechter",
        "waagrecht",
        "waagerecht",
        "waagerechte",
        "upper",
        "lower",
        "obere",
        "oberer",
        "untere",
        "unterer",
        "outer",
        "inner",
        "vordere",
        "vorderer",
        "hintere",
        "hinterer",
        "seitliche",
        "seitlicher",
        "mittlere",
        "mittlerer",
        "left",
        "right",
        "front",
        "rear",
        "top",
        "bottom",
        "oben",
        "unten",
        "vorne",
        "hinten",
        "links",
        "rechts",
        "middle",
        "center",
        "centre",
        "centered",
        "zentriert",
        "mittig",
    }
)


def _name_words(text: str) -> set[str]:
    """Content words of a feature name — alpha tokens, len > 2, minus
    positional/directional descriptors."""
    return {
        w for w in _NAME_WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in _NAME_SKIP
    }


def _plate_named_ok(name: str, features: dict[str, Any]) -> bool:
    """An ok plate_check entry names this feature — the declared footprint
    measured inside material, so 'missing entirely' cannot stand. Matches
    by shared name words, not a fuzzy haystack substring."""
    words = _name_words(name)
    if not words:
        return False
    for p in features.get("plates") or []:
        if not isinstance(p, dict) or not p.get("ok"):
            continue
        if _name_words(str(p.get("name") or "")) & words:
            return True
    return False


def _measured_presence_key(haystack: str, name: str, features: dict[str, Any]) -> str | None:
    """Measured-features key proving a claimed-missing feature exists —
    plate kinds need an ok plate_check naming the feature; scalar kinds
    only need a nonzero count. None = nothing proves presence."""
    for kw, key in _MEASURABLE_KINDS:
        if kw not in haystack:
            continue
        if key == "plates":
            if _plate_named_ok(name, features):
                return key
        elif _measured_presence(key, features):
            return key
    return None


_DIM_AXIS = {"length": "x", "width": "y", "height": "z"}
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
# void-ish feature words — removing material, so a matching bbox extent
# proves nothing about their existence (envelope refute is structure-only)
_VOID_WORDS = frozenset(
    {
        "hole",
        "bore",
        "slot",
        "pocket",
        "cutout",
        "groove",
        "thread",
        "countersink",
        "counterbore",
        "chamfer",
        "fillet",
        "radius",
        "loch",
        "bohrung",
        "nut",
        "tasche",
        "aussparung",
        "recess",
        "gewinde",
        "senkung",
        "senkbohrung",
        "fase",
    }
)


def _voidish(f: Feature) -> bool:
    return bool(set(_NAME_WORD_RE.findall(f"{f.type or ''} {f.name or ''}".lower())) & _VOID_WORDS)


def _declared_extents(f: Feature) -> dict[str, float]:
    """Per-axis extent the spec feature declares — from dims
    (length->x, width->y, height->z) and from position range spans
    ("x=-30 bis +30" -> 60)."""
    out: dict[str, float] = {}
    dims = f.dims or {}
    for k, ax in _DIM_AXIS.items():
        v = dims.get(k)
        if isinstance(v, int | float) and float(v) > 0:
            out[ax] = max(out.get(ax, 0.0), float(v))
    from cadx.agent.spechost import parse_ranges

    for ax, (lo, hi) in parse_ranges(str(f.position or "")).items():
        if ax in _AXIS_IDX and hi > lo:
            out[ax] = max(out.get(ax, 0.0), hi - lo)
    return out


def _spec_feature_for(spec: PartSpec, name: str) -> Feature | None:
    """Spec feature a verdict names — exact name match, else a single
    unambiguous containment ("leg" -> "vertical leg"). Ambiguity -> None:
    guessing which feature was meant must never drive a refute."""
    if not name:
        return None
    for f in spec.features:
        if (f.name or "").lower() == name:
            return f
    cand = [
        f for f in spec.features if f.name and (f.name.lower() in name or name in f.name.lower())
    ]
    return cand[0] if len(cand) == 1 else None


def _envelope_refutes(f: Feature | None, features: dict[str, Any]) -> bool:
    """A structure feature spanning a full measured axis cannot be
    'missing': the bbox proves material reaches that extent (observed:
    'the vertical leg is missing entirely' on a correct L-bracket —
    declared leg height 50, measured bbox z=50). Only a resolvable numeric
    extent matching within 5% refutes — never a guess, never a void."""
    if f is None or _voidish(f):
        return False
    bbox = features.get("bbox_size")
    if not isinstance(bbox, list | tuple) or len(bbox) < 3:
        return False
    try:
        meas = [float(bbox[0]), float(bbox[1]), float(bbox[2])]
    except (TypeError, ValueError):
        return False
    for ax, ext in _declared_extents(f).items():
        m = meas[_AXIS_IDX[ax]]
        if m > 0 and abs(m - ext) <= 0.05 * m:
            return True
    return False


# plate-like feature words for the volume sanity check — the structure
# whose declared dims must roughly explain the measured solid volume
_PLATE_LIKE = frozenset(
    {
        "foot",
        "feet",
        "leg",
        "legs",
        "wall",
        "walls",
        "base",
        "plate",
        "plates",
        "deck",
        "flange",
        "arm",
        "pad",
        "tab",
        "lug",
        "bracket",
        "winkel",
        "fuss",
        "fuß",
        "schenkel",
        "wand",
        "platte",
        "boden",
        "flansch",
        "lasche",
    }
)
_VOL_DIMS = ("length", "width", "height", "thickness")


def _plate_like(f: Feature) -> bool:
    words = set(_NAME_WORD_RE.findall(f"{f.type or ''} {f.name or ''}".lower()))
    return bool(words & _PLATE_LIKE)


def _declared_plate_volume(spec: PartSpec) -> float:
    """Sum of declared plate-like feature volumes — the product of the
    >= 2 plate dims (length*width*thickness or length*width*height) per
    declared instance count."""
    total = 0.0
    for f in spec.features:
        if not _plate_like(f):
            continue
        dims = f.dims or {}
        vals = [
            float(dims[k])
            for k in _VOL_DIMS
            if isinstance(dims.get(k), int | float) and float(dims[k]) > 0
        ]
        if len(vals) < 2:
            continue
        v = 1.0
        for d in vals:
            v *= d
        total += v * max(1, int(f.count or 1))
    return total


def _volume_consistent(spec: PartSpec, features: dict[str, Any]) -> bool:
    """Measured volume within ±30% of the declared plate volume — the
    structure demonstrably exists, so a 'missing' major on it is a
    misread, not absent geometry."""
    try:
        measured = float(features.get("volume") or 0.0)
    except (TypeError, ValueError):
        return False
    declared = _declared_plate_volume(spec)
    return measured > 0 and declared > 0 and abs(measured - declared) <= 0.30 * declared


def _structure_hit(what: str, spec: PartSpec) -> bool:
    """The discrepancy names a plate-like word or a plate-like spec
    feature by name."""
    words = set(_NAME_WORD_RE.findall(what.lower()))
    if words & _PLATE_LIKE:
        return True
    return any(_name_words(f.name) & words for f in spec.features if _plate_like(f))


def _drop_facet_dia_noise(critique: Critique, spec: PartSpec, features: dict[str, Any]) -> None:
    """On faceted mesh solids a hole Ø read off tessellated cylinders carries
    a few % of polygon noise (observed: Ø5.5 cutters measuring 5.47/5.91) —
    a "wrong diameter" discrepancy within 8% of a stated dia is a measurement
    artifact, not a defect. Dropping it stops the improve loop from chasing
    cutter radius back and forth; the observation stays in feature_verdicts."""
    if not features.get("approximate"):
        return
    stated = [
        float(f.dims["diameter"])
        for f in spec.features
        if isinstance(f.dims, dict) and f.dims.get("diameter")
    ]
    measured = [
        float(g["diameter"]) for g in features.get("hole_groups") or [] if g.get("diameter")
    ]
    if not stated or not measured:
        return
    close = any(abs(m - s) <= 0.08 * s for s in stated for m in measured)
    if not close:
        return
    keep = [
        d
        for d in critique.discrepancies
        if not (
            d.severity == "minor"
            and d.source == "spec"
            and any(k in d.what.lower() for k in ("diameter", "ø", "dia"))
        )
    ]
    critique.discrepancies = keep


def _measured_presence(kind_key: str, features: dict[str, Any]) -> bool:
    v = features.get(kind_key)
    if isinstance(v, list):
        return bool(v)
    return bool(v)


def _verify_missing_verdicts(critique: Critique, spec: PartSpec, features: dict[str, Any]) -> None:
    """Downgrade 'missing' verdicts contradicted by measured geometry.

    The vision model occasionally reports a feature absent that renders and
    measurements show present (observed: firesteel 'tube missing' while the
    tube spanned the plate — and 'the vertical leg is missing entirely' on
    a correct L-bracket). Three conservative refuters, each only flipping
    'missing' -> 'wrong' (the feature exists; position/shape may still
    mismatch):

    1. measured presence — holes/slots/fillets the counters see, and
       plate-like features an ok plate_check names,
    2. envelope — the spec feature's declared extent matching a measured
       bbox axis within 5% (material achieving it exists),
    3. volume sanity — discrepancy level only: measured volume ≈ the
       declared plate volumes makes 'missing' majors on structure
       implausible (severity downgrade, the verdict is NOT flipped).

    Unmeasurable kinds ('rib', 'texture', ...) stay LLM-judged.
    """
    spec_types = {f.name.lower(): (f.type or "") for f in spec.features}
    refuted: list[str] = []
    for fv in critique.feature_verdicts:
        if fv.status != "missing":
            continue
        name = (fv.name or "").lower()
        ftype = spec_types.get(name, "")
        haystack = f"{ftype} {name}"
        key = _measured_presence_key(haystack, name, features)
        if key is not None:
            fv.status = "wrong"
            fv.note = (fv.note or "") + (
                f" [measured: {key} present — 'missing' refuted, position/quality unverified]"
            )
            refuted.append(name)
        elif _envelope_refutes(_spec_feature_for(spec, name), features):
            fv.status = "wrong"
            fv.note = (fv.note or "") + (
                " [measured: declared extent matches the bbox — 'missing' "
                "refuted, position/quality unverified]"
            )
            refuted.append(name)
    vol_ok = _volume_consistent(spec, features)
    if not refuted and not vol_ok:
        return
    # discrepancies that repeat a refuted 'missing' claim lose their major
    # severity — they may still describe a real placement problem (minor)
    for d in critique.discrepancies:
        if d.severity != "major" or "missing" not in d.what.lower():
            continue
        if any(n in d.what.lower() for n in refuted):
            d.severity = "minor"
            d.what += " (feature measured present — likely a placement/shape issue)"
        elif vol_ok and _structure_hit(d.what, spec):
            d.severity = "minor"
            d.what += " (volume consistent with declared plates — likely a misread)"


def _verify_wrong_verdicts(critique: Critique, spec: PartSpec, features: dict[str, Any]) -> None:
    """Downgrade 'wrong' verdicts contradicted on every checkable dimension.

    Conservative — refuted only when the spec's own numbers match the
    measurement: a hole verdict 'wrong' stands unless a group matches the
    spec'd diameter AND axis; a slot verdict stands unless the slot count is
    met. Position/surface quality remain unverifiable, so the verdict falls
    back to 'unsure' (never 'present') and major discrepancies to minor.
    """
    spec_by_name = {f.name.lower(): f for f in spec.features}
    groups = [g for g in (features.get("hole_groups") or []) if not g.get("elongated")]
    refuted: list[str] = []
    for fv in critique.feature_verdicts:
        if fv.status != "wrong":
            continue
        name = (fv.name or "").lower()
        f = spec_by_name.get(name)
        if f is None:
            continue
        hay = f"{(f.type or '').lower()} {name}"
        ok = False
        if "slot" in hay:
            ok = int(features.get("slots") or 0) >= max(1, int(f.count or 1))
        elif "hole" in hay or "bore" in hay:
            want_n = max(1, int(f.count or 1))
            want_dia = None
            if f.dims:
                v = f.dims.get("diameter") or f.dims.get("dia") or f.dims.get("d")
                if isinstance(v, int | float) and v > 0:
                    want_dia = float(v)
            want_ax = str(getattr(f, "axis", "") or "").lstrip("+-").lower()
            matched = 0
            for g in groups:
                g_dia = float(g.get("diameter") or 0)
                if want_dia is not None and abs(g_dia - want_dia) > max(0.5, want_dia * 0.15):
                    continue
                g_ax = str(g.get("axis") or "").lstrip("+-").lower()
                if want_ax in ("x", "y", "z") and g_ax and g_ax != want_ax:
                    continue
                matched += int(g.get("count") or 1)
            ok = matched >= want_n
        if ok:
            fv.status = "unsure"
            fv.note = (fv.note or "") + (
                " [measured: spec dims match — 'wrong' refuted on the "
                "checkable axes, position unverified]"
            )
            refuted.append(name)
    if not refuted:
        return
    for d in critique.discrepancies:
        if d.severity == "major" and any(n in d.what.lower() for n in refuted):
            d.severity = "minor"
            d.what += " (spec dims measured correct — likely a position nuance)"


def _apply_verification_answers(critique: Critique, spec: PartSpec) -> None:
    """Map spec verification answers to deterministic issues (M6).

    The overall render can look plausible while a spec assertion is
    violated (slot on the wrong axis, rib outside the envelope) — the
    vision score stays high while verify_spec fails. A "no" answer, or an
    "unclear" backed by observed evidence, therefore becomes a major
    ``verify_question`` discrepancy regardless of the model's confidence
    and fails the verdict. A bare "unclear" (nothing suspicious in sight)
    adds nothing — renders that cannot decide a check must not stall the
    loop on their own.
    """
    if not spec.verification_questions or not critique.verification_answers:
        return
    added = False
    for a in critique.verification_answers:
        q = (a.question or "").strip()
        ev = (a.evidence or "").strip()
        if not q or (a.answer != "no" and not (a.answer == "unclear" and ev)):
            continue
        if any(q in d.what for d in critique.discrepancies):
            continue  # the critic already listed this exact problem
        critique.discrepancies.append(
            Discrepancy(
                what=q,
                severity="major",
                fix_hint=ev or "fix the part so this check answers yes",
                source="verify_question",
            )
        )
        added = True
    if added:
        # a failed spec check contradicts a passing verdict even when the
        # render looked plausible enough for a high score
        critique.passed = False


def fact_ceiling(dim_issues: list[str], feature_findings: list[dict[str, Any]]) -> int:
    """Upper bound from measured facts only — dims + major feature findings."""
    majors = sum(1 for f in feature_findings if f.get("severity") == "major")
    if dim_issues or majors:
        return max(0, 70 - 15 * (len(dim_issues) + majors))
    return 100


def fact_floor(
    critique: Critique,
    dim_issues: list[str],
    feature_findings: list[dict[str, Any]],
) -> int:
    """Lower bound when every measured fact passes and only unsourced
    remarks remain — invented flaws must not tank a clean part."""
    if dim_issues or any(f.get("severity") == "major" for f in feature_findings):
        return 0
    if critique.discrepancies and all(d.source in ("other", "dfm") for d in critique.discrepancies):
        return 60
    return 0


def normalize_score(critique: Critique) -> Critique:
    """Score consistency clamp: the LLM's number is untrusted when it
    contradicts its own discrepancy list (observed: '0/100 — 0 major,
    0 minor'). Deterministic ceiling so gates see an explainable score."""
    if critique.discrepancies:
        n_major = sum(1 for d in critique.discrepancies if d.severity == "major")
        n_minor = len(critique.discrepancies) - n_major
        # verify_question majors weigh less: a "no" answer usually duplicates
        # a discrepancy the critic already listed (observed: one missing-hole
        # pair -> 4 majors -> score clamped to 0, no gradient left)
        n_major_vq = sum(
            1
            for d in critique.discrepancies
            if d.severity == "major" and d.source == "verify_question"
        )
        ceiling = max(0, 100 - 25 * (n_major - n_major_vq) - 10 * n_major_vq - 10 * n_minor)
        if critique.score > ceiling:
            critique.score = ceiling
    elif critique.score < 60:
        # a low score with zero listed issues is meaningless — normalize so
        # measured facts (dim/feature findings, passed flag) decide instead
        critique.score = 60
    return critique
