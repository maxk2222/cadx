"""research step — auto-gate heuristic + brief→spec.assumptions wiring."""

from __future__ import annotations

from types import SimpleNamespace

from cadx.agent.steps import research
from cadx.agent.steps.intake import _sanitize_spec, research_block
from cadx.llm.schemas import PartSpec, ResearchBrief, ResearchFact


def test_needs_research_matches_product_and_norm_tokens():
    hits = [
        "heatsink for Intel i7-14700K",
        "Lüfterhalter für NEMA 23 Motor",
        "DIN 6885 keyway on a shaft",
        "mount for a 92mm fan",
        "bracket for LGA1700 socket cooler",
        "adapter plate ISO 4762 screw pattern",
        "threaded M6 insert host",
        "Gehäuse für Ryzen 7 7800X3D Kühlkörper",
        "wall mount case for a raspberry pi 5",
        "case for an Arduino Uno board",
        "adapter for a Noctua fan",
    ]
    misses = [
        "simple mounting bracket 100x60x8",
        "phone stand",
        "gear with 20 teeth",
        "M6 bolt",  # a bare fastener — no product/norm context
    ]
    for p in hits:
        assert research.needs_research(p), p
    for p in misses:
        assert not research.needs_research(p), p


def _ctx(prompt: str, research_mode: str = "auto", tools_ok: bool = True):
    return SimpleNamespace(
        prompt=prompt,
        spec=None,
        research=None,
        settings=SimpleNamespace(process="fdm"),
        runtime_settings=SimpleNamespace(
            agent_research=research_mode,
            agent_research_max_calls=8,
            agent_research_max_results=5,
        ),
        llm=SimpleNamespace(tools_mode="auto", tools_ok={"vision": tools_ok}),
        emit=None,
    )


async def test_run_gates():
    ctx = _ctx("heatsink for i7-14700K", research_mode="off")
    assert await research.run(ctx) is None
    ctx = _ctx("simple box 10x10x10", research_mode="auto")
    assert await research.run(ctx) is None


def _brief() -> ResearchBrief:
    return ResearchBrief(
        product="Intel Core i7-14700K",
        facts=[
            ResearchFact(
                claim="socket type",
                value="LGA1700",
                unit="",
                source="https://ark.intel.com/i7",
            ),
            ResearchFact(
                claim="mount hole spacing",
                value="78 x 78",
                unit="mm",
                source="https://noctua.at/lga1700",
            ),
        ],
        constraints=["cooler mount uses 4x M4 on 78x78 mm square"],
        open_questions=["IHS size: 38x38 vs 39x39 mm across sources"],
    )


def test_research_block_renders_facts_and_sources():
    text = research_block(_brief())
    assert "LGA1700" in text and "ark.intel.com" in text
    assert "78 x 78" in text and "open questions" in text


def test_sanitize_spec_lands_facts_in_assumptions_and_artifact():
    spec = PartSpec(
        name="heatsink",
        length_x={"value": 80},
        width_y={"value": 80},
        height_z={"value": 30},
    )
    ctx = _ctx("heatsink for i7-14700K")
    ctx.research = _brief()
    out = _sanitize_spec(spec, ctx)
    assert out.research is ctx.research
    sourced = [a for a in out.assumptions if a.startswith("researched:")]
    assert len(sourced) == 2
    assert any("LGA1700" in a and "ark.intel.com" in a for a in sourced)


def test_sanitize_spec_without_research_is_untouched():
    spec = PartSpec(
        name="bracket",
        length_x={"value": 100},
        width_y={"value": 50},
        height_z={"value": 20},
    )
    ctx = _ctx("bracket")
    out = _sanitize_spec(spec, ctx)
    assert out.research is None
    assert not any(a.startswith("researched:") for a in out.assumptions)


def test_researched_notes_survive_load_case_resanitize():
    """The orchestrator re-runs sanitize_load_cases on the recovery path —
    the _MAX_ASSUMPTION_NOTES trim must not cut the pipeline-side sourced
    notes that _sanitize_spec tail-appends."""
    from cadx.agent.steps.intake import _MAX_ASSUMPTION_NOTES, sanitize_load_cases

    spec = PartSpec(
        name="mount",
        length_x={"value": 48},
        width_y={"value": 24},
        height_z={"value": 52},
    )
    ctx = _ctx("mount for a nema 17 stepper")
    ctx.research = _brief()
    out = _sanitize_spec(spec, ctx)
    n_researched = sum(1 for a in out.assumptions if a.startswith("researched:"))
    assert n_researched == 2
    # model notes beyond the cap + re-sanitize — researched notes stay
    out.assumptions = [f"model note {i}" for i in range(_MAX_ASSUMPTION_NOTES + 3)] + [
        a for a in out.assumptions if a.startswith("researched:")
    ]
    sanitize_load_cases(out)
    sanitize_load_cases(out)  # second recovery pass — the observed killer
    assert [a for a in out.assumptions if a.startswith("researched:")] == [
        a for a in out.assumptions if a.startswith("researched:")
    ]
    assert sum(1 for a in out.assumptions if a.startswith("researched:")) == 2
    assert sum(1 for a in out.assumptions if a.startswith("model note")) == _MAX_ASSUMPTION_NOTES


def test_stated_dim_from_feature_spacing_downgraded():
    """Observed on the NEMA-17 eval: the model lifted 'two Ø4.5 holes about
    30 mm apart' into length_x=30 stated — verify_spec then enforced the
    guess and honestly FAILED a correct 48 mm part."""
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="nema mount",
        length_x={"value": 30, "source": "stated"},
        width_y={"value": 24, "source": "inferred"},
        height_z={"value": 52, "source": "inferred"},
    )
    notes = sanitize_stated_dims(
        spec,
        "motor mount for a NEMA 17 stepper, two dia 4.5 mm through-holes "
        "about 30 mm apart in the feet",
    )
    assert spec.length_x.source == "inferred"
    assert spec.width_y.source == "inferred" and spec.height_z.source == "inferred"
    assert notes and "length" not in notes[0]


def test_stated_dim_kept_when_prompt_states_envelope():
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="plate",
        length_x={"value": 48, "source": "stated"},
        width_y={"value": 24, "source": "stated"},
        height_z={"value": 80, "source": "stated"},
    )
    notes = sanitize_stated_dims(
        spec, "mounting plate 48 mm wide, 24 mm tall x 80 deep, 3 mm thick"
    )
    assert notes == []
    assert spec.length_x.source == "stated"
    assert spec.height_z.source == "stated"


def test_stated_dim_kept_from_triplet_and_diameter():
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="rotor",
        length_x={"value": 140, "source": "stated"},
        width_y={"value": 140, "source": "stated"},
        height_z={"value": 60, "source": "stated"},
    )
    assert sanitize_stated_dims(spec, "rotor Ø140, hub 60 tall") == []
    spec2 = PartSpec(
        name="block",
        length_x={"value": 120, "source": "stated"},
        width_y={"value": 70, "source": "stated"},
        height_z={"value": 90, "source": "stated"},
    )
    assert sanitize_stated_dims(spec2, "bracket 120 x 70 x 90 steel") == []


def test_stated_dim_untouched_without_prompt():
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="legacy",
        length_x={"value": 30, "source": "stated"},
        width_y={"value": 24},
        height_z={"value": 52},
    )
    assert sanitize_stated_dims(spec, "") == []
    assert spec.length_x.source == "stated"


def test_sanitize_dims_no_restate_from_feature_spacing():
    """The recovery path re-runs sanitize_dims after _sanitize_spec — its
    'largest mm = overall size' heuristic must not resurrect the '30 mm
    apart' spacing as a stated envelope (the second half of the NEMA bug)."""
    from cadx.agent.steps.intake import sanitize_dims, sanitize_stated_dims

    prompt = (
        "motor mount for a NEMA 17 stepper, two dia 4.5 mm through-holes "
        "about 30 mm apart in the feet"
    )
    spec = PartSpec(
        name="nema mount",
        length_x={"value": 30, "source": "stated"},
        width_y={"value": 24},
        height_z={"value": 52},
    )
    sanitize_stated_dims(spec, prompt)
    assert spec.length_x.source == "inferred"
    # the orchestrator recovery pass — must not re-state
    sanitize_dims(spec, prompt)
    assert spec.length_x.source == "inferred"
    assert float(spec.length_x.value) == 30.0


def test_sanitize_dims_still_restates_true_envelope():
    from cadx.agent.steps.intake import sanitize_dims

    spec = PartSpec(
        name="lever",
        length_x={"value": 40},
        width_y={"value": 20},
        height_z={"value": 10},
    )
    notes = sanitize_dims(spec, "a lever approx 130 mm long with a small knob")
    assert notes
    assert spec.length_x.source == "stated" and float(spec.length_x.value) == 130.0


def _feat(name, pos, count=1, dia=None, depth=None, ftype="hole"):
    from cadx.llm.schemas import Feature

    dims = {}
    if dia is not None:
        dims["diameter"] = dia
    if depth is not None:
        dims["depth"] = depth
    return Feature(
        name=name,
        type=ftype,
        description=name,
        position=pos,
        count=count,
        dims=dims,
        source="stated",
    )


def test_dedup_merges_rephrased_duplicate_features():
    """Observed spec: the 4xM3 pattern and the 2xM4 foot holes each declared
    twice under different phrasing — spec_features then honestly failed
    against the 3 measured groups."""
    from cadx.agent.steps.intake import dedup_spec_features

    spec = PartSpec(
        name="nema",
        length_x={"value": 20},
        width_y={"value": 48},
        height_z={"value": 52},
        features=[
            _feat(
                "pilot bore",
                "through-hole along y, centred at x=0, z=26",
                count=1,
                dia=22.0,
                depth=4.0,
            ),
            _feat(
                "M3 pattern",
                "through-holes along y at x=±15.5, z=26±15.5 (z = 10.5 and 41.5)",
                count=4,
                dia=3.4,
            ),
            _feat(
                "foot holes",
                "through-holes along z at y=10, x=±15 (30 mm apart)",
                count=2,
                dia=4.5,
            ),
            _feat(
                "M3 pattern again",
                "Through-holes along Y axis. Positions relative to motor centre: (±15.5, ±15.5).",
                count=4,
                dia=3.4,
            ),
            _feat(
                "foot holes again",
                "Through-holes along Z axis. Centred in the 20 mm flange depth (y=10). Spaced 30 mm apart in X (x=±15).",
                count=2,
                dia=4.5,
            ),
        ],
    )
    notes = dedup_spec_features(spec)
    assert len(spec.features) == 3
    assert len(notes) == 2
    assert [f.count for f in spec.features] == [1, 4, 2]


def test_dedup_keeps_distinct_patterns():
    from cadx.agent.steps.intake import dedup_spec_features

    spec = PartSpec(
        name="two patterns",
        length_x={"value": 80},
        width_y={"value": 40},
        height_z={"value": 40},
        features=[
            _feat("row A", "holes along y at x=±10, z=5", count=2, dia=4.0),
            _feat("row B", "holes along y at x=±20, z=35", count=2, dia=4.0),
        ],
    )
    assert dedup_spec_features(spec) == []
    assert len(spec.features) == 2


def test_dedup_keeps_different_count_or_dia():
    from cadx.agent.steps.intake import dedup_spec_features

    spec = PartSpec(
        name="mixed",
        length_x={"value": 80},
        width_y={"value": 40},
        height_z={"value": 40},
        features=[
            _feat("a", "holes along y at x=±10", count=2, dia=4.0),
            _feat("b", "holes along y at x=±10", count=4, dia=4.0),
            _feat("c", "holes along y at x=±10", count=2, dia=6.0),
        ],
    )
    assert dedup_spec_features(spec) == []
    assert len(spec.features) == 3


def test_stated_dim_wrong_axis_downgraded():
    """'~20 mm deep' is a real stated value — but assigned to X while the
    spec's own features declare x=±15.5 (need ≥31). The part built x=48
    correctly; the stated dim is a wrong-axis assignment, not an envelope."""
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="nema",
        length_x={"value": 20, "source": "stated"},
        width_y={"value": 24},
        height_z={"value": 52},
        features=[
            _feat("pilot", "through-hole along y, centred at x=0, z=26", dia=22.0),
            _feat(
                "M3",
                "through-holes along y at x=±15.5, z=26±15.5",
                count=4,
                dia=3.4,
            ),
            _feat("feet", "through-holes along z at y=10, x=±15", count=2, dia=4.5),
        ],
    )
    notes = sanitize_stated_dims(spec, "mount ~20 mm deep foot flange")
    assert spec.length_x.source == "inferred"
    assert notes and "wrong-axis" in notes[0]


def test_stated_dim_kept_when_features_fit():
    from cadx.agent.steps.intake import sanitize_stated_dims

    spec = PartSpec(
        name="ok",
        length_x={"value": 48, "source": "stated"},
        width_y={"value": 24},
        height_z={"value": 52},
        features=[
            _feat("M3", "holes along y at x=±15.5, z=26±15.5", count=4, dia=3.4),
        ],
    )
    assert sanitize_stated_dims(spec, "mounting plate 48 mm wide, 20 mm deep foot") == []
    assert spec.length_x.source == "stated"


def test_dedup_keeps_enumerated_instances_without_coords():
    """Tank run: 'foot lug 1/2/3' + 'foot bolt hole 1/2/3' — real placed
    instances with unparseable position text (empty shared axes) were
    collapsed to one feature each by the empty-all() bug."""
    from cadx.agent.steps.intake import dedup_spec_features

    spec = PartSpec(
        name="tank",
        length_x={"value": 90},
        width_y={"value": 90},
        height_z={"value": 120},
        features=[
            _feat("foot lug 1", "radial lug plate on the lower shell, angular position A", count=1),
            _feat("foot lug 2", "radial lug plate on the lower shell, angular position B", count=1),
            _feat("foot lug 3", "radial lug plate on the lower shell, angular position C", count=1),
            _feat("foot bolt hole 1", "through-Z bolt hole in lug plate A", count=1, dia=6.5),
            _feat("foot bolt hole 2", "through-Z bolt hole in lug plate B", count=1, dia=6.5),
            _feat("foot bolt hole 3", "through-Z bolt hole in lug plate C", count=1, dia=6.5),
        ],
    )
    assert dedup_spec_features(spec) == []
    assert len(spec.features) == 6


def test_dedup_still_merges_redeclared_pattern_without_coords():
    from cadx.agent.steps.intake import dedup_spec_features

    spec = PartSpec(
        name="nema",
        length_x={"value": 48},
        width_y={"value": 24},
        height_z={"value": 52},
        features=[
            _feat("M3 bolt circle", "through-holes along y on a square pattern", count=4, dia=3.4),
            _feat(
                "mounting hole group",
                "through-holes along y, positions per datasheet",
                count=4,
                dia=3.4,
            ),
        ],
    )
    notes = dedup_spec_features(spec)
    assert len(notes) == 1
    assert len(spec.features) == 1
