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
