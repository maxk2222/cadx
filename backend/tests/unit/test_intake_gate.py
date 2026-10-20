"""Degenerate-spec gate: tool-echo names + dropped prompt features (T1.2)."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import run_generation
from cadx.agent.steps.intake import spec_gaps
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import PartSpec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _spec(name: str = "plate", features: list | None = None) -> PartSpec:
    return PartSpec(
        name=name,
        summary="a plate" if name == "plate" else "",
        length_x={"value": 60.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 4.0, "source": "stated"},
        features=features or [],
        image_roles=[],
    )


def test_spec_gaps_tool_echo_name() -> None:
    gaps = spec_gaps(_spec(name="search_catalog", features=[]), "plain block")
    assert any("tool echo" in g for g in gaps)


def test_spec_gaps_catalog_summary() -> None:
    s = _spec()
    s.summary = "catalog search for parts"
    assert any("tool echo" in g for g in spec_gaps(s, "plain block"))


def test_spec_gaps_prompt_features_dropped() -> None:
    gaps = spec_gaps(_spec(features=[]), "Platte mit 2 Bohrungen und einer Fase")
    assert any("spec.features is empty" in g for g in gaps)


def test_spec_gaps_clean() -> None:
    assert spec_gaps(_spec(), "a simple smooth box") == []


def test_spec_gaps_features_present_ok() -> None:
    s = _spec(features=[{"name": "hole", "type": "hole"}])
    assert spec_gaps(s, "Platte mit Bohrung") == []


def _ctx(llm: FakeLLM, prompt: str, **kw) -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    kw.setdefault("settings", ProjectSettings(clarification_mode="assume", max_iterations=1))
    return AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt=prompt,
        gen_mode="cad",
        assets=[],
        **kw,
    )


async def test_degenerate_intake_triggers_feature_reask():
    """Intake emits a tool-echo spec with no features; the gate re-asks and
    recovers them before planning."""
    box_code = (FIXTURES / "simple_box.py").read_text()
    llm = FakeLLM(
        [
            # 1) intake: degenerate spec (tool echo, no features)
            {
                "name": "search_catalog",
                "summary": "catalog search",
                "units": "mm",
                "length_x": {"value": 60.0, "source": "stated"},
                "width_y": {"value": 40.0, "source": "stated"},
                "height_z": {"value": 4.0, "source": "stated"},
                "features": [],
                "open_questions": [],
            },
            # 2) intake_features re-ask: recover the holes
            {
                "name": "bracket plate",
                "summary": "plate with two holes",
                "features": [
                    {
                        "name": "holes",
                        "type": "hole",
                        "description": "two through holes",
                        "dims": {"diameter": 5.5},
                        "position": "in foot",
                        "count": 2,
                        "source": "stated",
                    }
                ],
            },
            # 3) plan
            {"approach": "cad", "approach_reason": "r", "steps": [], "params": []},
            # 4) codegen
            {
                "reasoning": "box",
                "params": [],
                "expected_bbox": [60, 40, 4],
                "code": box_code,
            },
            # 5) critique
            {
                "score": 30,
                "passed": False,
                "discrepancies": [
                    {"what": "missing holes", "severity": "major", "fix_hint": "add holes"}
                ],
                "matches": [],
                "confidence": 0.8,
            },
        ]
    )
    emits: list[tuple[str, str]] = []

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        emits.append((step, msg))

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt="Platte 60x40x4 mit 2 Bohrungen",
        settings=ProjectSettings(clarification_mode="assume", max_iterations=1),
        gen_mode="cad",
        assets=[],
    )
    outcome = await run_generation(ctx)
    spec = outcome.spec or {}
    assert len(spec.get("features") or []) == 1
    assert spec.get("features", [{}])[0]["type"] == "hole"
    assert spec.get("name") == "bracket plate"
    assert any("recovered" in m for _s, m in emits)


def _plate90(features) -> PartSpec:
    return PartSpec(
        name="lever",
        summary="",
        length_x={"value": 90.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 14.0, "source": "stated"},
        features=features,
        image_roles=[],
    )


def test_normalize_positions_rewrites_contradicting_coord() -> None:
    from cadx.agent.steps.intake import normalize_feature_positions
    from cadx.llm.schemas import Feature

    s = _plate90(
        [
            Feature(
                name="bore 2",
                type="bore",
                dims={"diameter": 10.0},
                position="75 mm from -x end, centered in y (x=20, y=0)",
            )
        ]
    )
    notes = normalize_feature_positions(s)
    assert notes
    assert "x=30" in s.features[0].position


def test_normalize_positions_keeps_consistent() -> None:
    from cadx.agent.steps.intake import normalize_feature_positions
    from cadx.llm.schemas import Feature

    s = _plate90(
        [
            Feature(
                name="bore 1",
                type="bore",
                dims={"diameter": 10.0},
                position="20 mm from -x end, centered in y (x=-25, y=0)",
            )
        ]
    )
    notes = normalize_feature_positions(s)
    assert not notes
    assert s.features[0].position == "20 mm from -x end, centered in y (x=-25, y=0)"


def test_normalize_positions_multi_offset_untouched() -> None:
    from cadx.agent.steps.intake import normalize_feature_positions
    from cadx.llm.schemas import Feature

    s = _plate90(
        [
            Feature(
                name="bores",
                type="bore",
                dims={"diameter": 10.0},
                count=2,
                position="20 and 75 mm from -x end, centered in y (x=5, y=0)",
            )
        ]
    )
    notes = normalize_feature_positions(s)
    assert not notes


def _handle_spec() -> PartSpec:
    """The observed mis-parse: '120 x 40 x 30' -> height_z=120 (echoing
    length_x) while every feature says z <= 30."""
    return PartSpec(
        name="Tragegriff",
        summary="arched carry handle",
        length_x={"value": 120.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 120.0, "source": "stated"},
        features=[
            {
                "name": "arch",
                "type": "arch",
                "dims": {"height": 30},
                "position": "zentriert, x=-45 bis +45, Scheitel bei z=30",
            },
            {
                "name": "tab -x",
                "type": "boss",
                "dims": {"length": 30, "thickness": 6},
                "position": "x=-60 bis -30, y=-10 bis +10, z=0 bis 6",
            },
            {
                "name": "tab +x",
                "type": "boss",
                "dims": {"length": 30, "thickness": 6},
                "position": "x=+30 bis +60, y=-10 bis +10, z=0 bis 6",
            },
        ],
        image_roles=[],
    )


def test_dim_echo_recovers_misread_height() -> None:
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    notes = sanitize_dim_echo(s, "Bügel ca. 120 x 40 x 30 mm")
    assert notes
    assert s.height_z.value == 30.0
    # x stays: feature extents -60..+60 match the declared 120
    assert s.length_x.value == 120.0


def test_dim_echo_ignores_consistent_dims() -> None:
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    s.height_z = s.height_z.model_copy(update={"value": 30.0})
    assert sanitize_dim_echo(s, "Bügel ca. 120 x 40 x 30 mm") == []


def test_dim_echo_needs_the_echo_signature() -> None:
    """A unique large dim is NOT rewritten — features legitimately may not
    span the full envelope."""
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    s.height_z = s.height_z.model_copy(update={"value": 95.0})  # unique, no echo
    assert sanitize_dim_echo(s, "Bügel ca. 120 x 40 x 30 mm") == []


def test_dim_echo_prefers_unmatched_prompt_value() -> None:
    """'120 x 40 x 90' mis-parsed as z=120: the prompt's own 90 is stronger
    evidence than the features' 30 — the prompt is ground truth."""
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    notes = sanitize_dim_echo(s, "Bügel ca. 120 x 40 x 90 mm")
    assert notes
    assert s.height_z.value == 90.0


def test_dim_echo_no_mm_prompt_uses_feature_extent() -> None:
    """A dim-free prompt gives nothing to check against — the spec's own
    feature extent is the best remaining evidence."""
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    notes = sanitize_dim_echo(s, "ein Griff mit Laschen")
    assert notes
    assert s.height_z.value == 30.0


def test_dim_echo_leaves_ambiguous_contradiction() -> None:
    """Prompt states dims, the suggested extent is none of them and no
    orphan value exists — rewriting would trade a guess for a guess."""
    from cadx.agent.steps.intake import sanitize_dim_echo

    s = _handle_spec()
    # prompt claims 70 for z; the feature extent 30 has no footing, and 70
    # is itself unmatched so it becomes the orphan -> rewritten to 70
    notes = sanitize_dim_echo(s, "Bügel ca. 120 x 40 x 70 mm")
    assert notes
    assert s.height_z.value == 70.0


def test_mirror_host_gap_detects_dropped_hole() -> None:
    """Observed E2E: intake emitted lugs at x=±45 but only the -x through-
    hole — 'je eine Bohrung' makes the +x hole a real spec gap."""
    s = PartSpec(
        name="Tragegriff",
        summary="handle",
        length_x={"value": 120.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
        features=[
            {"type": "arch", "name": "bail", "position": "x=-45 to +45, apex z=30"},
            {"type": "pad", "name": "lug -x", "position": "centered at x=-45, y=0, z=0..6"},
            {"type": "pad", "name": "lug +x", "position": "centered at x=+45, y=0, z=0..6"},
            {
                "type": "through_hole",
                "name": "bore -x",
                "dims": {"diameter": 5.5},
                "position": "x=-45, y=0, through z=0..6",
            },
        ],
        image_roles=[],
    )
    gaps = spec_gaps(s, "Tragegriff mit je einer Durchgangsbohrung in beiden Laschen")
    assert any("appears missing" in g for g in gaps)


def test_mirror_host_gap_silent_without_quantifier() -> None:
    """Same spec, but the prompt never promised a hole per side — an
    asymmetric design must not trip the gap."""
    s = PartSpec(
        name="h",
        summary="",
        length_x={"value": 120.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
        features=[
            {"type": "pad", "name": "lug -x", "position": "centered at x=-45"},
            {"type": "pad", "name": "lug +x", "position": "centered at x=+45"},
            {
                "type": "through_hole",
                "name": "bore -x",
                "dims": {"diameter": 5.5},
                "position": "x=-45, y=0",
            },
        ],
        image_roles=[],
    )
    assert spec_gaps(s, "Griff mit einer Bohrung") == []


def test_mirror_host_gap_silent_when_both_sides_present() -> None:
    s = PartSpec(
        name="h",
        summary="",
        length_x={"value": 120.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 30.0, "source": "stated"},
        features=[
            {"type": "pad", "name": "lug -x", "position": "x=-45"},
            {"type": "pad", "name": "lug +x", "position": "x=+45"},
            {
                "type": "through_hole",
                "name": "bore -x",
                "dims": {"diameter": 5.5},
                "position": "x=-45",
            },
            {
                "type": "through_hole",
                "name": "bore +x",
                "dims": {"diameter": 5.5},
                "position": "x=+45",
            },
        ],
        image_roles=[],
    )
    assert spec_gaps(s, "je eine Bohrung in beiden Laschen") == []
