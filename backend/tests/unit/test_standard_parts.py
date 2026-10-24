from __future__ import annotations

import pytest

from cadx.cad import catalog
from cadx.cad.allowlist import check_code
from cadx.cad.partspec import check_standard_part_calls
from cadx.cad.standard_parts import (
    build_standard_part_code,
    expected_thread_count,
    is_standard_only,
    standard_part_bbox,
)
from cadx.llm.schemas import Dim, Feature, PartSpec, StandardPartSpec


def _dims(length: float, width: float, height: float) -> tuple[Dim, Dim, Dim]:
    return (
        Dim(value=length, source="stated"),
        Dim(value=width, source="inferred"),
        Dim(value=height, source="inferred"),
    )


def test_m10_catalog_recipe_is_sandbox_code_and_norm_sized() -> None:
    part = StandardPartSpec(
        class_name="CounterSunkScrew",
        fastener_type="iso10642",
        size="M10-1.5",
        length_mm=150,
        covers_features=["head", "thread"],
    )
    spec = PartSpec(
        name="M10 countersunk screw",
        summary="M10x150 ISO 10642",
        length_x=_dims(20, 20, 150)[0],
        width_y=_dims(20, 20, 150)[1],
        height_z=_dims(20, 20, 150)[2],
        features=[
            Feature(name="head", type="cone"),
            Feature(name="thread", type="thread"),
        ],
        standard_parts=[part],
    )
    code = build_standard_part_code(spec.standard_parts)
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert check_standard_part_calls(code) == []
    assert is_standard_only(spec)
    # frame contract: standalone parts rest on XY with their axis up (+z)
    assert standard_part_bbox(part) == pytest.approx([20.0, 20.0, 150.0], abs=0.01)
    assert expected_thread_count(spec) == 1
    # LoD split: simple body always, real thread ridge only in export mode
    # (swept-helix block; bd simple=False as its fallback lane)
    assert 'CADX_DETAIL_MODE == "export"' in code
    assert "simple=True" in code
    assert "'length': 32.0" in code


def test_standard_thread_feature_does_not_hide_length_conflict() -> None:
    part = StandardPartSpec(
        class_name="CounterSunkScrew",
        fastener_type="iso10642",
        size="M10-1.5",
        length_mm=150,
        covers_features=["full-length thread"],
    )
    spec = PartSpec(
        name="long threaded screw",
        summary="M10 thread for full length",
        length_x=_dims(150, 20, 20)[0],
        width_y=_dims(150, 20, 20)[1],
        height_z=_dims(150, 20, 20)[2],
        features=[
            Feature(
                name="full-length thread",
                type="threaded rod",
                dims={"length": 150, "thread_pitch": 1.5},
                thread_spec="M10-1.5",
            )
        ],
        standard_parts=[part],
    )
    assert not is_standard_only(spec)


def test_catalog_lookup_supports_parametric_classes_and_rejects_unknown_args() -> None:
    gear = catalog.lookup_part(
        "SpurGear",
        parameters={
            "module": 2.0,
            "tooth_count": 18,
            "pressure_angle": 20.0,
            "thickness": 12.0,
        },
    )
    assert gear["bbox_size_mm"][2] == pytest.approx(12.0, abs=1e-3)
    try:
        catalog.lookup_part("SpurGear", parameters={"made_up": 2})
    except ValueError as exc:
        assert "unsupported parameter" in str(exc)
    else:
        raise AssertionError("unknown constructor arguments must be rejected")


def test_standard_pipe_recipe_uses_a_validated_path_length() -> None:
    part = StandardPartSpec(
        class_name="Pipe",
        length_mm=50,
        parameters={"nps": "1", "material": "steel", "identifier": "40"},
    )
    code = build_standard_part_code([part])
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert "Edge.make_line" in code
    assert check_standard_part_calls(code) == []


def test_standard_recipe_rejects_machining_helpers_as_parts() -> None:
    part = StandardPartSpec(class_name="InsertHole")
    try:
        build_standard_part_code([part])
    except ValueError as exc:
        assert "not a standalone part" in str(exc)
    else:
        raise AssertionError("context machining operations are not standalone parts")


async def test_standard_only_spec_uses_validated_recipe_without_llm():
    import uuid

    from cadx.agent.executor_client import InProcessExecutor
    from cadx.agent.steps import codegen as codegen_step
    from cadx.agent.trace import MemoryTrace
    from cadx.agent.types import AgentCtx
    from cadx.db.models import ProjectSettings
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    llm = FakeLLM([])

    async def emit(*_a, **_k) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="t",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="M8 socket head screw",
        mode="generate",
        settings=ProjectSettings(),
        runtime_settings=RuntimeSettings(),
    )
    part = StandardPartSpec(
        class_name="SocketHeadCapScrew",
        fastener_type="iso4762",
        size="M8-1.25",
        length_mm=30,
        covers_features=["head", "thread"],
    )
    spec = PartSpec(
        name="M8 socket head screw",
        summary="ISO 4762 M8x30",
        length_x=Dim(value=30, source="stated"),
        width_y=Dim(value=13, source="inferred"),
        height_z=Dim(value=13, source="inferred"),
        features=[Feature(name="head", type="cylinder"), Feature(name="thread", type="thread")],
        standard_parts=[part],
    )
    resp = await codegen_step.run(ctx, spec, None, iteration=1)
    assert "SocketHeadCapScrew" in resp.code
    assert "CADX_DETAIL_MODE" in resp.code
    assert resp.expected_bbox == [30.0, 13.0, 13.0]
    assert not llm.calls  # the validated recipe path never consumed the model


def test_expected_thread_count_ignores_clearance_fit_notes() -> None:
    """'clearance for M8' fills thread_spec as a fit note, not a thread
    callout — the robot_arm_mount eval failed because it counted 7
    'threads' (4x M8 + 2x M6 clearance + 1 real M12 tap)."""
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    spec = PartSpec(
        name="bracket",
        summary="",
        units="mm",
        length_x=d(100),
        width_y=d(80),
        height_z=d(40),
        features=[
            Feature(
                name="mounting_holes",
                type="through_hole",
                count=4,
                thread_spec="clearance for M8",
            ),
            Feature(
                name="m12_tapped_bore",
                type="threaded_hole",
                thread_spec="M12x1.75 ISO 6H,",
            ),
            Feature(
                name="flange_holes",
                type="through_hole",
                count=2,
                thread_spec="clearance for M6",
            ),
        ],
    )
    assert expected_thread_count(spec) == 1


def test_expected_thread_count_clearance_dia_disambiguates_m_spec() -> None:
    """Intake may write a valid-looking thread_spec ("M8x1.25-6H") onto a
    hole whose declared diameter is the M8 CLEARANCE size (8.5), not the
    tap drill (6.8) — the bolt's thread named, not a thread to build."""
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    spec = PartSpec(
        name="bracket",
        summary="",
        units="mm",
        length_x=d(100),
        width_y=d(80),
        height_z=d(40),
        features=[
            Feature(
                name="mounting holes",
                type="through_hole",
                count=4,
                thread_spec="M8x1.25-6H",
                dims={"diameter": 8.5},
            ),
            Feature(
                name="tapped bore",
                type="blind_hole",
                thread_spec="M12x1.75-6H",
                dims={"diameter": 10.2},
            ),
        ],
    )
    assert expected_thread_count(spec) == 1


def _m10_100_spec(**part_kwargs) -> PartSpec:
    part = StandardPartSpec(
        class_name="CounterSunkScrew",
        fastener_type="iso10642",
        size="M10-1.5",
        length_mm=100,
        covers_features=["head", "thread"],
        **part_kwargs,
    )
    d = lambda v: Dim(value=v, unit="mm", source="stated")  # noqa: E731
    return PartSpec(
        name="M10 countersunk screw",
        summary="M10x100 ISO 10642 full thread",
        length_x=d(20),
        width_y=d(20),
        height_z=d(100),
        features=[Feature(name="head", type="cone"), Feature(name="thread", type="thread")],
        standard_parts=[part],
    )


def test_full_thread_request_overrides_norm_reach() -> None:
    """The M10x100 failure mode: 'Vollgewinde' silently produced the ISO 888
    default b=2d+6=32 mm. `full_thread` must emit a calculate_thread_length
    override returning max_thread_length (=94.5: shank minus head)."""
    spec = _m10_100_spec(full_thread=True)
    code = build_standard_part_code(spec.standard_parts)
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert check_standard_part_calls(code) == []
    assert "def calculate_thread_length" in code
    assert "return self.max_thread_length" in code
    assert is_standard_only(spec)
    from cadx.cad.standard_parts import validate_standard_parts

    thread = validate_standard_parts(spec.standard_parts)[0]["thread"]
    assert thread["length"] == pytest.approx(94.5, abs=0.01)
    assert thread["thread_length_requested"] == "full"
    assert thread["thread_length_default"] == pytest.approx(32.0, abs=0.01)
    assert thread["max_thread_length"] == pytest.approx(94.5, abs=0.01)


def test_numeric_thread_length_emits_clamped_override() -> None:
    spec = _m10_100_spec(thread_length_mm=50)
    code = build_standard_part_code(spec.standard_parts)
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert "min(PARAMS[" in code and "self.max_thread_length" in code
    from cadx.cad.standard_parts import validate_standard_parts

    thread = validate_standard_parts(spec.standard_parts)[0]["thread"]
    assert thread["length"] == pytest.approx(50.0, abs=0.01)
    assert thread["thread_length_requested"] == 50.0


def test_thread_length_in_parameters_is_recipe_level_not_ctor_kwarg() -> None:
    """Models may write thread_length inside `parameters` — it must reach the
    method override, never the bd_warehouse constructor (unsupported kwarg)."""
    spec = _m10_100_spec(parameters={"thread_length": 60})
    code = build_standard_part_code(spec.standard_parts)
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert "thread_length=PARAMS" not in code


def test_thread_length_request_rejected_on_non_screw() -> None:
    part = StandardPartSpec(
        class_name="PlainWasher", fastener_type="iso7093", size="M10", full_thread=True
    )
    with pytest.raises(ValueError, match="only applies to screw parts"):
        build_standard_part_code([part])


def test_same_class_parts_with_distinct_thread_requests_get_own_subclass() -> None:
    from cadx.cad.standard_parts import validate_standard_parts

    parts = [
        StandardPartSpec(
            class_name="HexHeadScrew",
            fastener_type="iso4017",
            size="M8-1.25",
            length_mm=40,
            covers_features=["t1"],
            full_thread=True,
        ),
        StandardPartSpec(
            class_name="HexHeadScrew",
            fastener_type="iso4017",
            size="M8-1.25",
            length_mm=40,
            position_mm=[30, 0, 0],
            covers_features=["t2"],
            thread_length_mm=20,
        ),
    ]
    code = build_standard_part_code(parts)
    checked = check_code(code)
    assert checked.ok, checked.errors
    assert "class HexHeadScrew(" in code
    assert "class HexHeadScrew_2(" in code
    threads = [row["thread"]["length"] for row in validate_standard_parts(parts)]
    assert threads == pytest.approx([40.0, 20.0], abs=0.01)


def test_resolve_thread_intents_maps_vollgewinde_wording() -> None:
    from cadx.agent.steps.intake import resolve_thread_intents

    spec = _m10_100_spec()
    spec.features[1].description = "Vollgewinde über die gesamte Länge"
    notes = resolve_thread_intents(spec)
    assert spec.standard_parts[0].full_thread is True
    assert any("full" in n.lower() or "voll" in n.lower() for n in notes)
    # an untouched part keeps the catalog default — no knob, no note
    spec2 = _m10_100_spec()
    spec2.features[1].description = "ISO metric thread"
    assert resolve_thread_intents(spec2) == []
    assert spec2.standard_parts[0].full_thread is None


def test_resolve_thread_intents_ignores_ambiguous_features() -> None:
    """A thread feature owned by NO part's covers_features must not drive a
    screw-level override once a second screw exists — 'Vollgewinde' would be
    ambiguous between them and is better handled downstream than guessed."""
    from cadx.agent.steps.intake import resolve_thread_intents

    spec = _m10_100_spec()
    spec.standard_parts[0].covers_features = ["head"]
    spec.standard_parts.append(
        StandardPartSpec(
            class_name="SocketHeadCapScrew",
            fastener_type="iso4762",
            size="M8-1.25",
            length_mm=30,
            position_mm=[30, 0, 0],
            covers_features=["other head"],
        )
    )
    spec.features[1].description = "Vollgewinde"
    resolve_thread_intents(spec)
    assert spec.standard_parts[0].full_thread is None
    assert spec.standard_parts[1].full_thread is None


def test_resolve_thread_intents_uses_intrinsic_attribution() -> None:
    """...but with a single screw, an intrinsically-covered thread feature
    belongs to it even when covers_features forgot to list it."""
    from cadx.agent.steps.intake import resolve_thread_intents

    spec = _m10_100_spec()
    spec.standard_parts[0].covers_features = ["head"]
    spec.features[1].description = "Vollgewinde"
    resolve_thread_intents(spec)
    assert spec.standard_parts[0].full_thread is True
