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
        length_x=_dims(150, 20, 20)[0],
        width_y=_dims(150, 20, 20)[1],
        height_z=_dims(150, 20, 20)[2],
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
    assert standard_part_bbox(part) == pytest.approx([150.0, 20.0, 20.0], abs=0.01)
    assert expected_thread_count(spec) == 1
    assert 'CADX_DETAIL_MODE == "preview"' in code
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
