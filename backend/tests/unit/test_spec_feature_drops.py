from __future__ import annotations

import pytest

from cadx.agent.specdrops import drop_spec_feature
from cadx.cad.spec_features import active_feature_rows, dropped_feature_indices
from cadx.cad.verifyspec import verify_spec
from cadx.llm.schemas import Dim, Feature, PartSpec


def _spec() -> PartSpec:
    return PartSpec(
        name="mount",
        summary="",
        length_x=Dim(value=100, source="stated"),
        width_y=Dim(value=80, source="stated"),
        height_z=Dim(value=40, source="stated"),
        features=[
            Feature(name="bolt holes", type="hole", dims={"diameter": 8.5}),
            Feature(name="small fillet", type="fillet", dims={"fillet_radius": 2.0}),
        ],
    )


def test_feature_drop_is_hidden_from_llm_schema_and_survives_json_roundtrip() -> None:
    spec = _spec()
    result = drop_spec_feature(
        spec, 1, "kernel cannot build this edge round", iteration=3, session="codegen"
    )

    assert result["ok"] is True
    schema = PartSpec.model_json_schema()
    assert "feature_drops" not in schema.get("properties", {})

    restored = PartSpec.model_validate(spec.model_dump(mode="json"))
    assert len(restored.feature_drops) == 1
    assert restored.feature_drops[0].reason == "kernel cannot build this edge round"
    assert dropped_feature_indices(restored) == {1}
    assert [i for i, _feature in active_feature_rows(restored)] == [0]


def test_drop_requires_a_valid_unique_active_index_and_reason() -> None:
    spec = _spec()
    assert drop_spec_feature(spec, True, "valid enough reason", iteration=0, session="codegen")[
        "ok"
    ] is False
    assert drop_spec_feature(spec, 4, "valid enough reason", iteration=0, session="codegen")[
        "ok"
    ] is False
    assert drop_spec_feature(spec, 1, "too short", iteration=0, session="codegen")["ok"] is False
    assert drop_spec_feature(spec, 1, "not enough reason", iteration=0, session="codegen")["ok"]
    assert drop_spec_feature(spec, 1, "another detailed reason", iteration=1, session="repair")[
        "ok"
    ] is False


def test_verifyspec_skips_active_checks_but_reports_drops() -> None:
    spec = _spec()
    drop_spec_feature(
        spec, 0, "not present on the supplied reference mesh", iteration=1, session="codegen"
    )

    report = verify_spec(
        spec.model_dump(mode="json"),
        {"ok": True, "analysis": {}, "features": {}},
        feature_findings=[],
    )

    assert report["dropped_features"][0]["feature"]["name"] == "bolt holes"
    # the dropped bore (index 0) emits no per-feature checks; the active
    # fillet keeps its original spec index (feat_1_*), not a renumbered 0
    assert not any(c["name"].startswith("feat_0") for c in report["checks"])


@pytest.mark.asyncio
async def test_tool_session_drop_updates_submission_audit() -> None:
    from cadx.agent.toolspecs import ToolSession
    from cadx.llm.fake import FakeLLM
    from tests.unit.test_session_condense import _ctx

    spec = _spec()
    ctx = _ctx(FakeLLM([]))
    session = ToolSession(
        ctx,
        kind="codegen",
        spec_features=[
            {"index": index, "type": feature.type, "name": feature.name}
            for index, feature in active_feature_rows(spec)
        ],
        spec_drop_fn=lambda index, reason: drop_spec_feature(
            spec, index, reason, iteration=1, session="codegen"
        ),
    )

    assert "drop_spec_feature" in session.allowed_names()
    result = await session.execute(
        "drop_spec_feature",
        {"feature_index": 1, "reason": "cannot construct this concave fillet"},
    )
    assert result["ok"] is True
    assert [feature["name"] for feature in session.spec_features] == ["bolt holes"]


def test_tool_feature_audit_does_not_require_a_dropped_feature() -> None:
    from cadx.agent.toolspecs import feature_audit_missing

    spec = _spec()
    drop_spec_feature(spec, 1, "unsupported fillet geometry", iteration=1, session="codegen")
    active = [
        {"type": feature.type, "name": feature.name}
        for _index, feature in active_feature_rows(spec)
    ]

    assert feature_audit_missing("result = Box(10, 10, 10)", active) == []
