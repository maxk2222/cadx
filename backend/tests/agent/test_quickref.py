"""quickref — runtime-introspected signature digests (no LLM involved).

The digests must reflect the INSTALLED build123d/bd_warehouse/cadlib
packages, so assertions here pin real API surface (names, params, enum
members) rather than hand-maintained reference text.
"""

from __future__ import annotations

from typing import Any

import pytest

from cadx.agent import quickref as qr
from cadx.agent.toolspecs import INFO_TOOLS, ToolSession


def test_build123d_topic_core_signatures() -> None:
    text = qr.quickref("build123d")
    assert "Box(" in text
    assert "Cylinder(" in text
    assert "extrude(" in text
    assert "fillet(" in text
    assert "BuildPart(" in text
    # enum members are the most hallucinated surface
    assert "SUBTRACT" in text
    assert "Mode" in text
    assert "Align" in text


def test_cadlib_topic_lists_every_helper() -> None:
    text = qr.quickref("cadlib")
    for helper in ("through_hole", "safe_fillet", "filleted_box", "mounting_ear"):
        assert f"{helper}(" in text


def test_bd_warehouse_topic_classes_and_sizes() -> None:
    pytest.importorskip("bd_warehouse.fastener")
    text = qr.quickref("bd_warehouse")
    assert "CounterSunkScrew(" in text
    assert "fastener_type" in text
    assert "iso10642" in text
    assert "sizes[" in text  # deduped Class.sizes() tables per module


def test_dotted_path_function() -> None:
    text = qr.quickref("build123d.extrude")
    assert text.startswith("build123d.extrude(")
    assert "amount" in text


def test_dotted_path_class_and_module() -> None:
    pytest.importorskip("bd_warehouse.fastener")
    cls = qr.quickref("bd_warehouse.fastener.CounterSunkScrew")
    assert cls.startswith("bd_warehouse.fastener.CounterSunkScrew(")
    assert "size" in cls
    mod = qr.quickref("bd_warehouse.fastener")
    assert "CounterSunkScrew(" in mod
    cad = qr.quickref("cadlib.through_hole")
    assert cad.startswith("cadlib.through_hole(")


def test_dotted_path_bad_attribute() -> None:
    text = qr.quickref("build123d.NoSuchThingXYZ")
    assert "cannot resolve" in text


def test_unknown_topic_lists_valid_ones() -> None:
    for bad in ("nonsense", "", "os.system"):
        text = qr.quickref(bad)
        assert "unknown quickref topic" in text
        for t in qr.TOPICS:
            assert t in text


def test_results_are_cached_by_topic_and_versions() -> None:
    first = qr.quickref("build123d")
    key = ("build123d", qr._versions())
    assert key in qr._CACHE
    assert qr.quickref("build123d") is first  # identical cached object


def test_tool_schema_registered_in_info_tools() -> None:
    spec = next(t for t in INFO_TOOLS if t["function"]["name"] == "quickref")
    params = spec["function"]["parameters"]
    assert params["required"] == ["topic"]
    assert "topic" in params["properties"]


def _ctx(**overrides) -> Any:
    import uuid

    from cadx.agent.types import AgentCtx
    from cadx.db.models import ProjectSettings
    from cadx.runtime_config import RuntimeSettings

    kw = {
        "llm": None,
        "executor": None,
        "trace": None,
        "emit": None,
        "check_cancel": None,
        "job_id": "j",
        "project_id": None,
        "version_id": uuid.uuid4(),
        "mode": "generate",
        "prompt": "p",
        "settings": ProjectSettings(),
        "runtime_settings": RuntimeSettings(**overrides.pop("rs", {})),
    }
    kw.update(overrides)
    return AgentCtx(**kw)


def test_quickref_offered_in_code_edit_info_sessions() -> None:
    ctx = _ctx()
    for kind in ("edit", "codegen", "info", "inspect"):
        assert "quickref" in ToolSession(ctx, kind=kind).allowed_names()


def test_codegen_user_parts_gated_by_setting() -> None:
    from cadx.agent.steps.codegen import _user_parts
    from cadx.llm.schemas import PartSpec

    dim = {"value": 60.0, "unit": "mm", "source": "stated"}
    spec = PartSpec.model_validate(
        {
            "name": "part",
            "summary": "s",
            "units": "mm",
            "length_x": dim,
            "width_y": dim,
            "height_z": dim,
            "image_roles": [],
            "features": [],
        }
    )

    def _texts(ctx) -> str:
        return "\n".join(p.get("text", "") for p in _user_parts(ctx, spec, None, None))

    assert "API QUICKREF" not in _texts(_ctx())  # default off
    assert "API QUICKREF" in _texts(_ctx(rs={"agent_quickref_context": True}))


def test_tool_handler_dispatch() -> None:
    # _tool_quickref touches no session state — a bare self suffices
    res = ToolSession._tool_quickref(None, {"topic": "build123d"})  # type: ignore[arg-type]
    assert res["ok"] is True
    assert "Box(" in res["reference"]
    res = ToolSession._tool_quickref(None, {"topic": "nonsense"})  # type: ignore[arg-type]
    assert res["ok"] is False
    assert "Valid topics" in res["error"]


def test_context_block_is_bounded() -> None:
    text = qr.context_block()
    assert 0 < len(text) <= 2300
    assert "through_hole(" in text  # cadlib section
    assert "Mode" in text  # build123d enum surface survives truncation


def test_c_extension_objects_without_signature() -> None:
    # objects with no retrievable signature degrade to "(…)", never raise
    import math

    text = qr._signature(math.gcd)  # builtin — C impl may or may not have a sig
    assert text.startswith("(") and text.endswith(")")
