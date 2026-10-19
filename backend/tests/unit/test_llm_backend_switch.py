"""make_llm factory: external vs sidecar backend resolution."""

from __future__ import annotations

import pytest

from cadx.llm.client import LLMClient, make_llm
from cadx.runtime_config import RuntimeSettings


def test_external_backend_uses_llm_base_url():
    rt = RuntimeSettings(llm_base_url="https://llm.example.com/v1/")
    c = make_llm(rt)
    assert isinstance(c, LLMClient)
    assert c.base_url == "https://llm.example.com/v1"


def test_sidecar_backend_uses_mesh_base_url_no_auth():
    rt = RuntimeSettings(
        llm_backend="sidecar",
        mesh_base_url="http://sidecar:8765",
        llm_sidecar_model="qwen3-local",
    )
    c = make_llm(rt)
    # the sidecar serves chat under /v1/chat/completions
    assert c.base_url == "http://sidecar:8765/v1"
    assert c.api_key == ""  # local sidecar — no auth header
    # one pinned local model drives both roles
    assert c.models["vision"] == c.models["code"] == "qwen3-local"


def test_sidecar_base_url_v1_not_doubled():
    rt = RuntimeSettings(llm_backend="sidecar", mesh_base_url="http://sidecar:8765/v1")
    assert make_llm(rt).base_url == "http://sidecar:8765/v1"


def test_sidecar_without_sidecar_model_keeps_role_models():
    rt = RuntimeSettings(
        llm_backend="sidecar",
        mesh_base_url="http://sidecar:8765",
        llm_model_vision="vis-x",
        llm_model_code="code-y",
    )
    c = make_llm(rt)
    assert c.models == {"vision": "vis-x", "code": "code-y"}


def test_sidecar_backend_requires_mesh_base_url():
    with pytest.raises(ValueError, match="local sidecar not configured"):
        make_llm(RuntimeSettings(llm_backend="sidecar"))


def test_backend_param_overrides_runtime_default():
    # runtime default is external; the per-job override switches to sidecar
    rt = RuntimeSettings(mesh_base_url="http://s:1", llm_base_url="https://ext/v1")
    assert make_llm(rt, backend="sidecar").base_url == "http://s:1/v1"
    # ... and back again on a sidecar-configured runtime
    rt2 = RuntimeSettings(
        llm_backend="sidecar", mesh_base_url="http://s:1", llm_base_url="https://ext/v1"
    )
    assert make_llm(rt2, backend="external").base_url == "https://ext/v1"


def test_unknown_backend_falls_through_to_external():
    # defensive: anything but "sidecar" resolves to the external provider
    rt = RuntimeSettings(llm_base_url="https://ext/v1")
    assert make_llm(rt, backend="external").base_url == "https://ext/v1"
