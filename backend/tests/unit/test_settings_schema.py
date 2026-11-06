"""Settings API surface stays in lockstep with RuntimeSettings.

SettingsOut/SettingsUpdate are derived from the runtime config model — these
tests pin that derivation so a new RuntimeSettings field can never again
silently 422 on PUT or go missing from GET (the drift that blocked the
agent_codegen_samples eval variant).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from cadx.api.schemas import SettingsOut, SettingsUpdate
from cadx.db.session import create_all
from cadx.runtime_config import _SECRET_FIELDS, RuntimeSettings


@pytest.fixture(scope="module")
async def client():
    await create_all()
    from cadx.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# SettingsOut-only fields a client may legitimately echo back on PUT —
# declared on SettingsUpdate so extra="forbid" accepts them, ignored by
# the router (read-only computed values, never persisted)
_ECHO_FIELDS = {"data_dir", "resources_detected", "llm_detected"} | {
    f"{n}_set" for n in _SECRET_FIELDS
}


def test_update_schema_covers_runtime_fields():
    missing = set(RuntimeSettings.model_fields) - set(SettingsUpdate.model_fields)
    assert missing <= set(_SECRET_FIELDS)
    # every runtime field is writable; clear_* helpers and the SettingsOut
    # echo fields are the only extras
    extras = set(SettingsUpdate.model_fields) - set(RuntimeSettings.model_fields)
    assert extras == {f"clear_{n}" for n in _SECRET_FIELDS} | _ECHO_FIELDS


def test_out_schema_covers_runtime_fields():
    missing = set(RuntimeSettings.model_fields) - set(SettingsOut.model_fields)
    assert missing <= set(_SECRET_FIELDS)
    extras = set(SettingsOut.model_fields) - set(RuntimeSettings.model_fields)
    assert extras == {f"{n}_set" for n in _SECRET_FIELDS} | {
        "data_dir",
        "resources_detected",
        "llm_detected",
    }


def test_update_preserves_field_bounds():
    # agent_codegen_samples is Field(ge=1, le=8) on the runtime model — the
    # derived schema must keep the metadata, not just the type
    assert SettingsUpdate(agent_codegen_samples=3).agent_codegen_samples == 3
    with pytest.raises(ValidationError):
        SettingsUpdate(agent_codegen_samples=9)
    with pytest.raises(ValidationError):
        SettingsUpdate(fem_spsolve_dofs=0)  # ge=1_000


def test_update_forbids_unknown_and_secrets_stay_write_only():
    with pytest.raises(ValidationError):
        SettingsUpdate(nonsense_field=1)
    for name in _SECRET_FIELDS:
        assert name not in SettingsOut.model_fields or name.endswith("_set")


async def test_put_accepts_settings_out_echo(client):
    """The regression that blocked settings saves: a client round-tripping
    the GET body puts read-only echo fields (resources_detected,
    llm_detected, data_dir, *_set) into the PUT — they must be accepted and
    ignored, not 422 and never persisted."""
    g = (await client.get("/api/settings")).json()
    r = await client.put("/api/settings", json=g)
    assert r.status_code == 200, r.text
    # echo values are not writable: the fake detected value must not stick
    fake = dict(g, resources_detected={"cpus": 999}, llm_api_key_set=True)
    r = await client.put("/api/settings", json=fake)
    assert r.status_code == 200, r.text
    out = (await client.get("/api/settings")).json()
    assert out["resources_detected"].get("cpus") != 999
    assert out["llm_api_key_set"] is False


async def test_put_accepts_runtime_only_field(client):
    # the regression: agent_codegen_samples existed on RuntimeSettings (and
    # codegen reads it) but was missing from the PUT whitelist → 422
    r = await client.put("/api/settings", json={"agent_codegen_samples": 3})
    assert r.status_code == 200, r.text
    assert r.json()["agent_codegen_samples"] == 3
    r = await client.get("/api/settings")
    assert r.json()["agent_codegen_samples"] == 3
    # explicit null resets to the code default (sparse storage semantics)
    r = await client.put("/api/settings", json={"agent_codegen_samples": None})
    assert r.status_code == 200
    assert r.json()["agent_codegen_samples"] == RuntimeSettings().agent_codegen_samples


async def test_put_rejects_out_of_bounds(client):
    r = await client.put("/api/settings", json={"agent_codegen_samples": 99})
    assert r.status_code == 422


async def test_settings_response_hides_secret_values(client):
    await client.put("/api/settings", json={"llm_api_key": "sk-secret-value"})
    r = await client.get("/api/settings")
    body = r.json()
    assert body["llm_api_key_set"] is True
    assert "sk-secret-value" not in r.text
    await client.put("/api/settings", json={"clear_llm_api_key": True})
    assert (await client.get("/api/settings")).json()["llm_api_key_set"] is False
