"""DB-backed runtime settings: defaults, seeding, save/load roundtrip."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from cadx.db.models import AppSetting
from cadx.db.session import create_all, session_scope
from cadx.runtime_config import (
    RuntimeSettings,
    ensure_runtime_seeded,
    load_runtime_settings,
    save_runtime_settings,
)


@pytest.fixture()
async def clean_settings():
    await create_all()
    async with session_scope() as db:
        row = await db.get(AppSetting, "runtime")
        if row is not None:
            await db.delete(row)
        await db.commit()
    yield


async def test_defaults_without_row(clean_settings):
    async with session_scope() as db:
        s = await load_runtime_settings(db)
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.llm_tools == "auto"
    assert not s.llm_api_key.get_secret_value()


async def test_save_load_roundtrip_and_secret_not_public(clean_settings):
    s = RuntimeSettings(
        llm_model_code="my-model",
        mesh_base_url="http://x:1",
        llm_api_key=SecretStr("sk-secret"),
    )
    async with session_scope() as db:
        await save_runtime_settings(db, s)
        await db.commit()
        loaded = await load_runtime_settings(db)
    assert loaded.llm_model_code == "my-model"
    assert loaded.mesh_base_url == "http://x:1"
    assert loaded.llm_api_key.get_secret_value() == "sk-secret"
    public = loaded.public_values()
    assert public["llm_api_key_set"] is True
    assert "llm_api_key" not in public


async def test_seed_uses_env_once(clean_settings, monkeypatch):
    monkeypatch.setenv("MESH_BASE_URL", "http://seeded:8765")
    async with session_scope() as db:
        assert await ensure_runtime_seeded(db) is True
        await db.commit()
        assert (await load_runtime_settings(db)).mesh_base_url == "http://seeded:8765"
        # second call is a no-op — env changes afterwards are ignored
        monkeypatch.setenv("MESH_BASE_URL", "http://other:1")
        assert await ensure_runtime_seeded(db) is False
        await db.commit()
        assert (await load_runtime_settings(db)).mesh_base_url == "http://seeded:8765"


def test_executor_values_subset():
    vals = RuntimeSettings().executor_values()
    assert "exec_timeout_s" in vals and "fem_voxel_mm" in vals
    assert "llm_api_key" not in vals and "llm_base_url" not in vals
