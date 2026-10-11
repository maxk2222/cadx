"""DB-backed runtime settings: defaults, legacy import, seeding, precedence."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from cadagent.db.models import AppSetting
from cadagent.db.session import create_all, session_scope
from cadagent.runtime_config import (
    RuntimeSettings,
    ensure_runtime_seeded,
    import_legacy_env,
    legacy_env_values,
    load_runtime_settings,
    save_runtime_settings,
)


@pytest.fixture()
async def clean_settings():
    await create_all()
    async with session_scope() as db:
        for key in ("runtime", "llm", "mesh"):
            row = await db.get(AppSetting, key)
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


async def test_legacy_groups_merge_into_runtime(clean_settings):
    async with session_scope() as db:
        db.add(AppSetting(key="llm", value={"llm_base_url": "http://legacy:9/v1"}))
        db.add(AppSetting(key="mesh", value={"mesh_model": "hunyuan3d"}))
        await db.commit()
        s = await load_runtime_settings(db)
        # a real save removes the legacy groups
        await save_runtime_settings(db, s)
        await db.commit()
        s2 = await load_runtime_settings(db)
        assert s.llm_base_url == "http://legacy:9/v1"
        assert s2.mesh_model == "hunyuan3d"
        assert await db.get(AppSetting, "llm") is None


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


async def test_legacy_env_import_precedence(clean_settings, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "LLM_BASE_URL=http://legacy-llm/v1\n"
        "LLM_API_KEY=sk-old\n"
        "LLM_MODEL_CODE=legacy-coder\n"
        "DATABASE_URL=postgresql://ignored\n",
        encoding="utf-8",
    )
    async with session_scope() as db:
        # user-managed value wins over the legacy file
        await save_runtime_settings(db, RuntimeSettings(llm_model_code="db-model"))
        await db.commit()
        result = await import_legacy_env(db, env)
        await db.commit()
        s = await load_runtime_settings(db)
    assert "llm_base_url" in result["imported"]
    assert "llm_api_key" in result["imported"]
    assert "llm_model_code" not in result["imported"]  # DB value won
    assert s.llm_base_url == "http://legacy-llm/v1"
    assert s.llm_api_key.get_secret_value() == "sk-old"
    assert s.llm_model_code == "db-model"
    assert "DATABASE_URL" in result["ignored_infrastructure"]

    # idempotent
    async with session_scope() as db:
        result2 = await import_legacy_env(db, env)
    assert result2["already_imported"] is True


async def test_seeded_row_does_not_block_import(clean_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_BASE_URL", "http://seeded:8765")
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL_VISION=vision-legacy\n", encoding="utf-8")
    async with session_scope() as db:
        await ensure_runtime_seeded(db)
        await db.commit()
        await import_legacy_env(db, env)
        await db.commit()
        s = await load_runtime_settings(db)
    assert s.llm_model_vision == "vision-legacy"  # seed never blocks import
    assert s.mesh_base_url == "http://seeded:8765"


def test_legacy_env_values_unknown_key_raises(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TOTALLY_UNKNOWN_KEY=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TOTALLY_UNKNOWN_KEY"):
        legacy_env_values(env)


def test_executor_values_subset():
    vals = RuntimeSettings().executor_values()
    assert "exec_timeout_s" in vals and "fem_voxel_mm" in vals
    assert "llm_api_key" not in vals and "llm_base_url" not in vals
