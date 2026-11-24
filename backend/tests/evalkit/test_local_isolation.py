"""Local-eval per-run isolation — `evalkit/local.py::_rebind_guide_paths` +
`guides/repository.py` URL-aware sync engine.

Regression test for the cross-run contamination found in the
2026-11 no-pipes campaign: `guides.paths` snapshots DATA_DIR-derived
constants at import, `media_index` snapshots its own trio, and the
repository's sync engine singleton stayed bound to the FIRST run's sqlite.
Result: run N's publish committed into run 1's eval.db and wrote its guide
tree into run 1's data dir. Both bindings must be repointable per run.
"""

from __future__ import annotations

import sqlite3

import pytest

from kiara.evalkit.local import _rebind_guide_paths


@pytest.fixture
def _restore_guide_globals(monkeypatch):
    """_rebind_guide_paths and the repository engine mutate PROCESS-global
    module state — restore the pre-test bindings or later tests in the same
    worker inherit a tmp_path engine/data dir. Env patches get undone
    BEFORE the settings cache is rebuilt (fixture teardown order)."""
    import kiara.guides.paths as gp
    import kiara.guides.repository as repo

    saved_paths = {
        n: getattr(gp, n)
        for n in (
            "DATA_DIR",
            "INDEX_DIR",
            "PROJECT_DATA_DIR",
            "GLOBAL_ASSETS_DIR",
            "UPLOAD_ATTEMPTS_DIR",
            "ASSET_STORE_DIR",
            "ASSET_STORE_STAGING_DIR",
            "APP_RELEASES_DIR",
        )
    }
    saved_repo = (repo._sync_engine, repo._sync_sessionmaker, repo._sync_url_seen)
    yield
    for n, v in saved_paths.items():
        setattr(gp, n, v)
    repo._sync_engine, repo._sync_sessionmaker, repo._sync_url_seen = saved_repo
    # undo env patches BEFORE rebuilding the settings cache, then rebind
    # every importer to the real env
    monkeypatch.undo()
    from kiara.config import get_settings

    get_settings.cache_clear()
    _rebind_guide_paths()


def test_rebind_guide_paths_repoints_module_constants(
    tmp_path, monkeypatch, _restore_guide_globals
):
    import kiara.guides.paths as gp

    run_a = tmp_path / "run_a" / "data"
    run_b = tmp_path / "run_b" / "data"
    from kiara.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(run_a))
    get_settings.cache_clear()
    _rebind_guide_paths()
    assert run_a / "guides" == gp.PROJECT_DATA_DIR
    assert run_a / "global_assets" == gp.GLOBAL_ASSETS_DIR

    monkeypatch.setenv("DATA_DIR", str(run_b))
    get_settings.cache_clear()
    _rebind_guide_paths()
    assert run_b / "guides" == gp.PROJECT_DATA_DIR
    assert run_b / "upload_attempts" == gp.UPLOAD_ATTEMPTS_DIR
    # guide_dir() resolves through the module globals → follows the rebind
    assert str(gp.guide_dir("x")).startswith(str(run_b))

    # media_index binds its own constants (suite "project_data" naming)
    import kiara.guides.media_index as mi

    _rebind_guide_paths()
    assert run_b / "project_data" == mi.PROJECT_DATA_DIR
    assert run_b / "global_assets" == mi.GLOBAL_ASSETS_DIR


def test_repository_engine_rebinds_on_url_change(
    tmp_path, monkeypatch, _restore_guide_globals
):
    """The sync repository engine must commit to the CURRENT DATABASE_URL —
    not stay bound to the first engine ever built in this process."""
    import kiara.guides.repository as repo

    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_a}")
    from kiara.config import get_settings

    get_settings.cache_clear()
    eng_a = repo._engine()
    assert str(db_a) in str(eng_a.url)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_b}")
    get_settings.cache_clear()
    eng_b = repo._engine()
    assert str(db_b) in str(eng_b.url)
    assert eng_b is not eng_a
    # sessionmaker must follow the new engine too
    assert repo._sessionmaker().kw["bind"] is eng_b


def test_engine_writes_to_current_db(tmp_path, monkeypatch, _restore_guide_globals):
    """End-to-end: writes through the sync engine land in whichever
    DATABASE_URL is active — never a stale engine."""
    from sqlalchemy import text

    import kiara.guides.repository as repo
    from kiara.config import get_settings

    for name in ("a.db", "b.db"):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / name}")
        get_settings.cache_clear()
        with repo._engine().begin() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS probe (k TEXT, v TEXT)"))
            conn.execute(text("INSERT INTO probe (k, v) VALUES ('run', :n)"), {"n": name})

    for name in ("a.db", "b.db"):
        rows = list(sqlite3.connect(str(tmp_path / name)).execute("select v from probe"))
        assert rows == [(name,)], f"{name} holds {rows}"
