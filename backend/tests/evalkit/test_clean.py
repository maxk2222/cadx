"""Guarded cleanup: FK-safe wipe order, active-job guard, file purge."""

from __future__ import annotations

import pytest

from cadx.db.models import AgentTrace, Job, JobStatus, Project, Version
from cadx.db.session import async_session, create_all
from cadx.evalkit.clean import purge_files, wipe_db


@pytest.fixture(autouse=True)
async def _tables():
    await create_all()
    # the test-session DB is shared — reset every project-scoped table so
    # counts reflect only what this test seeded (raw deletes, FK-safe order)
    from sqlalchemy import delete as sa_delete

    from cadx.db.models import (
        AppSetting,
        Artifact,
        Asset,
        Chat,
        ChatMessage,
        LibraryPart,
    )

    async with async_session() as db:
        for m in (
            AgentTrace,
            Job,
            Artifact,
            Version,
            Asset,
            ChatMessage,
            Chat,
            LibraryPart,
            Project,
            AppSetting,
        ):
            await db.execute(sa_delete(m))
        await db.commit()


async def _seed(project_name="p"):
    async with async_session() as db:
        p = Project(name=project_name)
        v = Version(project_id=p.id, number=1, kind="generate", prompt="x")
        j = Job(version_id=v.id, type="generate", status=JobStatus.succeeded)
        t = AgentTrace(job_id=j.id, step="codegen", iteration=1)
        db.add_all([p, v, j, t])
        await db.commit()
        return p.id, v.id, j.id


async def _counts():
    from sqlmodel import func, select

    from cadx.db.models import Artifact, Asset, Chat, ChatMessage, LibraryPart

    out = {}
    async with async_session() as db:
        for name, m in [
            ("agenttrace", AgentTrace),
            ("job", Job),
            ("version", Version),
            ("project", Project),
            ("artifact", Artifact),
            ("asset", Asset),
            ("chat", Chat),
            ("chatmessage", ChatMessage),
            ("librarypart", LibraryPart),
        ]:
            out[name] = int((await db.execute(select(func.count()).select_from(m))).scalar_one())
    return out


async def test_wipe_db_dry_run_counts_but_keeps_rows():
    await _seed()
    rep = await wipe_db(dry_run=True)
    assert rep.dry_run
    assert rep.deleted["project"] == 1
    assert rep.deleted["agenttrace"] == 1
    counts = await _counts()
    assert counts["project"] == 1 and counts["agenttrace"] == 1


async def test_wipe_db_executes_fk_safe_and_preserves_settings():
    from sqlmodel import func, select

    from cadx.db.models import AppSetting

    await _seed()
    async with async_session() as db:
        db.add(AppSetting(key="runtime", value={"x": 1}))
        await db.commit()
    rep = await wipe_db(dry_run=False)
    counts = await _counts()
    assert all(n == 0 for n in counts.values())
    assert rep.deleted["project"] == 1 and rep.deleted["job"] == 1
    # settings preserved (not in the wipe set without --with-settings)
    async with async_session() as db:
        n = int((await db.execute(select(func.count()).select_from(AppSetting))).scalar_one())
    assert n == 1
    # --with-settings removes them too
    await wipe_db(dry_run=False, with_settings=True)
    async with async_session() as db:
        n = int((await db.execute(select(func.count()).select_from(AppSetting))).scalar_one())
    assert n == 0


async def test_wipe_db_refuses_with_active_jobs_unless_forced():
    _pid, _vid, jid = await _seed()
    async with async_session() as db:
        j = await db.get(Job, jid)
        j.status = JobStatus.running
        await db.commit()
    rep = await wipe_db(dry_run=False)
    assert rep.skipped and "active jobs" in rep.skipped[0]
    assert (await _counts())["project"] == 1
    rep2 = await wipe_db(dry_run=False, force=True)
    assert not rep2.skipped
    assert (await _counts())["project"] == 0


def test_purge_files_dry_run_and_execute(tmp_path):
    data = tmp_path / "data"
    (data / "projects" / "p1").mkdir(parents=True)
    (data / "eval_old.json").write_text("{}")
    keep = tmp_path / "evals" / "suite"
    keep.mkdir(parents=True)
    rep = purge_files(dry_run=True, data_dir=data)
    names = [f for f in rep.files_removed]
    assert any("projects" in n for n in names)
    assert any("eval_old.json" in n for n in names)
    assert (data / "projects" / "p1").is_dir()  # nothing removed
    rep2 = purge_files(dry_run=False, data_dir=data)
    assert rep2.files_removed
    assert not (data / "projects").exists()
    assert not (data / "eval_old.json").exists()
    assert keep.is_dir()  # suite untouched


def test_purge_files_never_touches_suite_or_baselines(tmp_path, monkeypatch):
    import cadx.evalkit.clean as cl

    suite = tmp_path / "evals" / "suite"
    base = tmp_path / "evals" / "baselines"
    res = tmp_path / "evals" / "results" / "case" / "run"
    scratch = tmp_path / "evals" / "scratch" / "case" / "run"
    for d in (suite, base, res, scratch):
        d.mkdir(parents=True, exist_ok=True)
    (suite / "x.json").write_text("{}")
    (base / "b.json").write_text("{}")
    monkeypatch.setattr(cl, "RESULTS_DIR", tmp_path / "evals" / "results")
    monkeypatch.setattr(cl, "SCRATCH_DIR", tmp_path / "evals" / "scratch")
    monkeypatch.setattr(cl, "REPORTS_DIR", tmp_path / "evals" / "reports")
    rep = purge_files(dry_run=False, data_dir=tmp_path / "data")
    assert all("suite" not in f and "baselines" not in f for f in rep.files_removed)
    assert (suite / "x.json").is_file() and (base / "b.json").is_file()
    assert not (tmp_path / "evals" / "results" / "case").exists()
    assert not (tmp_path / "evals" / "scratch" / "case").exists()
