"""Cleanup — guarded full DB reset + file purge.

`wipe_db` deletes every project-scoped row in FK-safe order (never touches
AppSetting — runtime settings and encrypted secrets survive unless
`with_settings` is passed). `purge_files` removes eval run trees and the
project data dir. Both default to dry-run; `--yes` executes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadx.evalkit.paths import REPORTS_DIR, RESULTS_DIR, SCRATCH_DIR
from cadx.logging import get_logger

log = get_logger(__name__)


@dataclass
class CleanReport:
    deleted: dict[str, int] = field(default_factory=dict)
    files_removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dry_run: bool = True


async def _active_jobs(db: Any) -> int:
    from sqlmodel import func, select

    from cadx.db.models import Job, JobStatus

    q = (
        select(func.count())
        .select_from(Job)
        .where(Job.status.in_([JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input]))
    )
    return int((await db.execute(q)).scalar_one())


async def wipe_db(
    *,
    dry_run: bool = True,
    force: bool = False,
    with_settings: bool = False,
    session: Any | None = None,
) -> CleanReport:
    """Delete all project-scoped rows in FK-safe order.

    Guard: refuses while jobs are queued/running/waiting unless `force`.
    Order follows the model FK graph (verified against delete_project):
    AgentTrace -> Job -> Artifact -> Version -> Asset -> ChatMessage ->
    Chat -> LibraryPart -> Project.
    """
    from sqlalchemy import delete as sa_delete
    from sqlmodel import func, select

    from cadx.db.models import (
        AgentTrace,
        AppSetting,
        Artifact,
        Asset,
        Chat,
        ChatMessage,
        Job,
        LibraryPart,
        Project,
        Version,
    )
    from cadx.db.session import session_scope

    rep = CleanReport(dry_run=dry_run)

    async def _wipe(db: Any) -> None:
        active = await _active_jobs(db)
        if active and not force:
            rep.skipped.append(f"{active} active jobs — aborting (use --force to override)")
            return
        # table -> count (counted first so dry-run reports what would go)
        tables = [
            ("agenttrace", AgentTrace),
            ("job", Job),
            ("artifact", Artifact),
            ("version", Version),
            ("asset", Asset),
            ("chatmessage", ChatMessage),
            ("chat", Chat),
            ("librarypart", LibraryPart),
            ("project", Project),
        ]
        if with_settings:
            tables.insert(0, ("appsetting", AppSetting))
        for name, model in tables:
            n = int((await db.execute(select(func.count()).select_from(model))).scalar_one())
            rep.deleted[name] = n
            if not dry_run and n:
                await db.execute(sa_delete(model))
        if not dry_run:
            await db.commit()

    if session is not None:
        await _wipe(session)
    else:
        async with session_scope() as db:
            await _wipe(db)
    return rep


def purge_files(*, dry_run: bool = True, data_dir: Path | None = None) -> CleanReport:
    """Remove eval file trees: data/projects, data/eval_*, evals run dirs.

    `evals/suite` and `evals/baselines` are committed and never touched.
    `data_dir` override exists for tests/scratch; default = storage root.
    """
    rep = CleanReport(dry_run=dry_run)
    if data_dir is None:
        from cadx import storage

        data_dir = storage.data_root()
    targets: list[Path] = []
    projects = data_dir / "projects"
    if projects.is_dir():
        targets.append(projects)
    for p in sorted(data_dir.glob("eval_*")):
        targets.append(p)
    for root in (RESULTS_DIR, REPORTS_DIR, SCRATCH_DIR):
        if root.is_dir():
            targets.extend(sorted(p for p in root.iterdir()))
    for t in targets:
        rep.files_removed.append(str(t))
        if not dry_run:
            if t.is_dir() and not t.is_symlink():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
    return rep
