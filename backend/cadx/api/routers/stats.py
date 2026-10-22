"""Aggregated generation statistics for the Stats page.

Read-only column-select queries (no heavy JSON/Text columns like code, log,
messages) — aggregated in Python so the same code works on Postgres and the
SQLite test database.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadx.api.deps import get_session
from cadx.api.schemas import (
    FemStats,
    IssueStat,
    JobTypeStat,
    MeshStats,
    ModelStat,
    ModeStat,
    NamedCount,
    ProjectStat,
    ScoreStats,
    StatsDayBucket,
    StatsOut,
    StatsTotals,
    StepStat,
)
from cadx.db.models import (
    AgentTrace,
    Artifact,
    Asset,
    Job,
    JobStatus,
    LibraryPart,
    Part,
    Project,
    Version,
    VersionStatus,
)

router = APIRouter(tags=["stats"])

_TERMINAL = (
    VersionStatus.succeeded,
    VersionStatus.needs_review,
    VersionStatus.failed,
    VersionStatus.cancelled,
)
_SCORE_BUCKETS = ("<40", "40-59", "60-79", "80-100")


def _score_bucket(score: int) -> str:
    if score < 40:
        return _SCORE_BUCKETS[0]
    if score < 60:
        return _SCORE_BUCKETS[1]
    if score < 80:
        return _SCORE_BUCKETS[2]
    return _SCORE_BUCKETS[3]


def _named(counter: Counter, limit: int = 0) -> list[NamedCount]:
    return [NamedCount(name=str(k), count=c) for k, c in counter.most_common(limit or None)]


_STEP_SUFFIX = re.compile(r"_\d+$")


def _step_base(step: str | None) -> str | None:
    """repair_2 / improve_3 → repair / improve — group per-iteration step ids."""
    return _STEP_SUFFIX.sub("", step) if step else None


def _err_head(error: str | None) -> str | None:
    """First meaningful line of a version error — stable grouping key."""
    if not error:
        return None
    for line in error.strip().splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return None


class _VersionAgg:
    def __init__(self) -> None:
        self.total = 0
        self.by_status: Counter = Counter()
        self.by_kind: Counter = Counter()
        self.by_mode_req: Counter = Counter()
        self.by_mode_res: Counter = Counter()
        self.score_buckets: Counter = Counter({b: 0 for b in _SCORE_BUCKETS})
        self.scores: list[int] = []
        self.iters: list[int] = []
        self.top_errors: Counter = Counter()
        self.day: Counter = Counter()
        self.mesh_models: Counter = Counter()
        self.mesh_count = 0
        self.mesh_cached = 0
        self.mesh_durations: list[float] = []
        self.modes: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "versions": 0,
                "succeeded": 0,
                "terminal": 0,
                "scores": [],
                "iters": [],
                "tokens": 0,
            }
        )
        self.issue_sources: dict[str, dict[str, int]] = defaultdict(
            lambda: {"major": 0, "minor": 0}
        )
        self.fem_versions = 0
        self.fem_worst_sfs: list[float] = []
        self.day_scores: dict[str, list[float]] = defaultdict(list)
        self.per_project: dict[Any, dict[str, Any]] = defaultdict(
            lambda: {"versions": 0, "succeeded": 0, "scores": [], "tokens": 0, "last": None}
        )


async def _versions_agg(db: AsyncSession, cutoff: datetime | None) -> _VersionAgg:
    q = select(
        Version.id,
        Version.project_id,
        Version.kind,
        Version.status,
        Version.gen_mode,
        Version.resolved_mode,
        Version.score,
        Version.iterations,
        Version.usage,
        Version.error,
        Version.mesh,
        Version.fem,
        Version.issues,
        Version.created_at,
    )
    if cutoff:
        q = q.where(col(Version.created_at) >= cutoff)
    rows = (await db.execute(q)).mappings().all()

    a = _VersionAgg()
    a.total = len(rows)
    for v in rows:
        _version_row(a, v)
    return a


def _v_mesh(a: _VersionAgg, v: Any) -> None:
    gen = (v["mesh"] or {}).get("generation") or {}
    if not gen:
        return
    a.mesh_count += 1
    if gen.get("cached"):
        a.mesh_cached += 1
    if isinstance(gen.get("duration_s"), (int, float)) and gen["duration_s"]:
        a.mesh_durations.append(float(gen["duration_s"]))
    if gen.get("model"):
        a.mesh_models[str(gen["model"])] += 1


def _v_mode(a: _VersionAgg, v: Any, ptok: int) -> None:
    if not v["resolved_mode"]:
        return
    m = a.modes[str(v["resolved_mode"])]
    m["versions"] += 1
    m["succeeded"] += 1 if v["status"] == VersionStatus.succeeded else 0
    m["terminal"] += 1 if v["status"] in _TERMINAL else 0
    if v["score"] is not None:
        m["scores"].append(v["score"])
    if v["iterations"]:
        m["iters"].append(v["iterations"])
    m["tokens"] += ptok


def _v_issues(a: _VersionAgg, v: Any) -> None:
    for i in v["issues"] or []:
        if not isinstance(i, dict) or not i.get("source"):
            continue
        sev = "minor" if i.get("severity") == "minor" else "major"
        a.issue_sources[str(i["source"])][sev] += 1


def _v_fem(a: _VersionAgg, v: Any) -> None:
    fem_cases = (v["fem"] or {}).get("cases") or []
    if not fem_cases:
        return
    a.fem_versions += 1
    sfs = [
        float(c["safety_factor"])
        for c in fem_cases
        if isinstance(c, dict) and isinstance(c.get("safety_factor"), (int, float))
    ]
    if sfs:
        a.fem_worst_sfs.append(min(sfs))


def _version_row(a: _VersionAgg, v: Any) -> None:
    a.by_status[str(v["status"])] += 1
    a.by_kind[str(v["kind"])] += 1
    if v["gen_mode"]:
        a.by_mode_req[str(v["gen_mode"])] += 1
    if v["resolved_mode"]:
        a.by_mode_res[str(v["resolved_mode"])] += 1
    if v["score"] is not None:
        a.scores.append(v["score"])
        a.score_buckets[_score_bucket(v["score"])] += 1
    if v["iterations"]:
        a.iters.append(v["iterations"])
    if (head := _err_head(v["error"])) is not None:
        a.top_errors[head] += 1
    _v_mesh(a, v)
    usage = v["usage"] or {}
    ptok = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    _v_mode(a, v, ptok)
    _v_issues(a, v)
    _v_fem(a, v)
    pp = a.per_project[v["project_id"]]
    pp["versions"] += 1
    pp["succeeded"] += 1 if v["status"] == VersionStatus.succeeded else 0
    if v["score"] is not None:
        pp["scores"].append(v["score"])
    pp["tokens"] += ptok
    if pp["last"] is None or v["created_at"] > pp["last"]:
        pp["last"] = v["created_at"]
    day = v["created_at"].date().isoformat()
    a.day[day] += 1
    if v["score"] is not None:
        a.day_scores[day].append(v["score"])


class _JobAgg:
    def __init__(self) -> None:
        self.total = 0
        self.by_status: Counter = Counter()
        self.by_type: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "failed": 0, "durs": []}
        )
        self.failed_steps: Counter = Counter()
        self.step_wall: Counter = Counter()
        self.day: Counter = Counter()
        self.done_day: dict[str, Counter] = defaultdict(Counter)  # finish-day → outcome
        self.compute_seconds = 0.0


async def _jobs_agg(db: AsyncSession, cutoff: datetime | None) -> _JobAgg:
    q = select(
        Job.version_id,
        Job.type,
        Job.status,
        Job.step,
        Job.started_at,
        Job.finished_at,
        Job.created_at,
        Job.live,
    )
    if cutoff:
        q = q.where(col(Job.created_at) >= cutoff)
    rows = (await db.execute(q)).mappings().all()

    a = _JobAgg()
    a.total = len(rows)
    for j in rows:
        a.by_status[str(j["status"])] += 1
        jt = a.by_type[str(j["type"])]
        jt["count"] += 1
        if j["status"] == JobStatus.failed:
            jt["failed"] += 1
            if j["step"]:
                a.failed_steps[_step_base(str(j["step"]))] += 1
        if j["started_at"] and j["finished_at"]:
            dur = max(0.0, (j["finished_at"] - j["started_at"]).total_seconds())
            jt["durs"].append(dur)
            a.compute_seconds += dur
        if j["finished_at"]:
            dkey = j["finished_at"].date().isoformat()
            if j["status"] == JobStatus.succeeded:
                a.done_day[dkey]["succeeded"] += 1
            elif j["status"] == JobStatus.failed:
                a.done_day[dkey]["failed"] += 1
            elif j["status"] == JobStatus.cancelled:
                a.done_day[dkey]["cancelled"] += 1
        for h in (j["live"] or {}).get("step_history") or []:
            if isinstance(h.get("dt_s"), (int, float)) and h.get("step"):
                a.step_wall[_step_base(str(h["step"]))] += float(h["dt_s"])
        a.day[j["created_at"].date().isoformat()] += 1
    return a


class _TraceAgg:
    def __init__(self) -> None:
        self.total = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.day: Counter = Counter()
        self.models: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "pt": 0, "ct": 0, "lat": []}
        )
        self.steps: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "tok": 0, "lat": []}
        )


async def _traces_agg(db: AsyncSession, cutoff: datetime | None) -> _TraceAgg:
    # never select messages/response — those JSON blobs are heavy
    q = select(
        AgentTrace.step,
        AgentTrace.model,
        AgentTrace.prompt_tokens,
        AgentTrace.completion_tokens,
        AgentTrace.latency_ms,
        AgentTrace.created_at,
    )
    if cutoff:
        q = q.where(col(AgentTrace.created_at) >= cutoff)
    rows = (await db.execute(q)).mappings().all()

    a = _TraceAgg()
    a.total = len(rows)
    for tr in rows:
        pt = int(tr["prompt_tokens"] or 0)
        ct = int(tr["completion_tokens"] or 0)
        a.prompt_tokens += pt
        a.completion_tokens += ct
        a.day[tr["created_at"].date().isoformat()] += pt + ct
        m = a.models[str(tr["model"] or "?")]
        m["calls"] += 1
        m["pt"] += pt
        m["ct"] += ct
        if tr["latency_ms"]:
            m["lat"].append(float(tr["latency_ms"]))
        s = a.steps[_step_base(str(tr["step"] or "?"))]
        s["calls"] += 1
        s["tok"] += pt + ct
        if tr["latency_ms"]:
            s["lat"].append(float(tr["latency_ms"]))
    return a


@router.get("/stats", response_model=StatsOut)
async def stats(days: int = Query(0, ge=0, le=3650), db: AsyncSession = Depends(get_session)):
    """Aggregate generation stats. `days=0` means all time; otherwise every
    entity is filtered on its own `created_at >= now - days`."""
    cutoff = datetime.now(UTC) - timedelta(days=days) if days else None

    async def _count(model: Any, date_col: Any | None = None) -> int:
        q = select(func.count()).select_from(model)
        if cutoff and date_col is not None:
            q = q.where(col(date_col) >= cutoff)
        return int((await db.execute(q)).scalar_one())

    va, ja, ta = (
        await _versions_agg(db, cutoff),
        await _jobs_agg(db, cutoff),
        await _traces_agg(db, cutoff),
    )
    n_projects = await _count(Project, Project.created_at)
    n_assets = await _count(Asset, Asset.created_at)
    n_parts = await _count(Part, Part.created_at)
    n_library = await _count(LibraryPart, LibraryPart.created_at)

    aq = select(Artifact.kind, func.count(), func.coalesce(func.sum(Artifact.size), 0)).group_by(
        Artifact.kind
    )
    if cutoff:
        aq = aq.where(col(Artifact.created_at) >= cutoff)
    artifact_rows = (await db.execute(aq)).all()

    sq = select(Asset.kind, func.count()).group_by(Asset.kind)
    if cutoff:
        sq = sq.where(col(Asset.created_at) >= cutoff)
    asset_rows = (await db.execute(sq)).all()

    project_names = {
        p["id"]: p["name"] for p in (await db.execute(select(Project.id, Project.name))).mappings()
    }

    # activity timeline — gapless over the window
    today = datetime.now(UTC).date()
    start = cutoff.date() if cutoff else None
    if start is None:
        dates = [date.fromisoformat(d) for d in (*va.day, *ja.day, *ta.day)]
        start = min(dates) if dates else today
    activity: list[StatsDayBucket] = []
    d = start
    while d <= today:
        key = d.isoformat()
        activity.append(
            StatsDayBucket(
                date=key,
                versions=va.day.get(key, 0),
                jobs=ja.day.get(key, 0),
                tokens=ta.day.get(key, 0),
                succeeded=ja.done_day.get(key, Counter()).get("succeeded", 0),
                failed=ja.done_day.get(key, Counter()).get("failed", 0),
                cancelled=ja.done_day.get(key, Counter()).get("cancelled", 0),
                score_avg=round(sum(ss) / len(ss), 1) if (ss := va.day_scores.get(key)) else None,
            )
        )
        d += timedelta(days=1)

    terminal = sum(va.by_status.get(str(s), 0) for s in _TERMINAL)
    succeeded = va.by_status.get(str(VersionStatus.succeeded), 0)

    return StatsOut(
        days=days,
        generated_at=datetime.now(UTC),
        totals=StatsTotals(
            projects=n_projects,
            versions=va.total,
            jobs=ja.total,
            assets=n_assets,
            artifacts=sum(int(r[1]) for r in artifact_rows),
            parts=n_parts,
            library_parts=n_library,
            artifact_bytes=sum(int(r[2]) for r in artifact_rows),
            prompt_tokens=ta.prompt_tokens,
            completion_tokens=ta.completion_tokens,
            llm_calls=ta.total,
            compute_seconds=round(ja.compute_seconds, 1),
        ),
        versions_by_status=_named(va.by_status),
        versions_by_kind=_named(va.by_kind),
        versions_by_mode_requested=_named(va.by_mode_req),
        versions_by_mode_resolved=_named(va.by_mode_res),
        jobs_by_status=_named(ja.by_status),
        job_types=[
            JobTypeStat(
                type=t,
                count=x["count"],
                failed=x["failed"],
                avg_duration_s=round(sum(x["durs"]) / len(x["durs"]), 1) if x["durs"] else None,
                p50_duration_s=round(sorted(x["durs"])[len(x["durs"]) // 2], 1)
                if x["durs"]
                else None,
                max_duration_s=round(max(x["durs"]), 1) if x["durs"] else None,
            )
            for t, x in sorted(ja.by_type.items(), key=lambda kv: -kv[1]["count"])
        ],
        mode_stats=[
            ModeStat(
                mode=m,
                versions=x["versions"],
                succeeded=x["succeeded"],
                pass_rate=round(x["succeeded"] / x["terminal"], 3) if x["terminal"] else None,
                avg_score=round(sum(x["scores"]) / len(x["scores"]), 1) if x["scores"] else None,
                avg_iterations=round(sum(x["iters"]) / len(x["iters"]), 1) if x["iters"] else None,
                tokens=x["tokens"],
            )
            for m, x in sorted(va.modes.items(), key=lambda kv: -kv[1]["versions"])
        ],
        issue_sources=[
            IssueStat(source=s, major=x["major"], minor=x["minor"], total=x["major"] + x["minor"])
            for s, x in sorted(
                va.issue_sources.items(), key=lambda kv: -(kv[1]["major"] + kv[1]["minor"])
            )
        ],
        fem=FemStats(
            versions=va.fem_versions,
            avg_worst_sf=round(sum(va.fem_worst_sfs) / len(va.fem_worst_sfs), 2)
            if va.fem_worst_sfs
            else None,
        ),
        score=ScoreStats(
            avg=round(sum(va.scores) / len(va.scores), 1) if va.scores else None,
            best=max(va.scores) if va.scores else None,
            pass_rate=round(succeeded / terminal, 3) if terminal else None,
            buckets=[NamedCount(name=b, count=va.score_buckets.get(b, 0)) for b in _SCORE_BUCKETS],
        ),
        iterations_avg=round(sum(va.iters) / len(va.iters), 1) if va.iters else None,
        activity=activity,
        models=[
            ModelStat(
                model=m,
                calls=x["calls"],
                prompt_tokens=x["pt"],
                completion_tokens=x["ct"],
                avg_latency_ms=round(sum(x["lat"]) / len(x["lat"]), 0) if x["lat"] else 0.0,
            )
            for m, x in sorted(ta.models.items(), key=lambda kv: -(kv[1]["pt"] + kv[1]["ct"]))
        ],
        steps=[
            StepStat(
                step=s,
                calls=x["calls"],
                tokens=x["tok"],
                avg_latency_ms=round(sum(x["lat"]) / len(x["lat"]), 0) if x["lat"] else None,
                wall_seconds=round(ja.step_wall.get(s, 0.0), 1),
            )
            for s, x in sorted(ta.steps.items(), key=lambda kv: -kv[1]["tok"])
        ],
        top_errors=_named(va.top_errors, limit=10),
        failed_steps=_named(ja.failed_steps, limit=10),
        mesh=MeshStats(
            generations=va.mesh_count,
            cached=va.mesh_cached,
            avg_duration_s=round(sum(va.mesh_durations) / len(va.mesh_durations), 1)
            if va.mesh_durations
            else None,
            models=_named(va.mesh_models),
        ),
        assets_by_kind=[NamedCount(name=str(r[0]), count=int(r[1])) for r in asset_rows],
        artifacts_by_kind=[NamedCount(name=str(r[0]), count=int(r[1])) for r in artifact_rows],
        projects=[
            ProjectStat(
                id=pid,
                name=project_names.get(pid, "?"),
                versions=x["versions"],
                succeeded=x["succeeded"],
                avg_score=round(sum(x["scores"]) / len(x["scores"]), 1) if x["scores"] else None,
                tokens=x["tokens"],
                last_activity=x["last"],
            )
            for pid, x in sorted(va.per_project.items(), key=lambda kv: -kv[1]["versions"])[:50]
        ],
    )
