"""Shared result types — identical schema for the http and local drivers.

`StageOutcome` is the normalized payload a driver returns per stage;
`StageResult` is the persisted verdict (outcome + gate results + metrics);
`CaseResult` is the per-run `result.json` content.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageOutcome:
    """Normalized per-stage payload produced by a driver dispatch.

    http driver fills it from VersionDetail + JobOut + AgentTrace rows +
    downloaded artifacts; local fills it from RunOutcome / task dicts +
    JsonlTrace records + files on disk.
    """

    status: str = "failed"  # succeeded|needs_review|failed|cancelled|timeout
    version_id: str | None = None
    job_id: str | None = None
    score: int | None = None
    iterations: int = 0
    resolved_mode: str | None = None
    mode_note: str = ""
    analysis: dict[str, Any] | None = None  # bbox_size, watertight*, holes, …
    features: dict[str, Any] | None = None  # measured feature dict for feature gates
    features_available: bool | None = None  # None=not attempted, False=unavailable
    features_error: str | None = None
    spec: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    code: str | None = None
    fem: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    dfm: dict[str, Any] | None = None  # DfmReport dump (process, warnings, summary)
    critique: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # kind|view -> path/relpath
    artifact_files: list[str] = field(default_factory=list)  # resolved stl/brep paths
    trace_records: list[dict[str, Any]] = field(default_factory=list)
    live: dict[str, Any] | None = None
    asset_ids: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)  # driver-specific extras
    wall_s: float = 0.0
    error: str | None = None


@dataclass
class GateResult:
    name: str
    ok: bool | None  # None = inconclusive/evidence unavailable
    measured: Any = None
    expected: Any = None
    detail: str = ""


@dataclass
class StageResult:
    stage_id: str
    stage_type: str
    status: str
    passed: bool | None
    wall_s: float
    version_id: str | None = None
    job_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class CaseResult:
    """Serialized as `result.json` per run — the report/baseline data unit."""

    case: str
    driver: str  # "http" | "local"
    variant: str
    run_id: str
    passed: bool | None
    wall_s: float
    project: str = ""
    project_id: str | None = None
    stages: list[StageResult] = field(default_factory=list)
    error: str | None = None
    finished_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    variant_overrides: dict[str, Any] = field(default_factory=dict)
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseResult:
        stages = [
            StageResult(
                stage_id=s.get("stage_id", "?"),
                stage_type=s.get("stage_type", "?"),
                status=s.get("status", "?"),
                passed=s.get("passed"),
                wall_s=float(s.get("wall_s") or 0.0),
                version_id=s.get("version_id"),
                job_id=s.get("job_id"),
                metrics=s.get("metrics") or {},
                gates=[GateResult(**g) for g in s.get("gates") or []],
                artifacts=s.get("artifacts") or [],
                error=s.get("error"),
            )
            for s in d.get("stages") or []
            if isinstance(s, dict)
        ]
        known = {
            "case",
            "driver",
            "variant",
            "run_id",
            "passed",
            "wall_s",
            "project",
            "project_id",
            "error",
            "finished_at",
            "meta",
            "variant_overrides",
            "settings_overrides",
            "schema_version",
        }
        kwargs = {k: d[k] for k in known if k in d}
        kwargs.setdefault("case", "?")
        kwargs.setdefault("driver", "?")
        kwargs.setdefault("variant", "default")
        kwargs.setdefault("run_id", "?")
        kwargs.setdefault("wall_s", 0.0)
        return CaseResult(stages=stages, **kwargs)
