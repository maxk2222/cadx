"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from cadagent.db.models import (
    ArtifactKind,
    AssetKind,
    JobStatus,
    JobType,
    ProjectSettings,
    VersionKind,
    VersionStatus,
)


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    settings: ProjectSettings | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    settings: ProjectSettings | None = None


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    settings: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version_count: int = 0
    latest_version_status: VersionStatus | None = None
    latest_thumbnail_url: str | None = None


class AssetUpdate(BaseModel):
    caption: str | None = None


class AssetOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    kind: AssetKind
    filename: str
    mime: str
    width: int | None
    height: int | None
    caption: str
    created_at: datetime
    url: str = ""


class GenerateRequest(BaseModel):
    prompt: str
    asset_ids: list[uuid.UUID] = Field(default_factory=list)
    mode: Literal["auto", "cad", "mesh", "hybrid"] | None = None


class EditRequest(BaseModel):
    prompt: str
    asset_ids: list[uuid.UUID] = Field(default_factory=list)
    mode: Literal["auto", "cad", "mesh", "hybrid"] | None = None
    regenerate_mesh: bool = False  # hybrid edit: regenerate the organic base mesh


class ParamsRequest(BaseModel):
    params: dict[str, Any]


class AnswerItem(BaseModel):
    id: str | None = None
    answer: str


class AnswerRequest(BaseModel):
    answers: list[AnswerItem]


class VersionSummary(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    parent_id: uuid.UUID | None
    number: int
    kind: VersionKind
    prompt: str
    status: VersionStatus
    score: int | None
    iterations: int
    error: str | None
    gen_mode: str = "auto"
    resolved_mode: str | None = None
    created_at: datetime
    thumbnail_url: str | None = None


class ArtifactOut(BaseModel):
    id: uuid.UUID
    kind: ArtifactKind
    view: str | None
    path: str
    size: int
    url: str
    meta: dict[str, Any] = Field(default_factory=dict)


class VersionDetail(VersionSummary):
    spec: dict[str, Any] | None
    plan: dict[str, Any] | None
    code: str | None
    params: dict[str, Any] | None
    params_schema: list[dict[str, Any]] | None
    analysis: dict[str, Any] | None
    critique: dict[str, Any] | None
    dfm: dict[str, Any] | None
    mesh: dict[str, Any] | None = None
    usage: dict[str, Any]
    asset_ids: list[str]
    artifacts: list[ArtifactOut] = Field(default_factory=list)


class JobOut(BaseModel):
    id: uuid.UUID
    version_id: uuid.UUID
    type: JobType
    status: JobStatus
    step: str | None
    iteration: int
    progress: float
    log: str
    questions: list[dict[str, Any]] | None
    cancel_requested: bool
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class EnqueueResponse(BaseModel):
    version_id: uuid.UUID
    job_id: uuid.UUID


class CatalogEntry(BaseModel):
    name: str
    category: str
    description: str
    signature: str = ""


class SettingsOut(BaseModel):
    llm_base_url: str
    llm_model_vision: str
    llm_model_code: str
    llm_json_mode: str
    llm_api_key_set: bool
    data_dir: str
    agent_max_iterations: int
    agent_max_repairs: int
    agent_pass_score: int
    agent_clarification_mode: str
    agent_default_gen_mode: str = "auto"
    exec_timeout_s: int
    exec_memory_mb: int
    mesh_base_url: str = ""
    mesh_model: str = "auto"


class SettingsUpdate(BaseModel):
    llm_base_url: str | None = None
    llm_model_vision: str | None = None
    llm_model_code: str | None = None
    llm_json_mode: str | None = None
    mesh_base_url: str | None = None
    mesh_model: str | None = None
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid"] | None = None


class MeshModelOut(BaseModel):
    id: str
    name: str = ""
    device: str = ""
    backend: str = ""
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True


class MeshHealthOut(BaseModel):
    ok: bool = False
    device: str = ""
    backend: str = ""
    error: str | None = None
    models: list[MeshModelOut] = Field(default_factory=list)
