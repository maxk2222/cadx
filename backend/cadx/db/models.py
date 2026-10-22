"""SQLModel database models. JSON columns via sqlalchemy.JSON."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, event, inspect
from sqlmodel import Field as SField
from sqlmodel import SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class AssetKind(enum.StrEnum):
    image = "image"
    generated_image = "generated_image"
    reference_step = "reference_step"
    reference_stl = "reference_stl"
    reference_dxf = "reference_dxf"


# Kinds that can serve as image inputs (LLM context, mesh source images).
IMAGE_ASSET_KINDS = (AssetKind.image, AssetKind.generated_image)


class VersionKind(enum.StrEnum):
    generate = "generate"
    edit = "edit"
    params = "params"
    optimize = "optimize"
    refine = "refine"  # post-TopOpt feature finishing child version
    mesh_edit = "mesh_edit"  # visual mesh-level edit (GUI ops stack)
    organic = "organic"  # procedural pseudo-topology organic restyle
    code = "code"  # externally-supplied build123d code, persisted+executed (MCP)
    meshgen = "meshgen"  # direct image->mesh generation via the sidecar (MCP)


class VersionStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    waiting_for_input = "waiting_for_input"
    succeeded = "succeeded"
    needs_review = "needs_review"
    failed = "failed"
    cancelled = "cancelled"


class JobType(enum.StrEnum):
    generate = "generate"
    edit = "edit"
    params = "params"
    reexport = "reexport"
    resume = "resume"
    optimize = "optimize"
    resample = "resample"  # density threshold re-extract — no new SIMP run
    refine = "refine"  # post-TopOpt feature finishing (exact bores/threads)
    mesh_edit = "mesh_edit"  # GUI ops stack -> tessellated child version
    fem = "fem"  # standalone FEM check on the same version
    organic = "organic"  # procedural pseudo-topology organic restyle (no FEM)
    code = "code"  # execute externally-supplied build123d code — no LLM (MCP)
    meshgen = "meshgen"  # image->mesh generation via sidecar — no LLM (MCP)


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    waiting_for_input = "waiting_for_input"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ArtifactKind(enum.StrEnum):
    step = "step"
    stl = "stl"
    three_mf = "3mf"
    glb = "glb"
    dxf = "dxf"
    svg = "svg"
    render = "render"
    contact_sheet = "contact_sheet"
    thumbnail = "thumbnail"
    mesh_raw = "mesh_raw"
    fem = "fem"  # FEM stress/deformation renders + reports


ProcessType = Literal["fdm", "sla", "cnc3", "cnc2d"]

GenMode = Literal["auto", "cad", "mesh", "hybrid", "topopt"]


class ProjectSettings(BaseModel):
    """Per-project overrides; every field optional, falls back to env defaults."""

    units: str = "mm"
    process: ProcessType = "fdm"
    material: str = ""
    clarification_mode: Literal["assume", "ask"] | None = None
    max_iterations: int | None = None
    max_repairs: int | None = None
    pass_score: int | None = None
    dim_tolerance_pct: float | None = None
    dim_tolerance_mm: float | None = None
    build_volume: list[float] = Field(default_factory=lambda: [256.0, 256.0, 256.0])
    min_wall_mm: float = 0.8
    nozzle_mm: float = 0.4
    tool_radius_mm: float = 1.5
    thread_compensation_mm: float = 0.0
    include_fasteners: bool = False
    gen_mode: GenMode = "auto"
    mesh_model: str = ""
    # provider backend overrides (None -> runtime default); RunOptions carries
    # the same keys so per-run options merge over these project defaults
    llm_backend: Literal["external", "sidecar"] | None = None
    mesh_backend: Literal["sidecar", "tripo"] | None = None
    image_backend: Literal["sidecar", "openai"] | None = None
    image_model: str | None = None  # empty/None -> runtime default
    image_edit_model: str | None = None
    mesh_steps: int | None = None
    mesh_guidance: float | None = None
    mesh_resolution: int | None = None
    mesh_seed: int | None = None
    # FEM / topology optimization overrides
    fem: bool = False  # run FEM structural analysis (fill spec.load_cases)
    fem_note: str = ""  # loads/fixtures description from the options panel
    fem_material: str = ""  # material table key: pla|petg|abs|asa|pa|pc|resin|aluminum|steel
    fem_voxel_mm: float | None = None
    topopt: bool = False  # apply topology optimization after base generation
    topopt_volfrac: float | None = None
    topopt_iterations: int | None = None


class Project(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    name: str
    description: str = ""
    settings: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = SField(default_factory=utcnow)
    updated_at: datetime = SField(default_factory=utcnow)


class Asset(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = SField(foreign_key="project.id", index=True)
    kind: AssetKind
    filename: str
    path: str  # relative to DATA_DIR
    mime: str = ""
    width: int | None = None
    height: int | None = None
    caption: str = ""
    derived_from: uuid.UUID | None = None  # source asset for edited copies
    gen_params: dict[str, Any] | None = SField(
        default=None, sa_column=Column(JSON)
    )  # provenance for generated_image assets
    created_at: datetime = SField(default_factory=utcnow)


class Version(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID = SField(foreign_key="project.id", index=True)
    parent_id: uuid.UUID | None = SField(default=None, foreign_key="version.id")
    number: int
    kind: VersionKind
    prompt: str = ""
    gen_mode: str = "auto"  # requested: auto | cad | mesh | hybrid | topopt
    resolved_mode: str | None = None  # effective: cad | mesh | hybrid | topopt
    options: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    asset_ids: list[str] = SField(default_factory=list, sa_column=Column(JSON))
    library_part_ids: list[str] = SField(default_factory=list, sa_column=Column(JSON))
    status: VersionStatus = VersionStatus.queued
    spec: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    plan: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    code: str | None = None
    params: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    params_schema: list[dict[str, Any]] | None = SField(default=None, sa_column=Column(JSON))
    analysis: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    critique: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    dfm: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    bom: list[dict[str, Any]] | None = SField(default=None, sa_column=Column(JSON))
    issues: list[dict[str, Any]] | None = SField(default=None, sa_column=Column(JSON))
    mesh: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    fem: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    iterations: int = 0
    score: int | None = None
    usage: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = SField(default_factory=utcnow)


class Artifact(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    version_id: uuid.UUID = SField(foreign_key="version.id", index=True)
    kind: ArtifactKind
    view: str | None = None
    path: str  # relative to DATA_DIR
    size: int = 0
    meta: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = SField(default_factory=utcnow)


class Job(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    version_id: uuid.UUID = SField(foreign_key="version.id", index=True)
    type: JobType
    status: JobStatus = JobStatus.queued
    step: str | None = None
    iteration: int = 0
    progress: float = 0.0
    log: str = ""
    questions: list[dict[str, Any]] | None = SField(default=None, sa_column=Column(JSON))
    resume_state: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    cancel_requested: bool = False
    finish_requested: bool = False
    live: dict[str, Any] | None = SField(
        default=None, sa_column=Column(JSON)
    )  # structured live state: max_iterations, scores, step timings, tokens
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = SField(default_factory=utcnow)


class AgentTrace(SQLModel, table=True):
    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID = SField(foreign_key="job.id", index=True)
    step: str
    iteration: int = 0
    model: str = ""
    messages: list[dict[str, Any]] = SField(default_factory=list, sa_column=Column(JSON))
    response: str = ""
    parsed: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    created_at: datetime = SField(default_factory=utcnow)


class AppSetting(SQLModel, table=True):
    key: str = SField(primary_key=True)
    value: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))


class ChatRole(enum.StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Chat(SQLModel, table=True):
    """A chat thread optionally bound to a project — the conversational
    agent front-end. Unbound chats can only talk and use catalog/library
    tools until the agent (or the UI) binds them to a project.

    `llm_messages` is the persisted, sanitised OpenAI-format conversation
    (no system prompt — rebuilt per turn; image parts stored as
    {"type": "asset_ref", "asset_id": ...} markers, rehydrated on load).
    """

    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    project_id: uuid.UUID | None = SField(default=None, foreign_key="project.id", index=True)
    title: str = ""
    llm_messages: list[dict[str, Any]] = SField(default_factory=list, sa_column=Column(JSON))
    turn_running: bool = False  # atomic turn claim (UPDATE ... WHERE NOT turn_running)
    cancel_requested: bool = False
    usage: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = SField(default_factory=utcnow)
    updated_at: datetime = SField(default_factory=utcnow)


class ChatMessage(SQLModel, table=True):
    """Rendered chat view: user/assistant/system rows. `meta` carries the
    tool-call log, spawned version/job ids and the streaming flag; `pending`
    marks user messages not yet consumed by a turn."""

    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    chat_id: uuid.UUID = SField(foreign_key="chat.id", index=True)
    role: ChatRole
    content: str = ""
    asset_ids: list[str] = SField(default_factory=list, sa_column=Column(JSON))
    meta: dict[str, Any] = SField(default_factory=dict, sa_column=Column(JSON))
    pending: bool = False
    created_at: datetime = SField(default_factory=utcnow)


class LibraryPart(SQLModel, table=True):
    """Project-independent reusable part: an uploaded STEP/STL body or a
    saved parametric code template from a successful version."""

    id: uuid.UUID = SField(default_factory=uuid.uuid4, primary_key=True)
    name: str
    kind: str = "step"  # step | stl | template
    filename: str = ""
    path: str = ""  # relative to DATA_DIR (file parts only)
    code: str | None = None  # build123d source (template parts only)
    params: dict[str, Any] | None = SField(default=None, sa_column=Column(JSON))
    caption: str = ""
    source_version_id: uuid.UUID | None = None
    created_at: datetime = SField(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# job lifecycle publishing — ONE chokepoint for every Job create/status change #
# (spawn, worker transitions, cancel endpoint, MCP — current and future).     #
# Fires inside the flush; the sync publish is ~0.3ms and failure-suppressed,  #
# so a redis outage never breaks a commit and a rolled-back row can at worst  #
# cost the client one extra refetch.                                          #
# --------------------------------------------------------------------------- #


def _job_lifecycle(job: Job) -> None:
    from cadx.events import publish_lifecycle_sync

    publish_lifecycle_sync(
        job_id=str(job.id),
        version_id=str(job.version_id),
        status=job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        job_type=job.type.value if isinstance(job.type, JobType) else str(job.type),
    )


@event.listens_for(Job, "after_insert")
def _job_after_insert(mapper: Any, connection: Any, job: Job) -> None:
    _job_lifecycle(job)


@event.listens_for(Job, "after_update")
def _job_after_update(mapper: Any, connection: Any, job: Job) -> None:
    # only real status transitions are lifecycle — progress/log/step writes on
    # the same row are per-job stream material, not global-list changes
    if inspect(job).attrs.status.history.has_changes():
        _job_lifecycle(job)
