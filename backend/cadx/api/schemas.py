"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cadx.db.models import (
    ArtifactKind,
    AssetKind,
    JobStatus,
    JobType,
    ProcessType,
    ProjectSettings,
    VersionKind,
    VersionStatus,
)


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    settings: ProjectSettings | None = None


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    latest_version_id: uuid.UUID | None = None
    running_version_id: uuid.UUID | None = None  # set while the latest version is active


class AssetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str | None = None
    role_hint: str | None = Field(default=None, max_length=40)  # e.g. "sketch"; "" clears


class AssetOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    kind: AssetKind
    filename: str
    mime: str
    width: int | None
    height: int | None
    caption: str
    derived_from: uuid.UUID | None = None
    gen_params: dict[str, Any] | None = None  # provenance for generated images
    created_at: datetime
    url: str = ""


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Per-run overrides chosen in the GUI options panel.

    Stored on the version (provenance) and merged over project settings in the
    worker: env defaults < project.settings < version.options.
    """

    process: ProcessType | None = None
    material: str | None = Field(default=None, max_length=40)
    include_fasteners: bool | None = None
    fem: bool | None = None  # request FEM structural analysis (spec.load_cases)
    fem_note: str | None = Field(default=None, max_length=200)  # loads/fixtures
    topopt: bool | None = None  # request topology optimization (gen_mode=topopt)
    topopt_volfrac: float | None = Field(default=None, ge=0.05, le=0.95)
    # agent budget overrides (merge over project settings / runtime defaults)
    max_iterations: int | None = Field(default=None, ge=1, le=200)
    max_repairs: int | None = Field(default=None, ge=0, le=200)
    pass_score: int | None = Field(default=None, ge=0, le=100)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    asset_ids: list[uuid.UUID] = Field(default_factory=list)
    library_part_ids: list[uuid.UUID] = Field(default_factory=list)
    mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"] | None = None
    options: RunOptions | None = None


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    asset_ids: list[uuid.UUID] = Field(default_factory=list)
    library_part_ids: list[uuid.UUID] = Field(default_factory=list)
    mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"] | None = None
    regenerate_mesh: bool = False  # hybrid edit: regenerate the organic base mesh
    options: RunOptions | None = None  # None inherits the parent's options


class LibraryPartOut(BaseModel):
    id: uuid.UUID
    name: str
    kind: str  # step | stl | template
    filename: str
    caption: str
    params: dict[str, Any] | None = None
    source_version_id: uuid.UUID | None = None
    created_at: datetime
    url: str = ""


class LibraryPartUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=120)
    caption: str | None = Field(default=None, max_length=400)


class LibraryFromVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Save an existing version into the reusable library."""

    name: str = Field(min_length=1, max_length=120)
    caption: str = Field(default="", max_length=400)
    as_template: bool = False  # True: keep the parametric code; False: copy the STEP


class ParamsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any]


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Manual topology-optimization trigger on an existing version.

    `load_cases` / `optimize` follow the LLM contract schemas
    (cadx.llm.schemas.LoadCase / OptimizeSpec). Empty values fall back to
    the parent spec's declared loads/optimize settings.
    """

    load_cases: list[dict[str, Any]] = Field(default_factory=list)
    optimize: dict[str, Any] = Field(default_factory=dict)
    material: str = ""
    # allow optimizing an already-optimized version again (e.g. warm-start
    # from the stored density, different void/keep regions, or a second pass
    # over the resampled mesh)
    allow_reopt: bool = False


class TopoptResampleRequest(BaseModel):
    """Re-extract a stored SIMP density field at a different threshold /
    min-member — no new SIMP run. Creates an optimize child version."""

    model_config = ConfigDict(extra="forbid")

    density_threshold: float = Field(default=0.5, ge=0.2, le=0.9)
    min_member_mm: float = Field(default=0.0, ge=0.0, le=50.0)
    load_cases: list[dict[str, Any]] = Field(default_factory=list)
    keep_regions: list[dict[str, Any]] = Field(default_factory=list)
    material: str = ""


class RefineRequest(BaseModel):
    """Post-TopOpt feature finishing: re-cut exact bores/threads/inserts at
    measured anchors and restore keep-region surfaces from the pre-optimization
    part. Creates a 'refine' child version.

    `features`  explicit selection — each entry matches a FeatureSpec subset
                ({kind|anchor_index|dia|pos|axis|...}). Empty → deterministic
                auto-resolution from the spec's declared features + scanned
                anchors.
    Empty `load_cases`/`keep_regions`/`material` fall back to the parent's
    declared values (the re-verify is skipped entirely when no cases exist).
    """

    model_config = ConfigDict(extra="forbid")

    features: list[dict[str, Any]] = Field(default_factory=list, max_length=40)
    load_cases: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    keep_regions: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    material: str = Field(default="", max_length=40)
    voxel_mm: float | None = Field(default=None, ge=0.3, le=8.0)
    restore_planes: bool | None = None  # None → OptimizeSpec.restore_planes (default on)


# --------------------------------------------------------------------------- #
# visual mesh editor + structural editor (GUI)                                  #
# --------------------------------------------------------------------------- #


class MeshOpIn(BaseModel):
    """One recorded mesh transform — validated server-side by OPS_REGISTRY."""

    model_config = ConfigDict(extra="forbid")

    op: str = Field(max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)


class MeshPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ops: list[MeshOpIn] = Field(default_factory=list, max_length=48)
    post: dict[str, Any] | None = None  # MeshPostParams overrides (mesh parents)


class MeshPreviewOut(BaseModel):
    ok: bool
    glb_url: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MeshEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ops: list[MeshOpIn] = Field(default_factory=list, max_length=48)
    post: dict[str, Any] | None = None  # MeshPostParams overrides (mesh parents)
    note: str = Field(default="", max_length=200)


class FemRegionRequest(BaseModel):
    """Region preview / standalone FEM check payload. `load_cases` and
    `keep_regions` follow the LLM contract (LoadCase / RegionSpec) and may
    contain painted `points` regions from the GUI."""

    model_config = ConfigDict(extra="forbid")

    load_cases: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    keep_regions: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    material: str = Field(default="", max_length=40)
    render_pngs: bool = False  # legacy 2D region renders; the 3D cells overlay is default


class FemPreviewOut(BaseModel):
    ok: bool
    pitch_mm: float | None = None
    dims: list[int] = Field(default_factory=list)
    solid_elements: int = 0
    cases: list[dict[str, Any]] = Field(default_factory=list)
    cells: list[dict[str, Any]] = Field(default_factory=list)  # 3D overlay point clouds
    anchors: list[dict[str, Any]] = Field(default_factory=list)  # resolved anchor table
    pngs: list[str] = Field(default_factory=list)  # /versions/<vid>/scratch/... URLs
    error: str | None = None


class AnchorsOut(BaseModel):
    """Detected/declared feature anchors — the table `hole_index` refers to."""

    ok: bool
    anchors: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class EditorStateRequest(BaseModel):
    """Persist GUI editor state (painted regions, ops draft) on the version —
    stored under options['ui_editor'], stripped before settings merge."""

    model_config = ConfigDict(extra="forbid")

    ui_editor: dict[str, Any] = Field(default_factory=dict)


class AnswerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    answer: str


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    # newest non-terminal job on this version (queued/running/waiting) —
    # {id, type, status, step, iteration, max_iterations, progress}
    active_job: dict[str, Any] | None = None


class ArtifactOut(BaseModel):
    id: uuid.UUID
    kind: ArtifactKind
    view: str | None
    path: str
    size: int
    url: str
    meta: dict[str, Any] = Field(default_factory=dict)


class VersionDetail(VersionSummary):
    options: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] | None
    plan: dict[str, Any] | None
    code: str | None
    params: dict[str, Any] | None
    params_schema: list[dict[str, Any]] | None
    analysis: dict[str, Any] | None
    critique: dict[str, Any] | None
    dfm: dict[str, Any] | None
    bom: list[dict[str, Any]] | None = None
    issues: list[dict[str, Any]] | None = None
    mesh: dict[str, Any] | None = None
    fem: dict[str, Any] | None = None
    usage: dict[str, Any]
    asset_ids: list[str]
    library_part_ids: list[str] = Field(default_factory=list)
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
    live: dict[str, Any] | None = None
    questions: list[dict[str, Any]] | None
    cancel_requested: bool
    finish_requested: bool
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class JobListItem(JobOut):
    """Job row with version/project context — for the global Jobs page."""

    version_number: int
    version_status: VersionStatus
    prompt: str = ""
    project_id: uuid.UUID
    project_name: str = ""


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
    llm_temperature_code: float
    llm_max_tokens: int
    llm_timeout_s: float
    llm_stream: bool
    llm_idle_timeout_s: float
    llm_total_timeout_s: float
    llm_extra_body: str
    llm_tools: Literal["auto", "on", "off"]
    agent_max_iterations: int
    agent_max_repairs: int
    agent_pass_score: int
    agent_dim_tolerance_pct: float
    agent_dim_tolerance_mm: float
    agent_clarification_mode: Literal["assume", "ask"]
    agent_max_input_images: int
    agent_max_tokens_per_job: int
    agent_max_wall_s: float
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"]
    agent_param_autofix: bool
    agent_patch_mode: Literal["auto", "always", "never"]
    agent_patch_retries: int
    agent_feature_feedback: bool
    agent_critique_reask: bool
    agent_tool_max_calls: int
    agent_tool_max_calls_info: int
    agent_tool_verify: bool
    agent_tool_verify_max: int
    agent_mesh_tools: bool
    agent_mesh_max_ops: int
    agent_mesh_preview_max: int
    agent_mesh_refine_rounds: int
    agent_mesh_hard_fail_score: int
    agent_mesh_hard_fail_limit: int
    agent_stall_patience: int
    fem_voxel_mm: float
    fem_max_elements: int
    fem_timeout_s: int
    topopt_timeout_s: int
    topopt_max_iterations: int
    topopt_default_volfrac: float
    topopt_default_rmin_mm: float
    mesh_base_url: str
    mesh_model: str
    image_model: str
    image_edit_model: str
    image_steps: int | None
    image_guidance: float | None
    image_max_side: int
    image_max_count: int
    image_autogen_mesh: bool
    image_tool_budget: int
    image_timeout_s: int
    image_default_strength: float
    mesh_timeout_s: int
    mesh_decimate_faces: int
    mesh_max_faces_for_cad: int
    exec_timeout_s: int
    exec_mem_limit_mb: int
    data_dir: str


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_base_url: str | None = None
    llm_api_key: str | None = Field(default=None, max_length=4000)
    clear_llm_api_key: bool = False
    llm_model_vision: str | None = Field(default=None, max_length=200)
    llm_model_code: str | None = Field(default=None, max_length=200)
    llm_json_mode: Literal["json_object", "json_schema", "none"] | None = None
    llm_temperature_code: float | None = Field(default=None, ge=0, le=2)
    llm_max_tokens: int | None = Field(default=None, ge=1, le=2_000_000)
    llm_timeout_s: float | None = Field(default=None, gt=0)
    llm_stream: bool | None = None
    llm_idle_timeout_s: float | None = Field(default=None, gt=0)
    llm_total_timeout_s: float | None = Field(default=None, gt=0)
    llm_extra_body: str | None = Field(default=None, max_length=20_000)
    llm_tools: Literal["auto", "on", "off"] | None = None
    agent_max_iterations: int | None = Field(default=None, ge=1, le=200)
    agent_max_repairs: int | None = Field(default=None, ge=0, le=200)
    agent_pass_score: int | None = Field(default=None, ge=0, le=100)
    agent_dim_tolerance_pct: float | None = Field(default=None, ge=0, le=100)
    agent_dim_tolerance_mm: float | None = Field(default=None, ge=0, le=1000)
    agent_clarification_mode: Literal["assume", "ask"] | None = None
    agent_max_input_images: int | None = Field(default=None, ge=0, le=100)
    agent_max_tokens_per_job: int | None = Field(default=None, ge=1)
    agent_max_wall_s: float | None = Field(default=None, gt=0)
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"] | None = None
    agent_param_autofix: bool | None = None
    agent_patch_mode: Literal["auto", "always", "never"] | None = None
    agent_patch_retries: int | None = Field(default=None, ge=0, le=20)
    agent_feature_feedback: bool | None = None
    agent_critique_reask: bool | None = None
    agent_tool_max_calls: int | None = Field(default=None, ge=1, le=100)
    agent_tool_max_calls_info: int | None = Field(default=None, ge=1, le=100)
    agent_tool_verify: bool | None = None
    agent_tool_verify_max: int | None = Field(default=None, ge=0, le=20)
    agent_mesh_tools: bool | None = None
    agent_mesh_max_ops: int | None = Field(default=None, ge=1, le=200)
    agent_mesh_preview_max: int | None = Field(default=None, ge=0, le=20)
    agent_mesh_refine_rounds: int | None = Field(default=None, ge=0, le=10)
    agent_mesh_hard_fail_score: int | None = Field(default=None, ge=0, le=100)
    agent_mesh_hard_fail_limit: int | None = Field(default=None, ge=1, le=20)
    agent_stall_patience: int | None = Field(default=None, ge=0, le=50)
    fem_voxel_mm: float | None = Field(default=None, gt=0, le=100)
    fem_max_elements: int | None = Field(default=None, ge=1, le=5_000_000)
    fem_timeout_s: int | None = Field(default=None, ge=1, le=86_400)
    topopt_timeout_s: int | None = Field(default=None, ge=1, le=86_400)
    topopt_max_iterations: int | None = Field(default=None, ge=1, le=1000)
    topopt_default_volfrac: float | None = Field(default=None, gt=0, le=1)
    topopt_default_rmin_mm: float | None = Field(default=None, gt=0, le=1000)
    mesh_base_url: str | None = None
    mesh_model: str | None = Field(default=None, max_length=200)
    image_model: str | None = Field(default=None, max_length=200)
    image_edit_model: str | None = Field(default=None, max_length=200)
    image_default_strength: float | None = Field(default=None, gt=0, le=1)
    image_steps: int | None = Field(default=None, ge=1, le=200)
    image_guidance: float | None = Field(default=None, ge=0, le=50)
    image_max_side: int | None = Field(default=None, ge=64, le=4096)
    image_max_count: int | None = Field(default=None, ge=1, le=16)
    image_autogen_mesh: bool | None = None
    image_tool_budget: int | None = Field(default=None, ge=0, le=50)
    image_timeout_s: int | None = Field(default=None, ge=1, le=86_400)
    mesh_timeout_s: int | None = Field(default=None, ge=1, le=86_400)
    mesh_decimate_faces: int | None = Field(default=None, ge=1)
    mesh_max_faces_for_cad: int | None = Field(default=None, ge=1)
    exec_timeout_s: int | None = Field(default=None, ge=1, le=86_400)
    exec_mem_limit_mb: int | None = Field(default=None, ge=256, le=1_048_576)


class MeshModelOut(BaseModel):
    id: str
    name: str = ""
    device: str = ""
    backend: str = ""
    kind: str = "mesh"  # "mesh" | "image"
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True
    last_error: str | None = None
    default_size: int | None = None
    max_count: int | None = None
    supports_edit: bool = False  # accepts an init image (img2img/instruction edit)
    requires_image: bool = False  # edit-only adapter, no txt2img


class ImageGenerateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    negative: str = ""
    model: str | None = None  # None → runtime image_model default
    seed: int = 0
    count: int = Field(default=1, ge=1, le=16)  # clamped by image_max_count
    width: int = Field(default=768, ge=64, le=4096)  # clamped by image_max_side
    height: int = Field(default=768, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0, le=50)
    caption: str = ""
    reference_asset_id: uuid.UUID | None = None  # init image for i2i/edit adapters
    strength: float | None = Field(default=None, gt=0, le=1)  # img2img denoise
    mark_sketch: bool = False  # tag results role_hint="sketch" for scribble adapters


class MeshHealthOut(BaseModel):
    ok: bool = False
    device: str = ""
    backend: str = ""
    error: str | None = None
    models: list[MeshModelOut] = Field(default_factory=list)
