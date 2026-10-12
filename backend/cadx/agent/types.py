"""Shared types for the agent layer (no DB imports — usable from CLI and worker)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cadx.db.models import ProjectSettings
from cadx.llm.schemas import ModelingPlan, PartSpec
from cadx.runtime_config import RuntimeSettings


@dataclass
class InputAsset:
    asset_id: str
    kind: str  # image | generated_image | reference_step | reference_stl | reference_dxf
    path: Path  # absolute path inside this process
    caption: str = ""
    filename: str = ""


# Asset kinds usable as image inputs (LLM context, mesh source images).
IMAGE_INPUT_KINDS = ("image", "generated_image")


@dataclass
class ParentContext:
    code: str = ""
    spec: PartSpec | None = None
    contact_sheet: Path | None = None
    number: int = 0
    resolved_mode: str | None = None  # cad | mesh | hybrid of the parent version
    mesh: dict[str, Any] | None = None  # parent version.mesh (orientation, model, seed...)
    mesh_solid: Path | None = None  # parent mesh/solid.brep when reusable (hybrid edit)
    analysis: dict[str, Any] | None = None  # parent version.analysis (anchors, measurements)


EmitFn = Callable[[str, int, str], Awaitable[None]]  # (step, iteration, message)
CancelFn = Callable[[], Awaitable[bool]]
UsageFn = Callable[[dict[str, Any]], Awaitable[None]]


class TraceSink(Protocol):
    async def record(
        self,
        *,
        step: str,
        iteration: int,
        model: str,
        messages: list[dict[str, Any]],
        response_text: str,
        parsed: dict[str, Any] | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> None: ...


class ExecutorPort(Protocol):
    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def export_all(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def process_mesh(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def fem_analyze(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def optimize(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AgentCtx:
    """Everything a step needs. Created by run_generation / run_edit / run_resume."""

    llm: Any  # LLMClient | FakeLLM
    executor: ExecutorPort
    trace: TraceSink
    emit: EmitFn
    check_cancel: CancelFn

    job_id: str
    project_id: uuid.UUID
    version_id: uuid.UUID
    mode: str  # "generate" | "edit" | "resume"
    prompt: str
    settings: ProjectSettings
    check_finish: CancelFn | None = None  # user asked to stop early, keep best
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    assets: list[InputAsset] = field(default_factory=list)
    # reusable library: file parts are appended to `assets` (load_reference),
    # code templates land here for the prompt only
    library_templates: list[dict[str, Any]] = field(default_factory=list)
    # ids the agent attached itself via attach_library_part during tool sessions
    library_attached: list[str] = field(default_factory=list)
    parent: ParentContext = field(default_factory=ParentContext)
    gen_mode: str = "auto"  # requested generation mode: auto | cad | mesh | hybrid
    resolved_mode: str | None = None  # effective mode: cad | mesh | hybrid
    mesh: Any = None  # MeshPort | None — mesh generator client
    mesh_available: bool = False
    image: Any = None  # ImagePort | None — text-to-image client (same sidecar)
    image_available: bool = False  # at least one image adapter is installed
    # asset ids the agent created itself (hero/concept images) — merged into
    # version.asset_ids by the worker so provenance survives restarts
    generated_asset_ids: list[str] = field(default_factory=list)
    # hybrid bridge: filled by _run_hybrid before the CAD loop starts
    hybrid_solid: Path | None = None  # path to solid.brep derived from the generated mesh
    hybrid_mesh_info: dict[str, Any] = field(default_factory=dict)  # bbox/volume/triangles
    mesh_info: dict[str, Any] = field(default_factory=dict)  # persisted to version.mesh
    mesh_orientation: dict[str, Any] | None = None  # {up, front, mirror_x} — orientation per job
    mesh_raw_reuse: Path | None = None  # already-generated raw GLB to reuse (edit/fallback)
    topopt_requested: bool = False  # gen_mode=topopt: base generation + SIMP phase
    # image measurement results (T5): asset_id -> ImageMeasurement dict
    measurements: dict[str, Any] = field(default_factory=dict)
    # reconciled hole/feature anchors (T9): FeatureAnchors dict
    anchors: dict[str, Any] | None = None

    # filled by steps
    spec: PartSpec | None = None
    plan: ModelingPlan | None = None
    budget_tokens: int = 400_000
    # True when max_iterations came from project/version config — an explicit
    # user cap defeats the adaptive iteration budget
    explicit_max_iterations: bool = False

    def tokens_used(self) -> int:
        u = getattr(self.llm, "total_usage", {})
        return int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
