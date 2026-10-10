"""Shared types for the agent layer (no DB imports — usable from CLI and worker)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cadagent.db.models import ProjectSettings
from cadagent.llm.schemas import ModelingPlan, PartSpec


@dataclass
class InputAsset:
    asset_id: str
    kind: str  # image | reference_step | reference_stl | reference_dxf
    path: Path  # absolute path inside this process
    caption: str = ""
    filename: str = ""


@dataclass
class ParentContext:
    code: str = ""
    spec: PartSpec | None = None
    contact_sheet: Path | None = None
    number: int = 0
    resolved_mode: str | None = None  # cad | mesh | hybrid of the parent version
    mesh: dict[str, Any] | None = None  # parent version.mesh (orientation, model, seed...)
    mesh_solid: Path | None = None  # parent mesh/solid.brep when reusable (hybrid edit)


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
    assets: list[InputAsset] = field(default_factory=list)
    parent: ParentContext = field(default_factory=ParentContext)
    gen_mode: str = "auto"  # requested generation mode: auto | cad | mesh | hybrid
    resolved_mode: str | None = None  # effective mode: cad | mesh | hybrid
    mesh: Any = None  # MeshPort | None — mesh generator client
    mesh_available: bool = False
    # hybrid bridge: filled by _run_hybrid before the CAD loop starts
    hybrid_solid: Path | None = None  # path to solid.brep derived from the generated mesh
    hybrid_mesh_info: dict[str, Any] = field(default_factory=dict)  # bbox/volume/triangles
    mesh_info: dict[str, Any] = field(default_factory=dict)  # persisted to version.mesh
    mesh_orientation: dict[str, str] | None = None  # {up, front} — LLM/heuristic choice per job
    mesh_raw_reuse: Path | None = None  # already-generated raw GLB to reuse (edit/fallback)

    # filled by steps
    spec: PartSpec | None = None
    plan: ModelingPlan | None = None
    budget_tokens: int = 400_000

    def tokens_used(self) -> int:
        u = getattr(self.llm, "total_usage", {})
        return int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
