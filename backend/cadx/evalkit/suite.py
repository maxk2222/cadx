"""Suite definitions — the committed benchmark spec layer.

Cases are JSON files under ``evals/suite/``; each case pins one canonical
object (difficulty 1..3), a fixed ``eval:*`` project name and an ordered
stage list. A stage maps to exactly one pipeline phase; gates are
declarative checks evaluated by ``evalkit.gates``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cadx.evalkit.paths import SUITE_DIR

# pipeline phases a stage can dispatch. `generate`/`edit` run the agent loop;
# the rest are deterministic job/asset stages.
STAGE_TYPES = {
    "generate",
    "edit",
    "params",
    "edit_feature",
    "optimize",
    "resample",
    "refine",
    "organic",
    "reexport",
    "fem",
    "mesh_edit",
    "meshgen",
    "imagegen",
    "imageedit",
}

# stages that produce a child Version+Job (vs same-version jobs / assets)
CHILD_VERSION_STAGES = {
    "generate",
    "edit",
    "params",
    "edit_feature",
    "optimize",
    "resample",
    "refine",
    "organic",
    "mesh_edit",
    "meshgen",
}
SAME_VERSION_STAGES = {"fem", "reexport"}
ASSET_STAGES = {"imagegen", "imageedit"}


class EvalStage(BaseModel):
    """One pipeline phase inside a case. Fields are type-dependent; the
    validator enforces the per-type requirements."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    enabled: bool = True
    prompt: str = ""
    mode: str | None = None  # auto|cad|mesh|hybrid|topopt (generate/edit)
    options: dict[str, Any] = Field(default_factory=dict)  # RunOptions
    images: list[str] = Field(default_factory=list)  # repo-relative image paths
    from_stage: str | None = None  # explicit parent stage id (default: previous)
    # stage payloads
    params: dict[str, Any] | None = None  # params override stage
    feature: dict[str, Any] | None = None  # edit_feature: {feature_id|dia, value}
    load_cases: list[dict[str, Any]] = Field(default_factory=list)
    keep_regions: list[dict[str, Any]] = Field(default_factory=list)
    material: str = ""
    optimize: dict[str, Any] = Field(default_factory=dict)  # OptimizeSpec
    resample: dict[str, Any] = Field(default_factory=dict)  # density_threshold etc
    features: list[dict[str, Any]] = Field(default_factory=list)  # refine selection
    ops: list[dict[str, Any]] = Field(default_factory=list)  # mesh_edit ops
    post: dict[str, Any] | None = None  # mesh post params
    organic: dict[str, Any] = Field(default_factory=dict)  # OrganicParams
    voxel_mm: float | None = None
    restore_planes: bool | None = None
    # image stages
    image_prompt: str = ""
    negative: str = ""
    count: int = 1
    width: int = 768
    height: int = 768
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    reference: str | None = None  # "prev-asset" | asset uuid (imageedit/meshgen)
    strength: float | None = None
    subject: str = ""  # meshgen subject hint
    model: str = ""  # mesh/image model override
    assets_from: str | None = None  # stage id whose image assets feed this stage
    # execution
    timeout_s: float = 7200.0
    use_parent_fallbacks: bool = True  # empty load_cases -> fall back to parent spec
    gates: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_type_fields(self) -> EvalStage:
        if self.type not in STAGE_TYPES:
            raise ValueError(f"unknown stage type {self.type!r}")
        if self.type in ("generate", "edit") and not self.prompt.strip():
            raise ValueError(f"stage {self.id}: {self.type} needs a prompt")
        if self.type == "params" and not self.params:
            raise ValueError(f"stage {self.id}: params stage needs `params`")
        if self.type == "mesh_edit" and not self.ops:
            raise ValueError(f"stage {self.id}: mesh_edit stage needs `ops`")
        if self.type in ("imagegen", "imageedit") and not self.image_prompt.strip():
            raise ValueError(f"stage {self.id}: {self.type} needs `image_prompt`")
        if self.type == "imageedit" and not (self.reference or self.assets_from):
            raise ValueError(f"stage {self.id}: imageedit needs `reference` or `assets_from`")
        if self.type == "edit_feature" and not self.feature:
            raise ValueError(f"stage {self.id}: edit_feature needs `feature`")
        return self


class EvalCase(BaseModel):
    """One canonical benchmark object — fixed eval project + stage chain."""

    model_config = ConfigDict(extra="allow")

    name: str
    project: str = ""  # defaults to eval:<name>
    difficulty: Literal[1, 2, 3] = 1
    description: str = ""
    images: list[str] = Field(default_factory=list)  # case-level input images
    settings: dict[str, Any] = Field(default_factory=dict)  # ProjectSettings overrides
    runtime: dict[str, Any] = Field(default_factory=dict)  # RuntimeSettings overrides
    stages: list[EvalStage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_stages(self) -> EvalCase:
        if not self.project:
            self.project = f"eval:{self.name}"
        ids = [s.id for s in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError(f"case {self.name}: duplicate stage ids")
        seen: set[str] = set()
        for s in self.stages:
            if s.from_stage and s.from_stage not in seen:
                raise ValueError(
                    f"case {self.name} stage {s.id}: from_stage {s.from_stage!r} "
                    "must reference an earlier stage"
                )
            if s.assets_from and s.assets_from not in seen:
                raise ValueError(
                    f"case {self.name} stage {s.id}: assets_from {s.assets_from!r} "
                    "must reference an earlier stage"
                )
            seen.add(s.id)
        if self.stages and self.stages[0].type not in ("generate", "meshgen"):
            raise ValueError(
                f"case {self.name}: first stage must be 'generate' or 'meshgen' "
                f"(got {self.stages[0].type})"
            )
        return self

    def enabled_stages(self, only: set[str] | None = None) -> list[EvalStage]:
        """Stages to execute: enabled, optionally filtered by --stages ids/types."""
        return [
            s for s in self.stages if s.enabled and (only is None or s.id in only or s.type in only)
        ]


def load_suite(suite_dir: Path | None = None) -> list[EvalCase]:
    """Load every case JSON under the suite dir, sorted by difficulty."""
    d = suite_dir or SUITE_DIR
    out: list[EvalCase] = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.setdefault("name", p.stem)
            out.append(EvalCase.model_validate(data))
        except Exception as e:
            raise ValueError(f"suite case {p.name}: {e}") from e
    out.sort(key=lambda c: c.difficulty)
    return out


def load_case(name: str, suite_dir: Path | None = None) -> EvalCase | None:
    for c in load_suite(suite_dir):
        if c.name == name:
            return c
    return None
