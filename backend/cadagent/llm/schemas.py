"""Pydantic schemas exchanged with the LLM (kept flat for provider compatibility)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Dim(BaseModel):
    value: float
    unit: str = "mm"
    source: Literal["stated", "inferred", "assumed"] = "inferred"


class Feature(BaseModel):
    name: str = ""
    type: str = "other"  # hole|slot|pocket|boss|fillet|chamfer|rib|thread|insert|bearing_seat|gear|text|pattern|cutout|other
    description: str = ""
    dims: dict[str, float] = Field(default_factory=dict)
    position: str = ""
    count: int = 1
    source: str = "stated"
    standard: str = ""  # e.g. "M4 ISO 10642 countersunk, normal fit"


class ImageRole(BaseModel):
    asset_id: str = ""
    role: str = "other"  # front|back|left|right|top|bottom|iso|detail|drawing|other
    notes: str = ""


class OpenQuestion(BaseModel):
    question: str
    why_it_matters: str = ""
    default_assumption: str = ""
    blocking: bool = False


# --------------------------------------------------------------------------- #
# structural / FEM contracts (intake fills these when loads are mentioned)     #
# --------------------------------------------------------------------------- #


class RegionSpec(BaseModel):
    """A geometric region of the part: fixtures, load application, keep-zones.

    Regions are resolved on a voxel grid inside the executor. `face` selects the
    outer element layer on one bbox side; `cylinder` selects material around an
    axis line (e.g. a bore); `box` selects a cube; `all` selects the whole part.
    """

    kind: Literal["face", "cylinder", "box", "all"] = "face"
    axis: str = ""  # +x|-x|+y|-y|+z|-z (face normal or cylinder axis)
    center: list[float] = Field(default_factory=list)  # optional mm hint (e.g. hole pos)
    diameter_mm: float | None = None  # cylinder diameter / box edge length
    margin_mm: float = 2.0  # selection depth/extent added around the region
    description: str = ""  # human label, e.g. "mounting holes"


class LoadSpec(BaseModel):
    """One applied load on a region. Give either force_n OR pressure_mpa."""

    region: RegionSpec = Field(default_factory=RegionSpec)
    force_n: list[float] = Field(default_factory=list)  # [fx, fy, fz] in Newton
    pressure_mpa: float | None = None  # uniform pressure on a face region


class LoadCase(BaseModel):
    name: str = "case1"
    fixed: list[RegionSpec] = Field(default_factory=list)  # fully constrained regions
    loads: list[LoadSpec] = Field(default_factory=list)


class OptimizeSpec(BaseModel):
    """Structural optimization request extracted by intake."""

    enabled: bool = False
    target_volume_fraction: float = 0.4  # 0.1..0.9 of the design space
    max_iterations: int | None = None  # SIMP iterations (default: runtime setting)
    safety_factor_min: float = 1.5
    keep_regions: list[RegionSpec] = Field(default_factory=list)  # non-design solid
    note: str = ""


class PartSpec(BaseModel):
    name: str = "part"
    summary: str = ""
    units: str = "mm"
    length_x: Dim
    width_y: Dim
    height_z: Dim
    orientation_note: str = ""
    features: list[Feature] = Field(default_factory=list)
    symmetry: str = ""
    image_roles: list[ImageRole] = Field(default_factory=list)
    process: str = "fdm"
    material: str = ""
    tolerances_note: str = ""
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    must_keep: list[str] = Field(default_factory=list)
    # structural requirements (filled when the prompt mentions loads/strength)
    load_cases: list[LoadCase] = Field(default_factory=list)
    optimize: OptimizeSpec | None = None


class ParamDef(BaseModel):
    name: str
    value: float | int | bool | str
    unit: str = "mm"
    min: float | None = None
    max: float | None = None
    description: str = ""


class ModelingPlan(BaseModel):
    approach: Literal["cad", "mesh", "hybrid"] = "cad"
    approach_reason: str = ""
    mesh_subject: str = ""  # short description of the organic/freeform body for the generator
    mesh_source_images: list[str] = Field(default_factory=list)  # asset_ids, hero first
    mesh_up_hint: str = ""  # e.g. "the figure stands upright, front faces the camera"
    hybrid_cad_features: list[str] = Field(default_factory=list)  # precise features to add in CAD
    optimize: bool = False  # run the FEM/TopOpt phase after base generation
    optimize_reason: str = ""
    steps: list[str] = Field(default_factory=list)
    params: list[ParamDef] = Field(default_factory=list)
    standard_parts: list[str] = Field(default_factory=list)
    notes: str = ""


class MeshOrientation(BaseModel):
    """Which source axes of the generated mesh correspond to up (+Z) and front (+Y)."""

    up: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+z"
    front: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+y"
    confidence: float = 0.0
    reason: str = ""


class CodeResponse(BaseModel):
    reasoning: str = ""
    params: list[ParamDef] = Field(default_factory=list)
    expected_bbox: list[float] = Field(default_factory=list)
    code: str


class PatchEdit(BaseModel):
    """One localised text replacement inside the current model code.

    `find` must match the code file exactly once (verbatim, including
    indentation); `replace` is the new text put in its place.
    """

    find: str
    replace: str


class PatchResponse(BaseModel):
    """Repair/improve answer: minimal edits preferred, full_code as fallback."""

    reasoning: str = ""
    params: list[ParamDef] = Field(default_factory=list)
    expected_bbox: list[float] = Field(default_factory=list)
    edits: list[PatchEdit] = Field(default_factory=list)
    full_code: str | None = None


class Discrepancy(BaseModel):
    what: str = ""
    where: str = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: str = ""


class FeatureVerdict(BaseModel):
    """Per-spec-feature verdict from the critic — one entry per spec feature."""

    name: str = ""
    status: Literal["present", "missing", "wrong", "unsure"] = "unsure"
    note: str = ""


class Critique(BaseModel):
    score: int = 0
    passed: bool = False
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    feature_verdicts: list[FeatureVerdict] = Field(default_factory=list)
    matches: list[str] = Field(default_factory=list)
    confidence: float = 0.0
