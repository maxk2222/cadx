"""Pydantic schemas exchanged with the LLM (kept flat for provider compatibility)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Short/Medium/Long string bounds — guided decoding honours `maxLength`, which
# stops a weak model looping inside an unbounded string field (observed: an
# intake run burned 8000 tokens inside one `notes`-style value).
S = Annotated[str, Field(max_length=80)]
M = Annotated[str, Field(max_length=300)]
L = Annotated[str, Field(max_length=1200)]

# `json_schema_extra={"required": [...]}` marks fields as REQUIRED in the JSON
# schema sent to the LLM while keeping the Python-side defaults: guided decoding
# (vLLM json_schema mode) then forces the model to actually emit those keys —
# weak models otherwise produce degenerate minimal JSON like
# {"score": 40, "passed": false} with no discrepancies, verdicts or features.
# Parsing stays tolerant (defaults still apply when a key is missing).
# `max_length` on list fields emits `maxItems` — a grammar-level bound that
# stops runaway array generation (observed: intake looping inside `features`
# until the 8000-token cap, killing the stream).


class Dim(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": ["value", "unit", "source"]})

    value: float
    unit: S = "mm"
    source: Literal["stated", "inferred", "assumed"] = "inferred"


class Feature(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["name", "type", "description", "dims", "position", "count", "source"]
        }
    )

    name: S = ""
    type: Annotated[str, Field(max_length=40)] = (
        "other"  # hole|slot|pocket|boss|fillet|chamfer|rib|thread|insert|bearing_seat|gear|text|pattern|cutout|other
    )
    description: M = ""
    # closed-key dim vocabulary in the *schema* (Python type stays dict):
    # free-form keys let weak models loop inside a dict KEY (unbounded string,
    # vLLM rejects `propertyNames` so it can't be grammar-bounded directly)
    dims: dict[str, float] = Field(
        default_factory=dict,
        json_schema_extra={
            "properties": {
                k: {"type": "number"}
                for k in (
                    "diameter",
                    "radius",
                    "depth",
                    "height",
                    "length",
                    "width",
                    "thickness",
                    "wall_thickness",
                    "depth_total",
                    "depth_partial",
                    "fillet_radius",
                    "chamfer",
                    "chamfer_distance",
                    "draft_angle",
                    "countersink_diameter",
                    "countersink_depth",
                    "countersink_angle",
                    "counterbore_diameter",
                    "counterbore_depth",
                    "slot_width",
                    "slot_length",
                    "distance",
                    "spacing",
                    "pitch",
                    "angle",
                    "count",
                    "head_diameter",
                    "head_depth",
                    "bore_diameter",
                    "thread_pitch",
                    "tolerance",
                )
            },
            "additionalProperties": False,
        },
    )
    position: M = ""
    axis: Annotated[str, Field(max_length=8)] = (
        ""  # x|y|z — direction a hole goes through / a slope or taper varies along
    )
    count: int = 1
    source: Annotated[str, Field(max_length=16)] = "stated"
    standard: Annotated[str, Field(max_length=160)] = (
        ""  # e.g. "M4 ISO 10642 countersunk, normal fit"
    )


class ImageRole(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": ["asset_id", "role"]})

    asset_id: S = ""
    role: Annotated[str, Field(max_length=40)] = (
        "other"  # front|back|left|right|top|bottom|iso|detail|drawing|other
    )
    notes: M = ""


class OpenQuestion(BaseModel):
    question: M
    why_it_matters: M = ""
    default_assumption: M = ""
    blocking: bool = False


# --------------------------------------------------------------------------- #
# structural / FEM contracts (intake fills these when loads are mentioned)     #
# --------------------------------------------------------------------------- #


class RegionSpec(BaseModel):
    """A geometric region of the part: fixtures, load application, keep-zones.

    Regions are resolved on a voxel grid inside the executor. Kinds:
    - face:     outer element layer on one bbox side (legacy alias of surface)
    - surface:  actual part surface facing `axis` — works on curved parts
    - area:     XY footprint sub-area (polygon/x_range/y_range/center+diameter);
                add `axis` to restrict it to the surface facing that way
    - cylinder: material within radius around an axis line (e.g. a bore)
    - box:      cube centred at `center` with edge `diameter_mm` (or 4*margin)
    - hole:     measured hole/feature region — `hole_index` (1-based) refers
                to the anchors listed in the context
    - all:      the whole solid
    """

    kind: Literal["face", "surface", "area", "cylinder", "box", "hole", "all"] = "face"
    axis: Annotated[str, Field(max_length=8)] = (
        ""  # +x|-x|+y|-y|+z|-z (face normal or cylinder axis)
    )
    center: list[float] = Field(default_factory=list, max_length=6)  # optional mm hint
    diameter_mm: float | None = None  # cylinder diameter / box edge length
    margin_mm: float = 2.0  # selection depth/extent added around the region
    polygon: list[list[float]] = Field(default_factory=list, max_length=40)  # XY mm polygon (area)
    x_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    y_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    hole_index: int | None = None  # 1-based index into the measured anchors (hole)
    description: M = ""  # human label, e.g. "mounting holes"


class LoadSpec(BaseModel):
    """One applied load on a region. Give either force_n OR pressure_mpa."""

    region: RegionSpec = Field(default_factory=RegionSpec)
    force_n: list[float] = Field(default_factory=list, max_length=6)  # [fx, fy, fz]
    pressure_mpa: float | None = None  # uniform pressure on a face region


class LoadCase(BaseModel):
    name: S = "case1"
    fixed: list[RegionSpec] = Field(default_factory=list, max_length=10)  # constrained
    loads: list[LoadSpec] = Field(default_factory=list, max_length=10)


class OptimizeSpec(BaseModel):
    """Structural optimization request extracted by intake."""

    enabled: bool = False
    target_volume_fraction: float = 0.4  # 0.1..0.9 of the design space
    max_iterations: int | None = None  # SIMP iterations (default: runtime setting)
    safety_factor_min: float = 1.5
    keep_regions: list[RegionSpec] = Field(default_factory=list, max_length=10)
    note: M = ""


class PartSpec(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "name",
                "summary",
                "length_x",
                "width_y",
                "height_z",
                "features",
                "image_roles",
            ]
        }
    )

    name: S = "part"
    summary: L = ""
    units: Annotated[str, Field(max_length=8)] = "mm"
    length_x: Dim
    width_y: Dim
    height_z: Dim
    orientation_note: M = ""
    features: list[Feature] = Field(default_factory=list, max_length=40)
    symmetry: S = ""
    image_roles: list[ImageRole] = Field(default_factory=list, max_length=20)
    process: Annotated[str, Field(max_length=40)] = "fdm"
    material: Annotated[str, Field(max_length=120)] = ""
    tolerances_note: M = ""
    assumptions: list[M] = Field(default_factory=list, max_length=20)
    open_questions: list[OpenQuestion] = Field(default_factory=list, max_length=12)
    must_keep: list[M] = Field(default_factory=list, max_length=20)
    # structural requirements (filled when the prompt mentions loads/strength)
    load_cases: list[LoadCase] = Field(default_factory=list, max_length=8)
    optimize: OptimizeSpec | None = None


class ParamDef(BaseModel):
    name: S
    value: float | int | bool | S
    unit: Annotated[str, Field(max_length=16)] = "mm"
    min: float | None = None
    max: float | None = None
    description: M = ""


class ModelingPlan(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["approach", "approach_reason", "strategy", "optimize", "steps", "params"]
        }
    )

    approach: Literal["cad", "mesh", "hybrid"] = "cad"
    approach_reason: M = ""
    strategy: list[Literal["cad", "mesh", "hybrid"]] = Field(
        default_factory=list, max_length=2
    )  # ordered approaches; [approach] when empty
    switch_rule: M = ""  # short note on when to switch to the next approach
    use_image_profile: bool = False  # build the body from the measured plan view
    mesh_subject: M = ""  # short description of the organic/freeform body for the generator
    mesh_source_images: list[S] = Field(default_factory=list, max_length=10)
    mesh_up_hint: M = ""  # e.g. "the figure stands upright, front faces the camera"
    hybrid_cad_features: list[M] = Field(default_factory=list, max_length=20)
    optimize: bool = False  # run the FEM/TopOpt phase after base generation
    optimize_reason: M = ""
    steps: list[M] = Field(default_factory=list, max_length=30)
    params: list[ParamDef] = Field(default_factory=list, max_length=40)
    standard_parts: list[S] = Field(default_factory=list, max_length=30)
    notes: L = ""


class MeshOrientation(BaseModel):
    """Which source axes of the generated mesh correspond to up (+Z) and front (+Y)."""

    model_config = ConfigDict(json_schema_extra={"required": ["up", "front", "confidence"]})

    up: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+z"
    front: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+y"
    confidence: float = 0.0
    reason: M = ""


class CodeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"required": ["reasoning", "expected_bbox", "code"]}
    )

    reasoning: L = ""
    params: list[ParamDef] = Field(default_factory=list, max_length=40)
    expected_bbox: list[float] = Field(default_factory=list, max_length=6)
    code: str


class PatchEdit(BaseModel):
    """One localised text replacement inside the current model code.

    `find` must match the code file exactly once (verbatim, including
    indentation); `replace` is the new text put in its place.
    """

    find: Annotated[str, Field(max_length=4000)]
    replace: Annotated[str, Field(max_length=4000)]


class PatchResponse(BaseModel):
    """Repair/improve answer: minimal edits preferred, full_code as fallback."""

    reasoning: L = ""
    params: list[ParamDef] = Field(default_factory=list, max_length=40)
    expected_bbox: list[float] = Field(default_factory=list, max_length=6)
    edits: list[PatchEdit] = Field(default_factory=list, max_length=40)
    full_code: str | None = None


class Discrepancy(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"required": ["what", "where", "severity", "fix_hint"]}
    )

    what: M = ""
    where: M = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: M = ""


class FeatureVerdict(BaseModel):
    """Per-spec-feature verdict from the critic — one entry per spec feature."""

    model_config = ConfigDict(json_schema_extra={"required": ["name", "status", "note"]})

    name: S = ""
    status: Literal["present", "missing", "wrong", "unsure"] = "unsure"
    note: M = ""


class Critique(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["score", "passed", "discrepancies", "feature_verdicts", "matches"]
        }
    )

    score: int = 0
    passed: bool = False
    discrepancies: list[Discrepancy] = Field(default_factory=list, max_length=20)
    feature_verdicts: list[FeatureVerdict] = Field(default_factory=list, max_length=50)
    matches: list[M] = Field(default_factory=list, max_length=20)
    confidence: float = 0.0
