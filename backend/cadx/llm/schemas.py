"""Pydantic schemas exchanged with the LLM (kept flat for provider compatibility)."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    - hole:     measured hole/feature anchor — `hole_index` (1-based) refers
                to the anchors listed in the context; the anchor's own axis
                (any of ±x/y/z) and 3D centre are used when present
    - anchor:   alias of hole — an indexed feature anchor (bore, insert,
                thread, bearing seat) on ANY axis
    - bore:     the bore WALL of a hole anchor — solid cells facing the void
                cylinder (for bearing loads, not a material ring around it)
    - points:   surface patch painted in the GUI — solid cells within
                `radius_mm` of any of `points` ([x,y,z] mm); `normal` is the
                averaged surface normal at the pick (drives pressure loads)
    - all:      the whole solid

    `fix_mode` applies when the region is used in `LoadCase.fixed`:
    - "all"    pins all 3 translational DOFs (encastre / glued, the default)
    - "normal" pins only the DOF along `axis` — a roller/sliding support or a
               symmetry plane (the part rests on a face but can still slide)
    `material` on a keep-region overrides the base material inside the region
    (e.g. "aluminum"/"brass" for a threaded insert) for FEM and TopOpt.
    """

    kind: Literal[
        "face", "surface", "area", "cylinder", "box", "hole", "anchor", "bore", "points", "all"
    ] = "face"
    axis: Annotated[str, Field(max_length=8)] = (
        ""  # +x|-x|+y|-y|+z|-z (face normal or cylinder axis)
    )
    center: list[float] = Field(default_factory=list, max_length=6)  # optional mm hint
    diameter_mm: float | None = None  # cylinder diameter / box edge length
    margin_mm: float = 2.0  # selection depth/extent added around the region
    polygon: list[list[float]] = Field(default_factory=list, max_length=40)  # XY mm polygon (area)
    x_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    y_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    hole_index: int | None = Field(
        default=None, ge=1
    )  # 1-based index into the measured anchors (hole)
    points: list[list[float]] = Field(
        default_factory=list, max_length=4000
    )  # painted surface points [x,y,z] mm (points)
    radius_mm: float | None = None  # brush radius (points); default max(2*margin_mm, 3)
    normal: list[float] = Field(default_factory=list, max_length=3)  # avg surface normal (points)
    fix_mode: Literal["all", "normal"] = "all"  # encastre vs roller/symmetry (fixed regions)
    material: Annotated[str, Field(max_length=40)] = ""  # material override (keep zones)
    description: M = ""  # human label, e.g. "mounting holes"

    @model_validator(mode="after")
    def _axis_from_description(self) -> Self:
        """Recover an intended axis that the model only wrote into the description.

        Directional kinds (face/surface/area/cylinder) silently fall back to +z when
        `axis` is empty — "underside (-Z)" describing a -z keep would otherwise pin
        the TOP skin. Word map is deliberately conservative: ambiguous terms like
        "back"/"rear" are not mapped.
        """
        if self.axis or self.kind in ("box", "points", "all"):
            return self
        desc = (self.description or "").lower()
        m = _AXIS_TOKEN.search(desc)
        if m:
            self.axis = m.group(0).strip("() ").lower()
            return self
        for ax, words in _AXIS_WORDS.items():
            if any(w in desc for w in words):
                self.axis = ax
                break
        return self


_AXIS_TOKEN = re.compile(r"(?<![A-Za-z0-9_+/-])([+-][xyz])(?![A-Za-z0-9_])", re.IGNORECASE)
_AXIS_WORDS: dict[str, tuple[str, ...]] = {
    "-z": ("underside", "bottom", "unterseite", "boden", "tang-contact"),
    "+z": ("top", "oberseite", "deckel"),
    "+y": ("front", "vorderseite"),
    "-x": ("left", "links"),
    "+x": ("right", "rechts"),
}


class LoadSpec(BaseModel):
    """One applied load on a region.

    Load kinds (pick ONE primary mechanism; combos are summed):
    - force_n [fx,fy,fz] — total force distributed over the region's nodes
    - pressure_mpa — uniform pressure acting against the region normal
    - bearing_n — bolt/bearing load: cosine-distributed radial force on a
      bore wall (hole/anchor/bore region), pointing along `bearing_dir`
      (or `force_n` when bearing_dir is empty)
    - moment_nm [mx,my,mz] — pure torque/couple distributed on the region
    - preload_n — bolt pretension: compressive axial pair on the two bore
      mouths of a hole/anchor region
    - remote_point [x,y,z] — force_n applied at that point instead of on the
      region: adds the moment (p-centroid)xF to the region
    - g_load [ax,ay,az] — body force in units of g on the region
      ([0,0,-1] = gravity; needs the region's material for mass)
    `direction_cone_deg` > 0 makes the optimizer evaluate the load over a cone
    of directions around force_n (robustness envelope for TopOpt).
    """

    region: RegionSpec = Field(default_factory=RegionSpec)
    force_n: list[float] = Field(default_factory=list, max_length=6)  # [fx, fy, fz]
    pressure_mpa: float | None = None  # uniform pressure on a face region
    bearing_n: float | None = None  # total bearing force on a bore wall (N)
    bearing_dir: list[float] = Field(
        default_factory=list, max_length=3
    )  # bearing dir (else force_n)
    moment_nm: list[float] = Field(default_factory=list, max_length=3)  # pure couple [N*m]
    preload_n: float | None = None  # bolt pretension along the bore axis (N)
    remote_point: list[float] = Field(default_factory=list, max_length=3)  # force lever point mm
    g_load: list[float] = Field(default_factory=list, max_length=3)  # body accel in units of g
    direction_cone_deg: float | None = Field(
        default=None, ge=0.0, le=180.0
    )  # robust TopOpt: sample load directions within this cone


class LoadCase(BaseModel):
    name: S = "case1"
    fixed: list[RegionSpec] = Field(default_factory=list, max_length=10)  # constrained
    loads: list[LoadSpec] = Field(default_factory=list, max_length=10)
    weight: float = 1.0  # relative importance for the TopOpt compliance sum

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: object) -> object:
        # API callers / LLM variants may use more natural names for `fixed`;
        # extra=ignore would otherwise silently drop them -> unsolvable case
        if isinstance(data, dict):
            for alias in ("fixed_regions", "supports", "constraints", "fixtures"):
                if alias in data:
                    if "fixed" not in data:
                        data = dict(data)
                        data["fixed"] = data[alias]
                    break
        return data


class OptimizeSpec(BaseModel):
    """Structural optimization request extracted by intake."""

    enabled: bool = False
    target_volume_fraction: float = 0.4  # 0.1..0.9 of the design space
    max_iterations: int | None = None  # SIMP iterations (default: runtime setting)
    safety_factor_min: float = 1.5
    keep_regions: list[RegionSpec] = Field(default_factory=list, max_length=10)
    void_regions: list[RegionSpec] = Field(
        default_factory=list, max_length=10
    )  # forbidden material zones — pinned empty (cable runs, swing space)
    keep_mode: Literal["manual", "auto"] = "manual"  # auto: hold every detected fastener anchor
    auto_keep_margin_mm: float | None = None  # ring around anchors when keep_mode=auto
    min_member_mm: float | None = None  # enforce: thicken members thinner than this
    density_threshold: float | None = None  # iso threshold 0.2..0.9 (default 0.5)
    warm_density_relpath: Annotated[str, Field(max_length=300)] = (
        ""  # density.npz to warm-start SIMP from
    )
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

    @field_validator("min", "max", mode="before")
    @classmethod
    def _bound_finite_or_none(cls, v: Any) -> Any:
        """Postgres JSON has no NaN/±Inf — drop non-finite bounds instead of crashing."""
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(float(v)):
            return None
        return v

    @field_validator("value", mode="before")
    @classmethod
    def _value_finite(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(float(v)):
            return 0.0
        return v


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
