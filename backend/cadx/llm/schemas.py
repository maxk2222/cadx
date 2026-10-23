"""Pydantic schemas exchanged with the LLM (kept flat for provider compatibility)."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

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


class FeatureCenter(BaseModel):
    """Absolute centre of one feature instance in mm — one entry per
    repetition of a repeated feature (holes, slots, bosses). X/Y are
    centred on the origin, Z measured up from 0 (the coordinate contract);
    an axis the spec never names stays ``None``."""

    x: float | None = None
    y: float | None = None
    z: float | None = None


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
    # structured per-instance centres ("two holes at x=-25 and x=+45, y=0")
    # — beats re-parsing `position` text downstream; NOT in `required` so
    # older specs/fixtures still parse
    centers: list[FeatureCenter] = Field(default_factory=list, max_length=24)
    axis: Annotated[str, Field(max_length=8)] = (
        ""  # x|y|z — direction a hole goes through / a slope or taper varies along
    )
    count: int = 1
    source: Annotated[str, Field(max_length=16)] = "stated"
    standard: Annotated[str, Field(max_length=160)] = (
        ""  # e.g. "M4 ISO 10642 countersunk, normal fit"
    )
    thread_spec: Annotated[str, Field(max_length=16)] = (
        ""  # 'M4' | 'M4x0.5' — re-cut as exact thread after topopt finishing
    )


class StandardPartSpec(BaseModel):
    model_config = ConfigDict(json_schema_extra={"required": ["class_name"]})

    class_name: S = ""
    fastener_type: S = ""
    size: S = ""
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=20)
    quantity: int = Field(default=1, ge=1, le=100)
    length_mm: float | None = Field(default=None, gt=0, le=10000)
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    placements_mm: list[tuple[float, float, float]] = Field(default_factory=list, max_length=100)
    axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] | None = None
    threaded: bool | None = None
    covers_features: list[S] = Field(default_factory=list, max_length=20)
    name: S = ""


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
    - cylinder: material within radius around an axis line (e.g. a bore);
                `inner_diameter_mm` makes it a hollow annulus/ring
    - rim:      outer radial band of width `margin_mm` following the part
                silhouette around `axis` (default +z) — the tooth ring of a
                gear/sprocket; `diameter_mm`+`inner_diameter_mm` select an
                exact annulus instead
    - border:   band of width `margin_mm` at the bbox outer edges along the
                axes named in `axis` ("x"/"xy"/"xyz" both sides, "+x" etc.
                one side only; default "xy")
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
        "face",
        "surface",
        "area",
        "cylinder",
        "rim",
        "border",
        "box",
        "hole",
        "anchor",
        "bore",
        "points",
        "all",
    ] = "face"
    axis: Annotated[str, Field(max_length=8)] = (
        ""  # +x|-x|+y|-y|+z|-z (face normal or cylinder axis)
    )
    center: list[float] = Field(default_factory=list, max_length=6)  # optional mm hint
    diameter_mm: float | None = None  # cylinder diameter / box edge length
    inner_diameter_mm: float | None = Field(
        default=None, ge=0.0
    )  # hollow cylinder/annulus inner diameter (cylinder/area/rim)
    margin_mm: float = 2.0  # selection depth/extent added around the region
    polygon: list[list[float]] = Field(default_factory=list, max_length=40)  # XY mm polygon (area)
    x_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    y_range: list[float] = Field(default_factory=list, max_length=2)  # [min,max] mm (area)
    z_range: list[float] = Field(
        default_factory=list, max_length=2
    )  # [min,max] mm (area/surface/rim/border/cylinder)
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
        # rim/border carry sensible axis defaults of their own — a gear rim
        # wraps around +z, a border band guards the lateral (XY) bbox edges
        if not self.axis:
            if self.kind == "rim":
                self.axis = "+z"
            elif self.kind == "border":
                self.axis = "xy"
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
    - normal_force_n — total force orthogonal to the region's surface:
      + pushes into the part, - pulls; the outward normal comes from the
      region axis/normal or the exposed skin of the selected cells
    - preload_n — bolt pretension: compressive axial pair on the two bore
      mouths of a hole/anchor region
    - remote_point [x,y,z] — force_n applied at that point instead of on the
      region: adds the moment (p-centroid)xF to the region (needs force_n)
    - g_load [ax,ay,az] — body force in units of g on the region
      ([0,0,-1] = gravity; needs the region's material for mass)
    `direction_cone_deg` > 0 makes the optimizer evaluate the load over a cone
    of directions around its primary direction (force_n, else moment_nm, else
    bearing_dir); the sampled rotation tilts ALL directional vectors of the
    load together so combined force+couple loads stay coherent.
    """

    region: RegionSpec = Field(default_factory=RegionSpec)
    force_n: list[float] = Field(default_factory=list, max_length=6)  # [fx, fy, fz]
    pressure_mpa: float | None = None  # uniform pressure on a face region
    bearing_n: float | None = None  # total bearing force on a bore wall (N)
    bearing_dir: list[float] = Field(
        default_factory=list, max_length=3
    )  # bearing dir (else force_n)
    moment_nm: list[float] = Field(default_factory=list, max_length=3)  # pure couple [N*m]
    normal_force_n: float | None = (
        None  # total N orthogonal to the region face (+ pushes in, - pulls out)
    )
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
    target_volume_fraction: float = Field(default=0.4, ge=0.05, le=0.95)  # of design space
    max_iterations: int | None = Field(default=None, ge=1, le=2000)  # SIMP iterations
    safety_factor_min: float = Field(default=1.5, ge=0.5, le=1e6)  # huge = always retry
    keep_regions: list[RegionSpec] = Field(default_factory=list, max_length=10)
    void_regions: list[RegionSpec] = Field(
        default_factory=list, max_length=10
    )  # forbidden material zones — pinned empty (cable runs, swing space)
    keep_mode: Literal["manual", "auto"] = "manual"  # auto: hold every detected fastener anchor
    auto_keep_margin_mm: float | None = Field(default=None, ge=0.0, le=50.0)
    min_member_mm: float | None = Field(default=None, ge=0.0, le=100.0)
    min_gap_mm: float | None = Field(default=None, ge=0.0, le=100.0)
    density_threshold: float | None = Field(default=None, ge=0.2, le=0.9)
    extract_smooth: bool = (
        True  # sub-voxel extraction: marching cubes on the smoothed density field
    )
    extract_sigma_px: float = Field(
        default=0.8, ge=0.0, le=5.0
    )  # Gaussian sigma in voxels for the iso field
    volume_preserve: bool = (
        True  # bisect the iso level so mesh volume matches the FEM-verified mask
    )
    warm_density_relpath: Annotated[str, Field(max_length=300)] = (
        ""  # density.npz to warm-start SIMP from
    )
    multires: bool = False  # coarse SIMP at 2x pitch, warm-start the fine grid (~3x faster)
    # manufacturability filters (SIMP)
    self_supporting: bool = False  # AM filter: cells need 45° support from below — no overhangs
    print_direction: Annotated[str, Field(max_length=3)] = (
        "+z"  # build direction for the self-supporting filter (+x/-x/...)
    )
    filter: Literal["cone", "heaviside", "helmholtz", "helmholtz+heaviside"] = (
        "helmholtz+heaviside"  # PDE filter (Lazarov & Sigmund) + 0/1
        # projection: crisp fields and consistent feature size right at the
        # domain boundary (cone thins edge members) — threshold-robust
        # extraction; "cone" remains the cheap fallback for debugging
    )
    robust: bool = (
        False  # robust formulation: worst-case over eroded/intermediate/dilated
        # Heaviside projections — print-tolerant, threshold-insensitive (~3x FEM)
    )
    robust_delta_eta: float = Field(
        default=0.25, ge=0.0, le=0.45
    )  # eta half-width of the eroded/dilated interval
    robust_designs: Literal["edi", "e"] = "edi"  # e = eroded only (~half the extra cost)
    symmetry: Literal["xy", "xz", "yz"] | None = (
        None  # mirror the design about a domain mid-plane — density
        # averaged with its reflection each SIMP iteration so the
        # printed part comes out symmetric
    )
    penal: float | None = Field(default=None, ge=1.0, le=8.0)  # SIMP exponent (default 3.0)
    penal_init: float | None = Field(default=None, ge=1.0, le=8.0)  # continuation start
    solver: Literal["oc", "mma", "beso"] = (
        "oc"  # oc (fast, volume-only) | mma (constraints) | beso (binary 0/1
        # evolutionary update — crisp binary designs, no gray/threshold
        # ambiguity; volume-only, ignores the constraint fields below)
    )
    case_objective: Literal["weighted", "minmax"] = (
        "weighted"  # minmax = robust worst-case over load cases (needs mma)
    )
    max_displacement_mm: float | None = Field(default=None, gt=0.0, le=1e4)
    max_stress_mpa: float | None = Field(default=None, gt=0.0, le=1e5)
    volfrac_init: float | None = Field(
        default=None, ge=0.05, le=0.95
    )  # volfrac continuation start (e.g. 0.7 → ramps to target) —
    # escapes the same local minima as penal_init; ignored when MMA
    # constraints are active (the volume bound must hold from iter 0)
    warmstart: Literal["uniform", "loadpath"] = (
        "uniform"  # loadpath: deterministic Dijkstra corridor field between
        # load centroids and fixed regions seeds x0 — skips the broad-search
        # phase, typically ~20-40% fewer iterations
    )
    volfrac_ramp_iters: int | None = Field(
        default=None, ge=1, le=200
    )  # total iterations for staged volfrac_init -> target continuation
    # (default 40); shorter schedules may advance before the stress layout settles
    loadpath_corridor_mm: float | None = Field(
        default=None, gt=0.0, le=50.0
    )  # corridor width for warmstart="loadpath" (default 2.5xvoxel pitch);
    # wider corridors seed thicker junctions — survives smoothing better
    mma_final_beta: float | None = Field(
        default=None, ge=1.0, le=32.0
    )  # solver="mma" only: project the converged blueprint at this
    # Heaviside sharpness for the returned density — collapses residual
    # gray fields from a cone-filtered run (e.g. 8.0)
    grow_mm: float = Field(
        default=0.0, ge=0.0, le=50.0
    )  # design-domain dilation: the optimizer may also ADD material up to
    # grow_mm beyond the part silhouette (reinforcement ribs, enlarged
    # walls) — regions still resolve on the true part surface
    min_cells_thinnest: int | None = Field(
        default=None, ge=1, le=32
    )  # voxel layers guaranteed through the thinnest spec axis when the
    # design pitch is snapped to spec dims (default 3 in the pipelines);
    # raise for finer thin-wall resolution, set 1 to keep the raw pitch
    aniso_material: bool = (
        False  # orthotropic FDM stiffness along print_direction (weaker
        # interlayer direction); ignored without a print_direction effect
    )
    aniso_factor: float | None = Field(
        default=None, ge=0.05, le=1.0
    )  # layer-normal stiffness factor when aniso_material is on (default 0.6 —
    # typical FDM interlayer bond strength vs in-plane)
    beso_er: float = Field(
        default=0.02, ge=0.005, le=0.2
    )  # BESO evolutionary ratio — volume-ramp step per iteration (solver=beso)
    # selectable structural strategy — "simp" (default, density TopOpt) or
    # "infill" (shell + physics-graded internal lattice; ~2 FEM solves)
    structure_mode: Literal["simp", "infill"] = "simp"
    infill_scope: Literal["full", "guided"] = (
        "guided"  # full: infill the whole part | guided: coarse SIMP picks
        # the retained topology, infill only fills what it keeps
    )
    infill_style: Literal["gyroid_graded", "struts_aligned"] = (
        "gyroid_graded"  # struts_aligned orients the lattice to the solved
        # principal stress directions
    )
    infill_shell_mm: float = Field(
        default=2.0, ge=0.0, le=20.0
    )  # closed outer skin thickness under which the lattice lives
    infill_period_mm: float = Field(
        default=8.0, ge=2.0, le=40.0
    )  # lattice cell size (modulated ±50% by local stiffness)
    infill_min_density: float = Field(
        default=0.25, ge=0.0, le=0.9
    )  # lattice occupancy floor — keeps the core connected and printable
    infill_max_density: float = Field(
        default=0.9, ge=0.1, le=1.0
    )  # lattice occupancy cap at energy hotspots (~solid below the skin)
    infill_alpha: float = Field(
        default=0.7, ge=0.1, le=3.0
    )  # energy→density grading exponent (<1 flattens, >1 concentrates)
    infill_seed: int = Field(
        default=0, ge=0, le=2_147_483_647
    )  # deterministic lattice phase — same seed + same mesh = same part
    infill_octet: bool | None = Field(
        default=None
    )  # body-diagonal strut families for struts_aligned (real 45° members
    # instead of stair-stepped axes); None -> runtime default infill_octet
    # verification hardening overrides — None follows the runtime settings:
    verify_robust: bool | None = Field(
        default=None
    )  # export-mask realizations (eroded/nominal/dilated) + coverage-weighted
    # boundary stiffness; None -> fem_verify_realizations/_coverage_e
    verify_homogenized: bool | None = Field(
        default=None
    )  # homogenized C_H(rho) infill verification model alongside the
    # resolved binary mask; None -> fem_homogenized_infill
    # post-optimization feature finishing (POST /versions/{id}/refine)
    auto_refine_features: bool = (
        False  # auto-enqueue feature re-cut (bores/threads/inserts) after optimize
    )
    restore_planes: bool = (
        True  # union original keep-region surfaces back (Taubin shrinks them ~1 pitch)
    )
    auto_plane_restore: bool = (
        False  # auto-detect dominant planar base faces and restore them after TopOpt
    )
    helical_threads: bool = False  # refine: cut real helical thread grooves (experimental, FDM)
    recut_bores: bool = True  # restore keep surfaces and deterministically re-cut real base bores
    note: M = ""

    @field_validator(
        "solver",
        "case_objective",
        "filter",
        "symmetry",
        "warmstart",
        "structure_mode",
        "infill_scope",
        "infill_style",
        mode="before",
    )
    @classmethod
    def _enum_or_default(cls, v: Any, info: ValidationInfo) -> Any:
        """Silent tolerance for hallucinated enum values: anything outside
        the Literal vocab (unknown strings, "", None on required fields)
        falls back to the documented default instead of failing the whole
        OptimizeSpec parse."""
        allowed: dict[str, tuple[str, ...]] = {
            "solver": ("oc", "mma", "beso"),
            "case_objective": ("weighted", "minmax"),
            "filter": ("cone", "heaviside", "helmholtz", "helmholtz+heaviside"),
            "symmetry": ("xy", "xz", "yz"),
            "warmstart": ("uniform", "loadpath"),
            "structure_mode": ("simp", "infill"),
            "infill_scope": ("full", "guided"),
            "infill_style": ("gyroid_graded", "struts_aligned"),
        }
        defaults: dict[str, Any] = {
            "solver": "oc",
            "case_objective": "weighted",
            "filter": "helmholtz+heaviside",
            "symmetry": None,
            "warmstart": "uniform",
            "structure_mode": "simp",
            "infill_scope": "guided",
            "infill_style": "gyroid_graded",
        }
        s = v.strip().lower() if isinstance(v, str) else v
        if s in allowed[info.field_name]:
            return s
        return defaults[info.field_name]


class OrganicParams(BaseModel):
    """Pseudo-topology-optimization request — a purely procedural organic
    restyling. NOT a structural optimization: no FEM, no load cases, no
    safety factor. Shared by the API request body, the `organic` MeshOp
    args (live preview) and the executor payload, so preview and committed
    child version run the identical parameter set.

    style:          procedural density field — gyroid (TPMS sheet),
                    trabecular (bone-like smoothed noise), voronoi (cell
                    walls), branches (floor-rooted struts/canopy)
    volfrac:        fraction of the design space kept solid (0.05..0.9)
    seed:           deterministic seed — same params + same mesh = same part
    period_mm:      characteristic feature size (cell size / blob size)
    surface_open:   fraction of the outer skin the field may carve (0..1):
                    0 = solid shell (structure hidden inside), 1 = open
                    skeleton (no skin protection). Default 0.35 keeps ~65%
                    of the skin — silhouette stays readable.
    shell_mm:       skin depth affected by surface_open (0 = auto:
                    ~0.2*period_mm, clamped to [1.6*pitch, 10] mm)
    floor_mm:       always-solid stand band above the part's min Z
    anchors:        resolved anchor table — auto-injected by apply_op when
                    auto_keep is on; excluded from serialized op args
    """

    style: Literal["gyroid", "trabecular", "voronoi", "branches"] = "gyroid"
    volfrac: float = Field(default=0.35, ge=0.05, le=0.9)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    period_mm: float = Field(default=12.0, gt=0, le=1000)
    surface_open: float = Field(default=0.35, ge=0.0, le=1.0)
    shell_mm: float = Field(default=0.0, ge=0, le=50)  # 0 = auto skin depth
    floor_mm: float = Field(default=2.0, ge=0, le=100)
    min_member_mm: float | None = Field(default=None, ge=0, le=50)
    min_gap_mm: float | None = Field(default=None, ge=0, le=50)
    density_threshold: float | None = Field(default=None, ge=0.2, le=0.9)
    auto_keep: bool = True  # pin every detected fastener anchor solid
    auto_keep_margin_mm: float = Field(default=3.0, gt=0, le=50)
    keep_regions: list[RegionSpec] = Field(default_factory=list, max_length=10)
    void_regions: list[RegionSpec] = Field(default_factory=list, max_length=10)
    voxel_mm: float | None = Field(default=None, gt=0, le=100)  # domain pitch override
    max_elements: int | None = Field(
        default=None, ge=1000, le=5_000_000
    )  # domain cell cap (smaller = coarser/faster)
    smooth_iterations: int | None = Field(default=None, ge=0, le=30)
    anchors: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


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
                "standard_parts",
                "image_roles",
                "verification_questions",
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
    standard_parts: list[StandardPartSpec] = Field(default_factory=list, max_length=24)
    # multi-image parts: how the body is built up (layers, gaps, joined parts)
    # — survives the spec->codegen handoff that otherwise flattens structure
    construction_notes: Annotated[str, Field(max_length=600)] = ""
    symmetry: S = ""
    image_roles: list[ImageRole] = Field(default_factory=list, max_length=20)
    process: Annotated[str, Field(max_length=40)] = "fdm"
    material: Annotated[str, Field(max_length=120)] = ""
    tolerances_note: M = ""
    assumptions: list[M] = Field(default_factory=list, max_length=20)
    open_questions: list[OpenQuestion] = Field(default_factory=list, max_length=12)
    must_keep: list[M] = Field(default_factory=list, max_length=20)
    # yes/no checks for the vision critic — one per placement-critical
    # feature (axis, envelope, orientation) so a plausible-looking render
    # cannot hide a spec violation (CADCodeVerify, M6)
    verification_questions: list[M] = Field(default_factory=list, max_length=6)
    # structural requirements (filled when the prompt mentions loads/strength)
    load_cases: list[LoadCase] = Field(default_factory=list, max_length=8)
    optimize: OptimizeSpec | None = None

    @field_validator("verification_questions", mode="before")
    @classmethod
    def _cap_verification_questions(cls, v: Any) -> Any:
        """Tolerant bound: drop blanks/dupes and truncate instead of failing
        the whole spec parse when a chatty model over-emits (the 3-6 guidance
        is a soft bound — grammar `maxItems` may not apply on every provider)."""
        if not isinstance(v, list):
            return v
        seen: set[str] = set()
        out: list[str] = []
        for q in v:
            if not isinstance(q, str) or not q.strip():
                continue
            key = q.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(q.strip())
            if len(out) >= 6:
                break
        return out


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


def _clamp_reasoning(v: Any) -> Any:
    """`reasoning` is advisory — guided decoders that ignore `maxLength` can
    overshoot L, and failing the whole response over a comment wastes the
    call (observed: improve patch dropped for a 1200+ char reasoning)."""
    return v[:1200] if isinstance(v, str) and len(v) > 1200 else v


class CodeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"required": ["reasoning", "expected_bbox", "code"]}
    )

    reasoning: L = ""
    params: list[ParamDef] = Field(default_factory=list, max_length=40)
    expected_bbox: list[float] = Field(default_factory=list, max_length=6)
    code: str

    _clamp = field_validator("reasoning", mode="before")(_clamp_reasoning)


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

    _clamp = field_validator("reasoning", mode="before")(_clamp_reasoning)


class Discrepancy(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"required": ["what", "where", "severity", "fix_hint"]}
    )

    what: M = ""
    where: M = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: M = ""
    # where the requirement comes from — keeps the critic from inventing
    # features that appear in neither the prompt, the spec nor the images
    source: Literal["spec", "prompt", "image", "dfm", "verify_question", "other"] = "other"


class FeatureVerdict(BaseModel):
    """Per-spec-feature verdict from the critic — one entry per spec feature."""

    model_config = ConfigDict(json_schema_extra={"required": ["name", "status", "note"]})

    name: S = ""
    status: Literal["present", "missing", "wrong", "unsure"] = "unsure"
    note: M = ""


class VerificationAnswer(BaseModel):
    """Critic's answer to one spec ``verification_questions`` entry.

    CADCodeVerify-style check: intake writes one yes/no question per
    placement-critical feature; the critic answers each against the
    renders. A ``no`` — or an ``unclear`` backed by evidence — becomes a
    deterministic ``verify_question`` discrepancy downstream, so a
    plausible-looking render cannot hide a spec violation.
    """

    model_config = ConfigDict(json_schema_extra={"required": ["question", "answer"]})

    question: M = ""
    answer: Literal["yes", "no", "unclear"] = "unclear"
    evidence: M = ""


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
    # deliberately NOT schema-required: zero-cost when the spec carries no
    # verification_questions — weak models then never see the field pinned
    verification_answers: list[VerificationAnswer] = Field(default_factory=list, max_length=12)
    confidence: float = 0.0

    @field_validator("verification_answers", mode="before")
    @classmethod
    def _cap_verification_answers(cls, v: Any) -> Any:
        """Truncate instead of failing the whole verdict parse on overflow."""
        if isinstance(v, list):
            return v[:12]
        return v
