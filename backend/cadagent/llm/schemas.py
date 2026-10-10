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


class Discrepancy(BaseModel):
    what: str = ""
    where: str = ""
    severity: Literal["major", "minor"] = "minor"
    fix_hint: str = ""


class Critique(BaseModel):
    score: int = 0
    passed: bool = False
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    matches: list[str] = Field(default_factory=list)
    confidence: float = 0.0
