"""Geometric analysis of a generated part (B-Rep + tessellation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh
from build123d import GeomType, Shape
from pydantic import BaseModel, Field


class CylFace(BaseModel):
    radius: float
    axis: list[float]
    center: list[float]


class Analysis(BaseModel):
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_size: list[float]
    volume: float
    surface_area: float = 0.0
    center_of_mass: list[float] = Field(default_factory=list)
    solid_count: int = 0
    is_valid: bool = False
    watertight: bool | None = None
    face_count: int = 0
    edge_count: int = 0
    cylindrical_faces: list[CylFace] = Field(default_factory=list)
    sits_on_xy: bool = False
    centered_xy: bool = False
    warnings: list[str] = Field(default_factory=list)
    source: str = "brep"  # "brep" | "mesh"
    triangle_count: int = 0
    roughness_deg: float = 0.0  # mean dihedral angle of near-vertical faces
    footprint_iou: float | None = None  # XY outline vs measured plan view


def _vec(v: Any) -> list[float]:
    out = []
    for attr in ("X", "Y", "Z"):
        c = getattr(v, attr)
        out.append(round(float(c() if callable(c) else c), 4))
    return out


def _cyl_axis(face: Any) -> list[float]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    axis = BRepAdaptor_Surface(face.wrapped).Cylinder().Axis().Direction()
    return _vec(axis)


def analyze(shape: Shape, stl_path: Path | None = None) -> Analysis:
    bb = shape.bounding_box()
    size = _vec(bb.max - bb.min)
    warnings: list[str] = []

    solid_count = len(shape.solids())
    is_valid = bool(shape.is_valid)
    if solid_count != 1:
        warnings.append(f"part has {solid_count} solids (expected 1)")
    if not is_valid:
        warnings.append("B-Rep is not valid")

    cyl_faces: list[CylFace] = []
    for f in shape.faces():
        try:
            if f.geom_type == GeomType.CYLINDER:
                cyl_faces.append(
                    CylFace(
                        radius=float(f.radius or 0.0), axis=_cyl_axis(f), center=_vec(f.center())
                    )
                )
        except Exception:
            continue

    watertight: bool | None = None
    if stl_path is not None and stl_path.exists():
        try:
            mesh = trimesh.load(str(stl_path))
            watertight = bool(
                getattr(mesh, "is_watertight", False)
                and getattr(mesh, "is_winding_consistent", False)
            )
            if not watertight:
                warnings.append("mesh is not watertight / winding inconsistent")
        except Exception:
            watertight = None

    sits = abs(float(bb.min.Z)) < 0.01
    cx, cy = float(bb.center().X), float(bb.center().Y)
    centered = (
        abs(cx) < max(size[0] * 0.01, 0.5) and abs(cy) < max(size[1] * 0.01, 0.5)
        if max(size[0], size[1]) > 0
        else False
    )
    if not sits:
        warnings.append("part does not rest on the XY plane (min Z != 0)")
    if not centered:
        warnings.append("part is not centred at the origin in XY")

    try:
        com = _vec(shape.center())
    except Exception:
        com = []

    return Analysis(
        bbox_min=_vec(bb.min),
        bbox_max=_vec(bb.max),
        bbox_size=size,
        volume=round(float(shape.volume), 3),
        surface_area=round(float(shape.area), 3),
        center_of_mass=com,
        solid_count=solid_count,
        is_valid=is_valid,
        watertight=watertight,
        face_count=len(shape.faces()),
        edge_count=len(shape.edges()),
        cylindrical_faces=cyl_faces,
        sits_on_xy=sits,
        centered_xy=centered,
        warnings=warnings,
    )
