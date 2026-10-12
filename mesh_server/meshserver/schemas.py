"""Shared pydantic models for the mesh-server HTTP API.

Contract documented in docs/mesh-server-api.md — the CADX backend keeps
its own copy of these types (no cross-import between the two projects).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenParams(BaseModel):
    model: str = "auto"
    seed: int = 0
    subject: str = ""
    image_roles: list[str] = Field(default_factory=list)  # front|back|left|right|iso|other
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None  # octree/voxel resolution hint


class ImageGenParams(BaseModel):
    """text-to-image request — plain JSON body, no image upload needed."""

    model: str = "auto"
    prompt: str = Field(min_length=1)
    negative: str = ""
    seed: int = 0
    count: int = Field(default=1, ge=1, le=4)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=768, ge=256, le=2048)
    steps: int | None = None
    guidance: float | None = None


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    progress: float = 0.0
    eta_s: float | None = None
    message: str = ""
    model: str = ""
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    duration_s: float = 0.0
    triangles: int | None = None
    images: int = 0  # image jobs: number of result PNGs available for download
    error: str | None = None


class ModelInfo(BaseModel):
    id: str
    name: str = ""
    device: str = ""
    backend: str = ""
    kind: Literal["mesh", "image"] = "mesh"
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True
    last_error: str | None = None  # badlisted after a load/OOM failure ('auto' skips it)
    default_size: int | None = None  # image adapters: default output side
    max_count: int | None = None  # image adapters: max images per job


class Health(BaseModel):
    ok: bool = True
    device: str = ""
    backend: str = ""
    models: list[ModelInfo] = Field(default_factory=list)
