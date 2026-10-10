"""Shared pydantic models for the mesh-server HTTP API.

Contract documented in docs/mesh-server-api.md — the cadagent backend keeps
its own copy of these types (no cross-import between the two projects).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenParams(BaseModel):
    model: str = "auto"
    seed: int = 0
    subject: str = ""
    image_roles: list[str] = Field(default_factory=list)  # front|back|left|right|iso|other
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None  # octree/voxel resolution hint


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
    error: str | None = None


class ModelInfo(BaseModel):
    id: str
    name: str = ""
    device: str = ""
    backend: str = ""
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True


class Health(BaseModel):
    ok: bool = True
    device: str = ""
    backend: str = ""
    models: list[ModelInfo] = Field(default_factory=list)
