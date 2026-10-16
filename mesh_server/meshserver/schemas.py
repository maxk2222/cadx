"""Shared pydantic models for the mesh-server HTTP API.

Contract documented in docs/mesh-server-api.md — the CADX backend keeps
its own copy of these types (no cross-import between the two projects).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenVariant(BaseModel):
    """One variant inside a batch mesh job — shares the job's model+images.

    Candidate screening submits N seeds in a single job so the subprocess
    loads the model once instead of N times (~30-60s each on torch/MPS)."""

    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    subject: str = ""


class GenParams(BaseModel):
    model: str = "auto"
    seed: int = 0
    subject: str = ""
    image_roles: list[str] = Field(default_factory=list)  # front|back|left|right|iso|other
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None  # octree/voxel resolution hint
    batch: list[GenVariant] | None = None  # >0 entries: one job, N outputs (out_i.glb)


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
    strength: float | None = None  # img2img edit strength 0..1


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    progress: float = 0.0
    eta_s: float | None = None
    message: str = ""
    # subprocess-mode liveness: True while the child process is actually
    # computing (>2% CPU recently) — lets client watchdogs distinguish a
    # silent-but-working generation from a frozen one
    busy: bool = False
    model: str = ""
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    duration_s: float = 0.0
    triangles: int | None = None
    images: int = 0  # image jobs: number of result PNGs available for download
    outputs: list[int] = Field(default_factory=list)  # batch: succeeded variant indices
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
    supports_edit: bool = False  # image adapters: accepts an init image
    requires_image: bool = False  # image adapters: edit-only, no txt2img
    default_steps: int | None = None  # adapter-native steps when request omits them
    canonical_up: str | None = None  # up-axis of the raw output frame ("y" = hunyuan)


class Health(BaseModel):
    ok: bool = True
    device: str = ""
    backend: str = ""
    models: list[ModelInfo] = Field(default_factory=list)
    mem_free_mb: int | None = None  # host RAM available to new jobs
