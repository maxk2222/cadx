"""Mesh generator client: talks to the external `mesh-server` sidecar over HTTP.

The sidecar (mesh_server/ in this repo) hosts local image->3D models
(Hunyuan3D, TRELLIS.2, TripoSG, ...). The worker calls it like an Ollama-style
service; the cad-executor never sees it — generated meshes are exchanged via
files under DATA_DIR.

Contract (see docs/mesh-server-api.md):
    GET  /v1/health            -> {ok, device, backend, models:[...]}
    GET  /v1/models            -> {models:[...]}
    POST /v1/jobs              -> multipart: images + json params -> {job_id}
    GET  /v1/jobs/{id}         -> {status, progress, eta_s, error, model}
    GET  /v1/jobs/{id}/mesh    -> GLB bytes
    DELETE /v1/jobs/{id}       -> cancel
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)


class MeshGenError(Exception):
    pass


class MeshGenRequest(BaseModel):
    """One image->mesh generation request."""

    images: list[Path] = Field(min_length=1)  # hero image first
    image_roles: list[str] = Field(default_factory=list)  # front|back|left|right|iso|other
    subject: str = ""  # short text description of the part (for logging / future t2i)
    model: str = "auto"
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    out_path: Path  # where the GLB is written
    job_id: str | None = None  # lets the server cancel the right job


class MeshGenResult(BaseModel):
    model: str
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    resolution: int | None = None
    duration_s: float = 0.0
    mesh_path: Path
    triangles: int | None = None


class MeshModelInfo(BaseModel):
    id: str
    name: str = ""
    device: str = ""
    backend: str = ""
    kind: str = "mesh"  # "mesh" | "image" — second adapter family on the sidecar
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True
    last_error: str | None = None  # badlisted after a load/OOM failure
    default_size: int | None = None  # image adapters: default output side
    max_count: int | None = None  # image adapters: max images per job


class MeshHealth(BaseModel):
    ok: bool = False
    device: str = ""
    backend: str = ""
    error: str | None = None
    models: list[MeshModelInfo] = Field(default_factory=list)


ProgressFn = Callable[[str, float], Awaitable[None]]  # (message, progress 0..1)


class MeshPort:
    """Async interface used by the orchestrator (Protocol-style base class)."""

    async def generate(
        self,
        req: MeshGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> MeshGenResult:
        raise NotImplementedError

    async def health(self) -> MeshHealth:
        raise NotImplementedError

    async def models(self) -> list[MeshModelInfo]:
        h = await self.health()
        return h.models


class HttpMeshClient(MeshPort):
    def __init__(self, base_url: str, *, timeout_s: float = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def health(self) -> MeshHealth:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/v1/health")
                r.raise_for_status()
                return MeshHealth(**r.json())
        except Exception as e:
            return MeshHealth(ok=False, error=str(e))

    async def models(self) -> list[MeshModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/v1/models")
                r.raise_for_status()
                return [MeshModelInfo(**m) for m in r.json().get("models", [])]
        except Exception:
            return []

    async def generate(
        self,
        req: MeshGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> MeshGenResult:
        params: dict[str, Any] = {
            "model": req.model,
            "seed": req.seed,
            "subject": req.subject,
            "image_roles": req.image_roles,
        }
        if req.steps is not None:
            params["steps"] = req.steps
        if req.guidance is not None:
            params["guidance"] = req.guidance
        if req.resolution is not None:
            params["resolution"] = req.resolution

        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("json", ("params.json", _dumps(params), "application/json"))
        ]
        for i, p in enumerate(req.images):
            data = p.read_bytes()
            files.append(("images", (p.name or f"image{i}.png", data, "application/octet-stream")))

        submit_timeout = min(60.0, self.timeout_s)
        async with httpx.AsyncClient(timeout=submit_timeout) as client:
            r = await client.post(f"{self.base_url}/v1/jobs", files=files)
            if r.status_code != 200:
                raise MeshGenError(f"mesh-server submit failed: {r.status_code} {r.text[:300]}")
            server_job = r.json()["job_id"]

        deadline = self.timeout_s
        poll = 2.0
        waited = 0.0
        status: dict[str, Any] = {}
        try:
            while waited < deadline:
                await asyncio.sleep(poll)
                waited += poll
                if should_abort is not None and await should_abort():
                    raise MeshGenError("job cancelled by user")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    st = await client.get(f"{self.base_url}/v1/jobs/{server_job}")
                if st.status_code != 200:
                    raise MeshGenError(f"mesh-server job poll failed: {st.status_code}")
                status = st.json()
                s = status.get("status")
                if s == "succeeded":
                    break
                if s in ("failed", "cancelled"):
                    raise MeshGenError(status.get("error") or f"mesh job {s}")
                if on_progress:
                    msg = status.get("message") or "mesh generation running"
                    await on_progress(msg, float(status.get("progress") or 0.0))
            else:
                raise MeshGenError(f"mesh generation timed out after {deadline}s")

            async with httpx.AsyncClient(timeout=300.0) as client:
                dl = await client.get(f"{self.base_url}/v1/jobs/{server_job}/mesh")
            if dl.status_code != 200:
                raise MeshGenError(f"mesh download failed: {dl.status_code}")
            req.out_path.parent.mkdir(parents=True, exist_ok=True)
            req.out_path.write_bytes(dl.content)
        finally:
            with (
                httpx.Client(timeout=10.0) as client,
                contextlib.suppress(Exception),
            ):
                client.delete(f"{self.base_url}/v1/jobs/{server_job}")

        return MeshGenResult(
            model=status.get("model") or req.model,
            backend=status.get("backend") or "",
            seed=int(status.get("seed", req.seed)),
            steps=status.get("steps") or req.steps,
            guidance=status.get("guidance") or req.guidance,
            resolution=status.get("resolution") or req.resolution,
            duration_s=float(status.get("duration_s") or waited),
            mesh_path=req.out_path,
            triangles=status.get("triangles"),
        )


def _dumps(d: dict[str, Any]) -> bytes:
    import json

    return json.dumps(d).encode()


class FakeMeshClient(MeshPort):
    """Deterministic generator for tests / compose E2E without a GPU.

    Produces a simple primitive blob whose bbox depends on the seed so
    different iterations are distinguishable. Requires trimesh.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def health(self) -> MeshHealth:
        return MeshHealth(
            ok=True,
            device="fake",
            backend="fake",
            models=[
                MeshModelInfo(
                    id="fake",
                    name="FakeMesh (test)",
                    device="fake",
                    backend="fake",
                    license_note="MIT (test fixture)",
                )
            ],
        )

    async def generate(
        self,
        req: MeshGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> MeshGenResult:
        import trimesh

        self.calls.append({"model": req.model, "seed": req.seed})
        if on_progress:
            await on_progress("fake mesh generation", 0.5)
        # deterministic blob: sphere squashed into a knob-ish shape; size varies with seed
        h = int(hashlib.sha256(f"{req.seed}-{req.subject}".encode()).hexdigest()[:8], 16)
        sx = 0.6 + (h % 40) / 100.0  # 0.6..0.99
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.apply_scale([sx, 0.8, 1.0])
        # normalize into a unit-ish box like real generators
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        req.out_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        req.out_path.write_bytes(buf.getvalue())
        return MeshGenResult(
            model="fake",
            backend="fake",
            seed=req.seed,
            steps=req.steps or 0,
            duration_s=0.1,
            mesh_path=req.out_path,
            triangles=len(mesh.faces),
        )


@dataclass
class CachedMeshAvailability:
    """One health probe per worker job; avoids hammering the sidecar."""

    client: MeshPort | None
    _health: MeshHealth | None = field(default=None, init=False)

    async def available(self) -> bool:
        if self.client is None:
            return False
        if self._health is None:
            self._health = await self.client.health()
        return self._health.ok

    async def health(self) -> MeshHealth:
        if self._health is None:
            self._health = await self.client.health() if self.client else MeshHealth()
        return self._health


def make_mesh_client(
    *,
    fake: bool = False,
    base_url: str | None = None,
    timeout_s: float | None = None,
    runtime_settings: RuntimeSettings | None = None,
) -> MeshPort | None:
    """Build the sidecar client from an explicit DB-backed runtime snapshot."""
    settings = runtime_settings or RuntimeSettings()
    if fake:
        return FakeMeshClient()
    url = (base_url if base_url is not None else settings.mesh_base_url).strip()
    if not url:
        return None
    return HttpMeshClient(url, timeout_s=timeout_s or settings.mesh_timeout_s)
