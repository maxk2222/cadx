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
    # controllable-generation conditioning (Hunyuan3D-Omni): "bbox" |
    # "voxel" | "point" | "pose"; control_input semantics follow the type
    # (e.g. bbox = [L, W, H] in mm — the adapter normalises). Adapters
    # without control support ignore both.
    control_type: str | None = None
    control_input: Any = None


class MeshGenVariant(BaseModel):
    """One output of a batch generation — same model+images, own seed/params.

    The server generates all variants inside one worker process so the model
    loads once instead of per candidate (~30-60s saved per extra variant)."""

    seed: int = 0
    steps: int | None = None
    resolution: int | None = None
    guidance: float | None = None
    out_path: Path


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
    kind: str = "mesh"  # "mesh" | "image" | "llm" — adapter families on the sidecar
    supports_multiview: bool = False
    max_images: int = 1
    license_note: str = ""
    loaded: bool = False
    available: bool = True
    last_error: str | None = None  # badlisted after a load/OOM failure
    default_size: int | None = None  # image adapters: default output side
    max_count: int | None = None  # image adapters: max images per job
    supports_edit: bool = False  # image adapters: accepts an init image
    requires_image: bool = False  # image adapters: edit-only, no txt2img
    default_steps: int | None = None  # adapter-native steps when request omits them
    canonical_up: str | None = None  # up-axis of the raw output frame
    # model-management fields (newer sidecars expose weight lifecycle state)
    downloaded: bool = False
    downloading: bool = False
    size_gb: float | None = None
    est_mem_mb: int | None = None
    supports_vision: bool = False  # llm adapters: accepts image inputs
    supports_tools: bool = False  # llm adapters: OpenAI tool_calls
    pinned: bool = False  # resident model pinned in memory
    custom: bool = False  # user-registered via POST /v1/models/register


class MeshHealth(BaseModel):
    ok: bool = False
    device: str = ""
    backend: str = ""
    error: str | None = None
    models: list[MeshModelInfo] = Field(default_factory=list)
    mem_free_mb: int | None = None  # host RAM available to new jobs


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

    async def generate_batch(
        self,
        req: MeshGenRequest,
        variants: list[MeshGenVariant],
        on_progress: ProgressFn | None = None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[MeshGenResult | None]:
        """Generate N variants; default falls back to sequential generate()."""
        out: list[MeshGenResult | None] = []
        for v in variants:
            vreq = req.model_copy(
                update={
                    "seed": v.seed,
                    "steps": v.steps,
                    "resolution": v.resolution,
                    "guidance": v.guidance,
                    "out_path": v.out_path,
                }
            )
            try:
                out.append(await self.generate(vreq, on_progress, should_abort))
            except Exception:
                out.append(None)
        return out


class HttpMeshClient(MeshPort):
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 900,
        stall_timeout_s: float = 300,
        min_free_mb: float = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.min_free_mb = min_free_mb

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

    async def _check_memory_gate(self) -> None:
        """Refuse to submit when the sidecar host reports too little free RAM.

        A generation launched under memory pressure stalls at 0% CPU and
        swaps the host into a watchdog panic. 0 disables the gate; servers
        without mem_free_mb (older version) never gate.
        """
        if self.min_free_mb <= 0:
            return
        h = await self.health()
        if h.ok and h.mem_free_mb is not None and h.mem_free_mb < self.min_free_mb:
            raise MeshGenError(
                f"mesh-server host has only {h.mem_free_mb}MB free "
                f"(min {self.min_free_mb:.0f}MB) — refusing to start a job"
            )

    async def generate(
        self,
        req: MeshGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> MeshGenResult:
        await self._check_memory_gate()
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
        if req.control_type is not None:
            params["control_type"] = req.control_type
            params["control_input"] = req.control_input

        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("json", ("params.json", _dumps(params), "application/json"))
        ]
        for i, p in enumerate(req.images):
            data = p.read_bytes()
            files.append(("images", (p.name or f"image{i}.png", data, "application/octet-stream")))

        server_job = await self._submit(files)

        try:
            status, waited = await self._poll(server_job, on_progress, should_abort)
            await self._download(server_job, req.out_path)
        finally:
            await self._delete_job(server_job)

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

    async def _submit(self, files: list[tuple[str, tuple[str, bytes, str]]]) -> str:
        submit_timeout = min(60.0, self.timeout_s)
        async with httpx.AsyncClient(timeout=submit_timeout) as client:
            r = await client.post(f"{self.base_url}/v1/jobs", files=files)
            if r.status_code != 200:
                raise MeshGenError(f"mesh-server submit failed: {r.status_code} {r.text[:300]}")
            return str(r.json()["job_id"])

    async def _poll(
        self,
        server_job: str,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None,
    ) -> tuple[dict[str, Any], float]:
        """Poll until succeeded; returns (status, waited_s). Raises on
        failed/cancelled/stall/timeout."""
        deadline = self.timeout_s
        poll = 2.0
        waited = 0.0
        status: dict[str, Any] = {}
        # stall watchdog: the server can freeze mid-generation (MPS/MLX memory
        # waits run at 0% CPU) while still reporting "running" — abort when
        # the (progress, message) pair stops moving
        last_sig: tuple[float, str] | None = None
        last_change_at = 0.0
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
                return status, waited
            if s in ("failed", "cancelled"):
                raise MeshGenError(status.get("error") or f"mesh job {s}")
            sig = (float(status.get("progress") or 0.0), str(status.get("message") or ""))
            # busy = the child is provably computing — silent stages
            # (hunyuan only reports 5% then 90%) are not stalls
            if status.get("busy") or sig != last_sig:
                last_sig = sig
                last_change_at = waited
            elif waited - last_change_at >= self.stall_timeout_s:
                raise MeshGenError(
                    f"mesh job stalled at {sig[0]:.0%} for "
                    f"{self.stall_timeout_s:.0f}s ({sig[1] or 'no message'})"
                )
            if on_progress:
                msg = status.get("message") or "mesh generation running"
                await on_progress(msg, sig[0])
        raise MeshGenError(f"mesh generation timed out after {deadline}s")

    async def _download(self, server_job: str, out_path: Path, i: int = 0) -> None:
        async with httpx.AsyncClient(timeout=300.0) as client:
            dl = await client.get(f"{self.base_url}/v1/jobs/{server_job}/mesh", params={"i": i})
        if dl.status_code != 200:
            raise MeshGenError(f"mesh download failed: {dl.status_code}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(out_path.write_bytes, dl.content)

    async def _delete_job(self, server_job: str) -> None:
        def _del() -> None:
            with httpx.Client(timeout=10.0) as client:
                client.delete(f"{self.base_url}/v1/jobs/{server_job}")

        with contextlib.suppress(Exception):
            await asyncio.to_thread(_del)

    async def generate_batch(
        self,
        req: MeshGenRequest,
        variants: list[MeshGenVariant],
        on_progress: ProgressFn | None = None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[MeshGenResult | None]:
        """One server job generating all variants in a single worker process —
        the model loads once instead of once per candidate."""
        if not variants:
            return []
        await self._check_memory_gate()
        params: dict[str, Any] = {
            "model": req.model,
            "subject": req.subject,
            "image_roles": req.image_roles,
            "batch": [
                {
                    k: v2
                    for k, v2 in {
                        "seed": v.seed,
                        "steps": v.steps,
                        "resolution": v.resolution,
                        "guidance": v.guidance,
                    }.items()
                    if v2 is not None
                }
                for v in variants
            ],
        }
        if req.control_type is not None:
            params["control_type"] = req.control_type
            params["control_input"] = req.control_input
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("json", ("params.json", _dumps(params), "application/json"))
        ]
        for i, p in enumerate(req.images):
            data = p.read_bytes()
            files.append(("images", (p.name or f"image{i}.png", data, "application/octet-stream")))

        server_job = await self._submit(files)
        results: list[MeshGenResult | None] = [None] * len(variants)
        try:
            status, waited = await self._poll(server_job, on_progress, should_abort)
            ok_idx = set(status.get("outputs") or [])
            # servers without batch support reply as single jobs — treat index 0
            if not ok_idx:
                ok_idx = {0}
            for i, v in enumerate(variants):
                if i not in ok_idx:
                    continue
                try:
                    await self._download(server_job, v.out_path, i=i)
                except MeshGenError:
                    continue
                results[i] = MeshGenResult(
                    model=status.get("model") or req.model,
                    backend=status.get("backend") or "",
                    seed=v.seed,
                    steps=v.steps if v.steps is not None else status.get("steps"),
                    guidance=(
                        v.guidance
                        if v.guidance is not None
                        else (status.get("guidance") or req.guidance)
                    ),
                    resolution=(
                        v.resolution if v.resolution is not None else status.get("resolution")
                    ),
                    duration_s=float(status.get("duration_s") or waited),
                    mesh_path=v.out_path,
                    triangles=status.get("triangles"),
                )
        finally:
            await self._delete_job(server_job)
        return results


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

        self.calls.append(
            {
                "model": req.model,
                "seed": req.seed,
                "control_type": req.control_type,
                "control_input": req.control_input,
            }
        )
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
    backend: str | None = None,
) -> MeshPort | None:
    """Build the mesh client from an explicit DB-backed runtime snapshot.

    ``backend`` overrides ``settings.mesh_backend``: ``"sidecar"`` talks to
    the local mesh-server (mesh_base_url required, else None), ``"tripo"``
    to the Tripo3D cloud API (tripo_api_key/base/model — no sidecar needed)."""
    settings = runtime_settings or RuntimeSettings()
    if fake:
        return FakeMeshClient()
    mode = backend or settings.mesh_backend
    if mode == "tripo":
        from cadx.agent.tripo_client import TripoMeshClient

        return TripoMeshClient(
            settings.tripo_api_key.get_secret_value(),
            base_url=settings.tripo_api_base,
            model_version=settings.tripo_model,
            timeout_s=timeout_s or settings.mesh_timeout_s,
        )
    url = (base_url if base_url is not None else settings.mesh_base_url).strip()
    if not url:
        return None
    return HttpMeshClient(
        url,
        timeout_s=timeout_s or settings.mesh_timeout_s,
        stall_timeout_s=settings.mesh_stall_timeout_s,
        min_free_mb=float(getattr(settings, "mesh_min_free_mb", 0) or 0),
    )
