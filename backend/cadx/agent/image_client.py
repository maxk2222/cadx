"""Text-to-image client: talks to the `mesh-server` sidecar over HTTP.

The sidecar hosts local text-to-image models (FLUX.1-schnell via mflux or
diffusers, SDXL/SD1.5 fallbacks, fake-image for tests) as a second adapter
family next to the mesh generators — same base URL, same job queue, one
resident heavyweight model at a time.

Contract (see docs/mesh-server-api.md):
    GET  /v1/health                    -> {ok, device, backend, models:[...]}
    GET  /v1/models                    -> {models:[{kind:"image",...}]}
    POST /v1/image-jobs                -> {job_id}         (plain JSON body)
    GET  /v1/image-jobs/{id}           -> {status, progress, images, error}
    GET  /v1/image-jobs/{id}/image?i=N -> PNG bytes
    DELETE /v1/image-jobs/{id}         -> cancel
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from cadx.agent.mesh_client import MeshHealth, MeshModelInfo, ProgressFn, _dumps
from cadx.logging import get_logger
from cadx.runtime_config import RuntimeSettings

log = get_logger(__name__)


class ImageGenError(Exception):
    pass


class ImageGenRequest(BaseModel):
    """One text-to-image generation request.

    `out_dir` is client-local (download target) and is NEVER serialised to
    the sidecar.
    """

    prompt: str
    negative: str = ""
    model: str = "auto"
    seed: int = 0
    count: int = Field(default=1, ge=1, le=4)
    width: int = 768
    height: int = 768
    steps: int | None = None
    guidance: float | None = None
    init_image: Path | None = None  # image-edit request (img2img/kontext)
    strength: float | None = None  # img2img noise level 0..1
    out_dir: Path


class ImageGenResult(BaseModel):
    model: str
    backend: str = ""
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    duration_s: float = 0.0
    image_paths: list[Path] = Field(default_factory=list)


class ImagePort:
    """Async interface used by the orchestrator (Protocol-style base class)."""

    async def generate(
        self,
        req: ImageGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> ImageGenResult:
        raise NotImplementedError

    async def health(self) -> MeshHealth:
        raise NotImplementedError

    async def models(self) -> list[MeshModelInfo]:
        return []


class HttpImageClient(ImagePort):
    def __init__(self, base_url: str, *, timeout_s: float = 300, hf_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.hf_token = hf_token

    def _headers(self) -> dict[str, str]:
        """Per-request credentials for the sidecar — its middleware applies
        X-HF-Token as HF_TOKEN so downloads and model loads can reach gated
        Hugging Face repos."""
        return {"X-HF-Token": self.hf_token} if self.hf_token else {}

    async def health(self) -> MeshHealth:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/v1/health", headers=self._headers())
                r.raise_for_status()
                return MeshHealth(**r.json())
        except Exception as e:
            return MeshHealth(ok=False, error=str(e))

    async def models(self) -> list[MeshModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
                r.raise_for_status()
                return [
                    MeshModelInfo(**m)
                    for m in r.json().get("models", [])
                    if m.get("kind", "mesh") == "image"
                ]
        except Exception:
            return []

    async def generate(
        self,
        req: ImageGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> ImageGenResult:
        params = req.model_dump(exclude={"out_dir", "init_image"}, exclude_none=True)

        submit_timeout = min(60.0, self.timeout_s)
        async with httpx.AsyncClient(timeout=submit_timeout) as client:
            if req.init_image is not None:
                files: list[tuple[str, tuple[str, bytes, str]]] = [
                    ("json", ("params.json", _dumps(params), "application/json")),
                    (
                        "image",
                        (
                            req.init_image.name or "init.png",
                            req.init_image.read_bytes(),
                            "application/octet-stream",
                        ),
                    ),
                ]
                r = await client.post(
                    f"{self.base_url}/v1/image-edit-jobs", files=files, headers=self._headers()
                )
            else:
                r = await client.post(
                    f"{self.base_url}/v1/image-jobs", json=params, headers=self._headers()
                )
            if r.status_code != 200:
                raise ImageGenError(f"image-server submit failed: {r.status_code} {r.text[:300]}")
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
                    raise ImageGenError("job cancelled by user")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    st = await client.get(
                        f"{self.base_url}/v1/image-jobs/{server_job}",
                        headers=self._headers(),
                    )
                if st.status_code != 200:
                    raise ImageGenError(f"image-server job poll failed: {st.status_code}")
                status = st.json()
                s = status.get("status")
                if s == "succeeded":
                    break
                if s in ("failed", "cancelled"):
                    raise ImageGenError(status.get("error") or f"image job {s}")
                if on_progress:
                    msg = status.get("message") or "image generation running"
                    await on_progress(msg, float(status.get("progress") or 0.0))
            else:
                raise ImageGenError(f"image generation timed out after {deadline}s")

            n_images = int(status.get("images") or 1)
            req.out_dir.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            for i in range(n_images):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    dl = await client.get(
                        f"{self.base_url}/v1/image-jobs/{server_job}/image",
                        params={"i": i},
                        headers=self._headers(),
                    )
                if dl.status_code != 200:
                    raise ImageGenError(f"image download failed: {dl.status_code}")
                p = req.out_dir / f"image{i}.png"
                p.write_bytes(dl.content)
                paths.append(p)
        finally:
            with (
                httpx.Client(timeout=10.0) as client,
                contextlib.suppress(Exception),
            ):
                client.delete(
                    f"{self.base_url}/v1/image-jobs/{server_job}", headers=self._headers()
                )

        return ImageGenResult(
            model=status.get("model") or req.model,
            backend=status.get("backend") or "",
            seed=int(status.get("seed", req.seed)),
            steps=status.get("steps") or req.steps,
            guidance=status.get("guidance") or req.guidance,
            duration_s=float(status.get("duration_s") or waited),
            image_paths=paths,
        )


class OpenAIImageClient(ImagePort):
    """OpenAI-compatible images API client (images/generations + edits).

    Used when ``runtime.image_backend == "openai"``. ``base_url`` resolves to
    ``image_api_base or llm_base_url`` and the key to
    ``image_api_key or llm_api_key`` — so any OpenAI-compatible provider
    (OpenAI, OpenRouter, Azure-style proxies) works. Synchronous API: no job
    queue, no server-side cancel — `should_abort` is checked before submit.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        model: str = "gpt-image-1",
        timeout_s: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def health(self) -> MeshHealth:
        if not self.api_key:
            return MeshHealth(ok=False, backend="openai", error="no api key configured")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._auth())
            if r.status_code == 200:
                return MeshHealth(ok=True, backend="openai", models=await self.models())
            return MeshHealth(ok=False, backend="openai", error=f"HTTP {r.status_code}")
        except Exception as e:
            return MeshHealth(ok=False, backend="openai", error=str(e))

    async def models(self) -> list[MeshModelInfo]:
        known = {
            "gpt-image-1": MeshModelInfo(
                id="gpt-image-1",
                name="gpt-image-1",
                backend="openai",
                kind="image",
                supports_edit=True,
                default_size=1024,
                max_count=4,
            ),
            "dall-e-3": MeshModelInfo(
                id="dall-e-3",
                name="dall-e-3",
                backend="openai",
                kind="image",
                supports_edit=False,
                default_size=1024,
                max_count=1,
            ),
            "dall-e-2": MeshModelInfo(
                id="dall-e-2",
                name="dall-e-2",
                backend="openai",
                kind="image",
                supports_edit=True,
                default_size=1024,
                max_count=10,
            ),
        }
        if self.model and self.model not in known:
            # custom deployment/endpoint model — surface it as the default entry
            return [
                MeshModelInfo(
                    id=self.model,
                    name=self.model,
                    backend="openai",
                    kind="image",
                    supports_edit=True,
                    default_size=1024,
                    max_count=4,
                ),
                *known.values(),
            ]
        return list(known.values())

    async def generate(
        self,
        req: ImageGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> ImageGenResult:
        if not self.api_key:
            raise ImageGenError(
                "no api key for image backend 'openai' (image_api_key/llm_api_key)"
            )
        model = req.model if req.model and req.model != "auto" else self.model
        n = max(1, req.count)
        size = f"{req.width}x{req.height}"
        if should_abort is not None and await should_abort():
            raise ImageGenError("job cancelled by user")
        if on_progress:
            await on_progress("image generation running", 0.2)
        t0 = time.monotonic()
        if req.init_image is not None:
            data = {
                "model": model,
                "prompt": req.prompt,
                "n": str(n),
                "size": size,
            }
            files = [
                (
                    "image",
                    (
                        req.init_image.name or "init.png",
                        req.init_image.read_bytes(),
                        "application/octet-stream",
                    ),
                )
            ]
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(
                    f"{self.base_url}/images/edits",
                    data=data,
                    files=files,
                    headers=self._auth(),
                )
        else:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": req.prompt,
                "n": n,
                "size": size,
            }
            if model.startswith("dall-e"):
                # gpt-image-1 always returns b64_json and rejects the param
                payload["response_format"] = "b64_json"
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(
                    f"{self.base_url}/images/generations",
                    json=payload,
                    headers=self._auth(),
                )
        if r.status_code != 200:
            raise ImageGenError(f"image api error: {r.status_code} {r.text[:300]}")
        entries = r.json().get("data") or []
        req.out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, entry in enumerate(entries[:n]):
            blob: bytes | None = None
            if entry.get("b64_json"):
                blob = base64.b64decode(entry["b64_json"])
            elif entry.get("url"):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    dl = await client.get(str(entry["url"]))
                if dl.status_code != 200:
                    raise ImageGenError(f"image download failed: {dl.status_code}")
                blob = dl.content
            if blob is None:
                continue
            p = req.out_dir / f"image{i}.png"
            p.write_bytes(blob)
            paths.append(p)
        if not paths:
            raise ImageGenError("image api returned no usable images")
        if on_progress:
            await on_progress("image generation done", 1.0)
        return ImageGenResult(
            model=model,
            backend="openai",
            seed=req.seed,
            steps=req.steps,
            guidance=req.guidance,
            duration_s=time.monotonic() - t0,
            image_paths=paths,
        )


class FakeImageClient(ImagePort):
    """Deterministic generator for tests — PIL gradient from seed+prompt."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def health(self) -> MeshHealth:
        return MeshHealth(
            ok=True,
            device="fake",
            backend="fake",
            models=[
                MeshModelInfo(
                    id="fake-image",
                    name="FakeImage (test)",
                    device="fake",
                    backend="fake",
                    kind="image",
                    supports_edit=True,  # echoes the init image back
                    license_note="MIT (test fixture)",
                )
            ],
        )

    async def models(self) -> list[MeshModelInfo]:
        return (await self.health()).models

    async def generate(
        self,
        req: ImageGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> ImageGenResult:
        import io

        from PIL import Image, ImageDraw

        self.calls.append(
            {
                "model": req.model,
                "seed": req.seed,
                "prompt": req.prompt,
                "init_image": str(req.init_image) if req.init_image else None,
            }
        )
        if on_progress:
            await on_progress("fake image generation", 0.5)
        req.out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        if req.init_image is not None and req.init_image.exists():
            # edit request: echo the init image — deterministic and keeps the
            # downstream silhouette intact for mesh-regen tests
            p = req.out_dir / "image0.png"
            p.write_bytes(req.init_image.read_bytes())
            paths.append(p)
            return ImageGenResult(
                model="fake-image-edit",
                backend="fake",
                seed=req.seed,
                steps=req.steps or 0,
                guidance=req.guidance,
                duration_s=0.1,
                image_paths=paths,
            )
        for i in range(max(1, req.count)):
            seed = req.seed + i * 1000
            h = int(hashlib.sha256(f"{seed}-{req.prompt}".encode()).hexdigest()[:8], 16)
            c0 = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
            img = Image.new("RGB", (req.width, req.height), c0)
            ImageDraw.Draw(img).text((8, 8), req.prompt[:60], fill=(255, 255, 255))
            p = req.out_dir / f"image{i}.png"
            buf = io.BytesIO()
            img.save(buf, "PNG")
            p.write_bytes(buf.getvalue())
            paths.append(p)
        return ImageGenResult(
            model="fake-image",
            backend="fake",
            seed=req.seed,
            steps=req.steps or 0,
            guidance=req.guidance,
            duration_s=0.1,
            image_paths=paths,
        )


def make_image_client(
    *,
    fake: bool = False,
    base_url: str | None = None,
    timeout_s: float | None = None,
    runtime_settings: RuntimeSettings | None = None,
    backend: str | None = None,
) -> ImagePort | None:
    """Build the image client for the configured backend.

    ``backend`` overrides ``settings.image_backend``: ``"sidecar"`` talks to
    the local mesh-server (mesh_base_url), ``"openai"`` to an
    OpenAI-compatible images API (image_api_base/key fall back to the llm_*
    provider)."""
    settings = runtime_settings or RuntimeSettings()
    if fake:
        return FakeImageClient()
    mode = backend or settings.image_backend
    if mode == "openai":
        base = (settings.image_api_base or settings.llm_base_url).strip()
        if not base:
            return None
        key = (
            settings.image_api_key.get_secret_value()
            or settings.llm_api_key.get_secret_value()
        )
        return OpenAIImageClient(
            base,
            key,
            model=settings.image_openai_model,
            timeout_s=timeout_s or settings.image_timeout_s,
        )
    url = (base_url if base_url is not None else settings.mesh_base_url).strip()
    if not url:
        return None
    return HttpImageClient(
        url,
        timeout_s=timeout_s or settings.image_timeout_s,
        hf_token=settings.hf_token.get_secret_value(),
    )
