"""Tripo3D cloud mesh generator — MeshPort implementation over the v2 openapi.

Used when ``runtime.mesh_backend == "tripo"`` (per-job override via
RunOptions.mesh_backend). No local sidecar required — auth is a bearer
``tripo_api_key``.

Flow (https://platform.tripo3d.ai — OpenAPI v2):
    POST {base}/upload          multipart image -> data.image_token
    POST {base}/task            {type: image_to_model, file: {...}} -> task_id
    GET  {base}/task/{id}       status + progress + result.model.url
    POST {base}/task/{id}/cancel  best-effort abort
    GET  {base}/balance         health probe

All Tripo responses wrap the payload in ``{"code": 0, "data": {...}}`` —
``code != 0`` is an API-level error even on HTTP 200.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from cadx.agent.mesh_client import (
    MeshGenError,
    MeshGenRequest,
    MeshGenResult,
    MeshHealth,
    MeshModelInfo,
    MeshPort,
    ProgressFn,
)
from cadx.logging import get_logger

log = get_logger(__name__)

_DONE = "success"
_FAILED = ("failed", "cancelled", "unknown")


class TripoMeshClient(MeshPort):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.tripo3d.ai/v2/openapi",
        model_version: str = "v2.5-20250123",
        timeout_s: float = 900,
        poll_s: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _require_key(self) -> None:
        if not self.api_key:
            raise MeshGenError("no tripo_api_key configured")

    @staticmethod
    def _check_code(body: dict[str, Any], what: str) -> None:
        code = body.get("code")
        if code not in (None, 0):
            msg = body.get("message") or body.get("error") or body
            raise MeshGenError(f"tripo {what} failed: {msg}")

    def _models(self) -> list[MeshModelInfo]:
        return [
            MeshModelInfo(
                id=self.model_version,
                name=f"Tripo3D {self.model_version}",
                backend="tripo3d",
                kind="mesh",
                supports_multiview=False,
                max_images=1,
            )
        ]

    async def models(self) -> list[MeshModelInfo]:
        return self._models()

    async def health(self) -> MeshHealth:
        if not self.api_key:
            return MeshHealth(
                ok=False, backend="tripo3d", error="no tripo_api_key", models=self._models()
            )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/balance", headers=self._headers())
            if r.status_code != 200:
                return MeshHealth(
                    ok=False,
                    backend="tripo3d",
                    error=f"HTTP {r.status_code}: {r.text[:200]}",
                    models=self._models(),
                )
            body = r.json()
            if body.get("code") not in (None, 0):
                return MeshHealth(
                    ok=False,
                    backend="tripo3d",
                    error=str(body.get("message") or body)[:200],
                    models=self._models(),
                )
            return MeshHealth(ok=True, backend="tripo3d", models=self._models())
        except Exception as e:
            return MeshHealth(ok=False, backend="tripo3d", error=str(e), models=self._models())

    async def generate(
        self,
        req: MeshGenRequest,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None = None,
    ) -> MeshGenResult:
        self._require_key()
        file_spec = await self._file_spec(req.images[0])
        task_id = await self._create_task(file_spec)
        status, waited = await self._poll(task_id, on_progress, should_abort)
        url = self._result_url(status)
        await self._download(url, req.out_path)
        return MeshGenResult(
            model=self.model_version,
            backend="tripo3d",
            seed=req.seed,
            steps=req.steps,
            duration_s=waited,
            mesh_path=req.out_path,
        )

    # -- internals --------------------------------------------------------- #

    async def _file_spec(self, src: Path) -> dict[str, Any]:
        """First source image -> Tripo `file` field. A path that IS an
        http(s) URL is passed through as a pre-hosted image; anything else is
        uploaded for an image_token."""
        s = str(src)
        # Path("https://h/x") normalises to "https:/h/x" — match the scheme
        # token rather than the "://" separator, and restore the lost slash
        if s.split(":", 1)[0] in ("http", "https"):
            scheme, _, rest = s.partition(":")
            url = f"{scheme}://{rest.lstrip('/')}"
            ext = url.rsplit("?", 1)[0].rsplit(".", 1)[-1].lower()
            return {"type": "jpg" if ext in ("jpg", "jpeg") else "png", "url": url}
        ext = src.suffix.lower()
        ftype = "jpg" if ext in (".jpg", ".jpeg") else "png"
        mime = "image/jpeg" if ftype == "jpg" else "image/png"
        data = await asyncio.to_thread(src.read_bytes)
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/upload",
                files=[("file", (src.name or f"image.{ftype}", data, mime))],
                headers=self._headers(),
            )
        if r.status_code != 200:
            raise MeshGenError(f"tripo upload failed: {r.status_code} {r.text[:300]}")
        body = r.json()
        self._check_code(body, "upload")
        token = (body.get("data") or {}).get("image_token")
        if not token:
            raise MeshGenError(f"tripo upload returned no image_token: {body}")
        return {"type": ftype, "image_token": token}

    async def _create_task(self, file_spec: dict[str, Any]) -> str:
        payload = {
            "type": "image_to_model",
            "file": file_spec,
            "model_version": self.model_version,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base_url}/task", json=payload, headers=self._headers()
            )
        if r.status_code != 200:
            raise MeshGenError(f"tripo task submit failed: {r.status_code} {r.text[:300]}")
        body = r.json()
        self._check_code(body, "task submit")
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise MeshGenError(f"tripo task submit returned no task_id: {body}")
        return str(task_id)

    async def _poll(
        self,
        task_id: str,
        on_progress: ProgressFn | None,
        should_abort: Callable[[], Awaitable[bool]] | None,
    ) -> tuple[dict[str, Any], float]:
        waited = 0.0
        while waited < self.timeout_s:
            await asyncio.sleep(self.poll_s)
            waited += self.poll_s
            if should_abort is not None and await should_abort():
                await self._cancel(task_id)
                raise MeshGenError("job cancelled by user")
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"{self.base_url}/task/{task_id}", headers=self._headers()
                )
            if r.status_code != 200:
                raise MeshGenError(f"tripo task poll failed: {r.status_code}")
            body = r.json()
            self._check_code(body, "task poll")
            status: dict[str, Any] = body.get("data") or {}
            s = str(status.get("status") or "")
            if s == _DONE:
                return status, waited
            if s in _FAILED:
                raise MeshGenError(
                    str(status.get("error") or status.get("message") or f"tripo task {s}")
                )
            if on_progress:
                p = float(status.get("progress") or 0.0)
                if p > 1.0:
                    p = p / 100.0  # tripo reports percent
                await on_progress("mesh generation running", max(0.0, min(1.0, p)))
        raise MeshGenError(f"mesh generation timed out after {self.timeout_s}s")

    @staticmethod
    def _result_url(status: dict[str, Any]) -> str:
        result = status.get("result") or {}
        url = (result.get("model") or {}).get("url") or (result.get("pbr_model") or {}).get(
            "url"
        )
        if not url:
            raise MeshGenError(f"tripo task succeeded without a model url: {status}")
        return str(url)

    async def _download(self, url: str, out_path: Path) -> None:
        async with httpx.AsyncClient(timeout=300.0) as client:
            dl = await client.get(url)
        if dl.status_code != 200:
            raise MeshGenError(f"tripo model download failed: {dl.status_code}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(out_path.write_bytes, dl.content)

    async def _cancel(self, task_id: str) -> None:
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self.base_url}/task/{task_id}/cancel", headers=self._headers()
                )
