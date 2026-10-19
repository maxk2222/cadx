"""Deterministic test adapter — always available, no GPU needed.

Produces simple watertight primitive meshes derived from the seed so
integration tests can exercise the whole API surface.
"""

from __future__ import annotations

import hashlib

import trimesh
from PIL import Image

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter, ProgressFn


class FakeAdapter(MeshAdapter):
    id = "fake"
    name = "FakeMesh (test fixture)"
    device = "fake"
    backend = "fake"
    supports_multiview = True
    max_images = 8
    license_note = "MIT (test fixture)"
    est_mem_mb = 100

    @classmethod
    def available(cls, cfg) -> bool:  # noqa: ARG003
        return True

    def __init__(self, cfg=None) -> None:  # noqa: ARG002
        self._loaded = True

    def generate(
        self,
        images: list[Image.Image],
        opts: GenOpts,
        progress: ProgressFn | None = None,
        cancelled=None,
    ) -> AdapterGenResult:
        if progress:
            progress(0.5, "fake generation")
        h = int(hashlib.sha256(f"{opts.seed}-{opts.subject}".encode()).hexdigest()[:8], 16)
        sx = 0.6 + (h % 40) / 100.0
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.apply_scale([sx, 0.8, 1.0])
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        return AdapterGenResult(
            mesh=mesh,
            model=self.id,
            backend=self.backend,
            seed=opts.seed,
            steps=opts.steps or 0,
            extra={"images": len(images)},
        )
