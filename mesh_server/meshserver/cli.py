"""meshserver CLI: serve | generate | models | health."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="mesh-server: local image-to-3D sidecar")


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Run the HTTP server."""
    import uvicorn

    from meshserver.app import app as fastapi_app
    from meshserver.config import get_config

    cfg = get_config()
    uvicorn.run(fastapi_app, host=host or cfg.host, port=port or cfg.port)


@app.command()
def models() -> None:
    """List adapters and their availability on this device."""
    from meshserver.config import get_config
    from meshserver.registry import Registry

    reg = Registry(get_config())
    print(json.dumps({"device": reg.device, "models": [m.model_dump() for m in reg.models()]}, indent=2))


@app.command()
def generate(
    images: list[Path] = typer.Argument(..., exists=True),
    model: str = typer.Option("auto", "--model"),
    seed: int = typer.Option(0, "--seed"),
    steps: Optional[int] = typer.Option(None, "--steps"),
    out: Path = typer.Option(Path("out.glb"), "-o", "--out"),
) -> None:
    """Generate a mesh in-process (no HTTP) — useful for adapter debugging."""
    from meshserver.adapters.base import GenOpts
    from meshserver.config import get_config
    from meshserver.jobs import JobManager
    from meshserver.postprocess import clean_mesh
    from meshserver.preprocess import prep_images
    from meshserver.registry import Registry

    cfg = get_config()
    reg = Registry(cfg)
    imgs = prep_images(list(images), cfg.bg_removal)
    cls = reg.select(model, len(imgs))
    if cls is None:
        print(f"no adapter available for '{model}' on device '{reg.device}'")
        raise typer.Exit(1)
    adapter = reg.get(cls)
    result = adapter.generate(imgs, GenOpts(seed=seed, steps=steps))
    mesh = clean_mesh(result.mesh)
    mesh.export(str(out))
    print(f"{out}  ({len(mesh.faces)} triangles, model={result.model})")


@app.command(name="health")
def health() -> None:
    """Print device/backend info."""
    from meshserver.config import get_config
    from meshserver.registry import Registry

    reg = Registry(get_config())
    avail = [c.id for c in reg.available()]
    print(json.dumps({"device": reg.device, "available": avail}, indent=2))


@app.command()
def download(
    model: str = typer.Argument("auto", help="adapter id or 'auto'"),
) -> None:
    """Pre-download model weights (adapter-specific; prints instructions)."""
    from meshserver.config import get_config
    from meshserver.registry import Registry

    reg = Registry(get_config())
    cls = reg.select(model) or reg.select_image(model)
    if cls is None:
        print(f"adapter '{model}' not installed — see mesh_server/README.md")
        raise typer.Exit(1)
    inst = cls.create(reg.cfg)
    print(f"loading {cls.name} (downloads weights on first use)…")
    inst.load()
    print("ok")


@app.command()
def image(
    prompt: str = typer.Argument(...),
    model: str = typer.Option("auto", "--model"),
    seed: int = typer.Option(0, "--seed"),
    count: int = typer.Option(1, "--count", min=1, max=4),
    width: int = typer.Option(768, "--width"),
    height: int = typer.Option(768, "--height"),
    out: Path = typer.Option(Path("."), "-o", "--out"),
) -> None:
    """Generate image(s) in-process (no HTTP) — adapter debugging."""
    from meshserver.adapters.image_base import ImageGenOpts
    from meshserver.config import get_config
    from meshserver.registry import Registry

    cfg = get_config()
    reg = Registry(cfg)
    cls = reg.select_image(model)
    if cls is None:
        print(f"no image adapter available for '{model}' on device '{reg.device}'")
        raise typer.Exit(1)
    adapter = reg.get(cls)
    result = adapter.generate(
        ImageGenOpts(prompt=prompt, seed=seed, count=count, width=width, height=height)
    )
    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(result.images):
        p = out / f"image{i}.png"
        img.save(p)
        print(f"{p}  (model={result.model}, seed={result.seed + i * 1000})")


if __name__ == "__main__":
    app()
