"""CADX CLI: api server, workers, migrations, local agent run."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, help="CADX")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("cadx.main:app", host=host, port=port, reload=reload)


@app.command()
def worker() -> None:
    """Run the agent-queue arq worker."""
    from arq import run_worker

    from cadx.queue import AgentWorkerSettings

    run_worker(AgentWorkerSettings)  # type: ignore[arg-type]


@app.command(name="cad-worker")
def cad_worker() -> None:
    """Run the cad-queue arq worker (CAD executor)."""
    from arq import run_worker

    from cadx.queue import CadWorkerSettings

    run_worker(CadWorkerSettings)  # type: ignore[arg-type]


@app.command()
def migrate() -> None:
    """Apply Alembic migrations."""
    import subprocess

    backend = Path(__file__).resolve().parent.parent
    subprocess.run(["alembic", "upgrade", "head"], cwd=backend, check=True)


@app.command()
def mcp(
    http: bool = typer.Option(False, "--http", help="Serve Streamable HTTP instead of stdio"),
    port: int = 8001,
) -> None:
    """Run the CADX MCP server (stdio default; --http for Streamable HTTP)."""
    if http:
        from contextlib import asynccontextmanager

        import uvicorn
        from starlette.applications import Starlette
        from starlette.middleware import Middleware

        from cadx.mcpserver.http import MCPDispatchMiddleware, mcp_asgi_app
        from cadx.mcpserver.server import session_manager

        @asynccontextmanager
        async def lifespan(app: Starlette):
            async with session_manager().run():
                yield

        starlette = Starlette(
            lifespan=lifespan,
            middleware=[Middleware(MCPDispatchMiddleware, prefix="/mcp", subapp=mcp_asgi_app())],
        )
        uvicorn.run(starlette, host="127.0.0.1", port=port)
        return
    from cadx.mcpserver.server import run_stdio

    asyncio.run(run_stdio())


@app.command(name="exec")
def exec_code(
    code_file: Path = typer.Argument(..., exists=True),
    out_project: str = "00000000-0000-0000-0000-000000000000",
    params: str = "{}",
) -> None:
    """Execute a model.py through the sandboxed pipeline (no LLM, no DB)."""
    from cadx.cad.tasks import run_execute_payload

    vid = uuid.uuid4()
    payload = {
        "job_id": "cli",
        "project_id": out_project,
        "version_id": str(vid),
        "iteration": 1,
        "code": code_file.read_text(encoding="utf-8"),
        "params_override": json.loads(params),
        "references": [],
        "settings": {},
        "full_exports": True,
        "name": "model",
    }
    result = run_execute_payload(payload)
    print(json.dumps(result, indent=2, default=str))


def _fake_script(mode: str = "cad") -> list[str | dict[str, Any]]:
    """Deterministic script for --fake runs; covers cad/mesh/hybrid call sequences."""
    spec: dict[str, Any] = {
        "name": "box" if mode == "cad" else "knob",
        "summary": "40x20x10 rounded block" if mode == "cad" else "organic knob ~40mm",
        "units": "mm",
        "length_x": {"value": 40.0, "source": "stated"},
        "width_y": {"value": 20.0, "source": "assumed" if mode != "cad" else "stated"},
        "height_z": {"value": 10.0, "source": "assumed" if mode != "cad" else "stated"},
        "open_questions": [],
    }
    box_code = (
        "from build123d import *\n"
        "PARAMS = {'length': 40.0, 'width': 20.0, 'height': 10.0, 'fillet_r': 1.5}\n"
        "with BuildPart() as p:\n"
        "    Box(PARAMS['length'], PARAMS['width'], PARAMS['height'], "
        "align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
        "    fillet(p.edges().filter_by(Axis.Z), radius=PARAMS['fillet_r'])\n"
        "result = p.part\n"
    )
    orient = {"up": "+z", "front": "+y", "confidence": 0.9, "reason": "dome up"}
    critique = {"score": 95, "passed": True, "discrepancies": [], "matches": [], "confidence": 0.9}

    if mode == "mesh":
        plan = {
            "approach": "mesh",
            "approach_reason": "organic freeform",
            "mesh_subject": "rounded block-like knob",
            "steps": [],
            "params": [],
            "standard_parts": [],
            "notes": "",
        }
        return [spec, plan, orient, critique]

    if mode == "hybrid":
        plan = {
            "approach": "hybrid",
            "approach_reason": "organic body + precise hole",
            "mesh_subject": "rounded block-like knob",
            "hybrid_cad_features": ["centered 5mm through-hole"],
            "steps": ["mesh", "hole"],
            "params": [],
            "standard_parts": [],
            "notes": "",
        }
        hybrid_code = (
            "from build123d import *\n"
            "PARAMS = {'hole_d': 5.0}\n"
            "base = load_mesh_solid()\n"
            "hole = Pos(0, 0, -100) * Cylinder(radius=PARAMS['hole_d'] / 2, height=300)\n"
            "result = base - hole\n"
        )
        return [
            spec,
            plan,
            orient,
            {
                "reasoning": "mesh + hole",
                "params": [],
                "expected_bbox": [40, 40, 40],
                "code": hybrid_code,
            },
            critique,
        ]

    plan = {"steps": ["box + fillet"], "params": [], "standard_parts": [], "notes": ""}
    return [
        spec,
        plan,
        {"reasoning": "box", "params": [], "expected_bbox": [40, 20, 10], "code": box_code},
        critique,
    ]


@app.command()
def run(
    prompt: str = typer.Argument(...),
    images: list[Path] = typer.Option([], "--image", "-i", exists=True),
    fake: bool = typer.Option(False, "--fake", help="Use FakeLLM (no API calls)"),
    fake_mesh: bool = typer.Option(
        False, "--fake-mesh", help="Use FakeMeshClient (no mesh server)"
    ),
    mode: str = typer.Option(
        "auto", "--mode", "-m", help="generation mode: auto | cad | mesh | hybrid"
    ),
    project_id: str = "00000000-0000-0000-0000-000000000000",
) -> None:
    """Run the agent locally (in-process executor, no DB/queues)."""

    async def _false() -> bool:
        return False

    async def _run() -> None:
        from cadx.agent.executor_client import InProcessExecutor
        from cadx.agent.mesh_client import make_mesh_client
        from cadx.agent.orchestrator import run_generation
        from cadx.agent.trace import MemoryTrace
        from cadx.agent.types import AgentCtx, InputAsset
        from cadx.db.models import ProjectSettings
        from cadx.llm.client import make_llm
        from cadx.llm.fake import FakeLLM
        from cadx.runtime_config import runtime_from_db_or_default

        vid = uuid.uuid4()
        assets = [
            InputAsset(
                asset_id=str(uuid.uuid4()),
                kind="image",
                path=p,
                caption="",
                filename=p.name,
            )
            for p in images
        ]
        runtime = await runtime_from_db_or_default()
        if fake:
            llm = FakeLLM(_fake_script(mode))
        else:
            from cadx.llm.capabilities import load_caps_session, record_caps_session

            llm = make_llm(runtime, caps=await load_caps_session(), caps_cb=record_caps_session)
        mesh = (
            make_mesh_client(fake=True) if fake_mesh else make_mesh_client(runtime_settings=runtime)
        )
        emitted: list[str] = []

        async def emit(step: str, iteration: int, message: str, **_fields: Any) -> None:
            line = f"[{step}:{iteration}] {message}"
            emitted.append(line)
            print(line)

        ctx = AgentCtx(
            llm=llm,
            executor=InProcessExecutor(),
            trace=MemoryTrace(),
            emit=emit,
            check_cancel=_false,
            job_id="cli",
            project_id=uuid.UUID(project_id),
            version_id=vid,
            mode="generate",
            prompt=prompt,
            settings=ProjectSettings(max_iterations=4, clarification_mode="assume"),
            runtime_settings=runtime,
            assets=assets,
            gen_mode=mode,
            mesh=mesh,
            mesh_available=(await mesh.health()).ok if mesh else False,
        )
        outcome = await run_generation(ctx)
        print(json.dumps(outcome.__dict__, indent=2, default=str))

    asyncio.run(_run())


@app.command(name="mesh-gen")
def mesh_gen(
    images: list[Path] = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("out.glb"), "-o", "--out"),
    model: str = typer.Option("auto", "--model"),
    seed: int = typer.Option(0, "--seed"),
    steps: int | None = typer.Option(None, "--steps"),
    base_url: str | None = typer.Option(None, "--url", help="mesh-server base URL"),
) -> None:
    """Generate a mesh directly via the mesh-server (no LLM, no DB)."""

    async def _run() -> None:
        from cadx.agent.mesh_client import MeshGenRequest, make_mesh_client
        from cadx.runtime_config import runtime_from_db_or_default

        client = make_mesh_client(
            base_url=base_url, runtime_settings=await runtime_from_db_or_default()
        )
        if client is None:
            print("mesh server not configured (set mesh_base_url in Settings or --url)")
            raise typer.Exit(1)

        async def prog(msg: str, p: float) -> None:
            print(f"{int(p * 100):3d}% {msg}")

        req = MeshGenRequest(
            images=list(images),
            subject="",
            model=model,
            seed=seed,
            steps=steps,
            out_path=out,
        )
        result = await client.generate(req, prog)
        print(
            f"mesh: {result.mesh_path} ({result.triangles} tris, model={result.model}, "
            f"seed={result.seed}, {result.duration_s:.1f}s)"
        )

    asyncio.run(_run())


@app.command(name="mesh-process")
def mesh_process(
    mesh_file: Path = typer.Argument(..., exists=True),
    dims: str = typer.Option("", "--dims", help="LxWxH in mm, e.g. 80x40x30"),
    up: str | None = typer.Option(None, "--up", help="+x -x +y -y +z -z"),
    front: str | None = typer.Option(None, "--front"),
    rotate_z: float = typer.Option(0.0, "--rotate-z"),
    flatten: float = typer.Option(0.0, "--flatten", help="cut/cap bottom at mm height"),
    decimate: int = typer.Option(0, "--decimate", help="target triangle count"),
    scale: float = typer.Option(1.0, "--scale"),
    solid: bool = typer.Option(False, "--solid", help="also convert to B-Rep solid"),
    project_id: str = "00000000-0000-0000-0000-000000000000",
) -> None:
    """Run the mesh post-pipeline (clean/orient/scale/renders/DfM/exports) locally."""
    import shutil

    from cadx import storage
    from cadx.cad.tasks import run_mesh_payload

    spec = None
    if dims:
        try:
            length, width, height = (float(x) for x in dims.lower().split("x"))
            spec = {
                "name": "mesh",
                "summary": "",
                "units": "mm",
                "length_x": {"value": length, "source": "stated"},
                "width_y": {"value": width, "source": "stated"},
                "height_z": {"value": height, "source": "stated"},
            }
        except ValueError:
            print("--dims must look like 80x40x30")
            raise typer.Exit(1) from None

    vid = uuid.uuid4()
    it_dir = storage.iteration_dir(uuid.UUID(project_id), vid, 1)
    it_dir.mkdir(parents=True, exist_ok=True)
    target = it_dir / mesh_file.name
    shutil.copy2(mesh_file, target)

    payload = {
        "job_id": "cli",
        "project_id": project_id,
        "version_id": str(vid),
        "iteration": 1,
        "mode": "final",
        "mesh_rel": f"projects/{project_id}/versions/{vid}/iterations/1/{mesh_file.name}",
        "post": {
            "up": up,
            "front": front,
            "rotate_z_deg": rotate_z,
            "flatten_bottom_mm": flatten,
            "decimate_faces": decimate or None,
            "scale": scale,
        },
        "spec": spec,
        "settings": {},
        "full_exports": True,
        "make_solid": solid,
        "name": "model",
    }
    result = run_mesh_payload(payload)
    print(json.dumps(result, indent=2, default=str))


@app.command(name="image-gen")
def image_gen(
    prompt: str = typer.Argument(...),
    out_dir: Path = typer.Option(Path.cwd(), "-o", "--out-dir"),
    model: str = typer.Option("auto", "--model"),
    seed: int = typer.Option(0, "--seed"),
    count: int = typer.Option(1, "--count", min=1, max=4),
    width: int = typer.Option(768, "--width"),
    height: int = typer.Option(768, "--height"),
    steps: int | None = typer.Option(None, "--steps"),
    reference: Path | None = typer.Option(
        None, "--reference", help="init image for img2img/instruction-edit adapters"
    ),
    strength: float | None = typer.Option(None, "--strength", min=0.0, max=1.0),
    base_url: str | None = typer.Option(None, "--url", help="mesh-server base URL"),
) -> None:
    """Generate images locally via the mesh-server image adapters (no DB).

    With --reference the request goes to the edit-capable adapters
    (img2img with --strength, instruction edit on kontext/qwen)."""

    async def _run() -> None:
        from cadx.agent.image_client import ImageGenRequest, make_image_client
        from cadx.runtime_config import runtime_from_db_or_default

        client = make_image_client(
            base_url=base_url, runtime_settings=await runtime_from_db_or_default()
        )
        if client is None:
            print("mesh server not configured (set mesh_base_url in Settings or --url)")
            raise typer.Exit(1)

        async def prog(msg: str, p: float) -> None:
            print(f"{int(p * 100):3d}% {msg}")

        req = ImageGenRequest(
            prompt=prompt,
            model=model,
            seed=seed,
            count=count,
            width=width,
            height=height,
            steps=steps,
            init_image=reference,
            strength=strength,
            out_dir=out_dir,
        )
        result = await client.generate(req, prog)
        paths = ", ".join(str(p) for p in result.image_paths)
        print(
            f"images: {paths} (model={result.model}, seed={result.seed}, {result.duration_s:.1f}s)"
        )

    asyncio.run(_run())


@app.command(name="mesh-health")
def mesh_health(
    base_url: str | None = typer.Option(None, "--url", help="mesh-server base URL"),
) -> None:
    """Check the mesh-server sidecar."""

    async def _run() -> None:
        from cadx.agent.mesh_client import make_mesh_client
        from cadx.runtime_config import runtime_from_db_or_default

        client = make_mesh_client(
            base_url=base_url, runtime_settings=await runtime_from_db_or_default()
        )
        if client is None:
            print("mesh server not configured (set mesh_base_url in Settings or --url)")
            raise typer.Exit(1)
        h = await client.health()
        print(json.dumps(h.model_dump(), indent=2))

    asyncio.run(_run())


def _mount_eval() -> None:
    """`cadx eval ...` sub-app — evalkit lives in cadx.evalkit; imported lazily
    so the `--local` bootstrap can set env before cadx.db.session binds."""
    from cadx.evalkit.cliapp import eval_app

    app.add_typer(eval_app, name="eval")


_mount_eval()


if __name__ == "__main__":
    app()
