"""Evaluation harness: run benchmarks through the real agent loop.

Benchmarks are JSON files in `eval/benchmarks/` with a prompt, expected
dimensions and (for deterministic runs) an embedded FakeLLM script.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cadagent import storage
from cadagent.agent.executor_client import InProcessExecutor
from cadagent.agent.orchestrator import RunOutcome, run_generation
from cadagent.agent.trace import MemoryTrace
from cadagent.agent.types import AgentCtx
from cadagent.db.models import ProjectSettings
from cadagent.llm.fake import FakeLLM

BENCH_DIR = Path(__file__).parent / "benchmarks"


@dataclass
class Benchmark:
    name: str
    prompt: str
    expected_bbox: list[float] | None = None
    tolerance_mm: float = 1.0
    llm_script: list[Any] | None = None  # FakeLLM responses; None -> real LLM
    settings: dict[str, Any] = field(default_factory=dict)
    gen_mode: str = "auto"  # auto | cad | mesh | hybrid
    fake_mesh: bool = False  # use FakeMeshClient instead of a real mesh server
    images: list[str] = field(default_factory=list)  # fixture image paths (repo-relative)


@dataclass
class EvalResult:
    name: str
    status: str
    score: int | None
    bbox: list[float] | None
    dim_ok: bool | None
    wall_s: float
    error: str | None
    version_dir: str = ""


def load_benchmarks() -> list[Benchmark]:
    out = []
    for p in sorted(BENCH_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append(Benchmark(name=p.stem, **d))
    return out


async def run_benchmark(bm: Benchmark, llm: Any | None = None) -> EvalResult:
    from cadagent.agent.mesh_client import make_mesh_client
    from cadagent.llm.client import LLMClient

    project_id = uuid.uuid4()
    version_id = uuid.uuid4()
    if bm.llm_script is not None:
        llm = FakeLLM(bm.llm_script)
    elif llm is None:
        llm = LLMClient()

    mesh = make_mesh_client(fake=True) if bm.fake_mesh else make_mesh_client()
    assets = []
    if bm.images:
        from cadagent.agent.types import InputAsset
        from cadagent.config import REPO_ROOT

        for rel in bm.images:
            p = (REPO_ROOT / rel).resolve()
            if p.exists():
                assets.append(
                    InputAsset(
                        asset_id=str(uuid.uuid4()),
                        kind="image",
                        path=p,
                        filename=p.name,
                    )
                )

    async def emit(step: str, it: int, msg: str) -> None:
        pass

    async def no_cancel() -> bool:
        return False

    ctx = AgentCtx(
        llm=llm,
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="eval",
        project_id=project_id,
        version_id=version_id,
        mode="generate",
        prompt=bm.prompt,
        settings=ProjectSettings(clarification_mode="assume", **bm.settings),
        assets=assets,
        gen_mode=bm.gen_mode,
        mesh=mesh,
        mesh_available=(await mesh.health()).ok if mesh else False,
    )
    t0 = time.monotonic()
    outcome: RunOutcome = await run_generation(ctx)
    wall = time.monotonic() - t0

    bbox = None
    dim_ok = None
    if outcome.analysis and outcome.analysis.get("bbox_size"):
        bbox = [round(float(x), 2) for x in outcome.analysis["bbox_size"]]
    if bm.expected_bbox and bbox:
        exp = sorted(bm.expected_bbox)
        act = sorted(bbox)
        dim_ok = all(
            abs(a - e) <= max(bm.tolerance_mm, e * 0.02) for a, e in zip(act, exp, strict=True)
        )

    return EvalResult(
        name=bm.name,
        status=outcome.status,
        score=outcome.score,
        bbox=bbox,
        dim_ok=dim_ok,
        wall_s=round(wall, 1),
        error=outcome.error,
        version_dir=str(storage.version_dir(project_id, version_id)),
    )


async def run_all(benchmarks: list[Benchmark] | None = None) -> list[EvalResult]:
    results = []
    for bm in benchmarks or load_benchmarks():
        try:
            results.append(await run_benchmark(bm))
        except Exception as e:
            results.append(EvalResult(bm.name, "crashed", None, None, None, 0.0, str(e)))
    return results
