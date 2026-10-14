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

from cadx import storage
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import RunOutcome, run_generation
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM

BENCH_DIR = Path(__file__).parent / "benchmarks"


@dataclass
class Benchmark:
    name: str
    prompt: str
    expected_bbox: list[float] | None = None
    tolerance_mm: float = 1.0
    llm_script: list[Any] | None = None  # FakeLLM responses; None -> real LLM
    llm_tools: str = "off"  # FakeLLM tools_mode: "auto" -> steps use tool sessions
    settings: dict[str, Any] = field(default_factory=dict)
    gen_mode: str = "auto"  # auto | cad | mesh | hybrid
    fake_mesh: bool = False  # use FakeMeshClient instead of a real mesh server
    images: list[str] = field(default_factory=list)  # fixture image paths (repo-relative)
    runtime: dict[str, Any] = field(default_factory=dict)  # RuntimeSettings overrides
    expected_features: list[dict[str, Any]] = field(default_factory=list)
    expect_watertight: bool = False
    max_wall_s: float | None = None
    max_prompt_tokens: int | None = None


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
    features_ok: bool | None = None
    watertight: bool | None = None
    prompt_tokens: int = 0
    calls: int = 0
    passed: bool | None = None


def _features_ok(expected: list[dict[str, Any]], features: dict[str, Any] | None) -> bool | None:
    """Compare expected feature entries against measured geometry.

    Supported entries:
      {"type": "hole", "count": N, "dia": D}  — ≥N holes in groups near D (±10%/0.5mm)
      {"type": "hole", "count": N}            — total_holes ≥ N
      {"type": "countersink"/"counterbore", "count": N}
      {"type": "fillet"} / {"type": "chamfer"} — presence check
    Returns None when nothing was expected.
    """
    if not expected:
        return None
    feats = features or {}
    groups = feats.get("hole_groups") or []

    def holes_near(dia: float | None) -> int:
        if dia is None:
            return int(feats.get("total_holes") or 0)
        tol = max(0.5, dia * 0.1)
        return sum(
            int(g.get("count") or 0)
            for g in groups
            if abs(float(g.get("diameter") or 0) - dia) <= tol
        )

    for ef in expected:
        ftype = str(ef.get("type", "")).lower()
        want = int(ef.get("count") or 1)
        if ftype in ("hole", "bore", "through-hole"):
            if holes_near(ef.get("dia")) < want:
                return False
        elif ftype == "countersink":
            if int(feats.get("countersinks") or 0) < want:
                return False
        elif ftype == "counterbore":
            if int(feats.get("counterbores") or 0) < want:
                return False
        elif ftype in ("fillet", "round"):
            if not (feats.get("rounds") or []):
                return False
        elif ftype == "chamfer" and int(feats.get("chamfer_faces") or 0) < 1:
            return False
    return True


def load_benchmarks() -> list[Benchmark]:
    out = []
    for p in sorted(BENCH_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append(Benchmark(name=p.stem, **d))
    return out


async def run_benchmark(bm: Benchmark, llm: Any | None = None) -> EvalResult:
    from cadx.agent.mesh_client import make_mesh_client
    from cadx.llm.client import LLMClient
    from cadx.runtime_config import runtime_from_db_or_default

    project_id = uuid.uuid4()
    version_id = uuid.uuid4()
    runtime = await runtime_from_db_or_default()
    if bm.runtime:
        runtime = runtime.model_copy(update=bm.runtime)
    if bm.llm_script is not None:
        llm = FakeLLM(bm.llm_script)
        llm.tools_mode = bm.llm_tools
    elif llm is None:
        llm = LLMClient(runtime_settings=runtime)

    mesh = (
        make_mesh_client(fake=True) if bm.fake_mesh else make_mesh_client(runtime_settings=runtime)
    )
    assets = []
    if bm.images:
        from cadx.agent.types import InputAsset
        from cadx.config import REPO_ROOT

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

    async def emit(step: str, it: int, msg: str, **_f) -> None:
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
        runtime_settings=runtime,
        assets=assets,
        gen_mode=bm.gen_mode,
        mesh=mesh,
        mesh_available=(await mesh.health()).ok if mesh else False,
    )
    t0 = time.monotonic()
    outcome: RunOutcome = await run_generation(ctx)
    wall = time.monotonic() - t0
    return _judge(bm, outcome, wall, storage.version_dir(project_id, version_id))


def _judge(bm: Benchmark, outcome: RunOutcome, wall: float, version_dir: Path) -> EvalResult:
    """Score a finished run against the benchmark's expectations."""
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

    analysis = outcome.analysis or {}
    watertight = analysis.get("watertight_after_repair")
    if watertight is None:
        watertight = analysis.get("watertight")
    usage = outcome.usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    calls = int(usage.get("calls") or 0)
    features_ok = _features_ok(bm.expected_features, outcome.features)

    passed: bool | None = outcome.status in ("succeeded", "needs_review")
    if passed:
        if dim_ok is False or features_ok is False:
            passed = False
        if bm.expect_watertight and watertight is not True:
            passed = False
        if bm.max_wall_s is not None and wall > bm.max_wall_s:
            passed = False
        if bm.max_prompt_tokens is not None and prompt_tokens > bm.max_prompt_tokens:
            passed = False

    return EvalResult(
        name=bm.name,
        status=outcome.status,
        score=outcome.score,
        bbox=bbox,
        dim_ok=dim_ok,
        wall_s=round(wall, 1),
        error=outcome.error,
        version_dir=str(version_dir),
        features_ok=features_ok,
        watertight=bool(watertight) if watertight is not None else None,
        prompt_tokens=prompt_tokens,
        calls=calls,
        passed=passed,
    )


async def run_all(benchmarks: list[Benchmark] | None = None) -> list[EvalResult]:
    results = []
    for bm in benchmarks or load_benchmarks():
        try:
            results.append(await run_benchmark(bm))
        except Exception as e:
            results.append(EvalResult(bm.name, "crashed", None, None, None, 0.0, str(e)))
    return results
