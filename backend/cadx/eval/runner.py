"""Evaluation harness: run benchmarks through the real agent loop.

Benchmarks are JSON files in `eval/benchmarks/` with a prompt, expected
dimensions and (for deterministic runs) an embedded FakeLLM script.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cadx import storage
from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.orchestrator import RunOutcome, run_generation
from cadx.agent.trace import JsonlTrace, MemoryTrace
from cadx.agent.types import AgentCtx
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM
from cadx.logging import get_logger

log = get_logger(__name__)

# backend package root — in the container image REPO_ROOT resolves to "/"
# while fixtures get injected under <backend>/tests, so benchmark image paths
# fall back here when the repo-relative lookup misses
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

BENCH_DIR = Path(__file__).parent / "benchmarks"

# un-named eval runs are grouped under this variant name in summaries
DEFAULT_VARIANT = "default"

_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_variant(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse a `--variant NAME=<json>` CLI value.

    `<json>` is a partial RuntimeSettings object merged UNDER each benchmark's
    own `runtime` overrides (benchmark-local keys win — variants set defaults,
    benchmarks may pin). Returns (name, overrides).
    """
    from cadx.runtime_config import RuntimeSettings

    name, sep, raw = spec.partition("=")
    name = name.strip()
    if not sep or not raw.strip():
        raise ValueError(f"--variant expects NAME=<json-object>, got {spec!r}")
    if not _VARIANT_NAME.fullmatch(name):
        raise ValueError(f"--variant name {name!r} must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if name == DEFAULT_VARIANT:
        raise ValueError(f"--variant name {name!r} is reserved for un-named runs")
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--variant {name}: malformed JSON: {e}") from e
    if not isinstance(overrides, dict):
        raise ValueError(f"--variant {name}: overrides must be a JSON object")
    unknown = sorted(set(overrides) - set(RuntimeSettings.model_fields))
    if unknown:
        raise ValueError(f"--variant {name}: unknown RuntimeSettings keys: {', '.join(unknown)}")
    try:
        RuntimeSettings.model_validate(overrides)
    except ValueError as e:
        raise ValueError(f"--variant {name}: invalid values: {e}") from e
    return name, overrides


def eval_out_root() -> Path:
    """Root for per-run eval summaries. CADX_EVAL_OUT overrides (tests/tools);
    otherwise lives under DATA_DIR like every other run artifact."""
    env = os.environ.get("CADX_EVAL_OUT")
    return Path(env) if env else storage.data_root() / "eval_out"


def run_out_dir(root: Path, benchmark: str, variant: str | None, run_id: str) -> Path:
    """Per-run output dir: `<root>/<bench>/<variant>/<run_id>/` for named
    variants; the un-named default keeps the flat `<root>/<bench>/<run_id>/`
    layout."""
    if variant:
        return root / benchmark / variant / run_id
    return root / benchmark / run_id


@contextlib.contextmanager
def _rss_sampler(benchmark: str, interval_s: float = 5.0) -> Any:
    """Background thread logging live RSS every `interval_s` seconds.

    Per-emit peak_rss only samples at step transitions — a >10GB spike can
    live and die between emits (observed: anon-rss 13.6GB at the OOM kill
    while the last step emit showed 3.2GB). The sampler catches inter-step
    growth and survives up to the kill itself.
    """
    import resource
    import threading

    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_s):
            cur = peak = 0
            try:
                # VmRSS = live footprint; getrusage only yields the peak mark
                for line in Path("/proc/self/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        cur = int(line.split()[1])
                    elif line.startswith("VmHWM:"):
                        peak = int(line.split()[1])
            except OSError:
                peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # KB on linux
            log.info("eval.rss", benchmark=benchmark, rss_mb=cur // 1024, peak_mb=peak // 1024)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


_rlimit_known_unsupported = False


def _apply_mem_limit() -> None:
    """Cap the eval process' heap so a runaway benchmark dies with MemoryError
    instead of taking the whole Docker VM down (observed: eval process ~12GB
    RSS in a 15.6GB VM -> container restart mid-run).

    RLIMIT_DATA over AS: file-backed mmaps (ONNX weights, venv) don't count —
    only the malloc heap that the workbench/trace growth actually lives in.
    """
    global _rlimit_known_unsupported
    if _rlimit_known_unsupported:
        return

    import resource

    limit_gb = float(os.environ.get("CADX_EVAL_RLIMIT_GB", "11"))
    lim = int(limit_gb * 1024**3)
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        # keep the current hard limit — raising it to RLIM_INFINITY needs
        # privileges unprivileged host runs don't have
        soft = lim if hard == resource.RLIM_INFINITY else min(lim, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
        log.info("eval.rlimit", rlimit_gb=limit_gb)
    except (ValueError, OSError) as e:
        # e.g. macOS caps RLIMIT_DATA below the reported RLIM_INFINITY — the
        # cap is best-effort anyway, so warn once and stop retrying
        _rlimit_known_unsupported = True
        log.warning("eval.rlimit_failed", err=str(e))


@dataclass
class Benchmark:
    name: str
    prompt: str
    expected_bbox: list[float] | None = None
    tolerance_mm: float = 1.0
    llm_script: list[Any] | None = None  # FakeLLM responses; None -> real LLM
    llm_tools: str = "off"  # FakeLLM tools_mode: "auto" -> steps use tool sessions
    real_only: bool = False  # skipped by `cadx eval`; runs only with --real
    settings: dict[str, Any] = field(default_factory=dict)
    gen_mode: str = "auto"  # auto | cad | mesh | hybrid
    fake_mesh: bool = False  # use FakeMeshClient instead of a real mesh server
    images: list[str] = field(default_factory=list)  # fixture image paths (repo-relative)
    # stored versions/<vid>/mesh/ dir to reuse — hybrid/mesh runs without
    # image input or a live mesh server (edit-style benchmarks)
    reuse_mesh_dir: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)  # RuntimeSettings overrides
    expected_features: list[dict[str, Any]] = field(default_factory=list)
    expect_watertight: bool = False
    expect_code: bool = False  # outcome must carry parametric code (parametrize)
    expect_issue_contains: str | None = None  # substring expected in final issues
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
    code_ok: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    iterations: int = 0  # deepest improvement-loop iteration seen in the trace
    variant: str = DEFAULT_VARIANT  # eval variant this run was executed under
    passed: bool | None = None
    issues: list[str] = field(default_factory=list)
    resolved_mode: str | None = None  # cad | mesh | hybrid actually used
    mode_reason: str = ""  # plan approach_reason + runtime mode_note
    # per-step LLM stats aggregated from the trace:
    # {step: {calls, prompt_tokens, completion_tokens, latency_ms}}
    step_stats: dict[str, Any] = field(default_factory=dict)
    execs: int = 0  # executor.execute calls (incl. session verifies, autofix)
    exec_s: float = 0.0  # total wall time inside executor.execute


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
    return all(_feature_entry_ok(ef, feats, groups) for ef in expected)


def _feature_entry_ok(  # noqa: C901
    ef: dict[str, Any], feats: dict[str, Any], groups: list[Any]
) -> bool:
    # elongated voids are slots — they must not satisfy a round-hole
    # expectation (a 20x6 slot would otherwise read as a big bore)
    round_groups = [g for g in groups if not g.get("elongated")]

    ftype = str(ef.get("type", "")).lower()
    want = int(ef.get("count") or 1)

    def _axis_ok(g: dict[str, Any], want_axis: str | None) -> bool:
        if want_axis is None:
            return True
        got = str(g.get("axis") or "").lstrip("+-").lower()
        return got == want_axis

    def _positions_ok(centers: list[list[float]], positions: list[Any]) -> bool:
        # every expected [x, y] must land within tol of a measured center
        tol = 1.5
        for ex in positions:
            if not isinstance(ex, list | tuple) or len(ex) < 2:
                continue
            if not any(
                len(c) >= 2
                and abs(float(c[0]) - float(ex[0])) <= tol
                and abs(float(c[1]) - float(ex[1])) <= tol
                for c in centers
            ):
                return False
        return True

    def _pcd_ok(centers: list[list[float]], pcd: float) -> bool:
        # bolt-circle diameter = 2 * mean distance of the hole centers to
        # their common centroid
        if len(centers) < 3:
            return False
        cx = sum(float(c[0]) for c in centers) / len(centers)
        cy = sum(float(c[1]) for c in centers) / len(centers)
        radii = [math.hypot(float(c[0]) - cx, float(c[1]) - cy) for c in centers if len(c) >= 2]
        if len(radii) < 3:
            return False
        tol = max(1.5, pcd * 0.03)
        if any(abs(r - radii[0]) > tol for r in radii):
            return False
        return abs(2 * sum(radii) / len(radii) - pcd) <= tol

    if ftype == "tunnel":
        # mesh-mode through-bores come from detect_mesh_features (feats["holes"]);
        # CAD-mode through-bores surface as depth_class="through" hole_groups —
        # a longitudinal BRep bore never gets kind="tunnel", so both sources
        # must satisfy the expectation
        axis = str(ef.get("axis") or "").lower() or None
        dia = ef.get("dia")
        dia_tol = max(0.5, float(dia) * 0.15) if dia is not None else None
        mesh_got = sum(
            1
            for h in feats.get("holes") or []
            if h.get("kind") == "tunnel"
            and (axis is None or h.get("axis") == axis)
            and (dia_tol is None or abs(float(h.get("dia") or 0) - float(dia)) <= dia_tol)
        )
        cad_got = sum(
            max(0, int(g.get("count") or 0) - int(g.get("blocked") or 0))
            for g in round_groups
            if g.get("depth_class") == "through"
            and _axis_ok(g, axis)
            and (dia_tol is None or abs(float(g.get("diameter") or 0) - float(dia)) <= dia_tol)
        )
        return mesh_got + cad_got >= want
    if ftype == "slot":
        return int(feats.get("slots") or 0) >= want
    if ftype in ("hole", "bore", "through-hole"):
        want_axis = str(ef.get("axis") or "").lstrip("+-").lower() or None
        matching = [g for g in round_groups if _axis_ok(g, want_axis)]
        dia = ef.get("dia")
        if dia is not None:
            tol = max(0.5, float(dia) * 0.1)
            matching = [
                g for g in matching if abs(float(g.get("diameter") or 0) - float(dia)) <= tol
            ]
        centers = [c for g in matching for c in (g.get("centers") or [])]
        n_matching = sum(int(g.get("count") or 0) for g in matching)
        if n_matching < want:
            return False
        positions = ef.get("positions")
        if positions and not _positions_ok(centers, positions):
            return False
        pcd = ef.get("pcd")
        return pcd is None or _pcd_ok(centers, float(pcd))
    if ftype == "countersink":
        return int(feats.get("countersinks") or 0) >= want
    if ftype == "counterbore":
        return int(feats.get("counterbores") or 0) >= want
    if ftype in ("fillet", "round"):
        return bool(feats.get("rounds") or [])
    if ftype == "chamfer":
        return int(feats.get("chamfer_faces") or 0) >= 1
    return True


def load_benchmarks() -> list[Benchmark]:
    out = []
    for p in sorted(BENCH_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append(Benchmark(name=p.stem, **d))
    return out


async def _eval_image_port(bm: Benchmark, runtime: Any) -> tuple[Any, bool]:
    """Production parity: worker_jobs wires the image port off the same sidecar
    — without it synth-view/concept paths silently no-op under eval."""
    from cadx.agent.image_client import make_image_client
    from cadx.agent.worker_jobs import _image_available

    image = make_image_client(fake=bm.fake_mesh, runtime_settings=runtime)
    return image, await _image_available(image)


async def _ensure_eval_project(project_id: uuid.UUID, name: str) -> None:
    """Production parity: DB-backed paths (generated-image persistence, asset
    bookkeeping) require a real project row — ephemeral eval ids violate
    asset_project_id_fkey (observed: synth-view persist crashed with FK).
    Bare-sqlite unit-test DBs have no schema — tolerate that, the warning
    still surfaces a real production failure."""
    from cadx.db.models import Project
    from cadx.db.session import session_scope

    try:
        async with session_scope() as db:
            db.add(Project(id=project_id, name=f"eval:{name}"))
            await db.commit()
    except Exception as e:
        log.warning("eval.project_insert_failed", error=str(e)[:200])


def _apply_reuse_mesh_dir(ctx: AgentCtx, mdir: Path | str | None) -> bool:
    """Wire a stored versions/<vid>/mesh/ dir into ctx for mesh-solid reuse.

    The benchmark's `reuse_mesh_dir` may be repo-relative (a checked-in
    fixture or the host's DATA_DIR at the repo root), DATA_DIR-relative
    (`projects/<pid>/versions/<vid>/mesh`), or backend-relative — same
    convention as benchmark image paths. Populates ctx.parent (the hybrid
    reuse gate in _prepare_hybrid_mesh reads mesh_solid + a truthy mesh dict)
    and ctx.mesh_raw_reuse, plus the stored reference mask when present.
    """
    from cadx.config import REPO_ROOT

    if not mdir:
        return False
    mdir = Path(mdir)
    candidates = [REPO_ROOT / mdir]
    with contextlib.suppress(Exception):  # paths escaping DATA_DIR raise here
        candidates.append(storage.resolve(str(mdir)))
    candidates.append(_BACKEND_ROOT / str(mdir).removeprefix("backend/"))
    src = next((p for p in candidates if p.is_dir()), None)
    if src is None:
        log.warning("eval.reuse_mesh_dir_missing", reuse_mesh_dir=str(mdir))
        return False

    solid = src / "solid.brep"
    if not solid.exists():
        log.warning("eval.reuse_mesh_dir_missing", reuse_mesh_dir=str(mdir), missing="solid.brep")
        return False
    ctx.parent.mesh_solid = solid
    ctx.parent.resolved_mode = "hybrid"

    mesh_info: dict[str, Any] = {}
    mj = src / "mesh.json"
    if mj.exists():
        with contextlib.suppress(Exception):
            data = json.loads(mj.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                mesh_info = data
    if not mesh_info:
        # the reuse gate requires a truthy parent.mesh — keep the minimal
        # keys _reuse_parent_mesh_assets reads (bbox_size, triangles)
        mesh_info = {"reused_solid": True, "bbox_size": mesh_info.get("bbox_size")}
    ctx.parent.mesh = mesh_info

    raw = src / "mesh_raw.glb"
    if raw.exists():
        ctx.mesh_raw_reuse = raw
    mask = src.parent / "_ref_mask.png"
    if mask.exists():
        dst = storage.ensure(storage.version_dir(ctx.project_id, ctx.version_id)) / "_ref_mask.png"
        shutil.copy2(mask, dst)
        # same wiring _ref_side_mask produces — process_mesh reuses the
        # stored mask for silhouette IoU instead of re-extracting it
        ctx.mesh_info["_ref_mask_rel"] = storage.relpath(dst)
    # reuse semantics are "the stored solid IS the reference" — attach its
    # contact sheet to the critique context so "unchanged vs reference"
    # questions are answerable from evidence instead of stalling the verdict
    ver_dir = src.parent
    sheets = sorted(
        ver_dir.glob("iterations/*/a*/contact_sheet.png"),
        key=lambda p: p.stat().st_mtime,
    ) or sorted(
        ver_dir.glob("iterations/*/orient_sheet.png"),
        key=lambda p: p.stat().st_mtime,
    )
    if sheets:
        ctx.parent.contact_sheet = sheets[-1]
    return True


async def run_benchmark(
    bm: Benchmark,
    llm: Any | None = None,
    *,
    queue_exec: bool = False,
    variant: str | None = None,
    variant_overrides: dict[str, Any] | None = None,
    eval_root: Path | None = None,
) -> EvalResult:
    from cadx.agent.mesh_client import make_mesh_client
    from cadx.llm.client import make_llm
    from cadx.runtime_config import runtime_from_db_or_default

    project_id = uuid.uuid4()
    version_id = uuid.uuid4()
    await _ensure_eval_project(project_id, bm.name)
    runtime = await runtime_from_db_or_default()
    # variant sets defaults; benchmark-local `runtime` keys pin on conflict
    runtime_overrides = {**(variant_overrides or {}), **bm.runtime}
    if runtime_overrides:
        runtime = runtime.model_copy(update=runtime_overrides)
    if bm.llm_script is not None:
        llm = FakeLLM(bm.llm_script)
        llm.tools_mode = bm.llm_tools
    elif llm is None:
        from cadx.llm.capabilities import load_caps_session, record_caps_session

        llm = make_llm(runtime, caps=await load_caps_session(), caps_cb=record_caps_session)

    mesh = (
        make_mesh_client(fake=True) if bm.fake_mesh else make_mesh_client(runtime_settings=runtime)
    )
    image, image_ok = await _eval_image_port(bm, runtime)
    assets = []
    if bm.images:
        from cadx.agent.types import InputAsset
        from cadx.config import REPO_ROOT

        for rel in bm.images:
            p = (REPO_ROOT / rel).resolve()
            if not p.exists():
                alt = _BACKEND_ROOT / rel.removeprefix("backend/")
                if alt.exists():
                    p = alt
            if p.exists():
                assets.append(
                    InputAsset(
                        asset_id=str(uuid.uuid4()),
                        kind="image",
                        path=p,
                        filename=p.name,
                    )
                )
            else:
                # a silently dropped fixture makes mesh/hybrid benchmarks
                # degrade to CAD with "no images attached" — invisible in
                # scores, so make it loud
                log.warning("eval.fixture_missing", benchmark=bm.name, path=str(p))

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        # step transitions are the only mid-flight signal in the log — the
        # OOM kill during hybrid prep was only localizable via these emits
        import resource

        log.info(
            "eval.step",
            benchmark=bm.name,
            step=step,
            iteration=it,
            msg=str(msg)[:160],
            # pinpoint which step grows the eval process — two worker-side
            # oom_kills happened mid-run with no allocation visibility
            peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        )

    async def no_cancel() -> bool:
        return False

    # --real in the stack should dispatch heavy OCC/mesh work to the
    # cad-executor service like production — the in-process executor keeps
    # accumulating native allocations until the container OOM-kills the eval
    if queue_exec:
        from cadx.agent.executor_client import QueueExecutor

        executor: Any = QueueExecutor(runtime_settings=runtime)
    else:
        executor = InProcessExecutor()

    # count + time every sandbox call (incl. tool-session verifies and the
    # autofix re-execute) — per-iteration cost only shows up here
    exec_stats = {"n": 0, "s": 0.0}
    _inner_exec = executor

    class _CountingExecutor:
        async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
            exec_stats["n"] += 1
            t = time.monotonic()
            try:
                return await _inner_exec.execute(payload)
            finally:
                exec_stats["s"] += time.monotonic() - t

        def __getattr__(self, name: str) -> Any:
            return getattr(_inner_exec, name)

    executor = _CountingExecutor()

    vdir = storage.version_dir(project_id, version_id)
    ctx = AgentCtx(
        llm=llm,
        executor=executor,
        # JSONL per record — an OOM/SIGKILL skips the end-of-run dump,
        # and the crash postmortem above showed the cost of losing it
        trace=JsonlTrace(vdir / "trace.jsonl"),
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
        image=image,
        image_available=image_ok,
        # production parity: the worker seeds the token budget and the
        # explicit-iterations flag — without them eval runs die at the 400k
        # default budget and ignore the benchmark's max_iterations cap intent
        budget_tokens=runtime.agent_max_tokens_per_job,
        explicit_max_iterations=bool(bm.settings.get("max_iterations")),
    )
    # reuse benchmarks seed the stored mesh intermediates — resolve_mode's
    # exemption then keeps hybrid/mesh viable without images or a mesh server
    _apply_reuse_mesh_dir(ctx, bm.reuse_mesh_dir)
    root = eval_root or eval_out_root()
    t0 = time.monotonic()
    try:
        _apply_mem_limit()
        with _rss_sampler(bm.name):
            outcome: RunOutcome = await run_generation(ctx)
    except Exception as e:
        # a crashed run still gets a summary — otherwise a whole variant cell
        # silently disappears from the compare report
        _write_run_summary(
            bm,
            None,
            project_id=project_id,
            version_id=version_id,
            vdir=vdir,
            variant=variant,
            variant_overrides=variant_overrides or {},
            runtime_overrides=runtime_overrides,
            wall_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
            root=root,
        )
        raise
    finally:
        # durable postmortem: the in-memory trace survives neither crashes
        # nor container exits — always dump what was captured
        _dump_trace(ctx.trace, vdir)
    wall = time.monotonic() - t0
    records = getattr(ctx.trace, "records", [])
    result = _judge(
        bm,
        outcome,
        wall,
        vdir,
        usage=dict(getattr(ctx.llm, "total_usage", {}) or {}),
        execs=exec_stats["n"],
        exec_s=exec_stats["s"],
        step_stats=_step_stats(records),
        iterations=_iterations(records),
        variant=variant or DEFAULT_VARIANT,
    )
    _write_run_summary(
        bm,
        result,
        project_id=project_id,
        version_id=version_id,
        vdir=vdir,
        variant=variant,
        variant_overrides=variant_overrides or {},
        runtime_overrides=runtime_overrides,
        wall_s=wall,
        root=root,
    )
    return result


def _write_run_summary(
    bm: Benchmark,
    result: EvalResult | None,
    *,
    project_id: uuid.UUID,
    version_id: uuid.UUID,
    vdir: Path,
    variant: str | None,
    variant_overrides: dict[str, Any],
    runtime_overrides: dict[str, Any],
    wall_s: float,
    root: Path,
    error: str | None = None,
) -> Path | None:
    """Persist the per-run summary `eval_compare` groups on.

    `<root>/<bench>/<variant>/<run_id>/result.json` — records the variant name
    and the effective runtime overrides so runs are self-describing.
    """
    out = run_out_dir(root, bm.name, variant, str(version_id))
    payload = {
        "schema_version": 1,
        "benchmark": bm.name,
        "variant": variant or DEFAULT_VARIANT,
        "variant_overrides": variant_overrides,
        "runtime_overrides": runtime_overrides,
        "run_id": str(version_id),
        "project_id": str(project_id),
        "version_dir": str(vdir),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_s": round(wall_s, 1),
        "error": error,
        "result": dataclasses.asdict(result) if result else None,
    }
    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "result.json"
        path.write_text(json.dumps(payload, default=str, indent=1), encoding="utf-8")
    except OSError as e:
        log.warning("eval.summary_failed", benchmark=bm.name, error=str(e))
        return None
    return path


def _iterations(records: list[dict[str, Any]]) -> int:
    """Deepest iteration index seen in the trace (0 = nothing recorded)."""
    it = 0
    for r in records:
        i = r.get("iteration")
        if isinstance(i, int):
            it = max(it, i)
    return it


def _step_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step LLM stats out of the trace records.

    `thinking_*` is the reasoning split: tokens spent on calls where the
    adaptive-thinking policy resolved to ON. Only tool-session records carry
    the flag today (toolloop stamps `parsed.thinking`), so the split is a
    lower bound for non-session steps.
    """
    agg: dict[str, dict[str, int]] = {}
    for r in records:
        step = r.get("step")
        if not step or "latency_ms" not in r:
            continue  # pure event records (tool_call emits) carry no latency
        a = agg.setdefault(
            step,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "thinking_calls": 0,
                "thinking_prompt_tokens": 0,
                "thinking_completion_tokens": 0,
            },
        )
        a["calls"] += 1
        a["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        a["completion_tokens"] += int(r.get("completion_tokens") or 0)
        a["latency_ms"] += int(r.get("latency_ms") or 0)
        if (r.get("parsed") or {}).get("thinking"):
            a["thinking_calls"] += 1
            a["thinking_prompt_tokens"] += int(r.get("prompt_tokens") or 0)
            a["thinking_completion_tokens"] += int(r.get("completion_tokens") or 0)
    return agg


def _dump_trace(trace: MemoryTrace, vdir: Path) -> None:
    try:
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "trace.json").write_text(
            json.dumps(trace.records, default=str, indent=1), encoding="utf-8"
        )
    except OSError as e:
        log.warning("eval.trace_dump_failed", error=str(e))


def _judge(
    bm: Benchmark,
    outcome: RunOutcome,
    wall: float,
    version_dir: Path,
    *,
    usage: dict[str, Any] | None = None,
    execs: int = 0,
    exec_s: float = 0.0,
    step_stats: dict[str, Any] | None = None,
    iterations: int = 0,
    variant: str = DEFAULT_VARIANT,
) -> EvalResult:
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
    # usage lives on the llm client; outcome.usage is only populated on the
    # success path, so failures would silently report calls=0
    usage = outcome.usage or usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    calls = int(usage.get("calls") or 0)
    features_ok = _features_ok(bm.expected_features, outcome.features)
    code_ok = (outcome.code is not None) if bm.expect_code else None
    issue_texts = [
        f"{i.get('what', '')} {i.get('fix_hint', '')}".strip() for i in outcome.issues or []
    ]
    issue_ok = None
    if bm.expect_issue_contains:
        needle = bm.expect_issue_contains.lower()
        issue_ok = any(needle in t.lower() for t in issue_texts)

    passed: bool | None = outcome.status in ("succeeded", "needs_review")
    if passed:
        if dim_ok is False or features_ok is False:
            passed = False
        if bm.expect_watertight and watertight is not True:
            passed = False
        if bm.expect_code and code_ok is not True:
            passed = False
        if issue_ok is False:
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
        code_ok=code_ok,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        calls=calls,
        iterations=iterations,
        variant=variant,
        passed=passed,
        issues=issue_texts,
        resolved_mode=outcome.resolved_mode or bm.gen_mode,
        execs=execs,
        exec_s=round(exec_s, 1),
        step_stats=step_stats or {},
        mode_reason=" | ".join(
            p
            for p in (
                str((outcome.plan or {}).get("approach_reason") or ""),
                outcome.mode_note,
            )
            if p
        ),
    )


async def run_all(benchmarks: list[Benchmark] | None = None) -> list[EvalResult]:
    results = []
    for bm in benchmarks or load_benchmarks():
        try:
            results.append(await run_benchmark(bm))
        except Exception as e:
            results.append(EvalResult(bm.name, "crashed", None, None, None, 0.0, str(e)))
    return results
