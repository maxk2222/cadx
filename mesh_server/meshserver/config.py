"""mesh-server configuration (env-driven + runtime overlay).

Env vars provide the boot defaults; a small set of scheduling knobs is
runtime-mutable via ``PUT /v1/config`` — values persist to
``model_dir/server_config.json`` (same overlay pattern as
``custom_models.json``) and ``cfg.effective(name)`` resolves
overlay → env-field → declared default at call time.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """int env var where EMPTY means unset — the run/mesh-server.env
    supervision contract writes declared-but-empty vars (VAR=) for
    optionals, and int('') would crash boot."""
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_keys() -> tuple[str, ...]:
    raw = os.environ.get("MESH_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    single = os.environ.get("MESH_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    return tuple(keys)


# keys a PUT /v1/config patch may carry — (type, applies-at). Everything
# "immediate" is re-read per call site; "next_model_load" is baked into the
# managed server's argv at load time.
RUNTIME_MUTABLE: dict[str, tuple[type, str]] = {
    "llm_max_concurrent": (int, "immediate"),
    "llm_queue_wait_s": (float, "immediate"),
    "llm_queue_fair_s": (float, "immediate"),
    "llm_queue_max": (int, "immediate"),
    "llm_stream_queue_wait_s": (float, "immediate"),
    "llm_keepalive_s": (float, "immediate"),
    "llm_prio_internal": (int, "immediate"),
    "llm_prio_external": (int, "immediate"),
    "llm_upstream_timeout_s": (float, "immediate"),
    "llm_stream_idle_s": (float, "immediate"),
    "llm_reasoning_parse": (bool, "immediate"),
    "llm_failed_ttl_s": (int, "immediate"),
    "llm_parallel": (int, "next_model_load"),
    "llm_pin": (bool, "immediate"),
    "openai_image_wait_s": (float, "immediate"),
    "unload_idle_s": (int, "immediate"),
    "job_min_free_mb": (int, "immediate"),
    "mem_headroom_mb": (int, "immediate"),
    # managed-server argv knobs — baked into the subprocess at spawn, so the
    # overlay value applies at the next model load
    "llm_ctx": (int, "next_model_load"),
    "llm_kv_bits": (int, "next_model_load"),
    "llm_prefill_step": (int, "next_model_load"),
    "llm_draft": (str, "next_model_load"),
    "llm_max_num_seqs": (int, "next_model_load"),
    "llm_prompt_cache_mb": (int, "next_model_load"),
    # llama.cpp binary suite (feature-probed against llama-server --help)
    "llm_gpu_layers": (int, "next_model_load"),
    "llm_kv_unified": (bool, "next_model_load"),
    "llm_kv_per_slot": (int, "next_model_load"),
    "llm_kv_cache_type": (str, "next_model_load"),
    "llm_flash_attn": (bool, "next_model_load"),
    "llm_apc": (bool, "next_model_load"),
    "llm_kv_scheme": (str, "next_model_load"),
}


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.environ.get("MESH_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("MESH_PORT", 8765))
    # device override: auto | cuda | rocm | mps | mlx | cpu
    device: str = field(default_factory=lambda: os.environ.get("MESH_DEVICE", "auto"))
    model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "MESH_MODEL_DIR",
                str(Path.home() / ".cache" / "meshserver" / "models"),
            )
        )
    )
    job_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MESH_JOB_DIR", "/tmp/meshserver-jobs")
        )
    )
    job_ttl_s: int = field(
        default_factory=lambda: _env_int("MESH_JOB_TTL_S", 3600)
    )
    # a running job whose (progress, message) pair hasn't changed for this
    # long is marked failed — MPS/MLX stalls freeze at 0% CPU forever and
    # would otherwise hold the single-flight queue indefinitely
    job_stall_s: int = field(
        default_factory=lambda: _env_int("MESH_JOB_STALL_S", 600)
    )
    # a client-aborted job whose child had already been idle this long is
    # recorded as *stalled* (adapter badlisted) rather than cancelled — the
    # abort is then the symptom of a frozen generation, not a user choice.
    # Kept below job_stall_s because the client watchdog fires first.
    job_cancel_idle_s: int = field(
        default_factory=lambda: _env_int("MESH_JOB_CANCEL_IDLE_S", 60)
    )
    # run each generation in its own subprocess (python -m meshserver.gen_worker)
    # so stall/cancel/RSS-cap can SIGKILL it and the OS reclaims all model
    # memory — a frozen adapter thread inside this process can't be killed
    jobs_subprocess: bool = field(
        default_factory=lambda: _env_bool("MESH_JOBS_SUBPROCESS", True)
    )
    # hard RSS cap for the generation subprocess (MB, 0 = off) — a runaway
    # model is killed before it can swap the host into a watchdog panic
    job_max_rss_mb: int = field(
        default_factory=lambda: _env_int("MESH_JOB_MAX_RSS_MB", 24576)
    )
    # refuse new jobs when the host reports less free RAM (MB, 0 = off)
    job_min_free_mb: int = field(
        default_factory=lambda: _env_int("MESH_JOB_MIN_FREE_MB", 0)
    )
    max_queue: int = field(
        default_factory=lambda: _env_int("MESH_MAX_QUEUE", 8)
    )
    unload_idle_s: int = field(
        default_factory=lambda: _env_int("MESH_UNLOAD_IDLE_S", 0)
    )
    fake: bool = field(default_factory=lambda: _env_bool("MESH_FAKE"))
    # enable adapters even when no GPU device was detected (e.g. slow CPU run)
    allow_cpu: bool = field(default_factory=lambda: _env_bool("MESH_ALLOW_CPU"))
    default_model: str = field(
        default_factory=lambda: os.environ.get("MESH_DEFAULT_MODEL", "auto")
    )
    bg_removal: str = field(
        default_factory=lambda: os.environ.get("MESH_BG_REMOVAL", "auto")
    )  # auto | always | never
    # text-to-image adapter family
    image_default_model: str = field(
        default_factory=lambda: os.environ.get("MESH_IMAGE_MODEL", "auto")
    )
    flux_mlx_quant: int = field(
        default_factory=lambda: _env_int("MESH_FLUX_MLX_QUANT", 4)
    )  # bit depth for the mflux weight file (4|8|16)
    # bake the LightX2V Lightning distillation LoRA into qwen-image-edit —
    # 8-step edits instead of ~28, ~4x faster synth views (Apache-2.0,
    # ungated). Set MESH_QWEN_EDIT_LIGHTNING=0 for the un-distilled model.
    qwen_edit_lightning: bool = field(
        default_factory=lambda: _env_bool("MESH_QWEN_EDIT_LIGHTNING", default=True)
    )
    # diffusers CPU-offload ladder: auto|model|sequential|none —
    # 'auto' enables model-level offload below cuda_offload_vram_gb free VRAM
    cuda_offload: str = field(
        default_factory=lambda: os.environ.get("MESH_CUDA_OFFLOAD", "auto")
    )
    cuda_offload_vram_gb: float = field(
        default_factory=lambda: _env_float("MESH_CUDA_OFFLOAD_VRAM_GB", 24)
    )
    # optional MLX unified-memory cap (0 = unlimited)
    mlx_mem_limit_gb: float = field(
        default_factory=lambda: _env_float("MESH_MLX_MEM_LIMIT_GB", 0)
    )

    # --- memory ledger / scheduling (Phase: multi-model residency) ---
    # host RAM kept free below this reserved amount when admitting models (MB)
    mem_headroom_mb: int = field(
        default_factory=lambda: _env_int("MESH_MEM_HEADROOM_MB", 4096)
    )
    # keep a warm per-model generation worker alive between jobs (saves the
    # model reload each subprocess job pays today)
    warm_workers: bool = field(
        default_factory=lambda: _env_bool("MESH_WARM_WORKERS", True)
    )
    warm_idle_s: int = field(
        default_factory=lambda: _env_int("MESH_WARM_IDLE_S", 300)
    )
    warm_max: int = field(
        default_factory=lambda: _env_int("MESH_WARM_MAX", 2)
    )
    # queue policy: a queued job older than this wins over model affinity
    queue_fair_s: int = field(
        default_factory=lambda: _env_int("MESH_QUEUE_FAIR_S", 120)
    )
    # queued job waiting on memory admission fails after this (s, 0 = never)
    queue_mem_wait_s: int = field(
        default_factory=lambda: _env_int("MESH_QUEUE_MEM_WAIT_S", 300)
    )

    # --- LLM adapter family (kind="llm") ---
    # a pinned LLM is never evicted by generation jobs; default lets the
    # scheduler reclaim its memory when a mesh/image job needs it
    llm_pin: bool = field(default_factory=lambda: _env_bool("MESH_LLM_PIN"))
    llm_load_timeout_s: int = field(
        default_factory=lambda: _env_int("MESH_LLM_LOAD_TIMEOUT_S", 300)
    )
    llm_ctx: int = field(
        default_factory=lambda: _env_int("MESH_LLM_CTX", 16384)
    )
    # llama-server binary override (empty = python -m llama_cpp.server)
    llama_server_bin: str = field(
        default_factory=lambda: os.environ.get("LLAMA_SERVER_BIN", "")
    )
    # interpreter for mlx LLM servers (empty = auto: mesh_server/.venv-llm,
    # else sys.executable) — the dedicated venv holds transformers>=5 stacks
    # (mlx-lm>=0.32, mlx-vlm) isolated from the shared torch/tf4 env
    llm_python: str = field(
        default_factory=lambda: os.environ.get("MESH_LLM_PYTHON", "")
    )

    # --- LAN / client access (Phase: openai/ollama-compatible exposure) ---
    # bearer tokens for Authorization: Bearer — REQUIRED for any
    # non-loopback bind (create_app refuses without them); loopback with no
    # keys keeps today's open-by-default behaviour
    api_keys: tuple[str, ...] = field(default_factory=_env_keys)
    # MESH_LAN=1 binds 0.0.0.0 when MESH_HOST wasn't given explicitly and
    # requires api_keys at startup
    lan: bool = field(default_factory=lambda: _env_bool("MESH_LAN"))
    # CORS allow-origins (comma list); empty = default "*" when lan else off
    cors_origins: str = field(
        default_factory=lambda: os.environ.get("MESH_CORS_ORIGINS", "")
    )

    # --- llm request scheduling (all runtime-mutable via /v1/config) ---
    # proxy-side semaphore bounding concurrent upstream calls (0 = unbounded)
    llm_max_concurrent: int = field(
        default_factory=lambda: _env_int("MESH_LLM_MAX_CONCURRENT", 4)
    )
    # bounded wait for a queue slot / memory admission before a 503
    llm_queue_wait_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_QUEUE_WAIT_S", 30)
    )
    # scheduling fairness window for the llm request queue: one affinity
    # match OR one priority step earns a +fair_s score bonus; anything
    # waited longer than the window wins regardless (starvation-free)
    llm_queue_fair_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_QUEUE_FAIR_S", 60)
    )
    # max requests held in the llm queue — beyond this an immediate 503
    # instead of unbounded pending work
    llm_queue_max: int = field(
        default_factory=lambda: _env_int("MESH_LLM_QUEUE_MAX", 32)
    )
    # bounded wait for streaming requests — longer than llm_queue_wait_s
    # because the client sees live queue/load progress the whole time
    llm_stream_queue_wait_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_STREAM_QUEUE_WAIT_S", 600)
    )
    # SSE comment keepalive cadence while a stream request is queued/loading
    llm_keepalive_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_KEEPALIVE_S", 5)
    )
    # scheduling priority per caller class — internal = KIARA backend /
    # loopback tooling, external = authenticated LAN clients. One step =
    # +llm_queue_fair_s score (external default jumps ~10 windows ahead)
    llm_prio_internal: int = field(
        default_factory=lambda: _env_int("MESH_LLM_PRIO_INTERNAL", 0)
    )
    llm_prio_external: int = field(
        default_factory=lambda: _env_int("MESH_LLM_PRIO_EXTERNAL", 10)
    )
    # total bound for a non-stream upstream call (0 = unbounded — the
    # historical default since generation latency is genuinely unbounded)
    llm_upstream_timeout_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_UPSTREAM_TIMEOUT_S", 0)
    )
    # bound on the gap BETWEEN upstream stream chunks — a wedged server
    # that holds a slot forever gets aborted after this silence (0 = off)
    llm_stream_idle_s: float = field(
        default_factory=lambda: _env_float("MESH_LLM_STREAM_IDLE_S", 0)
    )
    # normalize thinking-model output to the OpenAI wire format — raw
    # <think>…</think> regions inside content get split into
    # reasoning_content on both the streaming and the non-stream path
    # (mlx_lm/mlx_vlm always stream raw tags; llama.cpp depends on its
    # --reasoning-format build). Off = byte-verbatim relay.
    llm_reasoning_parse: bool = field(
        default_factory=lambda: _env_bool("MESH_LLM_REASONING_PARSE", True)
    )
    # badlist window for a failed adapter — escalates ×min(count,4) per
    # repeat failure so one-off transients heal but stall/OOM breakers
    # stay parked long enough not to flap
    llm_failed_ttl_s: int = field(
        default_factory=lambda: _env_int("MESH_LLM_FAILED_TTL_S", 300)
    )
    # llama.cpp --parallel slots (applies at next model load; KV cache
    # scales with slots × ctx, so the ledger estimate gains headroom)
    llm_parallel: int = field(
        default_factory=lambda: _env_int("MESH_LLM_PARALLEL", 1)
    )
    # sync-wait bound for OpenAI-style /v1/images/* wrappers
    openai_image_wait_s: float = field(
        default_factory=lambda: _env_float("MESH_OPENAI_IMAGE_WAIT_S", 600)
    )

    # --- managed-server argv tuning (mlx_lm / mlx_vlm / llama.cpp) ----------
    # KV-cache quantization bits (0 = server default/unquantized). Shrinks
    # resident KV memory ~4x at 8 bits → larger ctx/concurrency fits RAM.
    llm_kv_bits: int = field(
        default_factory=lambda: _env_int("MESH_LLM_KV_BITS", 0)
    )
    # prefill step size in tokens (0 = server default 2048)
    llm_prefill_step: int = field(
        default_factory=lambda: _env_int("MESH_LLM_PREFILL_STEP", 0)
    )
    # speculative decoding drafter: "" = off, "auto" = same repo (kind
    # auto-detected — Qwen3.5/3.6/3.8 bundle MTP weights), or
    # "hf/repo[:dflash|eagle3|mtp]" for an explicit drafter+family
    llm_draft: str = field(
        default_factory=lambda: os.environ.get("MESH_LLM_DRAFT", "")
    )
    # mlx_vlm continuous-batch bound (0 = unbounded) — backpressure instead
    # of unbounded peak memory when several requests queue upstream-side
    llm_max_num_seqs: int = field(
        default_factory=lambda: _env_int("MESH_LLM_MAX_NUM_SEQS", 0)
    )
    # prompt KV-cache budget in MB (0 = off) — repeated long prefixes
    # (system+spec) served from cache instead of re-prefilled per call;
    # mlx_lm → --prompt-cache-*, llama.cpp → --cache-ram + --cache-idle-slots
    llm_prompt_cache_mb: int = field(
        default_factory=lambda: _env_int("MESH_LLM_PROMPT_CACHE_MB", 0)
    )

    # --- llama.cpp binary suite (llama-server; RTX-class CUDA hosts) --------
    # GPU layer offload: -1 = all (default); a positive value caps the layer
    # count so an image/mesh model can share VRAM. ALSO fixes the
    # llama_cpp.server fallback, whose default 0 would run on CPU.
    llm_gpu_layers: int = field(
        default_factory=lambda: _env_int("MESH_LLM_GPU_LAYERS", -1)
    )
    # unified KV pool (-kvu): -c becomes one shared pool across all parallel
    # slots (vLLM-style) instead of a static ctx-per-slot split. Requires a
    # llama-server binary >= Nov-2025; feature-probe skips it on older builds.
    llm_kv_unified: bool = field(
        default_factory=lambda: _env_bool("MESH_LLM_KV_UNIFIED", True)
    )
    # per-sequence cap inside the unified pool (0 = pool limit only) →
    # --kv-unified-per-slot
    llm_kv_per_slot: int = field(
        default_factory=lambda: _env_int("MESH_LLM_KV_PER_SLOT", 0)
    )
    # KV cache quant type "" = f16 default; e.g. "q8_0"/"q4_0" halves/quarters
    # KV VRAM → --cache-type-k/--cache-type-v
    llm_kv_cache_type: str = field(
        default_factory=lambda: os.environ.get("MESH_LLM_KV_CACHE_TYPE", "")
    )
    # flash attention (-fa) — faster + less VRAM on long ctx; CUDA sm_120 ok
    llm_flash_attn: bool = field(
        default_factory=lambda: _env_bool("MESH_LLM_FLASH_ATTN", True)
    )
    # mlx_vlm automatic prefix cache (APC_ENABLED) — block-level shared prefix
    # reuse incl. disk store; the vLLM prefix-cache equivalent on Apple Silicon
    llm_apc: bool = field(default_factory=lambda: _env_bool("MESH_LLM_APC"))
    # mlx_vlm KV quantization backend "" = default; "uniform"|"turboquant" →
    # --kv-quant-scheme (only emitted together with llm_kv_bits)
    llm_kv_scheme: str = field(
        default_factory=lambda: os.environ.get("MESH_LLM_KV_SCHEME", "")
    )

    def __post_init__(self) -> None:
        # MESH_LAN=1 with no explicit MESH_HOST → bind all interfaces; the
        # create_app guard below refuses the start without api_keys either
        # way, so this can't silently open an unauthenticated LAN service
        if self.lan and not os.environ.get("MESH_HOST"):
            self.host = "0.0.0.0"
        self._overlay_cache: dict | None = None
        self._overlay_lock = threading.Lock()

    # -- runtime overlay (server_config.json) ----------------------------- #

    @property
    def overlay_path(self) -> Path:
        return self.model_dir / "server_config.json"

    def overlay(self) -> dict:
        # effective() is read from worker threads (registry.get in
        # asyncio.to_thread) while PUT /v1/config writes on the event loop —
        # the lock keeps the lazy file read + cache swap consistent
        with self._overlay_lock:
            if self._overlay_cache is None:
                try:
                    raw = json.loads(self.overlay_path.read_text())
                    self._overlay_cache = {
                        k: v for k, v in raw.items() if k in RUNTIME_MUTABLE
                    } if isinstance(raw, dict) else {}
                except (OSError, json.JSONDecodeError):
                    self._overlay_cache = {}
            return dict(self._overlay_cache)

    def set_overlay(self, overlay: dict) -> None:
        """Replace the overlay file wholesale (atomic write) — callers pass
        the complete desired dict so key removals persist."""
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.overlay_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(overlay, indent=1))
        tmp.replace(self.overlay_path)
        with self._overlay_lock:
            self._overlay_cache = dict(overlay)

    def effective(self, name: str, default=None):
        """overlay → dataclass field (env-booted) → default."""
        if name in RUNTIME_MUTABLE:
            overlay = self.overlay()
            if name in overlay:
                return overlay[name]
        return getattr(self, name, default)


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in ("127.0.0.1", "localhost", "::1")


def get_config() -> ServerConfig:
    return ServerConfig()
