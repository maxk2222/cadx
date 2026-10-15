"""mesh-server configuration (env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.environ.get("MESH_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("MESH_PORT", "8765")))
    # device override: auto | cuda | mps | mlx | cpu
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
        default_factory=lambda: Path(os.environ.get("MESH_JOB_DIR", "/tmp/meshserver-jobs"))
    )
    job_ttl_s: int = field(default_factory=lambda: int(os.environ.get("MESH_JOB_TTL_S", "3600")))
    # a running job whose (progress, message) pair hasn't changed for this
    # long is marked failed — MPS/MLX stalls freeze at 0% CPU forever and
    # would otherwise hold the single-flight queue indefinitely
    job_stall_s: int = field(default_factory=lambda: int(os.environ.get("MESH_JOB_STALL_S", "600")))
    # run each generation in its own subprocess (python -m meshserver.gen_worker)
    # so stall/cancel/RSS-cap can SIGKILL it and the OS reclaims all model
    # memory — a frozen adapter thread inside this process can't be killed
    jobs_subprocess: bool = field(default_factory=lambda: _env_bool("MESH_JOBS_SUBPROCESS", True))
    # hard RSS cap for the generation subprocess (MB, 0 = off) — a runaway
    # model is killed before it can swap the host into a watchdog panic
    job_max_rss_mb: int = field(
        default_factory=lambda: int(os.environ.get("MESH_JOB_MAX_RSS_MB", "24576"))
    )
    # refuse new jobs when the host reports less free RAM (MB, 0 = off)
    job_min_free_mb: int = field(
        default_factory=lambda: int(os.environ.get("MESH_JOB_MIN_FREE_MB", "0"))
    )
    max_queue: int = field(default_factory=lambda: int(os.environ.get("MESH_MAX_QUEUE", "8")))
    unload_idle_s: int = field(
        default_factory=lambda: int(os.environ.get("MESH_UNLOAD_IDLE_S", "0"))
    )
    fake: bool = field(default_factory=lambda: _env_bool("MESH_FAKE"))
    # enable adapters even when no GPU device was detected (e.g. slow CPU run)
    allow_cpu: bool = field(default_factory=lambda: _env_bool("MESH_ALLOW_CPU"))
    default_model: str = field(default_factory=lambda: os.environ.get("MESH_DEFAULT_MODEL", "auto"))
    bg_removal: str = field(
        default_factory=lambda: os.environ.get("MESH_BG_REMOVAL", "auto")
    )  # auto | always | never
    # text-to-image adapter family
    image_default_model: str = field(
        default_factory=lambda: os.environ.get("MESH_IMAGE_MODEL", "auto")
    )
    flux_mlx_quant: int = field(
        default_factory=lambda: int(os.environ.get("MESH_FLUX_MLX_QUANT", "4"))
    )  # bit depth for the mflux weight file (4|8|16)
    # diffusers CPU-offload ladder: auto|model|sequential|none —
    # 'auto' enables model-level offload below cuda_offload_vram_gb free VRAM
    cuda_offload: str = field(default_factory=lambda: os.environ.get("MESH_CUDA_OFFLOAD", "auto"))
    cuda_offload_vram_gb: float = field(
        default_factory=lambda: float(os.environ.get("MESH_CUDA_OFFLOAD_VRAM_GB", "24"))
    )
    # optional MLX unified-memory cap (0 = unlimited)
    mlx_mem_limit_gb: float = field(
        default_factory=lambda: float(os.environ.get("MESH_MLX_MEM_LIMIT_GB", "0"))
    )


def get_config() -> ServerConfig:
    return ServerConfig()
