"""llama.cpp flag suite (KVU pool, cache types, flash-attn, GPU offload,
prompt cache) + media-part capability gating + mlx server_env (APC)."""

from unittest.mock import patch

from meshserver.config import ServerConfig


def _gguf_inst(cfg, *, kv_bpt=0):
    from meshserver.adapters.llm_gguf import make_gguf_adapter

    cls = make_gguf_adapter("org/m", ["m.gguf"], id="t-gguf", kv_bytes_per_token=kv_bpt)
    return cls.create(cfg)


def _argv(inst, tmp_path, flavor="bin", flags=None):
    """server_argv with a stubbed resolver + flag-probe result."""
    prefix = ["llama-server"] if flavor == "bin" else ["python", "-m", "llama_cpp.server"]
    from meshserver.adapters import llm_gguf

    key = " ".join(prefix)
    saved = dict(llm_gguf._HELP_CACHE)
    if flags is not None:
        llm_gguf._HELP_CACHE[key] = flags
    try:
        with (
            patch.object(type(inst), "_resolve_server", return_value=(prefix, flavor)),
            patch.object(inst, "_model_path", return_value=tmp_path / "m.gguf"),
        ):
            return inst.server_argv(9999)
    finally:
        llm_gguf._HELP_CACHE.clear()
        llm_gguf._HELP_CACHE.update(saved)


_FULL_FLAGS = {
    "--kv-unified",
    "--kv-unified-per-slot",
    "--cache-type-k",
    "--cache-type-v",
    "--flash-attn",
    "-fa",
    "--cache-ram",
    "--cache-idle-slots",
}


def test_bin_unified_kv_and_slot_cap(tmp_path):
    cfg = ServerConfig(
        model_dir=tmp_path / "m",
        llm_kv_unified=True,
        llm_kv_per_slot=8192,
        llm_parallel=4,
    )
    argv = _argv(_gguf_inst(cfg), tmp_path, flags=_FULL_FLAGS)
    assert "--kv-unified" in argv
    assert argv[argv.index("--kv-unified-per-slot") + 1] == "8192"
    assert argv[argv.index("--parallel") + 1] == "4"
    # -c stays the shared pool size
    assert argv[argv.index("--ctx-size") + 1] == "16384"


def test_bin_kvu_skipped_on_old_binary(tmp_path):
    cfg = ServerConfig(model_dir=tmp_path / "m", llm_kv_unified=True)
    argv = _argv(_gguf_inst(cfg), tmp_path, flags={"--flash-attn"})
    assert "--kv-unified" not in argv


def test_bin_kvu_off_by_default_probe_failure(tmp_path):
    # probe failed (flags=None → real _server_flags path would run the
    # nonexistent binary; simulate via empty set) → classic flags only
    cfg = ServerConfig(model_dir=tmp_path / "m")
    argv = _argv(_gguf_inst(cfg), tmp_path, flags=set())
    for f in ("--kv-unified", "--cache-type-k", "-fa", "--flash-attn", "--cache-ram"):
        assert f not in argv


def test_bin_cache_type_flash_prompt_cache(tmp_path):
    cfg = ServerConfig(
        model_dir=tmp_path / "m",
        llm_kv_cache_type="q8_0",
        llm_flash_attn=True,
        llm_prompt_cache_mb=4096,
    )
    argv = _argv(_gguf_inst(cfg), tmp_path, flags=_FULL_FLAGS)
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
    assert "--flash-attn" in argv
    assert argv[argv.index("--cache-ram") + 1] == "4096"
    assert "--cache-idle-slots" in argv


def test_bin_cache_ram_skipped_with_mmproj(tmp_path):
    """host-RAM prompt cache is incompatible with mtmd (mmproj)."""
    from meshserver.adapters.llm_gguf import make_gguf_adapter

    (tmp_path / "models" / "gguf").mkdir(parents=True)
    (tmp_path / "models" / "gguf" / "mmproj.gguf").write_bytes(b"x")
    cls = make_gguf_adapter(
        "org/m", ["m.gguf"], mmproj="mmproj.gguf", id="t-vlm-gguf"
    )
    cfg = ServerConfig(model_dir=tmp_path / "models", llm_prompt_cache_mb=4096)
    inst = cls.create(cfg)
    argv = _argv(inst, tmp_path, flags=_FULL_FLAGS)
    assert "--mmproj" in argv
    assert "--cache-ram" not in argv
    assert "--cache-idle-slots" not in argv


def test_bin_gpu_layers_default_and_custom(tmp_path):
    argv = _argv(_gguf_inst(ServerConfig(model_dir=tmp_path / "m")), tmp_path, flags=set())
    assert argv[argv.index("--n-gpu-layers") + 1] == "-1"
    cfg = ServerConfig(model_dir=tmp_path / "m2", llm_gpu_layers=24)
    argv = _argv(_gguf_inst(cfg), tmp_path, flags=set())
    assert argv[argv.index("--n-gpu-layers") + 1] == "24"


def test_py_flavor_flags_and_cpu_fix(tmp_path):
    """llama_cpp.server fallback: n_gpu_layers must never default to 0
    (silent CPU run), cache-type + flash-attn map to its flag names."""
    cfg = ServerConfig(
        model_dir=tmp_path / "m",
        llm_kv_cache_type="q8_0",
        llm_parallel=4,  # meaningless on the serial py path
    )
    py_flags = {"--n_gpu_layers", "--cache_type_k", "--cache_type_v", "--flash_attn"}
    argv = _argv(_gguf_inst(cfg), tmp_path, flavor="py", flags=py_flags)
    assert argv[argv.index("--n_gpu_layers") + 1] == "-1"
    assert argv[argv.index("--cache_type_k") + 1] == "q8_0"
    assert "--flash_attn" in argv
    assert "--parallel" not in argv and "--kv-unified" not in argv
    # probe failure → optional flags skipped, n_gpu_layers still emitted
    argv2 = _argv(_gguf_inst(cfg), tmp_path, flavor="py", flags=set())
    assert argv2[argv2.index("--n_gpu_layers") + 1] == "-1"
    assert "--cache_type_k" not in argv2 and "--flash_attn" not in argv2


def test_mlx_apc_server_env(tmp_path):
    from meshserver.adapters.llm_mlx import make_mlx_adapter

    cls = make_mlx_adapter("org/m-4bit", id="t-mlx", vision=True)
    cfg = ServerConfig(model_dir=tmp_path / "m", llm_apc=True)
    assert cls.create(cfg).server_env() == {"APC_ENABLED": "1"}
    cfg2 = ServerConfig(model_dir=tmp_path / "m2")
    assert cls.create(cfg2).server_env() == {}


def test_mlx_kv_scheme_flag(tmp_path):
    """--kv-quant-scheme exists only on mlx_vlm.server, not mlx_lm."""
    from meshserver.adapters.llm_mlx import make_mlx_adapter

    vlm = make_mlx_adapter("org/m-4bit", id="t-vlm", vision=True)
    cfg = ServerConfig(model_dir=tmp_path / "m", llm_kv_bits=8, llm_kv_scheme="turboquant")
    argv = vlm.create(cfg).server_argv(9999)
    assert argv[argv.index("--kv-quant-scheme") + 1] == "turboquant"
    # lm flavor: scheme must not leak (mlx_lm CLI rejects the flag)
    lm = make_mlx_adapter("org/m-4bit", id="t-lm")
    argv_lm = lm.create(cfg).server_argv(9999)
    assert "--kv-quant-scheme" not in argv_lm
    # no scheme emitted without kv_bits
    cfg2 = ServerConfig(model_dir=tmp_path / "m2", llm_kv_scheme="turboquant")
    assert "--kv-quant-scheme" not in vlm.create(cfg2).server_argv(9999)


def test_media_part_gating():
    from meshserver.llm_proxy import _wants_audio, _wants_vision

    img = {"messages": [{"role": "u", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
    vid = {"messages": [{"role": "u", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}]}
    aud = {"messages": [{"role": "u", "content": [{"type": "input_audio", "input_audio": {"data": "x"}}]}]}
    txt = {"messages": [{"role": "u", "content": "hi"}]}
    assert _wants_vision(img) and _wants_vision(vid)
    assert not _wants_vision(aud) and not _wants_vision(txt)
    assert _wants_audio(aud) and not _wants_audio(vid)


def test_select_llm_audio_capability(tmp_path):
    """audio requests must not match a vision-only model on auto."""
    from meshserver.llm_models import llm_adapter_classes
    from meshserver.registry import Registry

    reg = Registry(ServerConfig(model_dir=tmp_path / "m"))
    reg._llm_classes = llm_adapter_classes(reg.cfg)
    # every builtin is vision-but-not-audio → audio request finds nothing
    body = {
        "model": "auto",
        "messages": [{"role": "u", "content": [{"type": "input_audio", "input_audio": {}}]}],
    }
    assert reg.select_llm("auto", need_audio=True) is None
    assert body  # signature-shape only; _select_llm handles the ladder
