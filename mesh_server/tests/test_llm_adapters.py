"""LlmAdapter lifecycle: subprocess spawn, readiness polling, group kill,
served_model_id, and the make_*_adapter class factories.

The "server" is a tiny stdlib HTTP stub — no mlx_lm / llama.cpp needed.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from meshserver.adapters.llm_base import LlmAdapter, free_port
from meshserver.adapters.llm_gguf import LlamaCppAdapter, make_gguf_adapter
from meshserver.adapters.llm_mlx import (
    MlxLmAdapter,
    _resolve_python,
    make_mlx_adapter,
)
from meshserver.config import ServerConfig

_STUB = textwrap.dedent(
    """
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {"object": "list", "data": [{"id": "stub-served-id", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
    """
)


class _StubLlm(LlmAdapter):
    id = "stub-llm"
    name = "Stub LLM"
    device = "cpu"
    backend = "stub"

    def server_argv(self, port: int) -> list[str]:
        import sys

        return [sys.executable, "-c", _STUB, str(port)]


class _NeverReady(LlmAdapter):
    id = "never-llm"

    def server_argv(self, port: int) -> list[str]:  # noqa: ARG002
        import sys

        return [sys.executable, "-c", "import time; time.sleep(60)"]


class _EarlyExit(LlmAdapter):
    id = "early-llm"

    def server_argv(self, port: int) -> list[str]:  # noqa: ARG002
        import sys

        return [sys.executable, "-c", "import sys; sys.exit(3)"]


def _cfg(tmp_path: Path, timeout: int = 15) -> ServerConfig:
    return ServerConfig(job_dir=tmp_path, model_dir=tmp_path / "models", llm_load_timeout_s=timeout)


def test_free_port_is_bindable():
    import socket

    port = free_port()
    assert isinstance(port, int) and port > 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_load_spawns_and_reports_ready(tmp_path):
    inst = _StubLlm(_cfg(tmp_path))
    try:
        inst.load()
        assert inst.loaded
        assert inst.port is not None
        assert inst.base_url() == f"http://127.0.0.1:{inst.port}/v1"
        # captured from the stub's GET /v1/models body during readiness
        assert inst.served_model_id() == "stub-served-id"
        assert (tmp_path / "llm-stub-llm.log").is_file()
    finally:
        inst.unload()
    assert not inst.loaded


def test_unload_kills_process_group(tmp_path):
    inst = _StubLlm(_cfg(tmp_path))
    inst.load()
    proc = inst._proc
    assert proc is not None and proc.poll() is None
    inst.unload()
    assert proc.poll() is not None  # dead — SIGTERM or SIGKILL
    assert inst.port is None
    with pytest.raises(RuntimeError):
        inst.base_url()


def test_load_readiness_timeout(tmp_path):
    inst = _NeverReady(_cfg(tmp_path, timeout=1))
    with pytest.raises(TimeoutError):
        inst.load()
    # timed-out child must be cleaned up, not left sleeping for 60s
    assert not inst.loaded


def test_load_early_exit(tmp_path):
    inst = _EarlyExit(_cfg(tmp_path))
    with pytest.raises(RuntimeError, match="exited early"):
        inst.load()


def test_served_model_id_fallback():
    inst = _StubLlm(ServerConfig())
    inst.hf_repo = "org/model"
    assert inst.served_model_id() == "org/model"
    inst.hf_repo = ""
    assert inst.served_model_id() == "stub-llm"


# -- factories ---------------------------------------------------------------- #


def test_make_mlx_adapter_class_attrs():
    cls = make_mlx_adapter(
        "mlx-community/Qwen3-8B-4bit",
        id="qwen3-mlx",
        name="Qwen3",
        vision=True,
        tools=False,
        ctx=8192,
        est_mb=5000,
    )
    assert issubclass(cls, MlxLmAdapter)
    assert cls.kind == "llm"
    assert cls.backend == "mlx-lm"
    assert cls.device == "mlx"
    assert cls.id == "qwen3-mlx"
    assert cls.hf_repo == "mlx-community/Qwen3-8B-4bit"
    assert cls.supports_vision is True
    assert cls.supports_tools is False
    assert cls.context_len == 8192
    assert cls.est_mem_mb == 5000


def test_make_gguf_adapter_class_attrs():
    cls = make_gguf_adapter(
        "org/repo",
        ["model-Q4_K_M.gguf"],
        mmproj="mmproj.gguf",
        id="g1",
        vision=False,
    )
    assert issubclass(cls, LlamaCppAdapter)
    assert cls.kind == "llm"
    assert cls.backend == "llama.cpp"
    assert cls.gguf_files == ["model-Q4_K_M.gguf"]
    assert cls.gguf_mmproj == "mmproj.gguf"
    assert cls.supports_vision is True  # implied by mmproj


def test_mlx_downloaded_probe(monkeypatch):
    cls = make_mlx_adapter("org/ok", id="x")
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo, fn, **kw: "/cache/x" if repo == "org/ok" else None,
    )
    assert cls.downloaded(ServerConfig()) is True
    cls2 = make_mlx_adapter("org/missing", id="y")
    assert cls2.downloaded(ServerConfig()) is False


def test_gguf_downloaded_and_argv(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.llama_server_bin = "/bin/echo"
    cfg.llm_ctx = 4096
    cls = make_gguf_adapter("org/r", ["m.gguf"], mmproj="v.gguf", id="g", ctx=8192)
    gdir = Path(cfg.model_dir) / "gguf"
    assert cls.downloaded(cfg) is False
    gdir.mkdir(parents=True)
    (gdir / "m.gguf").write_bytes(b"weights")
    assert cls.downloaded(cfg) is False  # mmproj still missing
    (gdir / "v.gguf").write_bytes(b"proj")
    assert cls.downloaded(cfg) is True
    assert cls.available(cfg) is True  # llama_server_bin resolves

    inst = cls(cfg)
    argv = inst.server_argv(9999)
    assert argv[0] == "/bin/echo"
    assert "-m" in argv and str(gdir / "m.gguf") in argv
    assert "--ctx-size" in argv and "4096" in argv  # cfg.llm_ctx wins over ctx attr
    assert "--jinja" in argv
    assert "--mmproj" in argv and str(gdir / "v.gguf") in argv


def test_gguf_available_falls_back_to_path(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setenv("LLAMA_SERVER_BIN", "")  # dataclass already built — set attr
    cfg.llama_server_bin = ""
    cls = make_gguf_adapter("org/r", ["m.gguf"], id="g")
    # PATH lookup happens via shutil.which — patch it to claim a binary exists
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-server")
    assert cls.available(cfg) is True


def test_gguf_unavailable_without_files(tmp_path):
    bare = type("Bare", (LlamaCppAdapter,), {"id": "bare"})
    assert bare.available(_cfg(tmp_path)) is False


def test_mlx_argv(tmp_path):
    cls = make_mlx_adapter("org/m", id="m1")
    inst = cls(_cfg(tmp_path))
    argv = inst.server_argv(7777)
    assert argv[1:3] == ["-m", "mlx_lm.server"]
    assert "org/m" in argv and "7777" in argv


def test_make_mlx_adapter_required_module():
    """vision adapters are served by mlx_vlm.server (mlx_lm strips image
    parts for the qwen3_5 family); text adapters stay on mlx_lm."""
    vision = make_mlx_adapter("org/x", id="v", vision=True)
    assert vision.required_module == "mlx_vlm"
    text = make_mlx_adapter("org/x", id="t", vision=False)
    assert text.required_module == "mlx_lm"
    assert MlxLmAdapter.required_module == "mlx_lm"


def test_mlx_argv_vision_module(tmp_path):
    cls = make_mlx_adapter("org/m", id="mv", vision=True)
    argv = cls(_cfg(tmp_path)).server_argv(7777)
    assert argv[1:3] == ["-m", "mlx_vlm.server"]
    assert "org/m" in argv and "7777" in argv


def test_resolve_python_cfg_override(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.llm_python = "/custom/llm/python"
    assert _resolve_python(cfg) == "/custom/llm/python"


def test_resolve_python_venv_llm(tmp_path, monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    venv_py = tmp_path / ".venv-llm" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.touch()
    monkeypatch.setattr(llm_mlx, "_llm_venv_python", lambda: venv_py)
    assert _resolve_python(_cfg(tmp_path)) == str(venv_py)


def test_resolve_python_falls_back_to_sys(tmp_path, monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    monkeypatch.setattr(
        llm_mlx, "_llm_venv_python", lambda: tmp_path / "absent" / "bin" / "python"
    )
    assert _resolve_python(_cfg(tmp_path)) == sys.executable


def test_module_importable_probes_and_caches(monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    calls: list[list[str]] = []

    class _Ok:
        returncode = 0

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        return _Ok()

    monkeypatch.setattr(llm_mlx.subprocess, "run", fake_run)
    llm_mlx._MODULE_IMPORT_CACHE.clear()
    assert llm_mlx._module_importable("/py", "mlx_lm") is True
    assert llm_mlx._module_importable("/py", "mlx_lm") is True
    assert len(calls) == 1  # second call served from the cache
    assert calls[0][:2] == ["/py", "-c"] and "import mlx_lm" in calls[0][2]

    class _Bad:
        returncode = 1

    monkeypatch.setattr(llm_mlx.subprocess, "run", lambda cmd, **kw: _Bad())
    llm_mlx._MODULE_IMPORT_CACHE.clear()
    assert llm_mlx._module_importable("/py2", "mlx_vlm") is False
    llm_mlx._MODULE_IMPORT_CACHE.clear()


def test_mlx_available_darwin_probe_ok(tmp_path, monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    monkeypatch.setattr(llm_mlx.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(llm_mlx.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(llm_mlx, "_module_importable", lambda py, mod: True)
    cls = make_mlx_adapter("org/m", id="av")
    assert cls.available(_cfg(tmp_path)) is True


def test_mlx_available_darwin_probe_fails(tmp_path, monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    monkeypatch.setattr(llm_mlx.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(llm_mlx.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(llm_mlx, "_module_importable", lambda py, mod: False)
    cls = make_mlx_adapter("org/m", id="av2", vision=True)
    assert cls.available(_cfg(tmp_path)) is False
    err = cls.available_error or ""
    assert "install_llm.sh" in err or "mlx_vlm" in err


def test_mlx_available_non_darwin(tmp_path, monkeypatch):
    import meshserver.adapters.llm_mlx as llm_mlx

    monkeypatch.setattr(llm_mlx.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llm_mlx.platform, "machine", lambda: "x86_64")
    # the import probe must not even run off Apple Silicon
    monkeypatch.setattr(
        llm_mlx,
        "_module_importable",
        lambda py, mod: pytest.fail("probe ran on non-darwin"),
    )
    cls = make_mlx_adapter("org/m", id="av3")
    assert cls.available(_cfg(tmp_path)) is False
    assert "Apple Silicon" in (cls.available_error or "")


def test_llm_kind_marked(tmp_path):
    """registry.models() reads kind off the class — llm adapters report 'llm'."""
    from meshserver.adapters.llm_mlx import MlxLmAdapter as M

    assert M.kind == "llm"
    assert LlamaCppAdapter.kind == "llm"


def test_downloaded_no_repo(tmp_path):
    bare = make_gguf_adapter("org/r", ["f.gguf"], id="b")
    assert bare.downloaded(_cfg(tmp_path)) is False
    norepo = make_mlx_adapter("", id="n")
    assert norepo.downloaded(_cfg(tmp_path)) is False


def test_unload_idempotent(tmp_path: Any):
    inst = _StubLlm(_cfg(tmp_path))
    inst.unload()  # never loaded — must not raise
    assert not inst.loaded
