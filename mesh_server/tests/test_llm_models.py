"""llm_models catalog: builtins, custom register/unregister, persistence."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from meshserver.config import ServerConfig
from meshserver.llm_models import (
    _entry_to_class,
    builtin_llm_classes,
    custom_llm_classes,
    llm_adapter_classes,
    register_custom_model,
    unregister_custom_model,
)
from meshserver.schemas import RegisterModelRequest


def _cfg(tmp_path: Path) -> ServerConfig:
    return ServerConfig(model_dir=tmp_path / "models", job_dir=tmp_path / "jobs")


def test_builtin_catalog_shape():
    classes = builtin_llm_classes()
    assert len(classes) >= 3
    ids = [c.id for c in classes]
    assert len(ids) == len(set(ids))
    assert all(c.kind == "llm" for c in classes)
    backends = {c.backend for c in classes}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon serves only the native mlx-lm variants
        assert backends == {"mlx-lm"}
        assert all(c.hf_repo for c in classes)
    else:
        # everywhere else: llama.cpp GGUF variants only
        assert backends == {"llama.cpp"}
        gguf = next(c for c in classes if c.backend == "llama.cpp")
        assert gguf.gguf_files and gguf.hf_repo


def test_builtin_catalog_platform_gated(monkeypatch):
    import meshserver.llm_models as lm

    # darwin-arm64 → mlx-lm builtins only
    monkeypatch.setattr(lm.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(lm.platform, "machine", lambda: "arm64")
    classes = lm.builtin_llm_classes()
    assert len(classes) == 5
    assert all(c.backend == "mlx-lm" for c in classes)
    assert all(c.supports_vision is True for c in classes)
    assert all(c.context_len == 262144 for c in classes)
    ids = {c.id for c in classes}
    assert "qwen3.8-flash-next-mlx" in ids
    assert not any("qwen2.5" in i for i in ids)

    # non-darwin → llama.cpp GGUF builtins only (mmproj → vision)
    monkeypatch.setattr(lm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(lm.platform, "machine", lambda: "x86_64")
    classes = lm.builtin_llm_classes()
    assert len(classes) == 4
    assert all(c.backend == "llama.cpp" for c in classes)
    assert all(c.supports_vision is True for c in classes)
    assert all(c.gguf_files and c.gguf_mmproj for c in classes)


def test_register_mlx_custom_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    req = RegisterModelRequest(
        repo="org/foo-4bit", backend="mlx-lm", name="Foo", vision=True, est_mem_mb=1234
    )
    info = register_custom_model(cfg, req)
    assert info.kind == "llm"
    assert info.custom is True
    assert info.supports_vision is True
    assert info.supports_tools is True
    assert info.est_mem_mb == 1234
    assert info.id  # non-empty

    # persisted
    data = json.loads((Path(cfg.model_dir) / "custom_models.json").read_text())
    assert data[0]["repo"] == "org/foo-4bit"
    assert data[0]["backend"] == "mlx-lm"

    # re-discovered by the catalog
    customs = custom_llm_classes(cfg)
    assert [c.id for c in customs] == [info.id]
    assert customs[0].hf_repo == "org/foo-4bit"
    assert customs[0].supports_vision is True

    # unregister removes entry + file
    assert unregister_custom_model(cfg, info.id) is True
    assert custom_llm_classes(cfg) == []
    assert not (Path(cfg.model_dir) / "custom_models.json").exists()
    assert unregister_custom_model(cfg, info.id) is False


@pytest.mark.parametrize("bad", ["noslash", "org/", "/name", "a//b", "org/na me", ""])
def test_register_rejects_bad_repo(tmp_path, bad):
    with pytest.raises(ValueError):
        register_custom_model(_cfg(tmp_path), RegisterModelRequest(repo=bad, backend="mlx-lm"))


def test_register_rejects_non_llm_kind(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        register_custom_model(
            _cfg(tmp_path), RegisterModelRequest(repo="org/x", kind="mesh")
        )


def test_register_gguf_infers_files(tmp_path, monkeypatch):
    import meshserver.llm_models as lm

    monkeypatch.setattr(
        lm, "_infer_gguf_files", lambda repo: (["model-Q4_K_M.gguf"], None)
    )
    cfg = _cfg(tmp_path)
    info = register_custom_model(
        cfg, RegisterModelRequest(repo="org/gguf-repo", backend="llama.cpp")
    )
    assert info.backend == "llama.cpp"
    customs = custom_llm_classes(cfg)
    assert len(customs) == 1 and customs[0].gguf_files == ["model-Q4_K_M.gguf"]


def test_register_gguf_without_files_fails(tmp_path, monkeypatch):
    import meshserver.llm_models as lm

    monkeypatch.setattr(lm, "_infer_gguf_files", lambda repo: ([], None))
    with pytest.raises(ValueError, match="incomplete"):
        register_custom_model(
            _cfg(tmp_path), RegisterModelRequest(repo="org/gg", backend="llama.cpp")
        )
    # nothing persisted
    assert not (Path(_cfg(tmp_path).model_dir) / "custom_models.json").exists()


def test_custom_entry_missing_files_skipped(tmp_path):
    """A llama.cpp entry without files is skipped+warned, not fatal."""
    cfg = _cfg(tmp_path)
    Path(cfg.model_dir).mkdir(parents=True)
    (Path(cfg.model_dir) / "custom_models.json").write_text(
        json.dumps([{"repo": "org/x", "backend": "llama.cpp"}])
    )
    assert custom_llm_classes(cfg) == []


def test_custom_entry_bad_repo_skipped(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg.model_dir).mkdir(parents=True)
    (Path(cfg.model_dir) / "custom_models.json").write_text(
        json.dumps([{"repo": "!!bad!!", "backend": "mlx-lm"}])
    )
    assert custom_llm_classes(cfg) == []


def test_entry_to_class_auto_backend():
    # files listed so the entry is valid whichever backend "" resolves to
    entry = {"repo": "org/auto", "backend": "", "name": "Auto", "files": ["m.gguf"]}
    cls = _entry_to_class(entry)
    assert cls is not None
    assert cls.custom is True
    assert cls.backend in ("mlx-lm", "llama.cpp")


def test_llm_adapter_classes_merges(tmp_path):
    cfg = _cfg(tmp_path)
    register_custom_model(
        cfg, RegisterModelRequest(repo="org/c1", backend="mlx-lm", name="C1")
    )
    all_cls = llm_adapter_classes(cfg)
    ids = {c.id for c in all_cls}
    builtin_ids = {c.id for c in builtin_llm_classes()}
    assert builtin_ids <= ids
    assert len(ids) == len(builtin_ids) + 1
