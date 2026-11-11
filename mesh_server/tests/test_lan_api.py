"""LAN/OpenAI surface: bearer auth, /v1/config overlay, /v1/models
dual-shape, /v1/images/* sync wrappers, llm concurrency knobs."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from meshserver.app import create_app
from meshserver.config import ServerConfig


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (40, 90, 200)).save(buf, "PNG")
    return buf.getvalue()


def _app(tmp_path, **kw):
    kw.setdefault("fake", True)
    kw.setdefault("job_dir", tmp_path / "jobs")
    kw.setdefault("model_dir", tmp_path / "models")
    kw.setdefault("bg_removal", "never")
    return create_app(ServerConfig(**kw))


# -- auth ----------------------------------------------------------------- #


def test_non_loopback_without_keys_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="API keys"):
        _app(tmp_path, host="0.0.0.0")


def test_loopback_open_by_default(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        assert c.get("/v1/models").status_code == 200


def test_keys_enforced_when_set(tmp_path):
    app = _app(tmp_path, api_keys=("k1", "k2"))
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer k2"}).status_code == 200
        # health stays open for monitoring
        assert c.get("/v1/health").status_code == 200
        # openai error shape
        body = c.get("/v1/queue").json()
        assert body["error"]["type"] == "authentication_error"


def test_lan_flag_sets_wildcard_host(monkeypatch):
    monkeypatch.delenv("MESH_HOST", raising=False)
    cfg = ServerConfig(lan=True, api_keys=("k",))
    assert cfg.host == "0.0.0.0"


# -- /v1/config ----------------------------------------------------------- #


def test_config_get_and_put(tmp_path):
    cfg_dir = tmp_path / "models"
    app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.get("/v1/config")
        assert r.status_code == 200
        cfg = r.json()["config"]
        assert cfg["llm_max_concurrent"]["value"] == 4
        assert cfg["llm_max_concurrent"]["source"] == "default"
        assert cfg["llm_parallel"]["applies"] == "next_model_load"

        r = c.put("/v1/config", json={"llm_max_concurrent": 2, "unload_idle_s": 300})
        assert r.status_code == 200, r.text
        assert r.json()["config"]["llm_max_concurrent"]["source"] == "overlay"
        # overlay persisted
        overlay = json.loads((cfg_dir / "server_config.json").read_text())
        assert overlay == {"llm_max_concurrent": 2, "unload_idle_s": 300}
        # null resets to default
        r = c.put("/v1/config", json={"llm_max_concurrent": None})
        assert r.json()["config"]["llm_max_concurrent"]["value"] == 4
        # unknown + wrong type rejected
        assert c.put("/v1/config", json={"bogus": 1}).status_code == 422
        assert c.put("/v1/config", json={"llm_parallel": "two"}).status_code == 422


def test_config_overlay_survives_new_instance(tmp_path):
    mdir = tmp_path / "models"
    c1 = ServerConfig(model_dir=mdir)
    c1.set_overlay({"llm_max_concurrent": 8})
    c2 = ServerConfig(model_dir=mdir)
    assert c2.effective("llm_max_concurrent") == 8


# -- /v1/models dual-shape ------------------------------------------------- #


def test_models_dual_shape(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body  # internal management view
        assert "data" in body  # openai listing
        assert all(e["object"] == "model" for e in body["data"])


# -- /v1/images/* ---------------------------------------------------------- #


def _wait_done(c, url: str, tries: int = 200) -> dict:
    import time

    for _ in range(tries):
        r = c.get(url)
        st = r.json()["status"]
        if st in ("succeeded", "failed", "cancelled"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_openai_images_generations(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.post(
            "/v1/images/generations",
            json={"model": "fake-image", "prompt": "a red gear", "n": 2, "size": "256x256"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["data"]) == 2
        assert all("b64_json" in e for e in body["data"])
        assert "X-Mesh-Job-Id" in r.headers


def test_openai_images_url_format(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.post(
            "/v1/images/generations",
            json={"model": "fake-image", "prompt": "p", "response_format": "url"},
        )
        assert r.status_code == 200, r.text
        url = r.json()["data"][0]["url"]
        assert "/v1/image-jobs/" in url and "/image?i=0" in url


def test_openai_images_edits(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.post(
            "/v1/images/edits",
            data={"model": "fake-image", "prompt": "make it blue"},
            files={"image": ("init.png", _png(), "image/png")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["b64_json"]


def test_openai_images_unknown_model(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.post(
            "/v1/images/generations",
            json={"model": "no-such-model", "prompt": "p"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "invalid_request_error"


def test_openai_images_missing_prompt(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        r = c.post("/v1/images/generations", json={"model": "fake-image", "prompt": "  "})
        assert r.status_code == 422


# -- llm argv knobs --------------------------------------------------------- #


def test_gguf_parallel_flag(tmp_path):
    from meshserver.adapters.llm_gguf import make_gguf_adapter
    from unittest.mock import patch

    cfg = ServerConfig(model_dir=tmp_path / "models", llm_parallel=4)
    cls = make_gguf_adapter("org/m", ["m.gguf"], id="t-gguf")
    inst = cls.create(cfg)
    with (
        patch.object(type(inst), "_resolve_server", return_value=(["llama-server"], "bin")),
        patch.object(inst, "_model_path", return_value=tmp_path / "m.gguf"),
    ):
        argv = inst.server_argv(9999)
    assert "--parallel" in argv
    assert argv[argv.index("--parallel") + 1] == "4"

    cfg2 = ServerConfig(model_dir=tmp_path / "models")
    inst2 = cls.create(cfg2)
    with (
        patch.object(type(inst2), "_resolve_server", return_value=(["llama-server"], "bin")),
        patch.object(inst2, "_model_path", return_value=tmp_path / "m.gguf"),
    ):
        argv2 = inst2.server_argv(9999)
    assert "--parallel" not in argv2  # default 1 → flag omitted


def _mlx_inst(cfg, *, vision: bool = False, repo: str = "org/m-4bit"):
    from meshserver.adapters.llm_mlx import make_mlx_adapter

    cls = make_mlx_adapter(repo, id="t-mlx" if not vision else "t-vlm", vision=vision)
    return cls.create(cfg)


def test_mlx_argv_tuning_flags(tmp_path):
    cfg = ServerConfig(
        model_dir=tmp_path / "models",
        llm_kv_bits=8,
        llm_prefill_step=8192,
        llm_prompt_cache_mb=4096,
    )
    argv = _mlx_inst(cfg).server_argv(9999)
    assert argv[argv.index("--kv-bits") + 1] == "8"
    assert argv[argv.index("--prefill-step-size") + 1] == "8192"
    assert argv[argv.index("--prompt-cache-bytes") + 1] == str(4096 * 1024 * 1024)
    # lm-only knob must not leak vlm flags
    assert "--max-kv-size" not in argv and "--max-num-seqs" not in argv

    argv_off = _mlx_inst(ServerConfig(model_dir=tmp_path / "m2")).server_argv(9999)
    for flag in ("--kv-bits", "--prefill-step-size", "--prompt-cache-bytes"):
        assert flag not in argv_off


def test_mlx_vlm_argv_ctx_seq_draft(tmp_path):
    cfg = ServerConfig(
        model_dir=tmp_path / "models",
        llm_ctx=131072,
        llm_max_num_seqs=2,
        llm_draft="auto",
    )
    argv = _mlx_inst(cfg, vision=True).server_argv(9999)
    assert argv[argv.index("--max-kv-size") + 1] == "131072"
    assert argv[argv.index("--max-num-seqs") + 1] == "2"
    assert argv[argv.index("--draft-model") + 1] == "org/m-4bit"
    assert "--draft-kind" not in argv  # auto → server detects
    # vlm instance must not emit the lm-only cache flags
    assert "--prompt-cache-bytes" not in argv


def test_mlx_draft_spec_parsing():
    from meshserver.adapters.llm_mlx import _draft_spec

    assert _draft_spec("", "r/x") == (None, None)
    assert _draft_spec("auto", "r/x") == ("r/x", None)
    assert _draft_spec("other/repo", "r/x") == ("other/repo", None)
    assert _draft_spec("other/repo:mtp", "r/x") == ("other/repo", "mtp")
    assert _draft_spec("mtp", "r/x") == ("r/x", "mtp")
    assert _draft_spec(":eagle3", "r/x") == ("r/x", "eagle3")


def test_config_new_knob_roundtrip(tmp_path):
    """Overlay-driven argv: PUT-able keys reach server_argv via effective()."""
    cfg = ServerConfig(model_dir=tmp_path / "models")
    cfg.set_overlay({"llm_kv_bits": 4, "llm_ctx": 262144, "llm_draft": "repo/d:eagle3"})
    argv = _mlx_inst(cfg, vision=True).server_argv(9999)
    assert argv[argv.index("--kv-bits") + 1] == "4"
    assert argv[argv.index("--max-kv-size") + 1] == "262144"
    assert argv[argv.index("--draft-model") + 1] == "repo/d"
    assert argv[argv.index("--draft-kind") + 1] == "eagle3"


def test_config_draft_validation(tmp_path):
    """apply_patch bounds: bad draft specs and out-of-range ints are
    rejected with 422; valid forms round-trip through the overlay."""
    import pytest
    from fastapi import HTTPException

    from meshserver.runtime_config import apply_patch, config_view

    cfg = ServerConfig(model_dir=tmp_path / "models")
    for ok in ("auto", "mtp", "hf/repo", "hf/repo:eagle3"):
        view = apply_patch(cfg, {"llm_draft": ok})
        assert view["config"]["llm_draft"]["value"] == ok
    # 'repo:kind' with an unknown kind tail or a non-str value must die;
    # 'dflash:mtp' parses as repo 'dflash' + kind 'mtp' — structurally valid
    for bad in ("hf/repo:wrong", "hf/repo:nope", 123):
        with pytest.raises(HTTPException):
            apply_patch(cfg, {"llm_draft": bad})
    for bad_kv in (-1, 12):
        with pytest.raises(HTTPException):
            apply_patch(cfg, {"llm_kv_bits": bad_kv})
    with pytest.raises(HTTPException):
        apply_patch(cfg, {"llm_ctx": 1024})  # below the 4096 floor
    apply_patch(cfg, {"llm_ctx": 262144})
    assert config_view(cfg)["config"]["llm_ctx"]["value"] == 262144
    # null resets to env/default
    apply_patch(cfg, {"llm_ctx": None})
    assert config_view(cfg)["config"]["llm_ctx"]["value"] == cfg.llm_ctx
