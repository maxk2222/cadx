"""Z-Image-Turbo (mflux MLX) adapter — availability probes, flags, ladder
position and generate() call plumbing. Mirrors test_qwen_image.py."""

from __future__ import annotations

import sys
import types

from meshserver.adapters.z_image_mlx import ZImageTurboAdapter

import pytest


class _Cfg:
    device = "mlx"
    flux_mlx_quant = 4


def test_available_false_on_non_darwin(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert ZImageTurboAdapter.available(_Cfg()) is False


def test_available_false_on_intel_mac(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert ZImageTurboAdapter.available(_Cfg()) is False


def test_available_true_when_import_works(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    mod = types.ModuleType("mflux.models.z_image.variants.z_image")
    mod.ZImage = object
    sys.modules["mflux.models.z_image.variants.z_image"] = mod
    assert ZImageTurboAdapter.available(_Cfg()) is True


def test_flags():
    assert ZImageTurboAdapter.id == "z-image-turbo"
    assert ZImageTurboAdapter.requires_image is False
    assert ZImageTurboAdapter.supports_edit is True
    assert ZImageTurboAdapter.default_steps == 8
    assert ZImageTurboAdapter.backend == "mflux"


def _stub_image(mid: str, *, edit: bool = False):
    class _S:
        id = mid
        kind = "image"
        requires_image = False
        supports_edit = edit
        est_mem_mb = 100

    return _S


def _registry(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter
    from meshserver.config import ServerConfig
    from meshserver.registry import Registry

    # fake=False explicitly: test_image_jobs.py sets MESH_FAKE=1 at module
    # level, which would flip select_image() into the fake-only pref ladder
    cfg = ServerConfig(device="mlx", fake=False, job_dir=tmp_path)
    reg = Registry(cfg)
    reg._probe = lambda cls: True
    reg._classes = [
        ZImageTurboAdapter,
        QwenImageMlxAdapter,
        QwenImageEditAdapter,
        _stub_image("flux-kontext-mlx", edit=True),
        _stub_image("flux-schnell-mlx", edit=True),
        _stub_image("sdxl", edit=True),
        _stub_image("sd15", edit=True),
    ]
    reg._llm_classes = []
    return reg


def test_registry_ladder_prefers_z_image(tmp_path):
    reg = _registry(tmp_path)
    assert reg.select_image("auto") is ZImageTurboAdapter


def test_registry_edit_ladder_still_prefers_qwen_edit(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter

    reg = _registry(tmp_path)
    assert reg.select_image("auto", need_edit=True) is QwenImageEditAdapter


def test_registry_falls_back_to_qwen_image(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    reg = _registry(tmp_path)
    reg._failed["z-image-turbo"] = "probe failed"
    assert reg.select_image("auto") is QwenImageMlxAdapter


class _Out:
    def __init__(self, img):
        self.image = img


class _Pipe:
    def __init__(self):
        self.calls = []

    def generate_image(self, **kw):
        self.calls.append(kw)
        from PIL import Image

        return _Out(Image.new("RGB", (8, 8)))


@pytest.fixture()
def adapter():
    a = ZImageTurboAdapter(_Cfg())
    a._pipe = _Pipe()
    a._loaded = True
    return a


def _opts(**kw):
    from meshserver.adapters.image_base import ImageGenOpts

    kw.setdefault("count", 1)
    kw.setdefault("width", 1024)
    kw.setdefault("height", 1024)
    return ImageGenOpts(prompt="a steel cube", seed=7, **kw)


def test_generate_uses_default_8_steps_and_no_guidance(adapter):
    res = adapter.generate(_opts(steps=None, guidance=7.5, negative="blur"))
    call = adapter._pipe.calls[0]
    assert call["num_inference_steps"] == 8
    assert call["guidance"] is None  # distilled — no CFG even when requested
    assert call["negative_prompt"] == "blur"
    assert res.steps == 8 and res.guidance is None


def test_generate_seed_offset_per_image(adapter):
    adapter.generate(_opts(count=2))
    assert [c["seed"] for c in adapter._pipe.calls] == [7, 1007]


def test_generate_img2img_passes_path_and_strength(adapter):
    adapter.generate(_opts(image_path="/tmp/in.png", image_strength=2.5))
    call = adapter._pipe.calls[0]
    assert call["image_path"] == "/tmp/in.png"
    assert call["image_strength"] == 1.0  # clamped


def test_generate_clamps_resolution_to_max_side(adapter):
    adapter.generate(_opts(width=2000, height=2000))
    call = adapter._pipe.calls[0]
    assert call["width"] == 1328 and call["height"] == 1328
