"""Qwen-Image (txt2img) MLX adapter: availability probe, auto-ladder and
generate() argument construction — all with fake pipes, no GPU/model load."""

from __future__ import annotations

import platform
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image

from meshserver.adapters.image_base import ImageAdapter, ImageGenOpts
from meshserver.config import ServerConfig
from meshserver.registry import Registry

QWEN_MOD = "mflux.models.qwen.variants.txt2img.qwen_image"


class _Pipe:
    """Records generate_image kwargs and returns a dummy PIL image."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def generate_image(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(image=Image.new("RGB", (16, 16)))


# ---------- available() ----------------------------------------------------


def test_available_false_off_arm64(monkeypatch, tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    cfg = ServerConfig(job_dir=tmp_path)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert QwenImageMlxAdapter.available(cfg) is False
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert QwenImageMlxAdapter.available(cfg) is False


def test_available_arm64_import_probe(monkeypatch, tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    cfg = ServerConfig(job_dir=tmp_path)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    # mflux present -> probe succeeds (sys.modules short-circuit, parents
    # are never touched)
    fake = ModuleType(QWEN_MOD)
    fake.QwenImage = object
    monkeypatch.setitem(sys.modules, QWEN_MOD, fake)
    assert QwenImageMlxAdapter.available(cfg) is True
    # None in sys.modules -> ModuleNotFoundError -> unavailable
    monkeypatch.setitem(sys.modules, QWEN_MOD, None)
    assert QwenImageMlxAdapter.available(cfg) is False


# ---------- class flags ----------------------------------------------------


def test_flags(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    assert a.requires_image is False
    assert a.supports_edit is True
    assert a.default_steps == 20


# ---------- registry ladder ------------------------------------------------


def _stub_image(adapter_id, **kw):
    return type(
        f"Stub{adapter_id}",
        (ImageAdapter,),
        {
            "id": adapter_id,
            "name": adapter_id,
            "kind": "image",
            "supports_edit": kw.pop("edit", False),
            "requires_image": kw.pop("requires_image", False),
            "available": classmethod(lambda cls, cfg: True),
            "generate": lambda self, *a, **k: None,
        },
    )


def _registry(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    # fake=False explicitly: test_image_jobs.py sets MESH_FAKE=1 at module
    # level, which would flip select_image() into the fake-only pref ladder
    cfg = ServerConfig(device="mlx", fake=False, job_dir=tmp_path)
    reg = Registry(cfg)
    # every class in the ladder reports available — only the preference
    # order decides
    reg._probe = lambda cls: True
    reg._classes = [
        QwenImageEditAdapter,
        QwenImageMlxAdapter,
        _stub_image("flux-kontext-mlx", edit=True),
        _stub_image("flux-schnell-mlx", edit=True),
        _stub_image("sdxl", edit=True),
        _stub_image("sd15", edit=True),
    ]
    # same isolation as test_registry_budget: host llm adapters must not
    # leak into the registry's class lists
    reg._llm_classes = []
    return reg


def test_mlx_ladder_prefers_qwen_image(tmp_path):
    """qwen-image wins the MLX txt2img ladder once the faster z-image-turbo
    is badlisted/absent — it stays the quality fallback."""
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    reg = _registry(tmp_path)
    reg._failed["z-image-turbo"] = "probe failed"
    assert reg.select_image("auto") is QwenImageMlxAdapter


def test_mlx_edit_ladder_prefers_qwen_edit(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    reg = _registry(tmp_path)
    cls = reg.select_image("auto", need_edit=True)
    assert cls is QwenImageEditAdapter
    assert cls is not QwenImageMlxAdapter


# ---------- generate() -----------------------------------------------------


def test_generate_txt2img_args(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True

    res = a.generate(ImageGenOpts(prompt="a mug", negative="blurry", seed=7, steps=12))
    assert res.model == "qwen-image"
    assert res.steps == 12
    kw = a._pipe.kwargs
    assert kw["seed"] == 7
    assert kw["prompt"] == "a mug"
    assert kw["num_inference_steps"] == 12
    assert kw["negative_prompt"] == "blurry"
    assert kw["guidance"] == 4.0
    assert "image_path" not in kw
    assert "image_strength" not in kw


def test_generate_seed_offset_and_default_steps(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True

    res = a.generate(ImageGenOpts(prompt="x", seed=5, count=2))
    assert len(res.images) == 2
    assert res.steps == 20  # default_steps for the non-distilled model
    assert a._pipe.kwargs["seed"] == (5 + 1000) & 0xFFFFFFFF  # i=1 offset
    assert a._pipe.kwargs["num_inference_steps"] == 20
    assert a._pipe.kwargs["negative_prompt"] is None


def test_generate_img2img_args(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    a.generate(ImageGenOpts(prompt="x", image_path=init, image_strength=0.45))
    kw = a._pipe.kwargs
    assert kw["image_path"] == str(init)
    assert kw["image_strength"] == 0.45


def test_generate_strength_clamped_and_default(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    a.generate(ImageGenOpts(prompt="x", image_path=init, image_strength=5.0))
    assert a._pipe.kwargs["image_strength"] == 1.0
    a.generate(ImageGenOpts(prompt="x", image_path=init))
    assert a._pipe.kwargs["image_strength"] == 0.6  # unset strength falls back


def test_generate_side_clamped(tmp_path):
    from meshserver.adapters.qwen_image_mlx import QwenImageMlxAdapter

    a = QwenImageMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True

    a.generate(ImageGenOpts(prompt="x", width=2048, height=512))
    assert a._pipe.kwargs["width"] == 1328  # max_side
    assert a._pipe.kwargs["height"] == 512
