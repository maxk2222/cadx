"""Edit-adapter argument construction with fake pipes (no GPU/model load)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from meshserver.adapters.image_base import ImageGenOpts
from meshserver.config import ServerConfig
from PIL import Image


class _Pipe:
    """Records generate_image kwargs and returns a dummy PIL image."""

    def __init__(self):
        self.kwargs: dict = {}

    def generate_image(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(image=Image.new("RGB", (16, 16)))


def test_flux_mlx_passes_image_path_and_strength(tmp_path):
    from meshserver.adapters.flux_mlx import FluxSchnellMlxAdapter

    a = FluxSchnellMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    res = a.generate(
        ImageGenOpts(prompt="make it black", image_path=init, image_strength=0.45)
    )
    assert res.images and res.model == "flux-schnell-mlx"
    assert a._pipe.kwargs["image_path"] == str(init)
    assert a._pipe.kwargs["image_strength"] == 0.45
    assert res.steps == 8  # i2i default > txt2img's 4


def test_flux_mlx_strength_clamped(tmp_path):
    from meshserver.adapters.flux_mlx import FluxSchnellMlxAdapter

    a = FluxSchnellMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    a.generate(ImageGenOpts(prompt="x", image_path=init, image_strength=5.0))
    assert a._pipe.kwargs["image_strength"] == 1.0


def test_flux_mlx_txt2img_unchanged(tmp_path):
    from meshserver.adapters.flux_mlx import FluxSchnellMlxAdapter

    a = FluxSchnellMlxAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    a.generate(ImageGenOpts(prompt="a mug"))
    assert "image_path" not in a._pipe.kwargs
    assert "image_strength" not in a._pipe.kwargs


def test_qwen_edit_requires_init_image(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter

    a = QwenImageEditAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    with pytest.raises(RuntimeError, match="init image"):
        a.generate(ImageGenOpts(prompt="x"))


def test_qwen_edit_passes_image_paths(tmp_path):
    from meshserver.adapters.qwen_edit_mlx import QwenImageEditAdapter

    a = QwenImageEditAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    res = a.generate(ImageGenOpts(prompt="add fins", image_path=init, steps=10))
    assert res.images and res.model == "qwen-image-edit"
    kw = a._pipe.kwargs
    assert kw["image_paths"] == [str(init)]
    assert kw["num_inference_steps"] == 10


def test_qwen_edit_lightning_pins_steps_and_guidance(tmp_path):
    """With the Lightning LoRA baked in the schedule is pinned to 8 steps at
    cfg 1.0 regardless of what the caller asked for."""
    from meshserver.adapters.qwen_edit_mlx import LIGHTNING_STEPS, QwenImageEditAdapter

    a = QwenImageEditAdapter(ServerConfig(job_dir=tmp_path))
    a._pipe = _Pipe()
    a._loaded = True
    a._lightning = True
    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)

    res = a.generate(
        ImageGenOpts(prompt="add fins", image_path=init, steps=28, guidance=4.0)
    )
    kw = a._pipe.kwargs
    assert kw["num_inference_steps"] == LIGHTNING_STEPS == 8
    assert kw["guidance"] == 1.0
    assert res.steps == 8 and res.guidance == 1.0


def test_qwen_edit_lightning_lora_wiring(tmp_path):
    """load() resolves the lightning LoRA file and passes it to the ctor."""
    import meshserver.adapters.qwen_edit_mlx as qe

    captured = {}

    class _FakeEdit:
        def __init__(self, **kw):
            captured.update(kw)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(qe, "_resolve_lightning_lora", lambda: "/cache/lora.safetensors")
    monkey.setattr(qe, "_patch_mflux_tf4_qwen2", lambda: None)
    import types

    mod = types.ModuleType("mflux.models.qwen.variants.edit.qwen_image_edit")
    mod.QwenImageEdit = _FakeEdit
    monkey.setitem(sys.modules, "mflux.models.qwen.variants.edit.qwen_image_edit", mod)

    cfg = ServerConfig(job_dir=tmp_path, qwen_edit_lightning=True)
    a = qe.QwenImageEditAdapter(cfg)
    a.load()
    assert a._lightning is True
    assert captured["lora_paths"] == ["/cache/lora.safetensors"]
    assert captured["lora_scales"] == [1.0]
    monkey.undo()


def test_qwen_edit_lightning_download_failure_falls_back(tmp_path):
    """A failed LoRA download must not kill load() — plain model, 28 steps."""
    import meshserver.adapters.qwen_edit_mlx as qe

    captured = {}

    class _FakeEdit:
        def __init__(self, **kw):
            captured.update(kw)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(qe, "_resolve_lightning_lora", lambda: None)
    monkey.setattr(qe, "_patch_mflux_tf4_qwen2", lambda: None)
    import types

    mod = types.ModuleType("mflux.models.qwen.variants.edit.qwen_image_edit")
    mod.QwenImageEdit = _FakeEdit
    monkey.setitem(sys.modules, "mflux.models.qwen.variants.edit.qwen_image_edit", mod)

    cfg = ServerConfig(job_dir=tmp_path, qwen_edit_lightning=True)
    a = qe.QwenImageEditAdapter(cfg)
    a.load()
    assert a._lightning is False
    assert "lora_paths" not in captured
    monkey.undo()


def test_flux_diffusers_i2i_clamps_strength(tmp_path):
    """img2img path: _pipe_edit receives PIL image + clamped strength."""
    from meshserver.adapters.flux_diffusers import FluxSchnellAdapter

    calls: dict = {}

    class EditPipe:
        def __call__(self, **kw):
            calls.update(kw)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16))])

    class FakeTorch:
        class Generator:
            def __init__(self, device=None):
                pass

            def manual_seed(self, s):
                return self

    a = FluxSchnellAdapter(ServerConfig(job_dir=tmp_path))
    a._loaded = True
    a._torch = FakeTorch
    a._pipe = lambda **kw: SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    a._pipe_edit = EditPipe()

    init = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(init)
    res = a.generate(ImageGenOpts(prompt="x", image_path=init, image_strength=2.0))
    assert res.model == "flux-schnell"
    assert calls["strength"] == 0.99
    assert isinstance(calls["image"], Image.Image)


def test_init_image_cleanup_on_delete(tmp_path):
    """Deleting an edit job removes the staged init file in incoming/."""
    import json as _json

    from fastapi.testclient import TestClient
    from meshserver.app import create_app

    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", bg_removal="never")
    src = tmp_path / "init.png"
    Image.new("RGB", (16, 16)).save(src)

    with TestClient(create_app(cfg)) as c:
        r = c.post(
            "/v1/image-edit-jobs",
            files={
                "json": ("p.json", _json.dumps({"prompt": "x"}), "application/json"),
                "image": ("init.png", src.read_bytes(), "image/png"),
            },
        )
        job_id = r.json()["job_id"]
        incoming = cfg.job_dir / "incoming"
        staged = list(incoming.glob("edit_*"))
        assert len(staged) == 1
        r = c.delete(f"/v1/image-jobs/{job_id}")
        assert r.status_code == 200
        assert list(incoming.glob("edit_*")) == []


def test_edit_filename_is_sanitized(tmp_path):
    """A hostile multipart filename must not escape the incoming dir."""
    import json as _json

    from fastapi.testclient import TestClient
    from meshserver.app import create_app

    cfg = ServerConfig(fake=True, job_dir=tmp_path / "jobs", bg_removal="never")

    with TestClient(create_app(cfg)) as c:
        r = c.post(
            "/v1/image-edit-jobs",
            files={
                "json": ("p.json", _json.dumps({"prompt": "x"}), "application/json"),
                "image": ("../evil.png", b"\x89PNG\r\n\x1a\n", "image/png"),
            },
        )
        assert r.status_code == 200, r.text
        assert list(tmp_path.glob("evil*")) == []  # nothing outside job_dir
        staged = list((cfg.job_dir / "incoming").glob("edit_*"))
        assert len(staged) == 1 and ".." not in staged[0].name
