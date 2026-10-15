"""ROCm device tier: detect_device, adapter ladders, CUDA-only guards.

No AMD hardware required — torch's ROCm interface is faked via
`torch.cuda.is_available() -> True` plus `torch.version.hip` set (exactly
what a real HIP-enabled torch build reports).
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from meshserver.adapters.base import MeshAdapter
from meshserver.adapters.image_base import ImageAdapter
from meshserver.config import ServerConfig
from meshserver.registry import Registry, detect_device
from meshserver.torch_util import TorchDevice, is_rocm, torch_backend


@pytest.fixture
def rocm_torch(monkeypatch):
    """Make the installed (CPU) torch look like a ROCm/HIP build."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4.2", raising=False)
    return torch


@pytest.fixture
def cuda_torch(monkeypatch):
    """Real NVIDIA build: cuda available, no hip."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    return torch


def test_is_rocm_detection(rocm_torch):
    assert is_rocm() is True


def test_is_rocm_false_on_nvidia(cuda_torch):
    assert is_rocm() is False


def test_torch_backend_rocm(rocm_torch):
    assert torch_backend() == "rocm"


def test_torch_backend_cuda(cuda_torch):
    assert torch_backend() == "cuda"


def test_detect_device_rocm(rocm_torch, tmp_path):
    cfg = ServerConfig(device="auto", job_dir=tmp_path)
    assert detect_device(cfg) == "rocm"


def test_detect_device_cuda_not_rocm(cuda_torch, tmp_path):
    cfg = ServerConfig(device="auto", job_dir=tmp_path)
    assert detect_device(cfg) == "cuda"


def test_detect_device_explicit_override(tmp_path):
    """MESH_DEVICE=rocm must survive even without torch."""
    cfg = ServerConfig(device="rocm", job_dir=tmp_path)
    assert detect_device(cfg) == "rocm"


def test_torch_device_descriptor(rocm_torch):
    class _A(MeshAdapter):
        device = TorchDevice("cuda")

        def generate(self, images, opts, progress=None, cancelled=None):
            raise NotImplementedError

    assert _A.device == "rocm"


def test_torch_device_descriptor_nvidia(cuda_torch):
    class _A(MeshAdapter):
        device = TorchDevice("cuda")

        def generate(self, images, opts, progress=None, cancelled=None):
            raise NotImplementedError

    assert _A.device == "cuda"


# ---------- adapter ladders ----------


def _stub_mesh(adapter_id, **kw):
    return type(
        f"Stub{adapter_id}",
        (MeshAdapter,),
        {
            "id": adapter_id,
            "name": adapter_id,
            "kind": "mesh",
            "supports_multiview": kw.pop("multiview", False),
            "max_images": kw.pop("max_images", 1),
            "available": classmethod(lambda cls, cfg: True),
            "generate": lambda self, *a, **k: None,
        },
    )


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


def _registry(device, tmp_path):
    cfg = ServerConfig(device=device, job_dir=tmp_path)
    reg = Registry(cfg)
    reg._classes = [
        _stub_mesh("trellis2"),
        _stub_mesh("triposg"),
        _stub_mesh("triposg-scribble"),
        _stub_mesh("hunyuan3d-2mini"),
        _stub_mesh("hunyuan3d-2mv", multiview=True, max_images=4),
        _stub_image("flux-schnell", edit=True),
        _stub_image("sdxl", edit=True),
        _stub_image("sd15", edit=True),
        _stub_image("qwen-image-edit", edit=True, requires_image=True),
    ]
    return reg


def test_rocm_mesh_ladder_prefers_triposg(rocm_torch, tmp_path):
    reg = _registry("rocm", tmp_path)
    cls = reg.select("auto", n_images=1)
    assert cls is not None and cls.id == "triposg"


def test_rocm_mesh_ladder_multiview(rocm_torch, tmp_path):
    """Multi-image requests should prefer the hunyuan mv adapter on rocm."""
    reg = _registry("rocm", tmp_path)
    cls = reg.select("auto", n_images=3)
    assert cls is not None and cls.id == "hunyuan3d-2mv"


def test_rocm_never_picks_trellis2(rocm_torch, tmp_path):
    reg = _registry("rocm", tmp_path)
    for n in (1, 2, 4):
        cls = reg.select("auto", n_images=n)
        assert cls is not None and cls.id != "trellis2"


def test_rocm_image_ladder(rocm_torch, tmp_path):
    reg = _registry("rocm", tmp_path)
    cls = reg.select_image("auto", need_edit=False)
    assert cls is not None and cls.id == "flux-schnell"


def test_rocm_image_edit_ladder(rocm_torch, tmp_path):
    """Edit jobs on ROCm go to flux-schnell img2img — the mlx-only edit
    adapters (qwen-image-edit, flux-kontext-mlx) are not in the ladder."""
    reg = _registry("rocm", tmp_path)
    cls = reg.select_image("auto", need_edit=True)
    assert cls is not None and cls.id == "flux-schnell"


def test_trellis2_unavailable_on_rocm(rocm_torch):
    """_cuda_ok must reject HIP builds — TRELLIS.2 extensions are CUDA-only."""
    from meshserver.adapters import trellis2_cuda

    assert trellis2_cuda._cuda_ok() is False


def test_trellis2_ok_on_real_cuda(cuda_torch):
    from meshserver.adapters import trellis2_cuda

    assert trellis2_cuda._cuda_ok() is True


def _triposg_capture(monkeypatch):
    """Wire a fake TripoSG pipe into the adapter and return (adapter, kwargs)."""
    from meshserver.adapters import triposg
    from PIL import Image

    captured: dict = {}

    class FakePipe:
        def __call__(self, **kw):
            captured.update(kw)
            return {"meshes": [([0, 0, 0], [0])]}

    class FakeTorch:
        # this host's torch is CPU-only — a real cuda device can't be built,
        # so only the generator/seed surface is faked
        class Generator:
            def __init__(self, device=None):
                self.device = device

            def manual_seed(self, seed):
                return self

        @staticmethod
        def manual_seed(seed):
            pass

    monkeypatch.setattr(triposg, "_device_ok", lambda: "cuda")

    inst = triposg.TripoSGAdapter(ServerConfig())
    inst._pipe = FakePipe()
    inst._torch = FakeTorch
    inst._device = "cuda"  # ROCm torch reports the cuda device namespace
    inst._loaded = True
    with contextlib.suppress(Exception):
        # mesh conversion failure is fine — we only need the kwargs
        inst.generate([Image.new("RGB", (8, 8))], triposg.GenOpts(seed=1))
    return captured


def test_triposg_rocm_disables_flash_decoder(rocm_torch, monkeypatch):
    """On ROCm the CUDA-only diso flash decoder must stay off."""
    captured = _triposg_capture(monkeypatch)
    assert captured.get("use_flash_decoder") is False
    assert "generator" in captured  # cuda generator still used on ROCm


def test_triposg_nvidia_keeps_flash_decoder(cuda_torch, monkeypatch):
    """Real CUDA builds keep the fast DiffDMC path."""
    captured = _triposg_capture(monkeypatch)
    assert "use_flash_decoder" not in captured
