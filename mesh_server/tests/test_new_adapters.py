"""WS-D adapters: registration, availability probing, GenOpts compat.

Runs without any adapter clone installed — available() must degrade to
False (with available_error) instead of raising.
"""

from __future__ import annotations

from meshserver.adapters.base import GenOpts
from meshserver.config import ServerConfig
from meshserver.registry import Registry


def _cfg() -> ServerConfig:
    return ServerConfig(fake=True)


def test_new_adapters_registered():
    reg = Registry(_cfg())
    ids = {cls.id for cls in reg._classes}
    assert "hunyuan3d-omni" in ids
    assert "step1x-3d" in ids


def test_adapter_metadata():
    from meshserver.adapters.hunyuan_omni import Hunyuan3DOmniAdapter
    from meshserver.adapters.step1x_3d import Step1X3DAdapter

    # single-image + control: no multiview slots
    assert Hunyuan3DOmniAdapter.max_images == 1
    assert Hunyuan3DOmniAdapter.supports_multiview is False
    assert "Hunyuan" in Hunyuan3DOmniAdapter.license_note
    assert Step1X3DAdapter.max_images == 1
    assert Step1X3DAdapter.license_note == "Apache-2.0"
    assert Step1X3DAdapter.device == "cuda"


def test_available_does_not_crash():
    """available() must return bool regardless of clone/deps presence."""
    from meshserver.adapters.hunyuan_omni import Hunyuan3DOmniAdapter
    from meshserver.adapters.step1x_3d import Step1X3DAdapter

    cfg = _cfg()
    for cls in (Hunyuan3DOmniAdapter, Step1X3DAdapter):
        result = cls.available(cfg)
        assert isinstance(result, bool)
        # when unavailable the failure may be explained via available_error
        # (registry._probe surfaces it as ModelInfo.last_error)
        err = getattr(cls, "available_error", None)
        assert err is None or isinstance(err, str)


def test_step1x_unavailable_without_cuda():
    """No NVIDIA CUDA on this dev box -> step1x-3d must stay unavailable."""
    import torch

    from meshserver.adapters.step1x_3d import Step1X3DAdapter

    if torch.cuda.is_available() and not getattr(torch.version, "hip", None):
        return  # real CUDA host: adapter may legitimately be available
    assert Step1X3DAdapter.available(_cfg()) is False


def test_genopts_backwards_compatible():
    """Old call sites without control fields keep working."""
    opts = GenOpts(seed=42, steps=10, guidance=5.0, resolution=256)
    assert opts.control_type is None
    assert opts.control_input is None
    # new fields usable when provided
    opts2 = GenOpts(control_type="bbox", control_input=[100.0, 50.0, 30.0])
    assert opts2.control_type == "bbox"


def test_bbox_dims_normalisation():
    from meshserver.adapters.hunyuan_omni import Hunyuan3DOmniAdapter

    # [x, y, z] mm -> relative proportions, largest dim = 1
    assert Hunyuan3DOmniAdapter._bbox_dims([100, 50, 25]) == [1.0, 0.5, 0.25]
    # 6-bounds form -> dims first
    assert Hunyuan3DOmniAdapter._bbox_dims([0, 0, 0, 80, 40, 40]) == [1.0, 0.5, 0.5]
    # nothing -> neutral cube
    assert Hunyuan3DOmniAdapter._bbox_dims(None) == [1.0, 1.0, 1.0]
    # degenerate -> neutral cube rather than division by zero
    assert Hunyuan3DOmniAdapter._bbox_dims([0, 0, 0]) == [1.0, 1.0, 1.0]


def test_registry_lists_new_models():
    """models() must not raise and must carry the new ids."""
    reg = Registry(_cfg())
    ids = {m.id for m in reg.models()}
    assert {"hunyuan3d-omni", "step1x-3d"} <= ids
