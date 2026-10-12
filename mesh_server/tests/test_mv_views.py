"""Role-to-view mapping for the multiview adapter (_images_arg)."""

from PIL import Image

from meshserver.adapters.base import GenOpts
from meshserver.adapters.hunyuan3d2_mps import Hunyuan3D2MvAdapter


def _imgs(n: int) -> list[Image.Image]:
    return [Image.new("RGB", (8, 8), (i * 40, i * 40, i * 40)) for i in range(n)]


def test_unlabeled_images_never_fill_wrong_view_slots():
    """3 front-face shots labeled other must not land in back/left/right."""
    imgs = _imgs(4)
    opts = GenOpts(image_roles=["front", "other", "other", "back"])
    mapping = Hunyuan3D2MvAdapter._images_arg(None, imgs, opts)
    assert mapping["front"] is imgs[0]
    assert mapping["back"] is imgs[3]
    assert "left" not in mapping and "right" not in mapping


def test_no_roles_falls_back_to_first_image_front():
    mapping = Hunyuan3D2MvAdapter._images_arg(None, _imgs(3), GenOpts())
    assert list(mapping) == ["front"]


def test_missing_front_filled_by_first_other_image():
    imgs = _imgs(3)
    opts = GenOpts(image_roles=["other", "back", "left"])
    mapping = Hunyuan3D2MvAdapter._images_arg(None, imgs, opts)
    assert mapping["front"] is imgs[0]
    assert mapping["back"] is imgs[1]
    assert mapping["left"] is imgs[2]
