"""Unit tests for preprocess/postprocess helpers (no models, no downloads)."""

import trimesh
from PIL import Image

from meshserver.postprocess import clean_mesh, export_glb
from meshserver.preprocess import load_image, prep_images, to_square


def test_to_square_pads_and_resizes(tmp_path):
    img = Image.new("RGB", (40, 20), (200, 30, 30))
    p = tmp_path / "a.png"
    img.save(p)
    out = to_square(load_image(p), size=128)
    assert out.size == (128, 128)
    # wider than tall → letterboxed vertically (content centered)
    assert out.getpixel((64, 64))[:3] == (200, 30, 30)
    assert out.getpixel((64, 0)) != (200, 30, 30)


def test_prep_images_no_download_in_auto(tmp_path):
    """bg_mode='auto' must not trigger a rembg model download."""
    img = Image.new("RGB", (64, 64), (10, 200, 10))
    p = tmp_path / "b.png"
    img.save(p)
    out = prep_images([p], bg_mode="auto")
    assert len(out) == 1
    assert out[0].size == (1024, 1024)


def test_clean_mesh_removes_floater():
    big = trimesh.creation.box(extents=[10, 10, 10])
    tiny = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    tiny.apply_translation([100, 0, 0])
    merged = trimesh.util.concatenate([big, tiny])
    out = clean_mesh(merged)
    assert out.extents[0] < 20


def test_export_glb_roundtrip():
    m = trimesh.creation.box()
    data = export_glb(m)
    assert data[:4] == b"glTF"
    back = trimesh.load(__import__("io").BytesIO(data), file_type="glb", force="mesh")
    assert len(back.faces) == 12
