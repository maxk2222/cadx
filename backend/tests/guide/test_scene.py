"""merged_glb frame bake — the glTF artifact must stand upright in Y-up
consumers. Regression: the CAD solver world is Z-up mm and trimesh writes
vertex data raw (no axis-convention node), so the merged assembly.glb lay
on its side in the version page's three.js viewer while the Z-up render
path looked correct."""

import numpy as np
import pytest
import trimesh

from cadx.guide.scene import merged_glb


def _mesh(tmp_path, name: str) -> str:
    p = tmp_path / name
    trimesh.creation.box(extents=(10, 20, 30)).export(str(p))
    return str(p)


def test_zup_world_bakes_into_gltf_frame(tmp_path) -> None:
    """A part placed 100 up in a Z-up world must land on glTF +Y (with the
    world's +Y mapped to -Z) — the file itself is Y-up, upright in any
    glTF consumer without viewer-side compensation."""
    src = _mesh(tmp_path, "a.glb")
    t = np.eye(4)
    t[:3, 3] = [5, 10, 100]
    out = merged_glb([("a", src, t.tolist())], tmp_path / "asm.glb", world_frame="z_up")
    g = trimesh.load(str(out)).to_geometry()
    center = np.asarray(g.bounds).mean(axis=0)
    assert center == pytest.approx([5, 100, -10], abs=1e-4)


def test_yup_world_passes_through(tmp_path) -> None:
    """Blender-mode placements already live in the glTF frame — baking a
    second conversion would tip them over."""
    src = _mesh(tmp_path, "a.glb")
    t = np.eye(4)
    t[:3, 3] = [5, 100, 10]
    out = merged_glb([("a", src, t.tolist())], tmp_path / "asm.glb", world_frame="y_up")
    g = trimesh.load(str(out)).to_geometry()
    center = np.asarray(g.bounds).mean(axis=0)
    assert center == pytest.approx([5, 100, 10], abs=1e-4)
