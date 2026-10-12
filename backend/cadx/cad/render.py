"""Off-screen rendering of the part into labelled view PNGs (PyVista).

Degrades gracefully: render_views() returns {} when VTK cannot initialise.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

VIEWS = ["front", "back", "left", "right", "top", "bottom", "iso"]
_RENDER_OK: bool | None = None


def render_available() -> bool:
    """Probe offscreen rendering once (renders a small sphere)."""
    global _RENDER_OK
    if _RENDER_OK is not None:
        return _RENDER_OK
    try:
        import pyvista as pv

        pl = pv.Plotter(off_screen=True, window_size=(64, 64))
        pl.add_mesh(pv.Sphere())
        pl.render()
        pl.close()
        _RENDER_OK = True
    except Exception:
        _RENDER_OK = False
    return _RENDER_OK


def _camera_for(view: str) -> tuple[str, tuple[float, float, float] | None]:
    """Return (camera_position, view_up) for pyvista."""
    up = (0, 0, 1)
    if view == "front":
        return "xy", up  # looking down -Y? pyvista 'xy' looks along -Z; use explicit below
    return view, up


def render_views(
    stl_path: Path, out_dir: Path, bbox_size: list[float], size: int = 1024
) -> dict[str, Path]:
    """Render the 7 standard views. Returns {view: path}."""
    if not render_available():
        return {}
    import pyvista as pv

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(str(stl_path))
    edges = mesh.extract_feature_edges(30)
    outline = mesh.outline()
    x, y, z = bbox_size
    title = f"L {x:.1f} x W {y:.1f} x H {z:.1f} mm"

    results: dict[str, Path] = {}
    for view in VIEWS:
        pl = pv.Plotter(off_screen=True, window_size=(size, size))
        pl.set_background("white")
        pl.add_mesh(mesh, color="#9aa3ad", smooth_shading=True)
        if edges.n_points:
            pl.add_mesh(edges, color="#222222", line_width=1.5)
        pl.add_mesh(outline, color="#888888", line_width=2)

        if view == "iso":
            pl.camera_position = "iso"
        else:
            positions = {
                "front": ((0, 1, 0), (0, 0, 1)),
                "back": ((0, -1, 0), (0, 0, 1)),
                "left": ((-1, 0, 0), (0, 0, 1)),
                "right": ((1, 0, 0), (0, 0, 1)),
                "top": ((0, 0, 1), (0, 1, 0)),
                "bottom": ((0, 0, -1), (0, -1, 0)),
            }
            pos, up = positions[view]
            c = mesh.center
            import math

            diag = math.sqrt(x * x + y * y + z * z)
            pl.camera.position = (c[0] + pos[0] * diag, c[1] + pos[1] * diag, c[2] + pos[2] * diag)
            pl.camera.focal_point = c
            pl.camera.up = up
            pl.camera.parallel_projection = True

        pl.reset_camera()
        pl.camera.zoom(1.15)
        pl.add_text(f"{view}  -  {title}", position="upper_edge", font_size=14, color="black")
        pl.add_axes(color="black")
        out = out_dir / f"{view}.png"
        pl.screenshot(str(out))
        pl.close()
        results[view] = out
    return results


def render_thumbnail(stl_path: Path, out_path: Path, size: int = 256) -> Path | None:
    if not render_available():
        return None
    import pyvista as pv

    mesh = pv.read(str(stl_path))
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    pl.add_mesh(mesh, color="#9aa3ad", smooth_shading=True)
    edges = mesh.extract_feature_edges(30)
    if edges.n_points:
        pl.add_mesh(edges, color="#333333", line_width=1)
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_path))
    pl.close()
    return out_path
