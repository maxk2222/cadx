"""Off-screen rendering of the part into labelled view PNGs (PyVista).

Degrades gracefully: render_views() returns {} when VTK cannot initialise.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

VIEWS = ["front", "back", "left", "right", "top", "bottom", "iso"]

# extra translucent "xray" shots appended after the opaque views — the
# critique's contact sheet gains tiles that reveal interior geometry
# (holes, cavities) without degrading the crisp opaque silhouette views
GHOST_VIEWS = ("iso", "front")
GHOST_SUFFIX = "_ghost"

_RENDER_OK: bool | None = None


def _ghost_opacity() -> float:
    """CADX_RENDER_GHOST_OPACITY (default 0.55); 0 or >=1 disables ghost shots."""
    try:
        return float(os.environ.get("CADX_RENDER_GHOST_OPACITY", "0.55"))
    except ValueError:
        return 0.55


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


def _thread_preview_lines(pv: Any, features: list[dict[str, Any]]) -> list[Any]:
    import numpy as np

    axes = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    def _num(v: Any) -> float | None:
        # generated CADX_FEATURES records are model output — values arrive
        # stringified or list-wrapped ([1.75], [[0,0,0]]); coerce what we can
        # and let the caller skip the feature rather than kill the render pass
        if isinstance(v, list | tuple):
            v = v[0] if len(v) == 1 else None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    actors = []
    for feature in features:
        try:
            axis = np.asarray(axes.get(str(feature.get("axis") or "+z"), (0.0, 0.0, 1.0)))
            pos_raw = feature.get("position_mm") or (0.0, 0.0, 0.0)
            if isinstance(pos_raw, list | tuple) and pos_raw and isinstance(
                pos_raw[0], list | tuple
            ):
                pos_raw = pos_raw[0]
            position = np.asarray(pos_raw, dtype=float)
            if position.shape != (3,):
                continue
            pitch = _num(feature.get("pitch"))
            length = _num(feature.get("length"))
            diameter = _num(feature.get("major_diameter"))
            if not pitch or not length or not diameter or pitch <= 0 or length <= 0 or diameter <= 0:
                continue
            start = _num(feature.get("start")) or 0.0
        except (TypeError, ValueError):
            continue
        reference = np.asarray((1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0))
        u = np.cross(axis, reference)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        turns = length / pitch
        steps = min(12000, max(32, math.ceil(turns * 12)))
        radius = diameter / 2 - 0.08 if feature.get("external") is False else diameter / 2 + 0.04
        hand = -1.0 if feature.get("hand") == "left" else 1.0
        points = []
        for i in range(steps + 1):
            fraction = i / steps
            distance = start + length * fraction
            angle = hand * math.tau * distance / pitch
            radial = radius * (u * math.cos(angle) + v * math.sin(angle))
            points.append(position + axis * distance + radial)
        line = pv.PolyData(np.asarray(points, dtype=float))
        line.lines = np.asarray([len(points), *range(len(points))], dtype=np.int64)
        actors.append(line)
    return actors


def render_views(
    stl_path: Path,
    out_dir: Path,
    bbox_size: list[float],
    size: int = 1024,
    *,
    thread_features: list[dict[str, Any]] | None = None,
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

    # one VTK render window for all seven views — creating/closing a Plotter
    # per view paid a fresh GL context + window init each time (most of the
    # per-view cost); only the camera + label change between shots
    positions = {
        "front": ((0, 1, 0), (0, 0, 1)),
        "back": ((0, -1, 0), (0, 0, 1)),
        "left": ((-1, 0, 0), (0, 0, 1)),
        "right": ((1, 0, 0), (0, 0, 1)),
        "top": ((0, 0, 1), (0, 1, 0)),
        "bottom": ((0, 0, -1), (0, -1, 0)),
    }
    diag = math.sqrt(x * x + y * y + z * z)
    c = mesh.center
    results: dict[str, Path] = {}
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    try:
        pl.set_background("white")
        actor = pl.add_mesh(mesh, color="#9aa3ad", smooth_shading=True, split_sharp_edges=True)
        if edges.n_points:
            pl.add_mesh(edges, color="#222222", line_width=1.5)
        pl.add_mesh(outline, color="#888888", line_width=2)
        for thread_line in _thread_preview_lines(pv, thread_features or []):
            pl.add_mesh(thread_line, color="#30343b", line_width=1.8)
        pl.add_axes(color="black")
        ghost = _ghost_opacity()
        shots: list[tuple[str, float, str]] = [(v, 1.0, "") for v in VIEWS]
        if 0.0 < ghost < 1.0:
            # edges + outline stay opaque — translucent faces over crisp
            # feature edges reads as a proper x-ray view
            shots += [(v, ghost, GHOST_SUFFIX) for v in GHOST_VIEWS]
        for view, opacity, suffix in shots:
            actor.prop.opacity = opacity
            if view == "iso":
                pl.camera_position = "iso"
            else:
                pos, up = positions[view]
                pl.camera.position = (
                    c[0] + pos[0] * diag,
                    c[1] + pos[1] * diag,
                    c[2] + pos[2] * diag,
                )
                pl.camera.focal_point = c
                pl.camera.up = up
                pl.camera.parallel_projection = True
            pl.reset_camera()
            pl.camera.zoom(1.15)
            tag = " (xray)" if suffix else ""
            label = pl.add_text(
                f"{view}{tag}  -  {title}"
                + ("  -  thread preview" if thread_features else ""),
                position="upper_edge",
                font_size=14,
                color="black",
            )
            out = out_dir / f"{view}{suffix}.png"
            pl.screenshot(str(out))
            pl.remove_actor(label)
            results[f"{view}{suffix}"] = out
        detail = _thread_detail_camera(thread_features or [])
        if detail is not None:
            focal, position, up, half_height = detail
            actor.prop.opacity = 1.0
            pl.camera.position = position
            pl.camera.focal_point = focal
            pl.camera.up = up
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = half_height
            label = pl.add_text(
                f"thread detail  -  {title}", position="upper_edge", font_size=14, color="black"
            )
            out = out_dir / "thread_detail.png"
            pl.screenshot(str(out))
            pl.remove_actor(label)
            results["thread_detail"] = out
    finally:
        pl.close()
    return results


def _thread_detail_camera(
    features: list[dict[str, Any]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float] | None:
    """Side-on close-up of the first thread feature — a 20 mm thread zone on a
    150 mm part is unreadable at part zoom. Axis up, camera 5 diameters out."""
    import numpy as np

    axes = {
        "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
    }
    for feature in features:
        axis = np.asarray(axes.get(str(feature.get("axis") or "+z"), (0.0, 0.0, 1.0)))
        position = np.asarray(feature.get("position_mm") or (0.0, 0.0, 0.0), dtype=float)
        diameter = float(feature.get("major_diameter") or 0.0)
        length = float(feature.get("length") or 0.0)
        if diameter <= 0 or length <= 0:
            continue
        start = float(feature.get("start") or 0.0)
        focal = position + axis * (start + length / 2)
        reference = np.asarray((1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0))
        side = np.cross(axis, reference)
        side /= np.linalg.norm(side)
        half_height = max(2.5 * diameter, 0.55 * min(length, 80.0))
        position_cam = focal + side * (5.0 * diameter + length)
        return (
            tuple(float(v) for v in focal),
            tuple(float(v) for v in position_cam),
            tuple(float(v) for v in axis),
            float(half_height),
        )
    return None


def render_thumbnail(
    stl_path: Path,
    out_path: Path,
    size: int = 256,
    *,
    thread_features: list[dict[str, Any]] | None = None,
) -> Path | None:
    if not render_available():
        return None
    import pyvista as pv

    mesh = pv.read(str(stl_path))
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    pl.add_mesh(mesh, color="#9aa3ad", smooth_shading=True, split_sharp_edges=True)
    edges = mesh.extract_feature_edges(30)
    if edges.n_points:
        pl.add_mesh(edges, color="#333333", line_width=1)
    for thread_line in _thread_preview_lines(pv, thread_features or []):
        pl.add_mesh(thread_line, color="#30343b", line_width=1.8)
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_path))
    pl.close()
    return out_path
