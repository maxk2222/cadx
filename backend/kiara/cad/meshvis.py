"""Viewer-safe mesh visuals — trimesh-only, importable in the cad-light
worker (no build123d/OCC dependency).

glTF defaults metallicFactor=1 when a mesh carries no explicit material, so
every material-less CAD GLB loads as a perfect mirror and renders BLACK in
any consumer without an environment map (all three.js viewers here). Every
GLB this pipeline writes — version previews, guide parts, normalized copies —
gets the same neutral dielectric PBR instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

from kiara.logging import get_logger

log = get_logger(__name__)


def viewer_pbr() -> Any:
    """Neutral dielectric PBR — flat light gray, no metalness."""
    from trimesh.visual.material import PBRMaterial

    return PBRMaterial(
        baseColorFactor=[168, 170, 178, 255],
        metallicFactor=0.0,
        roughnessFactor=0.75,
    )


def stamp_viewer_pbr(path: Path) -> Path:
    """Re-export a GLB with the neutral viewer PBR — no-op for non-GLB."""
    if path.suffix.lower() not in {".glb", ".gltf"}:
        return path
    from trimesh.visual import TextureVisuals

    try:
        m = trimesh.load(path)
        if hasattr(m, "to_geometry"):
            m = m.to_geometry()
        m.visual = TextureVisuals(material=viewer_pbr())
        m.export(str(path))
    except Exception as exc:
        log.warning("cad.pbr_stamp_failed", path=path.name, err=str(exc))
    return path
