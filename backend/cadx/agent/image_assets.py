"""Persist generated images as Asset rows with provenance.

Shared by the /images API router and the agent's hero-image autogeneration.
Files land directly under projects/<pid>/assets/ via storage.asset_path —
no intermediate staging directory.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cadx import storage
from cadx.db.models import Asset, AssetKind
from cadx.logging import get_logger

log = get_logger(__name__)


def persist_generated_images(
    db: AsyncSession,
    project_id: uuid.UUID,
    paths: list[Path],
    *,
    prompt: str,
    gen_params: dict[str, Any],
    caption: str = "",
    derived_from: uuid.UUID | None = None,
) -> list[Asset]:
    """Copy generated PNGs into the project asset dir and create Asset rows.

    `gen_params` must already contain everything needed to reproduce the
    image (prompt/model/seed/width/height/steps/guidance); this function adds
    the stored filename per image under `images[i].filename`.
    """
    from PIL import Image

    assets: list[Asset] = []
    for i, src in enumerate(paths):
        data = src.read_bytes()
        width: int | None = None
        height: int | None = None
        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except Exception:
            log.warning("image_asset.decode_failed", path=str(src))
        asset = Asset(
            project_id=project_id,
            kind=AssetKind.generated_image,
            filename=storage.safe_filename(f"gen-{prompt[:40].strip() or 'image'}-{i}.png"),
            path="",  # filled below
            mime="image/png",
            width=width,
            height=height,
            caption=caption or prompt[:200],
            derived_from=derived_from,
            gen_params={**gen_params, "image_index": i},
        )
        path = storage.asset_path(project_id, asset.id, asset.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        asset.path = storage.relpath(path)
        db.add(asset)
        assets.append(asset)
    return assets
