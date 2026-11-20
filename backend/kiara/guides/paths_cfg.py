"""Settings bridge for modules ported from the KIARA web suite.

The suite's `backend.core.config.settings` exposes DATA_DIR / MEDIA_MAX_* /
ASSIMP_*; our bootstrap settings live in `kiara.config`. This shim presents the
same attribute surface so the ported modules stay verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiara.config import get_settings


class _GuideSettings:
    """Suite-shaped settings surface backed by `kiara.config.get_settings()`."""

    @property
    def DATA_DIR(self) -> str:
        return str(get_settings().data_path)

    @property
    def DATA_PATH(self) -> Path:
        return get_settings().data_path

    @property
    def MEDIA_MAX_IMAGE_BYTES(self) -> int:
        return get_settings().media_max_image_bytes

    @property
    def MEDIA_MAX_VIDEO_BYTES(self) -> int:
        return get_settings().media_max_video_bytes

    @property
    def MEDIA_MAX_AUDIO_BYTES(self) -> int:
        return get_settings().media_max_audio_bytes

    @property
    def MEDIA_MAX_CAD_BYTES(self) -> int:
        return get_settings().media_max_cad_bytes

    @property
    def MEDIA_MAX_IMAGE_PIXELS(self) -> int:
        return get_settings().media_max_image_pixels

    @property
    def ASSIMP_DOCKER_IMAGE(self) -> str:
        return get_settings().assimp_docker_image

    @property
    def ASSIMP_TIMEOUT_SECONDS(self) -> int:
        return get_settings().assimp_timeout_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings = _GuideSettings()
