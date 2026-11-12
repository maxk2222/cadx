"""Shared id coercion helpers."""

from __future__ import annotations

import uuid


def to_uuid(s: str | uuid.UUID) -> uuid.UUID:
    """Accept either a uuid or its str form — every id column boundary
    (payloads, API paths, DB rows) funnels through this."""
    return s if isinstance(s, uuid.UUID) else uuid.UUID(str(s))
