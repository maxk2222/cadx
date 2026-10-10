"""Model adapters. Each adapter wraps one image-to-3D backend behind the
MeshAdapter interface in base.py — the API layer never imports model code
directly.
"""

from __future__ import annotations

from meshserver.adapters.base import AdapterGenResult, GenOpts, MeshAdapter
from meshserver.adapters.fake import FakeAdapter

__all__ = [
    "AdapterGenResult",
    "FakeAdapter",
    "GenOpts",
    "MeshAdapter",
]
