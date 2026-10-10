"""Shared fixtures: isolated DATA_DIR per test session."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cadagent-test-"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("LLM_API_KEY", "test")

from cadagent.config import get_settings  # noqa: E402

get_settings.cache_clear()
