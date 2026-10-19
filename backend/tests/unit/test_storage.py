"""cadx.storage path-safety: resolve() must reject sibling-prefix escapes."""

from __future__ import annotations

import pytest

from cadx import storage


def test_resolve_inside_data_dir():
    p = storage.resolve("tmp/inside.txt")
    assert p.parent == storage.data_root() / "tmp"
    # stored relpaths round-trip
    assert storage.resolve(storage.relpath(p)) == p


def test_resolve_rejects_dotdot_escape():
    with pytest.raises(ValueError, match="escapes data dir"):
        storage.resolve("../outside.txt")


def test_resolve_rejects_absolute_path():
    with pytest.raises(ValueError, match="escapes data dir"):
        storage.resolve("/etc/hostname")


def test_resolve_rejects_sibling_prefix():
    """`data_evil/` shares the `data` string prefix — a raw startswith check
    would pass it; the segment-wise check must not."""
    root = storage.data_root()
    with pytest.raises(ValueError, match="escapes data dir"):
        storage.resolve(f"../{root.name}_evil/x.step")


def test_resolve_accepts_nested_sibling_names():
    """Names that merely *start with* the root's name inside it are fine."""
    p = storage.resolve(f"{storage.data_root().name}_sub/file.step")
    assert p.is_relative_to(storage.data_root())
