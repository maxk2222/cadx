"""Kill-switch and fallback behavior — KIARA_NATIVE=0 and per-kind bisects
must leave every helper returning None so callers stay on Python paths."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


def test_per_kind_switches(monkeypatch):
    kiara_native = pytest.importorskip("kiara_native")
    from kiara.cad import native

    monkeypatch.setenv("KIARA_NATIVE_GEOM", "0")
    assert not native.native_enabled("geom")
    assert native.native_enabled("fem")

    monkeypatch.setenv("KIARA_NATIVE_FEM", "0")
    assert not native.native_enabled("fem")

    monkeypatch.delenv("KIARA_NATIVE_GEOM")
    assert native.native_enabled("geom")


def test_master_switch_blocks_import():
    """KIARA_NATIVE=0 in a fresh process leaves the adapter fully inert."""
    code = (
        "from kiara.cad import native; "
        "assert native.native_enabled('geom') is False; "
        "assert native.native_version() is None; "
        "assert native.mesh_index(object()) is None"
    )
    env = {**os.environ, "KIARA_NATIVE": "0"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_helpers_degrade_without_native(monkeypatch):
    """Even with the module importable, helpers return None on bad input —
    callers must never see an exception escape the adapter."""
    kiara_native = pytest.importorskip("kiara_native")
    from kiara.cad import native

    assert native.contains(None, np.zeros((1, 3))) is None
    assert native.ray_any(None, np.zeros((1, 3)), np.zeros((1, 3))) is None
    assert native.scanline_fill(None, np.zeros(3), (2, 2, 2), 1.0, 0.5) is None
    # malformed arrays → None, not an exception
    assert native.contains(object(), np.zeros((3, 2))) is None


def test_module_absent_is_valid(tmp_path, monkeypatch):
    """Reloading the adapter without kiara_native on the path is fine."""
    code = (
        "import sys; sys.modules.pop('kiara_native', None); "
        "sys.path.insert(0, '/nonexistent'); "
        "from kiara.cad import native; import importlib; importlib.reload(native); "
        "print(native.native_enabled())"
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    # either native still resolves (installed) or the reload path works —
    # both are valid, what must NOT happen is a crash
    assert r.returncode == 0, r.stderr
