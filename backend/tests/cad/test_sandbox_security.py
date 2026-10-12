"""Security spot checks: hostile generated code must be rejected or contained."""

import time

import pytest

from cadx.cad.sandbox import run_user_code

CASES = {
    "import_os": "import os\nPARAMS={}\nresult=os\n",
    "import_socket": "import socket\nPARAMS={}\nresult=None\n",
    "import_requests": "import requests\nPARAMS={}\nresult=None\n",
    "dunder_import": "PARAMS={}\nos=__import__('os')\nresult=None\n",
    "subclasses": "PARAMS={}\nx=[].__class__.__mro__\nresult=None\n",
    "open_file": "PARAMS={}\nf=open('/etc/passwd')\nresult=None\n",
    "eval_exec": "PARAMS={}\nresult=eval('1+1')\n",
    "importlib": "PARAMS={}\nimport importlib\nresult=None\n",
    "pathlib": "PARAMS={}\nfrom pathlib import Path\nresult=None\n",
    "getattr_bypass": "PARAMS={}\nresult=getattr(object,'__subclass''es__')()\n",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_hostile_code_blocked(name, tmp_path):
    out = run_user_code(CASES[name], tmp_path / name)
    assert not out.ok, name
    # static rejection preferred; a runtime ImportError/NameError is also acceptable
    assert out.error


def test_memory_bomb_killed(tmp_path):
    code = "PARAMS={}\nx = 'x' * (10**10)\nresult = None\n"
    out = run_user_code(code, tmp_path / "mem", timeout_s=20)
    assert not out.ok


def test_cpu_bomb_killed(tmp_path):
    code = "PARAMS={}\nwhile True: x=1\nresult=None\n"
    t0 = time.monotonic()
    out = run_user_code(code, tmp_path / "cpu", timeout_s=5)
    assert not out.ok
    assert any(k in (out.error or "") for k in ("timed out", "CPU limit", "killed", "crashed"))
    assert time.monotonic() - t0 < 30


def test_anchor_helpers(tmp_path):
    """anchor_holes + load_image_profile + profile_body run inside the sandbox."""
    code = """
import cadlib
from build123d import *

PARAMS = {"thickness": 8.0}

holes = anchor_holes()
assert len(holes) == 3 and holes[0]["x"] == -40.0
assert ANCHOR_HOLES == holes
assert len(IMAGE_PROFILE) >= 3

face = load_image_profile()
result = cadlib.profile_body(face, 8.0)
for h in holes:
    result -= Pos(h["x"], h["y"], -1) * Cylinder(radius=2.5, height=10)
"""
    anchors = {
        "holes": [
            {"index": 0, "x": -40.0, "y": 2.0, "dia": 5.0, "source": "image"},
            {"index": 1, "x": 0.0, "y": 0.0, "dia": 5.0, "source": "image"},
            {"index": 2, "x": 40.0, "y": -2.0, "dia": 5.0, "source": "image"},
        ],
        "image_outline": [
            [-60, -15],
            [-30, -19],
            [0, -17],
            [30, -19],
            [60, -12],
            [60, 12],
            [30, 19],
            [0, 17],
            [-30, 19],
            [-60, 15],
            [-60, -15],
        ],
    }
    out = run_user_code(code, tmp_path / "anchors", anchors=anchors, timeout_s=120)
    assert out.ok, out.error
    assert out.brep_path is not None


def test_load_image_profile_empty(tmp_path):
    """load_image_profile raises a clear error without measured outlines."""
    code = "PARAMS = {}\nresult = load_image_profile()"
    out = run_user_code(code, tmp_path / "noprofile", anchors={}, timeout_s=60)
    assert not out.ok
    assert "no measured plan view" in (out.error or "")
