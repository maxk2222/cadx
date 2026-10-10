"""Security spot checks: hostile generated code must be rejected or contained."""

import time

import pytest

from cadagent.cad.sandbox import run_user_code

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
