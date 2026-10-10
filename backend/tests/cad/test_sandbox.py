import time
from pathlib import Path

import pytest

from cadagent.cad.sandbox import run_user_code

FIXTURES = Path(__file__).parent.parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def workdir(tmp_path):
    return tmp_path / "work"


def test_ok_case(workdir):
    out = run_user_code(read("simple_box.py"), workdir)
    assert out.ok, out.error
    assert out.brep_path and out.brep_path.exists()
    assert out.params_used["length"] == 40.0


def test_syntax_error(workdir):
    out = run_user_code("PARAMS = {}\nresult = Box(1,\n", workdir)
    assert not out.ok
    assert "syntax" in (out.error or "").lower()


def test_static_rejection_no_exec(workdir):
    out = run_user_code("import os\nPARAMS={}\nresult = os.system('x')\n", workdir)
    assert not out.ok
    assert "static" in (out.error or "")
    assert out.static_errors


def test_runtime_error_traceback(workdir):
    code = "from build123d import *\nPARAMS = {}\nresult = Box(1,1,1)\nresult = 1/0\n"
    out = run_user_code(code, workdir)
    assert not out.ok
    assert "ZeroDivisionError" in (out.traceback or "")


def test_timeout_kills(workdir):
    code = "PARAMS = {}\nwhile True: pass\nresult = Box(1,1,1)\n"
    t0 = time.monotonic()
    out = run_user_code(code, workdir, timeout_s=3)
    assert not out.ok
    assert any(k in (out.error or "") for k in ("timed out", "CPU limit", "killed"))
    assert time.monotonic() - t0 < 25


def test_params_override(workdir):
    out = run_user_code(read("simple_box.py"), workdir, params_override={"length": 55.0})
    assert out.ok, out.error
    assert out.params_used["length"] == 55.0


def test_unknown_param_rejected(workdir):
    out = run_user_code(read("simple_box.py"), workdir, params_override={"nope": 1})
    assert not out.ok
    assert "unknown param" in (out.error or "")


def test_bd_warehouse_fixture(workdir):
    out = run_user_code(read("plate_holes.py"), workdir)
    assert out.ok, (out.error, out.traceback)
    assert out.params_used["screw"] == "M4-0.7"


def test_load_reference(workdir, tmp_path):
    from build123d import Box, export_step

    ref = tmp_path / "ref.step"
    export_step(Box(5, 5, 5), str(ref))
    code = "PARAMS = {}\nref = load_reference('a1')\nresult = ref\n"
    out = run_user_code(code, workdir, references=[])  # missing ref -> error
    assert not out.ok
    from cadagent.cad.sandbox import Reference

    out2 = run_user_code(code, workdir, references=[Reference(asset_id="a1", path=str(ref))])
    assert out2.ok, (out2.error, out2.traceback)
