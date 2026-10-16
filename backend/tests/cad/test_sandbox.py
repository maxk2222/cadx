import time
from pathlib import Path

import pytest

from cadx.cad.sandbox import run_user_code

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
    from cadx.cad.sandbox import Reference

    out2 = run_user_code(code, workdir, references=[Reference(asset_id="a1", path=str(ref))])
    assert out2.ok, (out2.error, out2.traceback)


def test_geometry_at_failure_reports_builder_bbox(workdir):
    """A mid-build crash must report where the builder's shape actually is —
    the repair loop can't see renders, this is its ground truth."""
    code = """from build123d import *
PARAMS = {}
with BuildPart() as p:
    Box(60, 40, 4)
raise ValueError("boom")
result = p.part
"""
    out = run_user_code(code, workdir)
    assert not out.ok
    geo = {g["name"]: g for g in out.geometry_at_failure}
    assert "p" in geo, out.geometry_at_failure
    assert geo["p"]["bbox_min"] == [-30.0, -20.0, -2.0]
    assert geo["p"]["solids"] == 1


def test_geometry_at_failure_none_result(workdir):
    code = "from build123d import *\nPARAMS = {}\nresult = None\n"
    out = run_user_code(code, workdir)
    assert not out.ok
    assert out.geometry_at_failure == []


def test_warm_cold_parity(workdir, tmp_path):
    """Warm fork-server and cold subprocess must produce identical results."""
    code = read("simple_box.py")
    warm = run_user_code(code, workdir, warm=True)
    cold = run_user_code(code, tmp_path / "cold", warm=False)
    assert warm.ok and cold.ok
    assert warm.params_used == cold.params_used
    assert cold.brep_path and cold.brep_path.exists()


def test_warm_zygote_survives_child_timeout(workdir, tmp_path):
    """A runaway fork child gets killed; the zygote itself stays usable —
    the next request must still be served by the warm path."""
    loop = "PARAMS = {}\nwhile True: pass\nresult = None\n"
    out = run_user_code(loop, workdir, timeout_s=3, warm=True)
    assert not out.ok
    assert any(k in (out.error or "") for k in ("timed out", "CPU limit", "killed"))
    out2 = run_user_code(read("simple_box.py"), tmp_path / "w2", warm=True)
    assert out2.ok, (out2.error, out2.traceback)


def test_cold_fallback_when_zygote_broken(workdir, monkeypatch):
    """If the zygote cannot spawn/serve, run_user_code must silently fall
    back to the cold subprocess — the security model is identical."""
    import cadx.cad.sandbox as sb

    sb._ZYGOTE.kill()  # force a spawn attempt on next run

    def boom(self):
        raise RuntimeError("no zygote")

    monkeypatch.setattr(sb._Zygote, "spawn", boom)
    out = run_user_code(read("simple_box.py"), workdir, warm=True)
    assert out.ok, (out.error, out.traceback)  # cold subprocess took over
