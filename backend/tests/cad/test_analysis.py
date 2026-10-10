from pathlib import Path

from build123d import import_brep

from cadagent.cad.analysis import analyze
from cadagent.cad.sandbox import run_user_code

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _shape(tmp_path):
    out = run_user_code((FIXTURES / "simple_box.py").read_text(), tmp_path / "w")
    assert out.ok, out.error
    return import_brep(str(out.brep_path))


def test_box_analysis(tmp_path):
    a = analyze(_shape(tmp_path))
    # 40x20x10 box with 1.5 fillet on vertical edges
    assert a.bbox_size == [40.0, 20.0, 10.0]
    assert a.solid_count == 1
    assert a.is_valid
    assert a.sits_on_xy
    assert a.centered_xy
    assert 7000 < a.volume < 8000  # 8000 - fillet volume


def test_cylindrical_faces_detected(tmp_path):
    out = run_user_code((FIXTURES / "plate_holes.py").read_text(), tmp_path / "w")
    assert out.ok, out.error
    a = analyze(import_brep(str(out.brep_path)))
    m4 = [c for c in a.cylindrical_faces if abs(c.radius - 2.25) < 0.01]
    assert len(m4) == 4  # 4x M4 normal clearance holes
