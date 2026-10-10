from pathlib import Path

from cadagent.cad.engine import run_pipeline
from cadagent.db.models import ProjectSettings

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_full_pipeline(tmp_path):
    res = run_pipeline(
        code=(FIXTURES / "plate_holes.py").read_text(),
        out_dir=tmp_path / "it1",
        settings=ProjectSettings(process="fdm"),
        full_exports=True,
    )
    assert res.ok, res.error
    assert res.analysis is not None
    assert res.analysis.bbox_size == [100.0, 60.0, 8.0]
    assert len(res.analysis.cylindrical_faces) >= 4
    for fmt in ("step", "stl", "3mf", "glb"):
        assert res.exports.get(fmt), fmt
    assert res.dfm is not None
    assert (tmp_path / "it1" / "contact_sheet.png").exists()
    assert (tmp_path / "it1" / "thumbnail.png").exists()


def test_pipeline_failure_reports(tmp_path):
    res = run_pipeline(
        code="PARAMS = {}\nresult = undefined_name\n",
        out_dir=tmp_path / "bad",
    )
    assert not res.ok
    assert res.error
