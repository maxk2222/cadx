"""Error diagnosis — newer classifier categories.

Unit-level mirror of tests/cad/test_errordiag.py: pure `diagnose` rules
(the Plane `normal=`/`z_dir` and sweep `profile=`/`sections` hallucinations,
the mesh-boolean no-op and the OCC frozen-shape / open-wire loft failures)
— no OCC runtime or sandbox needed here.
"""

from __future__ import annotations

from cadx.cad.errordiag import diagnose

CODE = "line1\nline2\nline3\nline4\nline5\nline6\nline7\n"


def test_plane_normal_kwarg_is_api_plane_normal():
    """build123d Plane has no `normal` kwarg — the hallucination maps to
    `z_dir` and gets its own category, not the generic api-args bucket."""
    d = diagnose(
        "TypeError: Plane.__init__() got an unexpected keyword argument 'normal'",
        'File "model.py", line 4\nTypeError',
        CODE,
    )
    assert d.category == "api-plane-normal"
    assert "z_dir" in d.hint


def test_sweep_profile_kwarg_is_api_sweep_profile():
    """build123d sweep() takes `sections=` — `profile=` is the recurring
    wrong guess."""
    d = diagnose(
        "TypeError: sweep() got an unexpected keyword argument 'profile'",
        'File "model.py", line 6\nTypeError',
        CODE,
    )
    assert d.category == "api-sweep-profile"
    assert "sections" in d.hint


def test_mesh_cut_noop_is_hybrid_boolean_noop():
    """mesh_cut's no-material guard needs a fix hint that points at the
    envelope clip helper, not a generic argument-type guess."""
    d = diagnose(
        "ValueError: mesh_cut removed no material — the tool does not overlap the body",
        'File "model.py", line 7\nValueError',
        CODE,
    )
    assert d.category == "hybrid-boolean-noop"
    assert "clip_to_box" in d.hint


def test_occ_frozen_shape():
    d = diagnose(
        "OCP.Standard_Failure.Standard_Failure: TopoDS_FrozenShape: "
        "the shape is frozen and cannot be modified",
        'File "model.py", line 9\nStandard_Failure',
        CODE,
    )
    assert d.category == "occ-frozen"


def test_occ_loft_open_wire():
    """'closed wires' StdFail_NotDone is a loft/sweep face failure — it must
    not be swallowed by the fillet-radius hint."""
    d = diagnose(
        "StdFail_NotDone: Face can only be created with closed wires",
        'File "model.py", line 11\n    loft(...)\nStdFail_NotDone',
        CODE,
    )
    assert d.category == "occ-loft"
