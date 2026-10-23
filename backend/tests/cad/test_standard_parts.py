from __future__ import annotations

import numpy as np
import pytest
from build123d import Axis, Plane, import_brep, import_step
from PIL import Image

from cadx.cad import catalog
from cadx.cad.allowlist import check_code
from cadx.cad.exporters import export_all_3d, quick_stl
from cadx.cad.partspec import check_standard_part_calls
from cadx.cad.render import render_available, render_views
from cadx.cad.sandbox import run_user_code
from cadx.cad.standard_parts import build_standard_part_code
from cadx.llm.schemas import StandardPartSpec


def _m10_code() -> str:
    return build_standard_part_code(
        [
            StandardPartSpec(
                class_name="CounterSunkScrew",
                fastener_type="iso10642",
                size="M10-1.5",
                length_mm=150,
            )
        ]
    )


def test_m10_preview_and_exact_export_have_distinct_thread_geometry(tmp_path):
    preview = run_user_code(
        _m10_code(),
        tmp_path / "preview",
        detail_mode="preview",
        warm=False,
        threads=1,
        timeout_s=180,
        mem_mb=4096,
    )
    assert preview.ok, preview.error
    assert preview.detail_mode == "preview"
    assert preview.thread_count == 0
    assert preview.preview_thread_count == 1

    exact = run_user_code(
        _m10_code(),
        tmp_path / "exact",
        detail_mode="export",
        warm=False,
        threads=1,
        timeout_s=180,
        mem_mb=4096,
    )
    assert exact.ok, exact.error
    assert exact.detail_mode == "export"
    assert exact.thread_count == 1
    assert exact.bom == [
        {
            "part_class": "CounterSunkScrew",
            "type": "iso10642",
            "size": "M10-1.5",
            "count": 1,
            "name": None,
        }
    ]
    preview_shape = import_brep(str(preview.brep_path))
    exact_shape = import_brep(str(exact.brep_path))
    preview_bb = preview_shape.bounding_box().size
    exact_bb = exact_shape.bounding_box().size
    assert pytest.approx([20, 20, 150], abs=0.2) == [
        preview_bb.X,
        preview_bb.Y,
        preview_bb.Z,
    ]
    assert pytest.approx([20, 20, 150], abs=0.2) == [
        exact_bb.X,
        exact_bb.Y,
        exact_bb.Z,
    ]
    # frame contract: the screw stands on its countersunk head — the catalog
    # builds it head-datum-first (body extends local -Z), so the recipe must
    # flip it, not just translate it: min z = 0 and the wide head sits at the
    # bottom, the M10 shank tip at the top
    for shape in (preview_shape, exact_shape):
        bb = shape.bounding_box()
        assert bb.min.Z == pytest.approx(0.0, abs=0.05)
        head = shape.faces().filter_by(Plane.XY).sort_by(Axis.Z)[0]
        assert head.bounding_box().size.X == pytest.approx(20.0, abs=0.5)
    assert exact_shape.volume > preview_shape.volume

    exports = export_all_3d(exact_shape, tmp_path / "exports")
    assert all(exports.get(fmt) for fmt in ("step", "stl", "3mf", "glb"))
    for fmt in ("step", "stl", "3mf", "glb"):
        assert (tmp_path / "exports" / exports[fmt]).stat().st_size > 0
    step = import_step(str(tmp_path / "exports" / exports["step"]))
    assert step.volume == pytest.approx(exact_shape.volume, rel=0.01)


@pytest.mark.render
def test_thread_preview_metadata_produces_visible_render_lines(tmp_path):
    if not render_available():
        pytest.skip("offscreen rendering unavailable")
    out = run_user_code(
        _m10_code(),
        tmp_path / "render",
        detail_mode="preview",
        warm=False,
        threads=1,
        timeout_s=180,
        mem_mb=4096,
    )
    assert out.ok, out.error
    stl = quick_stl(import_brep(str(out.brep_path)), tmp_path / "render" / "render_mesh")
    thread_features = [row for row in out.design_features if row.get("kind") == "thread"]
    assert len(thread_features) == 1
    renders = render_views(
        stl,
        tmp_path / "render" / "thread_views",
        [150.0, 20.0, 20.0],
        size=256,
        thread_features=thread_features,
    )
    plain = render_views(
        stl,
        tmp_path / "render" / "plain_views",
        [150.0, 20.0, 20.0],
        size=256,
    )
    assert "iso" in renders and "iso" in plain
    with Image.open(renders["iso"]) as threaded_img, Image.open(plain["iso"]) as plain_img:
        threaded = np.asarray(threaded_img.convert("RGB"), dtype=np.int16)
        unthreaded = np.asarray(plain_img.convert("RGB"), dtype=np.int16)
    assert float(np.abs(threaded - unthreaded).mean()) > 0.1
    assert "thread_detail" in renders


# --------------------------------------------------------------------------- #
# catalog breadth — every registered family has a validated path or a
# documented exception
# --------------------------------------------------------------------------- #

# Classes whose only required params are host objects (a fastener/bearing/ring/
# key to cut a feature FOR) — context helpers, never standalone parts.
_CONTEXT_CLASSES = {
    "ClearanceHole",
    "InsertHole",
    "Keyway",
    "PressFitHole",
    "RetainingRingGroove",
    "TapHole",
    "ThreadedHole",
}

# Required numeric params without choices the catalog cannot invent.
_EXAMPLE_PARAMS: dict[str, dict[str, object]] = {
    "IsoThread": {"major_diameter": 10.0, "pitch": 1.5},
    "TrapezoidalThread": {"diameter": 10.0, "pitch": 2.0, "thread_angle": 30.0},
    "WhitworthThread": {"major_diameter": 10.0, "pitch": 1.5},
    "HelicalGear": {
        "module": 2.0,
        "tooth_count": 18,
        "pressure_angle": 20.0,
        "helix_angle": 25.0,
        "thickness": 12.0,
    },
    "RackGear": {"module": 2.0, "tooth_count": 18, "pressure_angle": 20.0, "thickness": 12.0},
    "SpurGear": {"module": 2.0, "tooth_count": 18, "pressure_angle": 20.0, "thickness": 12.0},
    "WormGear": {
        "module": 2.0,
        "starts": 1,
        "diametral_quotient": 10.0,
        "tooth_count": 1,
        "worm_length": 30.0,
        "wheel_thickness": 10.0,
    },
    "WormWheel": {"module": 2.0, "tooth_count": 30, "thickness": 10.0},
    "Sprocket": {"num_teeth": 16},
    "Pipe": {"nps": "1", "material": "stainless", "identifier": "40S"},
    "LappedFlangeStub": {"stub_length": 50.0},
    "CBeamAssembly": {"rail_length": 100.0},
    "XtremeSolidVWheelAssembly": {"eccentric": True},
    "ShaftKey": {"shaft_diameter": 8.0},
    "HexFlangedEccentricBushing": {
        "bore_diameter": 5.0,
        "body_diameter": 10.0,
        "body_length": 12.0,
        "flange_height": 2.0,
        "hex_size": 13.0,
        "eccentricity": 0.7,
    },
}


def _spec_for(name: str, entry: dict) -> StandardPartSpec:
    """Auto-fill a valid StandardPartSpec: real type/size from the norm table,
    choice params take their first option, remaining numerics come from the
    documented example table."""
    schema = entry.get("parameters") or {}
    types = entry.get("types") or {}
    real_types = [t for t in types if t != "default"]
    ftype = real_types[0] if real_types else None
    size = None
    if "size" in schema:
        pool = types.get(ftype) or next(iter(types.values()), [])
        size = pool[0] if pool else None
    parameters = dict(_EXAMPLE_PARAMS.get(name, {}))
    length_mm = None
    for pname, row in schema.items():
        if not row.get("required") or pname in parameters:
            continue
        choices = row.get("choices") or []
        if pname == "size":
            continue
        if pname in {"fastener_type", "bearing_type", "o_ring_type", "bottle_type"} and ftype:
            continue  # covered by the fastener_type field via the selector map
        if pname == "length":
            if choices:
                parameters["length"] = choices[0]
            else:
                length_mm = 25.0
            continue
        if choices:
            parameters[pname] = choices[0]
            continue
        raise AssertionError(
            f"{name}: required param '{pname}' has no choices and no example — "
            "add it to _EXAMPLE_PARAMS"
        )
    return StandardPartSpec(
        class_name=name,
        fastener_type=ftype or "",
        size=size or "",
        length_mm=length_mm if entry.get("category") != "pipes" else 50.0,
        parameters=parameters,
    )


def test_every_catalog_family_has_a_validated_recipe_or_documented_exception():
    failures: dict[str, str] = {}
    for name in sorted(catalog.registry()):
        entry = catalog.find_entry(name) or {}
        if name in _CONTEXT_CLASSES:
            with pytest.raises(ValueError, match="not a standalone part"):
                build_standard_part_code([StandardPartSpec(class_name=name)])
            continue
        try:
            spec = _spec_for(name, entry)
            code = build_standard_part_code([spec])
            checked = check_code(code)
            assert checked.ok, f"{name}: allowlist rejected recipe: {checked.errors[:3]}"
            violations = check_standard_part_calls(code)
            assert not violations, f"{name}: {violations[:3]}"
        except Exception as exc:
            failures[name] = str(exc)[:200]
    assert not failures, f"catalog families without a validated path: {failures}"


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("SocketHeadCapScrew", {"fastener_type": "iso4762", "size": "M8-1.25", "length_mm": 30}),
        ("HexNut", {"fastener_type": "iso4032", "size": "M8-1.25"}),
        ("HeatSetNut", {"fastener_type": "ruthex", "size": "M4-0.7-4.0"}),
        ("ExternalSnapRing", {"size": "5"}),
        ("SingleRowDeepGrooveBallBearing", {"fastener_type": "SKT", "size": "M4-11-4"}),
        ("ShaftKey", {"size": "8", "length_mm": 25}),
        ("ORing", {"fastener_type": "iso3601", "size": "005"}),
        ("SpurGear", _EXAMPLE_PARAMS["SpurGear"]),
        ("Pipe", {**_EXAMPLE_PARAMS["Pipe"], "length_mm": 60}),
        ("WeldNeckFlange", {"parameters": {"nps": "1", "flange_class": "150"}}),
        ("VSlotLinearRail", {"parameters": {"rail_size": "20x20"}, "length_mm": 100}),
    ],
)
def test_representative_families_instantiate_with_norm_dimensions(name, kwargs):
    kwargs = dict(kwargs)
    params = kwargs.pop("parameters", None) or _EXAMPLE_PARAMS.get(name)
    data = catalog.lookup_part(
        name,
        fastener_type=kwargs.get("fastener_type") or None,
        size=kwargs.get("size") or None,
        length=kwargs.get("length_mm"),
        parameters=params,
        simple=True,
    )
    bbox = data.get("bbox_size_mm")
    assert isinstance(bbox, list) and len(bbox) == 3
    assert all(float(v) > 0 for v in bbox)
