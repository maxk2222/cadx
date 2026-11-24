"""Unity conversion golden tests — mm/Z-up RH → m/Y-up LH.

Golden vectors derived by hand: CAD up (+Z) must become Unity up (+Y);
the manual's working face (+Y) lands on Unity -Z and operator-left (+X)
on Unity -X — the suite's published convention (verified on a reference
guide: connectors at -Z, manual-"links" at -X). Rotations conjugate
through the det-1 spatial map so RH→LH is handled — a CAD rotation
about Z must appear as a Unity rotation about Y.
"""

import numpy as np

from kiara.guide import unity


def test_position_mm_zup_to_m_yup() -> None:
    # CAD (x=10mm len, y=20mm width, z=30mm height) → Unity (-0.01, 0.03, -0.02) m
    assert unity.to_unity_position([10, 20, 30]) == [-0.01, 0.03, -0.02]
    assert unity.to_unity_position([0, 0, 0]) == [-0.0, 0.0, -0.0]


def test_identity_rotation_stays_identity() -> None:
    q = unity.to_unity_quaternion(np.eye(3))
    assert q == [0.0, 0.0, 0.0, 1.0]


def test_z_up_yaw_becomes_unity_yaw() -> None:
    """CAD 90° rotation about Z (yaw) → Unity quaternion about +Y."""
    theta = np.deg2rad(90)
    rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    q = unity.to_unity_quaternion(rz)
    # quaternion [x,y,z,w]: pure Y rotation → x=z=0, |y|=sin(45°), w=cos(45°)
    assert abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6
    assert abs(abs(q[1]) - np.sin(np.deg2rad(45))) < 1e-6
    assert abs(q[3] - np.cos(np.deg2rad(45))) < 1e-6


def test_transform_maps_position_and_rotation() -> None:
    """CAD transform: translate (100,200,300)mm + identity rot → Unity
    pos (-0.1, 0.3, -0.2)m, identity quaternion."""
    t = np.eye(4)
    t[0:3, 3] = [100, 200, 300]
    out = unity.transform_to_unity(t.tolist())
    assert out["position"] == [-0.1, 0.3, -0.2]
    assert out["quaternion"] == [0.0, 0.0, 0.0, 1.0]
    assert out["scale"] == [1.0, 1.0, 1.0]


def _quat_to_mat(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_handedness_preserved_through_conjugation() -> None:
    """Arbitrary proper CAD rotation stays proper in Unity (det +1) even
    though the change-of-basis itself is a reflection."""
    rng = np.random.default_rng(7)
    a, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(a) < 0:
        a[:, 0] = -a[:, 0]
    q = unity.to_unity_quaternion(a)
    ru = _quat_to_mat(q)
    assert np.linalg.det(ru) > 0.99


def test_build_unity_guide_smoke() -> None:
    from kiara.guide.state import (
        AssemblyInstance,
        GuidePart,
        GuideState,
        GuideStep,
    )

    st = GuideState(language="de")
    st.parts.append(GuidePart(id="base", name="Basis", step_path="/x/base.step", source="uploaded"))
    st.assembly.instances.append(
        AssemblyInstance(part_id="base", transform=np.eye(4).tolist(), placed_in_step=1)
    )
    st.assembly.step_instances["1"] = ["base"]
    st.steps.append(
        GuideStep(number=1, text="Basis auf den Tisch legen.", display_text="Basis hinlegen.")
    )
    guide = unity.build_unity_guide(st)
    assert guide["units"] == "meters"
    assert guide["coordinate_system"] == "unity-left-handed-y-up"
    assert guide["steps"][0]["display_text"] == "Basis hinlegen."
    assert guide["steps"][0]["placements"][0]["part_id"] == "base"
    assert guide["steps"][0]["visible_instance_ids"] == ["base"]


# --------------------------------------------------------------------------- #
# arrows / texts
# --------------------------------------------------------------------------- #


def _arrow_state() -> "object":
    """Two mated parts: `host` fixed at origin with an upward-facing port,
    `plug` mated onto it — the joint's approach_axis is the fixed port's
    outward normal (+Z local = +Z world under identity)."""
    from kiara.guide.state import (
        AssemblyInstance,
        GuidePart,
        GuideState,
        GuideStep,
        PortInfo,
    )

    st = GuideState(language="de")
    st.parts.append(
        GuidePart(
            id="host",
            name="Host",
            source="uploaded",
            bbox_min=[-20, -20, 0],
            bbox_max=[20, 20, 20],
            ports=[PortInfo(name="mouth", pos=[0, 0, 20], axis=[0, 0, 1], gender="female")],
        )
    )
    st.parts.append(
        GuidePart(
            id="plug",
            name="Plug",
            source="uploaded",
            bbox_min=[-5, -5, 0],
            bbox_max=[5, 5, 100],
            ports=[PortInfo(name="tip", pos=[0, 0, 100], axis=[0, 0, -1], gender="male")],
        )
    )
    st.assembly.instances.append(
        AssemblyInstance(
            instance_id="host#1", part_id="host", transform=np.eye(4).tolist(), placed_in_step=1
        )
    )
    st.assembly.instances.append(
        AssemblyInstance(
            instance_id="plug#1", part_id="plug", transform=np.eye(4).tolist(), placed_in_step=2
        )
    )
    st.assembly.step_instances["1"] = ["host#1"]
    st.assembly.step_instances["2"] = ["plug#1"]
    st.assembly.joints.append(
        {
            "step": 2,
            "moving_part": "plug",
            "fixed_part": "host",
            "moving_instance": "plug#1",
            "fixed_instance": "host#1",
            "moving_port": "tip",
            "fixed_port": "mouth",
            "approach_axis": [0.0, 0.0, 1.0],  # fixed port axis, fixed-part local
        }
    )
    st.steps.append(GuideStep(number=1, text="Host hinstellen.", display_text="Host."))
    st.steps.append(GuideStep(number=2, text="Plug einstecken.", display_text="Plug."))
    return st


def test_joint_arrow_points_at_the_port() -> None:
    """Arrow semantics = tip lands on the mate: direction is the NEGATED
    fixed-port outward axis (approach-from → travel-to), tail stands off
    along the approach direction scaled by the moving part's extent."""
    st = _arrow_state()
    guide = unity.build_unity_guide(st)
    arrow = guide["steps"][1]["arrows"][0]
    # CAD +Z approach → unity direction maps to (0, -1, 0): pointing DOWN at the port
    assert arrow["direction"] == [-0.0, -1.0, -0.0]
    # tail above the port (unity y = CAD z): port at z=20mm → 0.02m;
    # plug is 100mm tall along the axis → standoff = 50+15 = 65mm → 0.065m
    assert arrow["origin"][1] == 0.085
    # label sits mid-shaft between tip and tail
    assert 0.02 < arrow["label_position"][1] < 0.085


def test_annotation_arrow_tail_clears_bbox() -> None:
    """A 500mm part must not swallow the arrow tail: standoff scales with
    the bbox extent along the arrow axis, not a fixed 30mm."""
    st = _arrow_state()
    host = st.part_by_id("host")
    host.bbox_min, host.bbox_max = [-10, -10, 0], [10, 10, 500]
    st.annotations.append(
        {"kind": "arrow", "step": 2, "part_id": "host", "approach_axis": [0, 0, -1]}
    )
    guide = unity.build_unity_guide(st)
    arrow = guide["steps"][1]["arrows"][-1]  # annotation arrow appended after joint arrow
    # bbox center z=250mm; down-arrow tail at center - axis*(250+15) → z=515mm → unity y 0.515
    assert arrow["direction"] == [-0.0, -1.0, -0.0]
    assert arrow["origin"][1] == 0.515


def test_text_annotations_emit_constant_rotation() -> None:
    """Every text field carries the same identity rotation — orientation is
    a fixed contract; only position varies."""
    st = _arrow_state()
    st.annotations.append(
        {"kind": "text", "step": 2, "position": [10.0, 20.0, 30.0], "text": "Vorsicht heiß"}
    )
    st.annotations.append({"kind": "text", "step": 2, "text": "no position"})
    guide = unity.build_unity_guide(st)
    texts = guide["steps"][1]["texts"]
    assert len(texts) == 2
    assert texts[0]["position"] == [-0.01, 0.03, -0.02]
    assert all(t["rotation"] == [0.0, 0.0, 0.0, 1.0] for t in texts)
    # unset position → deterministic fallback above this step's placed bbox
    # (plug at origin spans z 0..100 → 100+45=145mm → unity y 0.145)
    assert texts[1]["position"] == [0.0, 0.145, 0.0]


def test_preassembly_joint_axis_is_the_fixed_port_axis() -> None:
    """pN.get('axis') bug guard: bridge/premount joints must store the real
    fixed-port axis, never the [0,0,1] fallback."""
    from unittest.mock import MagicMock

    from kiara.guide.preassembly import _port_axis
    from kiara.guide.state import GuidePart, PortInfo

    sess = MagicMock()
    sess.state.part_by_id.side_effect = lambda pid: GuidePart(
        id=pid,
        name=pid,
        ports=[
            PortInfo(name="a", axis=[1.0, 0.0, 0.0]),
            PortInfo(name="b", axis=[0.0, -1.0, 0.0]),
        ],
    )
    inst = MagicMock(part_id="host")
    assert _port_axis(sess, inst, "b") == [0.0, -1.0, 0.0]
    assert _port_axis(sess, inst, "missing") == [0.0, 0.0, 1.0]
