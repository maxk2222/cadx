"""Unity conversion golden tests — mm/Z-up RH → m/Y-up LH.

Golden vectors derived by hand: CAD up (+Z) must become Unity up (+Y);
CAD front (+Y) becomes Unity forward (+Z). Rotations conjugate through
the det-1 spatial map so RH→LH is handled — a CAD rotation about Z must
appear as a Unity rotation about Y.
"""

import numpy as np

from cadx.guide import unity


def test_position_mm_zup_to_m_yup() -> None:
    # CAD (x=10mm len, y=20mm width, z=30mm height) → Unity (0.01, 0.03, 0.02) m
    assert unity.to_unity_position([10, 20, 30]) == [0.01, 0.03, 0.02]
    assert unity.to_unity_position([0, 0, 0]) == [0.0, 0.0, 0.0]


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
    pos (0.1, 0.3, 0.2)m, identity quaternion."""
    t = np.eye(4)
    t[0:3, 3] = [100, 200, 300]
    out = unity.transform_to_unity(t.tolist())
    assert out["position"] == [0.1, 0.3, 0.2]
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
    from cadx.guide.state import (
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
