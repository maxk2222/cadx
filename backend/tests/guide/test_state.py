"""GuideState persistence + mission bookkeeping."""

from pathlib import Path

from kiara.guide.state import (
    AssemblyInstance,
    DocumentIndex,
    GuidePart,
    GuideState,
    GuideStep,
    PageSummary,
    PortInfo,
)


def _state() -> GuideState:
    return GuideState(
        project_id="p1",
        guide_version_id="v1",
        output_dir="/tmp/x",
        document=DocumentIndex(
            pdf_path="/m.pdf",
            pdf_asset_id="a1",
            page_count=2,
            pages=[
                PageSummary(page=1, image_path="/p1.png", text="Seite 1"),
                PageSummary(page=2, image_path="/p2.png", text="Seite 2"),
            ],
            detected_part_names=["Rohr", "Winkel"],
        ),
        steps=[
            GuideStep(
                number=1,
                title="Rohr einstecken",
                text="Insert the pipe into the elbow until it bottoms out.",
                display_text="Rohr in den Winkel stecken",
                required_parts=["pipe_l500", "elbow_90"],
                source_pages=[1],
            )
        ],
        parts=[
            GuidePart(
                id="pipe_l500",
                name="Pipe L500",
                ports=[
                    PortInfo(
                        name="port_0", pos=[0, 0, 0], axis=[1, 0, 0], radius=11.0, gender="male"
                    )
                ],
            )
        ],
    )


def test_atomic_save_load_roundtrip(tmp_path: Path) -> None:
    s = _state()
    s.output_dir = str(tmp_path)
    path = s.save()
    assert path == tmp_path / "state.json"
    # tmp files are gone
    assert not list(tmp_path.glob(".state-*"))
    loaded = GuideState.load(tmp_path)
    assert loaded is not None
    assert loaded.document is not None
    assert loaded.document.page_count == 2
    assert loaded.steps[0].display_text == "Rohr in den Winkel stecken"
    assert loaded.parts[0].ports[0].radius == 11.0


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert GuideState.load(tmp_path) is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{not json")
    assert GuideState.load(tmp_path) is None


def test_mission_bookkeeping_resume(tmp_path: Path) -> None:
    s = _state()
    s.output_dir = str(tmp_path)
    s.mark_done("doc")
    s.mark_done("steps")
    s.mark_failed("assembly")
    s.save()
    loaded = GuideState.load(tmp_path)
    assert loaded is not None
    assert loaded.missions_done == ["doc", "steps"]
    assert loaded.missions_failed == ["assembly"]
    # re-doing a failed mission clears the failure
    loaded.mark_done("assembly")
    assert "assembly" in loaded.missions_done
    assert "assembly" not in loaded.missions_failed


def test_lookups() -> None:
    s = _state()
    assert s.part_by_id("pipe_l500") is not None
    assert s.part_by_id("nope") is None
    assert s.step_by_number(1) is not None
    assert s.step_by_number(99) is None
    s.assembly.instances.append(AssemblyInstance(part_id="pipe_l500", placed_in_step=1))
    assert s.placed_part_ids() == {"pipe_l500"}


def test_world_frame_from_mode() -> None:
    """The transform world frame is derived, never stored: cad/auto runs
    solve in Z-up mm, blender runs already carry the glTF Y-up frame."""
    assert GuideState(guide_mode="cad").world_frame == "z_up"
    assert GuideState(guide_mode="auto").world_frame == "z_up"
    assert GuideState(guide_mode="blender").world_frame == "y_up"
