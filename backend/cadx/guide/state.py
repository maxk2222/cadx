"""GuideState — the pipeline blackboard (ported from KIARA v4 state/models.py).

One JSON file per guide run, rewritten atomically after every
state-changing tool. The state dir is keyed by ``guide_version_id`` — NOT
job_id — because resuming a parked job spawns a NEW job row/id; the
version id is stable across resumes (masterplan §5.5).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from cadx.logging import get_logger

log = get_logger(__name__)

STATE_FILENAME = "state.json"


class PortInfo(BaseModel):
    """A CAD mating interface candidate (KIARA PortInfo)."""

    name: str
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # mm, part-local
    axis: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])  # unit, outward
    radius: float = 0.0  # mm — rim lip radius (bore mouth for sockets)
    gender: str = "unknown"  # male | female | flat | unknown
    kind: str = "cylinder"  # cylinder | plane | slot
    depth: float = 0.0  # measured bore depth inward (0 = solid/unknown)
    inner_radius: float = 0.0  # clear bore radius inside the port (0 = unmeasured)
    outer_radius: float = 0.0  # rim's outer lip (tube OD / collar OD; 0 = unmeasured)
    nested_in: str | None = None  # enclosing socket port name — a stub recessed
    # inside a boss/nut opening (union-fitting thread site), not a collar edge


class GuidePart(BaseModel):
    """One physical part in the guide (KIARA PartInfo)."""

    id: str  # slug, unique within the guide
    name: str
    source: str = "uploaded"  # uploaded | generated
    asset_id: str | None = None
    version_id: str | None = None  # generated Version id
    mesh_path: str | None = None  # normalized GLB (mm) for scene/solver
    step_path: str | None = None
    ports: list[PortInfo] = Field(default_factory=list)
    bbox_min: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    status: str = "registered"  # registered | analyzed | error
    error: str = ""


class GuideSubStep(BaseModel):
    """One atomic action inside a step (KIARA SubStep)."""

    action: str  # place | attach | orient | insert | fasten | check
    part: str = ""  # GuidePart.id or free label
    target: str = ""
    target2: str = ""  # second mate for bridging parts (elbow across two pipes)
    text: str = ""
    side: str = ""


class GuideStep(BaseModel):
    """One ordered instruction step (KIARA GuideStepDraft).

    Dual-text contract: ``text`` is the detailed internal description that
    drives downstream tools; ``display_text`` is the short AR-facing label.
    """

    number: int
    title: str = ""
    text: str = ""
    display_text: str = ""
    action_type: str = "assembly"  # assembly | check | safety | prep
    required_parts: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    safety_hint: str = ""
    media_image: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    sub_steps: list[GuideSubStep] = Field(default_factory=list)
    parts_delta: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
    """One rendered PDF page (KIARA PageSummary)."""

    page: int
    image_path: str = ""
    text: str = ""
    classification: str = "unknown"  # cover | toc | steps | parts_list | appendix
    figures: list[dict] = Field(default_factory=list)
    vlm_summary: str = ""


class DocumentIndex(BaseModel):
    """Prepared PDF artifact index (KIARA DocumentIndex)."""

    pdf_path: str = ""
    pdf_asset_id: str = ""
    dpi: int = 200
    page_count: int = 0
    language: str = "de"
    doc_type: str = "assembly_manual"
    pages: list[PageSummary] = Field(default_factory=list)
    detected_part_names: list[str] = Field(default_factory=list)


class AssemblyInstance(BaseModel):
    """One placed occurrence of a part (KIARA AssemblyInstance).

    A registered part may be placed several times (left/right sides,
    repeated fittings) — each placement is its own instance keyed by
    ``instance_id`` ("<part_id>#<n>")."""

    part_id: str
    instance_id: str = ""  # "<part_id>#<n>", unique among live instances
    transform: list[list[float]] = Field(default_factory=list)  # 4x4, mm, CAD frame
    placed_in_step: int = 0
    substep_idx: int = 0
    # editor overrides — rendering-only, keep pipeline palette semantics
    color: str | None = None
    opacity: float | None = None


class AssemblyState(BaseModel):
    instances: list[AssemblyInstance] = Field(default_factory=list)
    joints: list[dict] = Field(default_factory=list)
    step_instances: dict[str, list[str]] = Field(default_factory=dict)  # step -> instance_ids
    status: str = "empty"  # empty | partial | complete
    planner: dict = Field(default_factory=dict)  # deterministic pre-assembly report


class ReviewFinding(BaseModel):
    severity: str = "note"  # blocker | warning | note
    mission: str = ""  # mission that must fix it
    step: int | None = None
    text: str = ""
    evidence: str = ""


class ReviewReport(BaseModel):
    score: float = 0.0  # 0..10
    findings: list[ReviewFinding] = Field(default_factory=list)
    geometry_ok: bool | None = None
    iteration: int = 0


class GuideState(BaseModel):
    """The single source of truth — KIARA Blackboard equivalent."""

    project_id: str = ""
    guide_version_id: str = ""  # state-dir key — stable across job resumes
    language: str = "de"
    user_notes: list[str] = Field(default_factory=list)
    # agent-authored masternotes (tips/assumptions/extra info for the guide
    # user) — emitted as a project "note" asset at job end
    master_notes: list[str] = Field(default_factory=list)
    pending_answers: list[str] = Field(default_factory=list)
    # spawn-time flag: False = all parts must come from uploads
    # (generate_missing_part refuses; the agent may only alias)
    allow_part_generation: bool = True
    document: DocumentIndex | None = None
    steps: list[GuideStep] = Field(default_factory=list)
    parts: list[GuidePart] = Field(default_factory=list)
    assembly: AssemblyState = Field(default_factory=AssemblyState)
    review: ReviewReport = Field(default_factory=ReviewReport)
    annotations: list[dict] = Field(default_factory=list)
    # rewritten text variants — key "<audience>:<language>" (plus
    # "_original": the pre-rewrite snapshot), each a list of
    # {number,title,text,display_text}. The canonical steps always carry
    # the LAST applied variant so unity_guide.json stays current.
    variants: dict[str, list[dict]] = Field(default_factory=dict)
    missions_done: list[str] = Field(default_factory=list)
    missions_failed: list[str] = Field(default_factory=list)
    output_dir: str = ""
    unity_guide_path: str | None = None

    # -- persistence -------------------------------------------------------- #

    def save(self, state_dir: Path | None = None) -> Path:
        """Atomically write state.json (tmp + replace)."""
        base = state_dir or Path(self.output_dir)
        base.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=base)
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.model_dump_json(indent=1))
            tmp_path.replace(base / STATE_FILENAME)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
        return base / STATE_FILENAME

    @classmethod
    def load(cls, state_dir: Path) -> GuideState | None:
        path = state_dir / STATE_FILENAME
        if not path.exists():
            return None
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as e:
            # silent None would reset a half-built guide to scratch with no
            # trace — keep the corrupt file for diagnosis
            log.warning("guide.state_load_failed", dir=str(state_dir), err=str(e)[:300])
            return None

    # -- mission bookkeeping ------------------------------------------------- #

    def mark_done(self, mission: str) -> None:
        if mission not in self.missions_done:
            self.missions_done.append(mission)
        if mission in self.missions_failed:
            self.missions_failed.remove(mission)

    def mark_failed(self, mission: str) -> None:
        if mission not in self.missions_failed:
            self.missions_failed.append(mission)

    # -- lookups -------------------------------------------------------------- #

    def part_by_id(self, part_id: str) -> GuidePart | None:
        for p in self.parts:
            if p.id == part_id:
                return p
        return None

    def step_by_number(self, number: int) -> GuideStep | None:
        for s in self.steps:
            if s.number == number:
                return s
        return None

    def placed_part_ids(self) -> set[str]:
        return {i.part_id for i in self.assembly.instances}
