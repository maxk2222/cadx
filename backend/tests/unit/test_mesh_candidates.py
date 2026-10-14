"""Candidate selection rules: heightfield only as rescue (T4.2) and
multiview role filtering (T4.4)."""

from __future__ import annotations

import uuid
from pathlib import Path

from cadx.agent.executor_client import InProcessExecutor
from cadx.agent.trace import MemoryTrace
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import ProjectSettings
from cadx.llm.fake import FakeLLM

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _ctx() -> AgentCtx:
    async def emit(step: str, it: int, msg: str, **_f) -> None:
        ctx_msgs.append(msg)

    async def no_cancel() -> bool:
        return False

    ctx_msgs: list[str] = []
    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=InProcessExecutor(),
        trace=MemoryTrace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="test",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        prompt="test",
        mode="generate",
        settings=ProjectSettings(clarification_mode="assume"),
    )
    ctx._test_msgs = ctx_msgs  # type: ignore[attr-defined]
    return ctx


async def test_select_candidate_viable_generator_kept_over_heightfield():
    from cadx.agent.orchestrator import _select_candidate

    ctx = _ctx()
    scores = {(1, "a"): 0.6}
    hf = ("x.glb", 0.75)
    seed = await _select_candidate(ctx, 1, scores, hf)
    assert seed == 1
    assert "_heightfield_rel" not in ctx.mesh_info
    assert any("organic detail wins" in m for m in ctx._test_msgs)  # type: ignore[attr-defined]


async def test_select_candidate_weak_generator_uses_heightfield():
    from cadx.agent.orchestrator import _select_candidate

    ctx = _ctx()
    scores = {(1, "a"): 0.2}
    hf = ("x.glb", 0.75)
    seed = await _select_candidate(ctx, 1, scores, hf)
    assert seed is None
    assert ctx.mesh_info["_heightfield_rel"] == "x.glb"


async def test_arrange_gen_images_drops_unlabelled_views():
    """Multiview adapter: 'other'-role images are not fed as view slots."""
    from cadx.agent.orchestrator import _arrange_gen_images
    from cadx.llm.schemas import PartSpec

    img = FIXTURES / "images" / "knob.png"
    imgs = [InputAsset(asset_id=f"a{i}", kind="image", path=img) for i in range(4)]
    roles = ["front", "other", "other", "back"]

    ctx = _ctx()

    class _Mesh:
        async def health(self):
            class H:
                device = "cpu"

                class M:
                    id = "mv"
                    available = True
                    max_images = 4
                    kind = "mesh"

                models = [M()]

            return H()

    ctx.mesh = _Mesh()
    spec = PartSpec(
        name="x",
        summary="",
        length_x={"value": 40.0, "source": "stated"},
        width_y={"value": 40.0, "source": "stated"},
        height_z={"value": 20.0, "source": "stated"},
    )
    model, out_imgs, out_roles = await _arrange_gen_images(
        ctx, 1, imgs, roles, spec, prefer="mv"
    )
    assert out_roles == ["front", "back"]
    assert len(out_imgs) == 2
    assert model == "mv"
