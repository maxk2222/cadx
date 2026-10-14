"""Agent image integration: hero-image autogen + generate_concept_image tool."""

from __future__ import annotations

import uuid

from sqlmodel import select

from cadx.agent.image_client import FakeImageClient
from cadx.agent.orchestrator import _autogen_hero_image
from cadx.agent.toolspecs import ToolSession
from cadx.agent.types import AgentCtx, InputAsset
from cadx.db.models import Asset, AssetKind, Project, ProjectSettings
from cadx.db.session import create_all, session_scope
from cadx.llm.fake import FakeLLM
from cadx.llm.schemas import ModelingPlan, PartSpec
from cadx.runtime_config import RuntimeSettings


class _Exec:
    async def execute(self, payload):  # pragma: no cover - unused here
        return {"ok": True}


class _Trace:
    async def record(self, **kw):  # pragma: no cover
        pass


def _ctx(**kw) -> AgentCtx:
    emitted: list[tuple[str, str]] = []

    async def emit(step: str, it: int, msg: str, **_f) -> None:
        emitted.append((step, msg))

    async def no_cancel() -> bool:
        return False

    kw.setdefault("runtime_settings", RuntimeSettings())
    ctx = AgentCtx(
        llm=FakeLLM([]),
        executor=_Exec(),
        trace=_Trace(),
        emit=emit,
        check_cancel=no_cancel,
        job_id="imgtest",
        project_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        mode="generate",
        prompt="a fantasy bottle opener shaped like a dragon",
        settings=ProjectSettings(),
        **kw,
    )
    ctx._emitted = emitted  # type: ignore[attr-defined]
    return ctx


def _plan(approach="mesh", subject="dragon bottle opener") -> ModelingPlan:
    return ModelingPlan(steps=["generate mesh"], approach=approach, mesh_subject=subject)


def _spec() -> PartSpec:
    return PartSpec(
        name="dragon bottle opener",
        length_x={"value": 120.0, "unit": "mm", "source": "inferred"},
        width_y={"value": 60.0, "unit": "mm", "source": "inferred"},
        height_z={"value": 40.0, "unit": "mm", "source": "inferred"},
        open_questions=[],
    )


async def _mk_project(project_id: uuid.UUID) -> None:
    await create_all()
    async with session_scope() as db:
        db.add(Project(id=project_id, name="t", description=""))
        await db.commit()


async def test_hero_autogen_for_mesh_mode(tmp_path):
    ctx = _ctx(gen_mode="mesh", image=FakeImageClient(), image_available=True)
    await _mk_project(ctx.project_id)
    await _autogen_hero_image(ctx, _spec(), _plan())

    gen = [a for a in ctx.assets if a.kind == "generated_image"]
    assert len(gen) == 1
    assert gen[0].path.exists()
    assert len(ctx.generated_asset_ids) == 1

    async with session_scope() as db:
        rows = (
            (await db.execute(select(Asset).where(Asset.project_id == ctx.project_id)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    a = rows[0]
    assert a.kind == AssetKind.generated_image
    assert a.gen_params["source"] == "hero_autogen"
    assert "dragon bottle opener" in a.gen_params["subject"]
    assert a.gen_params["model"] == "fake-image"


async def test_hero_autogen_skips_when_images_exist():
    ctx = _ctx(gen_mode="mesh", image=FakeImageClient(), image_available=True)
    from pathlib import Path

    from cadx.agent.types import InputAsset

    ctx.assets.append(InputAsset(asset_id="x", kind="image", path=Path("/tmp/x.png")))
    await _autogen_hero_image(ctx, _spec(), _plan())
    assert not ctx.generated_asset_ids
    assert not ctx.image.calls


async def test_hero_autogen_skips_for_cad():
    ctx = _ctx(gen_mode="cad", image=FakeImageClient(), image_available=True)
    await _autogen_hero_image(ctx, _spec(), _plan("cad"))
    assert not ctx.image.calls


async def test_hero_autogen_skips_when_disabled():
    ctx = _ctx(
        gen_mode="mesh",
        image=FakeImageClient(),
        image_available=True,
        runtime_settings=RuntimeSettings(image_autogen_mesh=False),
    )
    await _autogen_hero_image(ctx, _spec(), _plan())
    assert not ctx.image.calls


async def test_hero_autogen_skips_without_image_client():
    ctx = _ctx(gen_mode="mesh")
    await _autogen_hero_image(ctx, _spec(), _plan())  # must not raise


async def test_hero_autogen_failure_degrades(tmp_path):
    class FailClient(FakeImageClient):
        async def generate(self, req, on_progress, should_abort=None):
            raise RuntimeError("VRAM exhausted")

    ctx = _ctx(gen_mode="mesh", image=FailClient(), image_available=True)
    await _autogen_hero_image(ctx, _spec(), _plan())
    assert not ctx.generated_asset_ids
    assert any("failed" in m for _s, m in ctx._emitted)  # type: ignore[attr-defined]


async def test_concept_image_tool(tmp_path):
    ctx = _ctx(
        gen_mode="mesh",
        image=FakeImageClient(),
        image_available=True,
        runtime_settings=RuntimeSettings(image_tool_budget=2),
    )
    await _mk_project(ctx.project_id)
    session = ToolSession(ctx, kind="info")
    names = {t["function"]["name"] for t in session.tools()}
    assert "generate_concept_image" in names

    res = await session.execute(
        "generate_concept_image", {"prompt": "organic dragon opener", "count": 2}
    )
    assert res["ok"], res
    assert len(res["assets"]) == 2
    assert len(res["_images"]) == 2
    assert res["calls_left"] == 1
    assert len(ctx.generated_asset_ids) == 2
    # generated images become image inputs
    assert sum(1 for a in ctx.assets if a.kind == "generated_image") == 2


async def test_concept_image_tool_budget(tmp_path):
    ctx = _ctx(
        image=FakeImageClient(),
        image_available=True,
        runtime_settings=RuntimeSettings(image_tool_budget=1),
    )
    await _mk_project(ctx.project_id)
    session = ToolSession(ctx, kind="info")
    res = await session.execute("generate_concept_image", {"prompt": "a"})
    assert res["ok"]
    res = await session.execute("generate_concept_image", {"prompt": "b"})
    # after the first call the tool leaves the schema (image_left == 0) →
    # the second call is rejected without hitting the generator
    assert not res["ok"]
    assert len(ctx.image.calls) == 1
    assert len(ctx.generated_asset_ids) == 1


async def test_concept_image_tool_hidden_without_client():
    ctx = _ctx()  # no image client
    session = ToolSession(ctx, kind="info")
    names = {t["function"]["name"] for t in session.tools()}
    assert "generate_concept_image" not in names


def _ref_asset(tmp_path) -> InputAsset:
    p = tmp_path / "ref.png"
    from PIL import Image

    Image.new("RGB", (32, 32), (200, 10, 10)).save(p)
    return InputAsset(asset_id=str(uuid.uuid4()), kind="image", path=p)


async def test_concept_image_tool_with_reference(tmp_path):
    ctx = _ctx(
        image=FakeImageClient(),
        image_available=True,
        runtime_settings=RuntimeSettings(image_tool_budget=2),
    )
    await _mk_project(ctx.project_id)
    ref = _ref_asset(tmp_path)
    ctx.assets.append(ref)
    session = ToolSession(ctx, kind="info")

    res = await session.execute(
        "generate_concept_image",
        {
            "prompt": "make it matte black",
            "reference_asset_id": ref.asset_id,
            "strength": 0.4,
            "mark_sketch": True,
        },
    )
    assert res["ok"], res
    call = ctx.image.calls[0]
    assert call["init_image"] == str(ref.path)
    # new asset derives from the reference and carries the sketch hint
    gen = [a for a in ctx.assets if a.kind == "generated_image"]
    assert len(gen) == 1 and gen[0].role_hint == "sketch"
    async with session_scope() as db:
        row = await db.get(Asset, uuid.UUID(gen[0].asset_id))
    assert row is not None
    assert row.derived_from == uuid.UUID(ref.asset_id)
    assert row.gen_params["edit"] is True
    assert row.gen_params["role_hint"] == "sketch"
    assert row.gen_params["strength"] == 0.4


async def test_concept_image_tool_bad_reference_keeps_budget(tmp_path):
    ctx = _ctx(
        image=FakeImageClient(),
        image_available=True,
        runtime_settings=RuntimeSettings(image_tool_budget=1),
    )
    await _mk_project(ctx.project_id)
    session = ToolSession(ctx, kind="info")
    res = await session.execute(
        "generate_concept_image",
        {"prompt": "x", "reference_asset_id": str(uuid.uuid4())},
    )
    assert not res["ok"]
    assert res["calls_left"] == 1  # invalid ref must not burn the budget
    assert not ctx.image.calls
