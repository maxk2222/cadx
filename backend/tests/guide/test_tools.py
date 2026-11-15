"""Tool registry + session dispatch tests — allowlists and schemas."""

from unittest.mock import MagicMock

import pytest

from cadx.guide import tools as gtools
from cadx.guide.session import SESSION_SCHEMAS, GuideToolSession
from cadx.guide.state import GuideState


def test_every_mission_tool_has_schema_and_dispatch() -> None:
    schemas = gtools.merged_schemas()
    handlers = gtools.merged_handlers()
    for mission, names in gtools.MISSION_TOOLS.items():
        for name in names:
            in_session = name in SESSION_SCHEMAS
            in_module = name in schemas
            assert in_session or in_module, f"{mission}: '{name}' has no schema"
            if in_module:
                assert name in handlers, f"{mission}: '{name}' has no handler"


def test_mission_allowlists_filter_tools() -> None:
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    sess = GuideToolSession(
        ctx,
        mission="doc",
        state=GuideState(),
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    doc_tools = {t["function"]["name"] for t in sess.tools()}
    assert "prepare_pdf" in doc_tools
    assert "create_step" not in doc_tools
    sess.mission = "assembly"
    asm_tools = {t["function"]["name"] for t in sess.tools()}
    assert "place_instance" in asm_tools and "solve_attachment_preview" in asm_tools
    assert "prepare_pdf" not in asm_tools


def test_place_instance_rejects_unknown_candidate() -> None:
    """The LLM cannot inject transforms — only server-issued candidate_ids."""
    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    from cadx.guide.state import GuidePart, GuideStep

    state.parts.append(GuidePart(id="a", name="A", mesh_path="/tmp/a.glb"))
    state.steps.append(GuideStep(number=1, text="t", display_text="d"))
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    from cadx.guide.assembly_tools import place_instance

    res = pytest.importorskip("asyncio").run(
        place_instance(sess, {"part_id": "a", "candidate_id": "invented", "step": 1})
    )
    assert res["ok"] is False
    assert "candidate" in res["error"]


def test_detect_language_respects_lock() -> None:
    """A wizard-pinned output language must survive doc-mission
    autodetect — only document.language records the manual's language."""
    from cadx.guide.doc_tools import detect_language
    from cadx.guide.state import DocumentIndex

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState(language="fr", language_locked=True)
    state.document = DocumentIndex()
    sess = GuideToolSession(
        ctx,
        mission="doc",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    import asyncio

    res = asyncio.run(detect_language(sess, {"language": "de"}))
    assert res["ok"] is True and res["locked"] is True
    assert state.language == "fr"  # output language untouched
    assert state.document.language == "fr"

    # unlocked: the detected language wins as before
    state2 = GuideState()
    state2.document = DocumentIndex()
    sess.state = state2
    res2 = asyncio.run(detect_language(sess, {"language": "en"}))
    assert res2["ok"] is True and "locked" not in res2
    assert state2.language == "en"
    assert state2.document.language == "en"


def test_done_guard_blocks_incomplete_steps() -> None:
    from cadx.guide.step_tools import HANDLERS  # noqa: F401 — registers guard

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    sess = GuideToolSession(
        ctx,
        mission="steps",
        state=GuideState(),
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    res = sess._tool_done_mission({"reasoning": "x"})
    assert res["ok"] is False


def test_match_required_suggests_renamed_fittings() -> None:
    """Manuals rename the same fitting between parts list and step text —
    match_required_parts must suggest registered candidates (category+size),
    never silently mark them missing."""
    from cadx.guide import part_tools
    from cadx.guide.state import DocumentIndex, GuidePart

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.document = DocumentIndex(
        detected_part_names=[
            "Elbow DN20, end 2x external",
            "Elbow DN20, end 1x external 1x internal",
            "Pipe DN20, l = 300 mm",
        ]
    )
    state.parts.extend(
        [
            GuidePart(
                id="elbow-ss",
                name="Elbow DN20 socket to spigot",
                mesh_path="/x/elbow_rn20_socket_to_spigot.glb",
            ),
            GuidePart(
                id="elbow-ff",
                name="Elbow DN20 socket to socket",
                mesh_path="/x/elbow_rn20_soket_to_socket.glb",
            ),
            GuidePart(id="pipe-300", name="Pipe DN20", mesh_path="/x/pipe_l300.glb"),
        ]
    )
    sess = GuideToolSession(
        ctx,
        mission="parts",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    import asyncio

    res = asyncio.run(part_tools.match_required_parts(sess, {}))
    assert res["required_count"] == 3
    elbow_suggs = {
        r["name"]: set(r.get("suggested_part_ids", []))
        for r in res["matches"]
        if "Elbow" in r["name"]
    }
    # both elbow naming variants suggest both registered elbows, never the pipe
    for s in elbow_suggs.values():
        assert s == {"elbow-ss", "elbow-ff"}
    pipe_row = next(r for r in res["matches"] if "Pipe" in r["name"])
    assert pipe_row["matched_part_id"] == "pipe-300"  # substring match, no suggestion needed

    # generation refused -> refusal also carries the same candidates
    state.allow_part_generation = False
    res2 = asyncio.run(
        part_tools.generate_missing_part(
            sess, {"part_id": "x", "name": "Elbow DN20, end 2x external", "description": "y" * 30}
        )
    )
    assert res2["ok"] is False
    assert set(res2["suggested_part_ids"]) == {"elbow-ss", "elbow-ff"}


def test_same_part_placed_twice_creates_instances() -> None:
    """Left/right sides and repeated fittings: one registered part must be
    placeable multiple times — each placement is its own instance."""
    from cadx.guide.assembly_tools import _completeness_issues, place_instance
    from cadx.guide.state import DocumentIndex, GuidePart, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.document = DocumentIndex(detected_part_names=["Adapter"])
    state.parts.append(GuidePart(id="adapter", name="Adapter DN20"))
    # two assembly steps each needing one adapter instance
    for n in (1, 2):
        state.steps.append(
            GuideStep(
                number=n,
                text="t",
                display_text="d",
                required_parts=["Adapter"],
                sub_steps=[{"action": "attach", "part": "Adapter", "target": "Gehäuse"}],
            )
        )
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    import asyncio

    cand = {"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    sess._preview_cache["c1"] = dict(cand)
    sess._preview_cache["c2"] = dict(cand)
    r1 = asyncio.run(place_instance(sess, {"part_id": "adapter", "candidate_id": "c1", "step": 1}))
    assert r1["ok"], r1
    r2 = asyncio.run(place_instance(sess, {"part_id": "adapter", "candidate_id": "c2", "step": 2}))
    assert r2["ok"], r2  # second placement of the SAME part must succeed
    ids = [i.instance_id for i in state.assembly.instances]
    assert ids == ["adapter#1", "adapter#2"]
    assert _completeness_issues(sess) == []
    # and removing only step 2 leaves step 1's instance untouched
    from cadx.guide.assembly_tools import reset_step

    asyncio.run(reset_step(sess, {"step": 2}))
    assert [i.instance_id for i in state.assembly.instances] == ["adapter#1"]


def test_preassembly_places_deterministically() -> None:
    """The planner resolves the step plan without any LLM: base part via
    env fallback, then the mating part attached onto its port."""
    import asyncio

    from cadx.guide.preassembly import run_preassembly
    from cadx.guide.state import GuidePart, GuideStep, GuideSubStep, PortInfo

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts += [
        GuidePart(
            id="base",
            name="Base",
            ports=[PortInfo(name="p0", pos=[0, 0, 10], axis=[0, 0, 1], radius=10, gender="female")],
        ),
        GuidePart(
            id="plug",
            name="Plug",
            ports=[PortInfo(name="m0", pos=[0, 0, 0], axis=[0, 0, -1], radius=10, gender="male")],
        ),
    ]
    state.steps += [
        GuideStep(
            number=1,
            text="Base aufstellen",
            display_text="d",
            required_parts=["Base"],
            sub_steps=[GuideSubStep(action="place", part="Base")],
        ),
        GuideStep(
            number=2,
            text="Plug einstecken",
            display_text="d",
            required_parts=["Plug"],
            sub_steps=[GuideSubStep(action="attach", part="Plug", target="Base, linke obere Seite")],
        ),
    ]
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,
        guide_assets=[],
        executor=None,
    )
    report = asyncio.run(run_preassembly(sess))
    insts = {i.instance_id for i in state.assembly.instances}
    assert insts == {"base#1", "plug#1"}, (insts, report)
    plug = next(i for i in state.assembly.instances if i.part_id == "plug")
    assert plug.transform[2][3] != 0.0  # attached ABOVE the base, not at origin
    assert state.assembly.joints[0]["moving_instance"] == "plug#1"
    assert state.assembly.planner["placed"] == 2

    # resume-safe: a second run skips the satisfied requirements
    report2 = asyncio.run(run_preassembly(sess))
    assert len(state.assembly.instances) == 2
    assert all(r["reason"] == "already satisfied" for r in report2["skipped"])


def test_emit_assembly_artifacts_merges_glb(tmp_path) -> None:
    """The finished assembly ships as one glb artifact — the version page
    viewer keys off kind=glb, so a guide with placed instances must produce
    exactly that."""
    import asyncio

    import numpy as np
    import trimesh

    from cadx.guide.guide_runner import _emit_assembly_artifacts
    from cadx.guide.state import AssemblyInstance, GuidePart

    mesh = tmp_path / "part.glb"
    trimesh.creation.box(extents=(10, 10, 10)).export(str(mesh))
    state = GuideState()
    state.parts.append(GuidePart(id="cube", name="Cube", mesh_path=str(mesh)))
    t = np.eye(4)
    t[0, 3] = 50.0
    state.assembly.instances.append(
        AssemblyInstance(
            part_id="cube", instance_id="cube#1", placed_in_step=1, transform=t.tolist()
        )
    )
    from cadx import storage

    arts = asyncio.run(
        _emit_assembly_artifacts(state, storage.data_root() / "tmp_test_emit", executor=None)
    )
    glb_rel = arts.get("export:glb")
    assert glb_rel, arts

    sc = trimesh.load(str(storage.resolve(glb_rel)))
    assert "cube#1" in sc.geometry
    assert sc.bounds[0][0] > 40.0  # transform applied


def test_emit_assembly_artifacts_empty(tmp_path) -> None:
    import asyncio

    from cadx import storage
    from cadx.guide.guide_runner import _emit_assembly_artifacts

    assert asyncio.run(
        _emit_assembly_artifacts(GuideState(), storage.data_root() / "tmp_test_emit2", executor=None)
    ) == {}


def test_preassembly_pairs_symmetric_targets_1to1() -> None:
    """Two placed adapters + two pipe attach-tasks must pair adapter#1↔pipe#1
    and adapter#2↔pipe#2 — never both pipes onto the newest adapter."""
    from types import SimpleNamespace

    import numpy as np

    from cadx.guide.preassembly import _pick_target_instance
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState

    state = GuideState()
    state.parts.append(GuidePart(id="adapter", name="Adapter"))
    state.parts.append(GuidePart(id="pipe", name="Rohr"))
    for i in (1, 2):
        state.assembly.instances.append(
            AssemblyInstance(
                part_id="adapter",
                instance_id=f"adapter#{i}",
                placed_in_step=i,
                transform=np.eye(4).tolist(),
            )
        )
    # pipe#1 already joined adapter#1 — the next pipe must take adapter#2
    state.assembly.joints.append(
        {
            "moving_instance": "pipe#1",
            "moving_part": "pipe",
            "fixed_instance": "adapter#1",
            "fixed_port": "rim_0",
        }
    )
    sess = SimpleNamespace(state=state, _aliases_store={})
    assert _pick_target_instance(sess, "Adapter", moving_pid="pipe") == "adapter#2"


def test_free_placement_spreads_beside() -> None:
    """No-hint env placement lays parts out in a row — never a tower."""
    import numpy as np

    from cadx.guide.solver import solve_free_placement
    from cadx.guide.state import GuidePart

    p = GuidePart(id="p", name="P", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    placed = [{"bbox_min_world": [0, 0, 0], "bbox_max_world": [10, 10, 10]}]
    t = np.asarray(solve_free_placement(p, placed, ""), dtype=float).reshape(4, 4)
    assert t[0, 3] > 5.0  # pushed +X beside the placed part
    assert t[2, 3] == 0.0  # resting on the same level — not on top


def test_add_arrow_resolves_forgiving_ids() -> None:
    """Arrows must accept instance ids, names and manual labels — a 4x
    error streak on resolvable ids killed live annotate runs."""
    import asyncio

    from cadx.guide.annotate_tools import add_arrow
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(
        GuidePart(id="pipe_l70", name="Rohr DN22, l = 70 mm", mesh_path="/tmp/a.glb")
    )
    state.steps.append(GuideStep(number=2, text="t", display_text="d"))
    state.assembly.instances.append(
        AssemblyInstance(instance_id="pipe_l70#1", part_id="pipe_l70", placed_in_step=2)
    )
    sess = GuideToolSession(
        ctx, mission="annotate", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,  # type: ignore[arg-type]
    )
    for pid in ("pipe_l70#1", "Rohr DN22", "pipe_l70"):
        res = asyncio.run(add_arrow(sess, {"step": "step 2", "part_id": pid, "direction": "down"}))
        assert res["ok"] is True, pid
        assert state.annotations[-1]["part_id"] == "pipe_l70"
    res = asyncio.run(add_arrow(sess, {"step": 99, "part_id": "pipe_l70", "direction": "down"}))
    assert res["ok"] is False and "valid steps" in res["error"]


def test_resolve_asset_rescues_hallucinated_uuid() -> None:
    """The LLM passed a stale/invented asset UUID 5x in a row and the
    parts mission died on error_streak — the resolver must fall back to
    an unambiguous filename token overlap from part_id+name."""
    from cadx.guide.part_tools import _resolve_asset
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuideState

    assets = [
        {"id": "aaa-1", "filename": "Casing.glb", "kind": "cad", "path": "/x/Casing.glb"},
        {"id": "bbb-2", "filename": "Adapter_rn22_internal_thread_to_plain.glb",
         "kind": "cad", "path": "/x/Adapter_rn22_internal_thread_to_plain.glb"},
        {"id": "ccc-3", "filename": "Elbow_90deg_RN22_socket_to_spigot.glb",
         "kind": "cad", "path": "/x/Elbow_90deg_RN22_socket_to_spigot.glb"},
        {"id": "ddd-4", "filename": "Elbow_90deg_RN22_soket_to_socket.glb",
         "kind": "cad", "path": "/x/Elbow_90deg_RN22_soket_to_socket.glb"},
    ]
    sess = GuideToolSession(
        MagicMock(), mission="parts", state=GuideState(), state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=assets, executor=None,
    )
    # exact id still wins
    assert _resolve_asset(sess, "aaa-1", part_id="x", name="y")["id"] == "aaa-1"
    # filename reference resolves
    assert _resolve_asset(sess, "Casing.glb", part_id="c", name="Casing")["id"] == "aaa-1"
    # hallucinated uuid + descriptive part fields → unambiguous filename match
    got = _resolve_asset(
        sess, "4f636566-9525-4d99-865a-9364d449219b",
        part_id="adapter_rn22_ig_plain", name="Adapter DN22, Innengewinde/Muffe",
    )
    assert got and got["id"] == "bbb-2"
    # ambiguous part fields (generic elbow name matches both) stay an error
    assert _resolve_asset(
        sess, "deadbeef-0000-1111-2222-333344445555",
        part_id="bogen", name="Rohrbogen",
    ) is None


@pytest.mark.asyncio
async def test_image_lookup_circuit_breaker() -> None:
    """Repeated invalid figure/page lookups short-circuit after the cap —
    the missing-part eval run burned its whole review mission on these."""
    from cadx.guide.session import _IMAGE_FAIL_CAP

    sess = GuideToolSession(
        MagicMock(), mission="review", state=GuideState(), state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[], executor=None,
    )
    for _ in range(_IMAGE_FAIL_CAP):
        res = await sess._dispatch(
            "get_figure_image", {"page": 99, "figure_index": 99}
        )
        assert res["ok"] is False
    assert sess._bad_image_calls == _IMAGE_FAIL_CAP
    res = await sess._dispatch("get_figure_image", {"page": 99, "figure_index": 99})
    assert res["ok"] is False
    assert "disabled" in res["error"]
    res = await sess._dispatch("get_page_image", {"page": 99})
    assert res["ok"] is False and "disabled" in res["error"]


def test_resolve_name_digraph_fold() -> None:
    """'Waermepumpe'/'Rohr DN 22, l = 70 mm' style labels resolve the
    umlaut/punctuation part names — the blocker pattern from eval run 2."""
    from types import SimpleNamespace

    from cadx.guide.state import GuidePart

    state = GuideState()
    state.parts = [
        GuidePart(id="casing", name="Wärmepumpe Gehäuse"),
        GuidePart(id="pipe_l70", name="Rohr DN22 l=70mm"),
    ]
    sess = SimpleNamespace(state=state, _aliases_store={})
    from cadx.guide.assembly_tools import _resolve_name

    assert _resolve_name(sess, "Waermepumpe") == "casing"
    assert _resolve_name(sess, "Wärmepumpe Gehäuse") == "casing"
    assert _resolve_name(sess, "Rohr DN 22, l = 70 mm") == "pipe_l70"


@pytest.mark.asyncio
async def test_solve_attachment_empty_ports_names_fix() -> None:
    """Empty port lists mean 'features not extracted' — the error must name
    get_part_features, not just echo empty lists."""
    from types import SimpleNamespace

    from cadx.guide.assembly_tools import solve_attachment_preview
    from cadx.guide.state import AssemblyInstance, GuidePart

    state = GuideState()
    state.parts = [
        GuidePart(id="pipe", name="Pipe", ports=[]),
        GuidePart(id="elbow", name="Elbow", ports=[]),
    ]
    state.assembly.instances = [
        AssemblyInstance(
            instance_id="elbow#1", part_id="elbow",
            transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
    ]
    sess = SimpleNamespace(state=state, _aliases_store={})
    res = await solve_attachment_preview(
        sess,
        {
            "part_id": "pipe",
            "target_part_id": "elbow#1",
            "moving_port": "rim_0",
            "target_port": "rim_1",
        },
    )
    assert res["ok"] is False
    assert "get_part_features" in res["error"]


def test_register_duplicate_part_id_upserts_quantity() -> None:
    """Re-registering the same part_id+asset is an idempotent update — the
    parts agent re-issues the call after reading the Stückliste and must
    not eat an error for it."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.part_tools import register_uploaded_part
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuidePart, GuideState

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(GuidePart(id="pipe", name="Rohr", asset_id="a1"))
    sess = GuideToolSession(
        ctx, mission="parts", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[{"id": "a1", "filename": "pipe.glb", "path": "/nope/pipe.glb"}],
        executor=None,
    )
    r = asyncio.run(
        register_uploaded_part(
            sess, {"part_id": "pipe", "name": "Rohr DN22", "asset_id": "a1", "quantity": 2}
        )
    )
    assert r["ok"] and r.get("updated") and state.parts[0].quantity == 2, r


def test_register_duplicate_asset_auto_aliases() -> None:
    """Same FILE under a new id — the agent's real intent is 'make this
    name resolve'; fulfilling it as an alias beats the error-streak death
    spiral observed in the bare eval."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.part_tools import register_uploaded_part
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuidePart, GuideState

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(GuidePart(id="pipe_l70", name="Rohr DN22, l = 70 mm", asset_id="a1"))
    sess = GuideToolSession(
        ctx, mission="parts", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[{"id": "a1", "filename": "pipe.glb", "path": "/nope/pipe.glb"}],
        executor=None,
    )
    r = asyncio.run(
        register_uploaded_part(
            sess, {"part_id": "rohr_70", "name": "Rohr 70mm", "asset_id": "a1"}
        )
    )
    assert r["ok"] and r["aliased_to"] == "pipe_l70" and "Rohr 70mm" in r["aliases_added"], r
    assert len(state.parts) == 1  # no duplicate part row
    from cadx.guide.part_tools import _norm

    assert sess._aliases_store.get(_norm("Rohr 70mm")) == "pipe_l70"


def test_place_instance_respects_quantity() -> None:
    """Kit quantity is a hard bound — the error must route the cause to
    the plan, not invite blind unplacing."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(GuidePart(id="elbow", name="Bogen", quantity=1))
    for n in (1, 2):  # the plan needs 2 copies; the kit ships 1
        state.steps.append(
            GuideStep(
                number=n, text="t", display_text="d",
                required_parts=["Bogen"],
                sub_steps=[{"action": "attach", "part": "Bogen", "target": ""}],
            )
        )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    cand = {"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    sess._preview_cache["c1"] = dict(cand)
    sess._preview_cache["c2"] = dict(cand)
    r1 = asyncio.run(place_instance(sess, {"part_id": "elbow", "candidate_id": "c1", "step": 1}))
    assert r1["ok"], r1
    # step 2 legitimately needs a copy — but the kit's single elbow is
    # already placed, so the kit bound fires before any step quota
    r2 = asyncio.run(
        place_instance(sess, {"part_id": "elbow", "candidate_id": "c2", "step": 2})
    )
    assert r2["ok"] is False and "ships only 1x" in r2["error"], r2


def test_place_instance_premount_lives_outside_kit_bound() -> None:
    """BOM "2 + 2 vormontiert" ships 2 kit copies AND 2 device-mounted —
    the planner's premount tasks must place instances 3/4 even though the
    kit bound (2) is already exhausted. Regression: the early kit check
    fired before the premount-aware cap and killed every premount with
    'ships only Nx' (masked as 'no candidate' in the planner report)."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(GuidePart(id="adapter", name="Adapter DN22", quantity=2))
    for n in (1, 2):
        state.assembly.instances.append(
            AssemblyInstance(
                part_id="adapter", instance_id=f"adapter#{n}",
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                placed_in_step=n, substep_idx=0,
            )
        )
        state.steps.append(
            GuideStep(
                number=n, text="t", display_text="d",
                required_parts=["Adapter DN22"],
                sub_steps=[{"action": "attach", "part": "Adapter DN22", "target": ""}],
            )
        )
    state.steps.append(
        GuideStep(
            number=5, text="t", display_text="d",
            required_parts=["Bogen"],
            sub_steps=[{"action": "insert", "part": "Bogen", "target": "vormontierter Adapter"}],
        )
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    tf = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, -500, 1]]
    sess._preview_cache["pm1"] = {"transform": tf, "collision_free": True}
    sess._preview_cache["pm2"] = {"transform": tf, "collision_free": True}
    sess._preview_cache["pm3"] = {"transform": tf, "collision_free": True}
    sess._premount_quota = {"adapter": 2}
    for cid, want_no in (("pm1", 3), ("pm2", 4)):
        r = asyncio.run(place_instance(sess, {
            "part_id": "adapter", "candidate_id": cid, "step": 5,
            "premount": True, "_premount_task": True,
        }))
        assert r["ok"], r
        inst = sess.state.assembly.instances[-1]
        assert inst.premounted and inst.instance_id == f"adapter#{want_no}"
    # the premount quota itself stays a hard bound
    r3 = asyncio.run(place_instance(sess, {
        "part_id": "adapter", "candidate_id": "pm3", "step": 5,
        "premount": True, "_premount_task": True,
    }))
    assert r3["ok"] is False and "premounted" in r3["error"], r3


def test_place_instance_rejects_candidate_on_wrong_target() -> None:
    """Plan says 'insert pipe into adapter' but the preview candidate mates
    to the casing (pipe fits its rim port too → buried-in-host geometry).
    place_instance must refuse and name the right fixed_part."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts += [
        GuidePart(id="casing", name="Gehäuse"),
        GuidePart(id="adapter", name="Adapter DN22"),
        GuidePart(id="pipe", name="Rohr 70"),
    ]
    state.assembly.instances.append(
        AssemblyInstance(
            part_id="adapter", instance_id="adapter#1",
            transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            placed_in_step=1, substep_idx=0,
        )
    )
    state.steps.append(
        GuideStep(
            number=2, text="t", display_text="d",
            required_parts=["Rohr 70"],
            sub_steps=[{"action": "insert", "part": "Rohr 70", "target": "Adapter DN22"}],
        )
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    tf = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # candidate mates pipe→casing (wrong: plan wants adapter)
    sess._preview_cache["bad"] = {
        "transform": tf, "moving_port": "p_out", "fixed_port": "rim_0",
        "fixed_part": "casing", "fixed_instance": "casing#1",
    }
    r_bad = asyncio.run(
        place_instance(sess, {"part_id": "pipe", "candidate_id": "bad", "step": 2})
    )
    assert r_bad["ok"] is False and "requires target" in r_bad["error"], r_bad
    assert "adapter" in r_bad["error"]
    # same candidate shape but mating pipe→adapter passes the target check
    sess._preview_cache["good"] = {
        "transform": tf, "moving_port": "p_out", "fixed_port": "sock",
        "fixed_part": "adapter", "fixed_instance": "adapter#1",
    }
    r_good = asyncio.run(
        place_instance(sess, {"part_id": "pipe", "candidate_id": "good", "step": 2})
    )
    assert r_good["ok"], r_good


def test_place_instance_skips_target_check_for_env_targets() -> None:
    """'linkes oberes Anschlussrohr' is a casing feature, not a registered
    part — the candidate's fixed_part cannot be verified and must pass."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts += [
        GuidePart(id="casing", name="Gehäuse"),
        GuidePart(id="adapter", name="Adapter DN22"),
    ]
    state.steps.append(
        GuideStep(
            number=1, text="t", display_text="d",
            required_parts=["Adapter DN22"],
            sub_steps=[{
                "action": "fasten", "part": "Adapter DN22",
                "target": "linkes oberes Anschlussrohr",
            }],
        )
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    sess._preview_cache["c"] = {
        "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "moving_port": "thr", "fixed_port": "rim_0",
        "fixed_part": "casing", "fixed_instance": "casing#1",
    }
    r = asyncio.run(
        place_instance(sess, {"part_id": "adapter", "candidate_id": "c", "step": 1})
    )
    assert r["ok"], r


def test_alias_part_rejects_contradictory_size_and_category() -> None:
    """'Rohr DN22 l=500mm' aliased onto the 70mm part silently corrupts
    every later resolution — the alias must rename the SAME part."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.part_tools import alias_part
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuidePart, GuideState

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(GuidePart(id="rohr_70", name="Rohr DN22, l = 70 mm"))
    sess = GuideToolSession(
        ctx, mission="parts", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    # different length marker → reject
    r = asyncio.run(
        alias_part(sess, {"part_id": "rohr_70", "alias": "Rohr DN22, l = 500 mm"})
    )
    assert r["ok"] is False and "size" in r["error"], r
    # elbow name onto a pipe part → reject
    r2 = asyncio.run(
        alias_part(sess, {"part_id": "rohr_70", "alias": "Rohrbogen DN22 Muffe"})
    )
    assert r2["ok"] is False and "elbow" in r2["error"], r2
    # same-size same-cat rename → allowed
    r3 = asyncio.run(
        alias_part(sess, {"part_id": "rohr_70", "alias": "kurzes Rohr DN22"})
    )
    assert r3["ok"], r3


def test_place_instance_rejects_buried_candidate() -> None:
    """A mate whose transform sinks the body inside a NON-partner instance is
    physically wrong even when port gap/normals pass — the long pipe must not
    slide through its fitting into the host behind it (observed: pipe_l500
    100% inside the casing AABB after preassembly)."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(
        GuidePart(id="casing", name="Gehäuse", bbox_min=[0, 0, 0], bbox_max=[400, 300, 600])
    )
    state.parts.append(
        GuidePart(id="adapter", name="Adapter", bbox_min=[0, 0, 0], bbox_max=[40, 40, 40])
    )
    state.parts.append(
        GuidePart(id="pipe", name="Rohr", bbox_min=[0, 0, 0], bbox_max=[25, 25, 500])
    )
    ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    state.assembly.instances.append(
        AssemblyInstance(part_id="casing", instance_id="casing#1", transform=ident)
    )
    state.assembly.instances.append(
        AssemblyInstance(
            part_id="adapter",
            instance_id="adapter#1",
            transform=[[1, 0, 0, 800], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
    )
    state.steps.append(
        GuideStep(
            number=1,
            text="t",
            display_text="d",
            required_parts=["Rohr"],
            sub_steps=[{"action": "insert", "part": "Rohr", "target": "Adapter"}],
        )
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    # candidate mated to the adapter (partner excluded from burial counting)
    # but the pipe AABB lies entirely INSIDE the casing box
    sess._preview_cache["cb"] = {
        "transform": [[1, 0, 0, 50], [0, 1, 0, 50], [0, 0, 1, 50], [0, 0, 0, 1]],
        "collision_free": True,
        "moving_port": "rim_0",
        "fixed_port": "rim_5",
        "fixed_part": "adapter",
        "fixed_instance": "adapter#1",
    }
    r = asyncio.run(
        place_instance(sess, {"part_id": "pipe", "candidate_id": "cb", "step": 1})
    )
    assert r["ok"] is False and "buries" in r["error"] and "casing#1" in r["error"], r
    # a candidate whose body stays outside passes the gate
    sess._preview_cache["cg"] = {
        "transform": [[1, 0, 0, 800], [0, 1, 0, 60], [0, 0, 1, 0], [0, 0, 0, 1]],
        "collision_free": True,
        "moving_port": "rim_0",
        "fixed_port": "rim_5",
        "fixed_part": "adapter",
        "fixed_instance": "adapter#1",
    }
    r2 = asyncio.run(
        place_instance(sess, {"part_id": "pipe", "candidate_id": "cg", "step": 1})
    )
    assert r2["ok"], r2


def test_create_step_rejects_duplicate_title() -> None:
    """Re-creating an identical-titled step is the observed degeneration
    (duplicate batches re-numbering the same actions) — the tool must point
    at the existing step instead of silently stacking duplicates."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuideState, GuideStep
    from cadx.guide.step_tools import create_step

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.steps.append(
        GuideStep(number=1, title="Adapter links schrauben", text="t", display_text="d")
    )
    sess = GuideToolSession(
        ctx, mission="steps", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    r = asyncio.run(
        create_step(
            sess,
            {"number": 5, "title": "Adapter links schrauben", "text": "t", "display_text": "d"},
        )
    )
    assert r["ok"] is False and "step 1" in r["error"], r
    # a different side keeps a distinct title → allowed
    r2 = asyncio.run(
        create_step(
            sess,
            {"number": 5, "title": "Adapter rechts schrauben", "text": "t", "display_text": "d"},
        )
    )
    assert r2["ok"], r2


def test_place_instance_rejects_swallowing_candidate() -> None:
    """Reverse-direction burial: a candidate whose AABB swallows an already
    placed foreign instance lands ON TOP of built hardware — the elbow that
    mates casing rim_1 while covering the adapter+pipe stack keeps valid
    port mates yet duplicates a branch at the same spot (observed: elbows
    buried 75% of adapters)."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import place_instance
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import AssemblyInstance, GuidePart, GuideState, GuideStep

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts.append(
        GuidePart(id="casing", name="Gehäuse", bbox_min=[0, 0, 0], bbox_max=[400, 300, 600])
    )
    state.parts.append(
        GuidePart(id="adapter", name="Adapter", bbox_min=[0, 0, 0], bbox_max=[30, 30, 30])
    )
    state.parts.append(
        GuidePart(id="elbow", name="Rohrbogen", bbox_min=[0, 0, 0], bbox_max=[60, 60, 60])
    )
    ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    state.assembly.instances.append(
        AssemblyInstance(part_id="casing", instance_id="casing#1", transform=ident)
    )
    # adapter placed OUTSIDE the casing box — the elbow candidate drops on it
    state.assembly.instances.append(
        AssemblyInstance(
            part_id="adapter",
            instance_id="adapter#1",
            transform=[[1, 0, 0, 500], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
    )
    state.steps.append(
        GuideStep(
            number=1,
            text="t",
            display_text="d",
            required_parts=["Rohrbogen"],
            sub_steps=[{"action": "insert", "part": "Rohrbogen", "target": "Gehäuse"}],
        )
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    # elbow candidate mated to the casing but its box covers adapter#1 fully
    sess._preview_cache["sw"] = {
        "transform": [[1, 0, 0, 485], [0, 1, 0, -15], [0, 0, 1, -15], [0, 0, 0, 1]],
        "collision_free": True,
        "moving_port": "rim_14",
        "fixed_port": "rim_1",
        "fixed_part": "casing",
        "fixed_instance": "casing#1",
    }
    r = asyncio.run(
        place_instance(sess, {"part_id": "elbow", "candidate_id": "sw", "step": 1})
    )
    assert r["ok"] is False and "lands on top of" in r["error"], r


def test_create_steps_upserts_existing_numbers() -> None:
    """A reopened steps mission resends the whole plan — each existing
    number must UPDATE (fields + fresh sub_steps), not error and leave a
    half-stacked plan (observed: 13→19 duplicated steps, then
    set_step_order fights)."""
    import asyncio
    from unittest.mock import MagicMock

    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import GuideState, GuideStep
    from cadx.guide.step_tools import create_steps

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    st = GuideStep(number=1, title="Adapter links", text="old", display_text="old")
    st.sub_steps.append(
        __import__("cadx.guide.state", fromlist=["GuideSubStep"]).GuideSubStep(
            action="fasten", part="Adapter", target="Gehäuse"
        )
    )
    state.steps.append(st)
    sess = GuideToolSession(
        ctx, mission="steps", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    r = asyncio.run(
        create_steps(
            sess,
            {
                "steps": [
                    {
                        "number": 1,
                        "title": "Adapter links",
                        "text": "new text body here",
                        "display_text": "new display",
                        "sub_steps": [
                            {"action": "insert", "part": "Rohr", "target": "Adapter"}
                        ],
                    },
                    {
                        "number": 2,
                        "title": "Rohr einstecken",
                        "text": "step two text body",
                        "display_text": "step two display",
                    },
                ]
            },
        )
    )
    assert r["created"] == [1, 2], r
    step = state.step_by_number(1)
    assert step.text == "new text body here"
    assert len(step.sub_steps) == 1 and step.sub_steps[0].part == "Rohr"
    assert len(state.steps) == 2


def test_generate_missing_part_rejects_premounted() -> None:
    """'Vormontiert' components are initial-state host geometry, never
    generated — the guard must fire on the label, not just the name."""
    import asyncio

    from cadx.guide.part_tools import generate_missing_part
    from cadx.guide.state import GuidePart

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.allow_part_generation = True
    state.parts.append(GuidePart(id="casing", name="Gehäuse", mesh_path="/tmp/c.glb"))
    sess = GuideToolSession(
        ctx,
        mission="parts",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    for args in (
        {"part_id": "anschlussrohr", "name": "Anschlussrohr (vormontiert)",
         "description": "straight pipe stub DN22 welded to the casing"},
        {"part_id": "verkleidung", "name": "Verkleidung",
         "description": "housing cover, premounted on the heat pump"},
    ):
        r = asyncio.run(generate_missing_part(sess, args))
        assert r["ok"] is False and "premounted" in r["error"], r


def test_repeat_reject_breaker() -> None:
    """Identical failing placement calls short-circuit after the cap —
    a verbatim retry can only ever return the same rejection."""
    import asyncio

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    sess = GuideToolSession(
        ctx,
        mission="assembly",
        state=state,
        state_dir="/tmp/x",
        project_id=None,  # type: ignore[arg-type]
        guide_assets=[],
        executor=None,
    )
    args = {"part_id": "ghost", "candidate_id": "nope"}
    for _ in range(3):
        r = asyncio.run(sess._dispatch("place_instance", dict(args)))
        assert r["ok"] is False and not r.get("_streak_exempt"), r
    r = asyncio.run(sess._dispatch("place_instance", dict(args)))
    assert r["ok"] is False and r["_streak_exempt"], r
    # different args are unaffected
    r2 = asyncio.run(sess._dispatch("place_instance", {"part_id": "ghost2"}))
    assert not r2.get("_streak_exempt"), r2


def test_alias_discriminativity_gate() -> None:
    """A generic family name ('anschlussrohr') aliased onto ONE of two
    same-kind parts is a silent guess — every later 'anschlussrohr'
    reference then binds the l500 even when the manual means the l70.
    The gate demands the alias carry the target's discriminator."""
    from cadx.guide.part_tools import _name_part_conflict

    sibs = [
        ("pipe_l500", "Rohr DN22, l = 500 mm", "Pipe_l500_mm.glb"),
        ("pipe_l70", "Rohr DN22, l = 70 mm", "Pipe_l70_mm.glb"),
    ]
    # generic compound → blocked
    hit = _name_part_conflict(
        "Anschlussrohr", "Rohr DN22, l = 500 mm", "pipe_l500", sibs
    )
    assert hit and "generic" in hit
    # qualified alias carries the discriminator → allowed
    assert _name_part_conflict(
        "oberes Anschlussrohr l=70", "Rohr DN22, l = 70 mm", "pipe_l70", sibs
    ) is None
    # manual-side feature names discriminate too — 'Muffe auf Spitzende'
    # is the BOM's own name for the spigot elbow
    esibs = [
        ("elbow_io", "Rohrbogen DN22, 1x aussen 1x innen", "Elbow_RN22_socket_to_spigot.glb"),
        ("elbow_oo", "Rohrbogen DN22, 2x aussen", "Elbow_RN22_socket_to_socket.glb"),
    ]
    assert _name_part_conflict(
        "Rohrbogen DN 22, Muffe auf Spitzende",
        "Rohrbogen DN22, 1x aussen 1x innen",
        "elbow_io",
        esibs,
    ) is None
    assert _name_part_conflict(
        "Rohrbogen", "Rohrbogen DN22, 1x aussen 1x innen", "elbow_io", esibs
    )
    # single-part category → no gate
    assert _name_part_conflict(
        "Verkleidung", "Gehäuse", "casing", [*sibs, ("casing", "Gehäuse", "C.glb")]
    ) is None


def test_sig_compound_rohr_is_pipe() -> None:
    from cadx.guide.part_tools import _sig

    assert _sig("Anschlussrohr")[0] == "pipe"
    assert _sig("Anschlussrohre")[0] == "pipe"
    assert _sig("Rohrbogen DN22")[0] == "elbow"  # elbow still wins


def test_declared_mates_audits_orient_substep_leg() -> None:
    """The l70 leg was declared on an `orient` sub-step (non-placement)
    with a clean target_pid — the elbow only jointed to l500. The audit
    must read EVERY sub-step naming the part, not only the placing one,
    or the dangling socket passes as a clean single mate."""
    from unittest.mock import MagicMock

    from cadx.guide.assembly_tools import _declared_mates
    from cadx.guide.session import GuideToolSession
    from cadx.guide.state import (
        AssemblyInstance,
        GuidePart,
        GuideState,
        GuideStep,
    )

    ctx = MagicMock()
    ctx.runtime_settings = MagicMock(guide_image_budget=4)
    state = GuideState()
    state.parts += [
        GuidePart(id="elbow_oo", name="Rohrbogen DN22 2x aussen"),
        GuidePart(id="pipe_l500", name="Rohr DN22 l=500mm"),
        GuidePart(id="pipe_l70", name="Rohr DN22 l=70mm"),
    ]
    state.steps.append(
        GuideStep(
            number=8, text="t", display_text="d",
            required_parts=["Rohrbogen DN22 2x aussen"],
            sub_steps=[
                {"action": "insert", "part": "Rohrbogen DN22 2x aussen",
                 "part_pid": "elbow_oo",
                 "target": "oberes Ende des linken vertikalen Rohrs",
                 "target_pid": "pipe_l500"},
                {"action": "orient", "part": "Rohrbogen DN22 2x aussen",
                 "part_pid": "elbow_oo",
                 "target": "linkes oberes Rohr DN22 l=70",
                 "target_pid": "pipe_l70"},
            ],
        )
    )
    state.assembly.instances += [
        AssemblyInstance(instance_id="elbow_oo#1", part_id="elbow_oo",
                         placed_in_step=8, substep_idx=0),
        AssemblyInstance(instance_id="pipe_l500#1", part_id="pipe_l500",
                         placed_in_step=7, substep_idx=0),
        AssemblyInstance(instance_id="pipe_l70#1", part_id="pipe_l70",
                         placed_in_step=3, substep_idx=0),
    ]
    state.assembly.joints.append(
        {"step": 8, "moving_instance": "elbow_oo#1",
         "fixed_instance": "pipe_l500#1"}
    )
    sess = GuideToolSession(
        ctx, mission="assembly", state=state, state_dir="/tmp/x",
        project_id=None, guide_assets=[], executor=None,
    )
    dm = _declared_mates(sess)
    miss = [m for m in dm["unfulfilled_mates"] if m["instance"] == "elbow_oo#1"]
    assert miss and miss[0]["missing"] == ["pipe_l70"], dm
