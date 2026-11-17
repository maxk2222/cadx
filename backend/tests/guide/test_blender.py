"""Blender-mode guide tests — client protocol, path bridge, scene
read-back, scene gate, and the mode-aware mission ordering.

No real Blender needed: the socket side is a thread-backed fake JSON
server; execute_code paths get stubbed clients.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadx.guide import blender_check, blender_client, blender_tools
from cadx.guide.blender_client import BlenderClient, BlenderError
from cadx.guide.state import AssemblyState, GuidePart, GuideState

# -- fake socket server ------------------------------------------------------ #


class _FakeServer:
    """Minimal BlenderMCP wire peer: one JSON blob in, one JSON blob out."""

    def __init__(self, handler=None, split_replies: int = 0, status: str = "success"):
        self.handler = handler or (lambda cmd: {"pong": True})
        self.status = status
        self.split_replies = split_replies  # chunk the reply into N writes
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.received: list[dict] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                self.sock.settimeout(0.2)
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    try:
                        cmd = json.loads(buf.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    self.received.append(cmd)
                    if self.status == "error":
                        reply = json.dumps(
                            {"status": "error", "message": str(self.handler(cmd))}
                        ).encode()
                    else:
                        result = self.handler(cmd)
                        reply = json.dumps(
                            {"status": "success", "result": result}
                        ).encode()
                    if self.split_replies > 1:
                        step = max(1, len(reply) // self.split_replies)
                        for i in range(0, len(reply), step):
                            conn.sendall(reply[i : i + step])
                    else:
                        conn.sendall(reply)
                    break

    def stop(self):
        self._stop.set()
        self.sock.close()
        self.thread.join(timeout=2)


@pytest.fixture()
def fake_server():
    servers = []
    yield lambda **kw: servers.append(_FakeServer(**kw)) or servers[-1]
    for s in servers:
        s.stop()


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    blender_client.reset_client()
    yield
    blender_client.reset_client()


# -- client protocol --------------------------------------------------------- #


def test_ping_and_command(fake_server):
    srv = fake_server(handler=lambda c: {"pong": True})
    c = BlenderClient("127.0.0.1", srv.port)
    assert c.ping() is True
    assert srv.received[0]["type"] == "ping"


def test_execute_code_returns_stdout(fake_server):
    srv = fake_server(handler=lambda c: {"executed": True, "result": "hello"})
    c = BlenderClient("127.0.0.1", srv.port)
    assert c.execute_code("print(1)") == "hello"
    assert srv.received[0]["params"]["code"] == "print(1)"


def test_split_reply_reassembled(fake_server):
    big = {"result": "x" * 3000}
    srv = fake_server(handler=lambda c: big, split_replies=8)
    c = BlenderClient("127.0.0.1", srv.port)
    assert c.command("whatever") == big


def test_error_status_raises(fake_server):
    srv = fake_server(handler=lambda c: "boom", status="error")
    c = BlenderClient("127.0.0.1", srv.port)
    with pytest.raises(BlenderError, match="boom"):
        c.command("ping")


def test_unreachable_raises_with_remediation():
    c = BlenderClient("127.0.0.1", 1)  # nothing listens on port 1
    with pytest.raises(BlenderError, match="unreachable"):
        c.command("ping", timeout_s=2.0)


# -- path bridge ------------------------------------------------------------- #


def test_path_bridge_roundtrip(tmp_path, monkeypatch):
    from cadx import storage

    host_root = tmp_path / "hostdata"
    host_root.mkdir()
    monkeypatch.setenv("CADX_DATA_HOST_ROOT", str(host_root))
    # worker sees /data as storage.data_root — emulate by pointing the
    # bridge at the same root (identity mapping in tests)
    monkeypatch.setattr(storage, "data_root", lambda: host_root)
    inside = host_root / "projects" / "p" / "guide" / "blender" / "x.glb"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"g")
    assert blender_client.to_host_path(inside) == str(inside)
    assert blender_client.to_worker_path(str(inside)) == inside
    outside = tmp_path / "elsewhere" / "x.glb"
    outside.parent.mkdir(parents=True)
    with pytest.raises(BlenderError, match="outside the data root"):
        blender_client.to_host_path(outside)


def test_path_bridge_requires_root(monkeypatch):
    monkeypatch.delenv("CADX_DATA_HOST_ROOT", raising=False)
    with pytest.raises(BlenderError, match="CADX_DATA_HOST_ROOT"):
        blender_client.to_host_path(Path("/data/anything"))


# -- scene read-back --------------------------------------------------------- #


def _sess(state_dir: Path, parts=None, steps=None) -> SimpleNamespace:
    state = GuideState(output_dir=str(state_dir))
    state.parts = parts or []
    state.steps = steps or []
    state.assembly = AssemblyState()
    return SimpleNamespace(state=state, state_dir=state_dir)


def _part(pid: str) -> GuidePart:
    return GuidePart(id=pid, name=pid, source="blender")


def test_build_instances_bindings():
    sess = _sess(Path("/tmp/x"), parts=[_part("pipe"), _part("elbow")])
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    objects = {
        "TubeA": {"part": "pipe", "inst": "", "step": 3, "premount": False,
                  "matrix": ident},
        "TubeB": {"part": "pipe", "inst": "pipe#9", "step": 0, "premount": True,
                  "matrix": ident},
        "Elbow.001": {"part": "elbow", "inst": "", "step": 4, "premount": False,
                      "matrix": ident},
        "Stray": {"part": "nope", "inst": "", "step": 1, "premount": False,
                  "matrix": ident},
        "BadMat": {"part": "pipe", "inst": "", "step": 1, "premount": False,
                   "matrix": [[0.0]]},
    }
    insts, per_step, counts, errors = blender_tools._build_instances(sess, objects)
    assert len(insts) == 3  # Stray (unknown part) + BadMat (matrix) rejected
    by_iid = {i.instance_id: i for i in insts}
    # auto-numbered per part in sorted object-name order
    assert "pipe#1" in by_iid and "pipe#9" in by_iid and "elbow#1" in by_iid
    assert by_iid["pipe#1"].placed_in_step == 3
    assert by_iid["pipe#9"].premounted is True  # explicit premount flag
    assert by_iid["elbow#1"].blender_object == "Elbow.001"
    assert per_step == {"3": ["pipe#1"], "4": ["elbow#1"]}
    assert counts == {"pipe": 2, "elbow": 1}
    assert len(errors) == 2
    assert any("nope" in e for e in errors)
    assert any("matrix_world" in e for e in errors)


# -- scene gate -------------------------------------------------------------- #


def _inst(iid: str, pid: str, x: float = 0.0, step: int = 1, premount=False):
    from cadx.guide.state import AssemblyInstance

    t = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    t[0][3], t[1][3], t[2][3] = x, 0.0, 0.0
    return AssemblyInstance(
        part_id=pid, instance_id=iid, transform=t,
        placed_in_step=step, premounted=premount,
    )


def _boxed_part(pid: str, size: float = 100.0) -> GuidePart:
    p = _part(pid)
    p.bbox_min = [-size / 2, -size / 2, 0.0]
    p.bbox_max = [size / 2, size / 2, size]
    return p


def test_overlap_fraction_and_world_aabb():
    part = _boxed_part("p", 100.0)
    a = _inst("p#1", "p", x=0.0)
    b = _inst("p#2", "p", x=75.0)  # 25mm of the 100mm box overlap
    aa = blender_check._world_aabb(a, part)
    bb = blender_check._world_aabb(b, part)
    frac = blender_check._overlap_fraction(aa, bb)
    assert 0.24 < frac < 0.26  # 25/100 on x
    far = _inst("p#3", "p", x=500.0)
    assert blender_check._overlap_fraction(aa, blender_check._world_aabb(far, part)) == 0.0


@pytest.mark.asyncio
async def test_scene_gate_blockers(tmp_path, monkeypatch):
    """Gate blocks: empty scene, unplaced registered part, degenerate
    transform, buried overlap — all as honest blockers."""
    from cadx.guide.state import GuideStep

    part = _boxed_part("casing")
    sess = _sess(tmp_path, parts=[part, _part("loose"), _part("stray")])
    sess._aliases_store = {}
    # 'loose' is required by a step -> missing instance blocks;
    # 'stray' is required by nothing -> orphan warning, emit-excluded
    sess.state.steps = [GuideStep(number=1, required_parts=["loose"])]

    class FakeClient:
        def execute_code(self, code, **kw):
            return '__CADX__{"Obj": {"part": "casing", "step": 1, "matrix": ' + \
                json.dumps([[1.0 if i == j else 0.0 for j in range(4)]
                            for i in range(4)]) + "}}"

    monkeypatch.setattr(blender_client, "get_client", lambda: FakeClient())
    blockers, manifest = await blender_check.run_scene_gate(sess)
    assert manifest["reachable"] is True
    # 'loose' is step-required but never placed
    assert any("loose" in b and "no tagged instance" in b for b in blockers)
    # 'stray' is an orphan registration — warning, not a blocker
    assert not any("stray" in b for b in blockers)
    assert manifest["orphan_parts"] == ["stray"]
    assert any("stray" in w for w in manifest["warnings"])
    assert "Obj" in manifest["objects"]


@pytest.mark.asyncio
async def test_scene_gate_unreachable(tmp_path, monkeypatch):
    sess = _sess(tmp_path)
    def boom():
        raise BlenderError("Blender MCP not reachable")
    monkeypatch.setattr(blender_client, "get_client", boom)
    blockers, manifest = await blender_check.run_scene_gate(sess)
    assert manifest["reachable"] is False
    assert any("unreachable" in b for b in blockers)


@pytest.mark.asyncio
async def test_scene_gate_buried_blocker(tmp_path, monkeypatch):
    """A non-premounted instance whose solid sits inside another part's
    solid blocks; bbox overlap alone only warns."""
    import trimesh

    big = _boxed_part("casing", 200.0)
    small = _boxed_part("nut", 10.0)
    for p, ext in ((big, (200, 200, 200)), (small, (10, 10, 10))):
        mesh = trimesh.creation.box(extents=ext)
        mp = tmp_path / f"{p.id}.glb"
        mesh.export(mp)
        p.mesh_path = str(mp)
        p.bbox_min = [-ext[0] / 2, -ext[1] / 2, -ext[2] / 2]
        p.bbox_max = [ext[0] / 2, ext[1] / 2, ext[2] / 2]
    sess = _sess(tmp_path, parts=[big, small])
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    scene = {
        "C": {"part": "casing", "step": 0, "premount": True, "matrix": ident},
        "N": {"part": "nut", "step": 1, "premount": False,
              "matrix": ident},  # same origin -> 100% inside casing
    }
    class FakeClient:
        def execute_code(self, code, **kw):
            return "__CADX__" + json.dumps(scene)
    monkeypatch.setattr(blender_client, "get_client", lambda: FakeClient())
    blockers, manifest = await blender_check.run_scene_gate(sess)
    assert any("solid" in b for b in blockers)
    assert manifest["overlaps"][0]["solid_fraction"] == 1.0


# -- mode plumbing ----------------------------------------------------------- #


def test_mission_order_modes():
    from cadx.guide.guide_runner import BLENDER_MISSION_ORDER, MISSION_ORDER, _mission_order

    s = GuideState(output_dir="/tmp/x")
    assert _mission_order(s) == MISSION_ORDER
    s.guide_mode = "blender"
    assert _mission_order(s) == BLENDER_MISSION_ORDER
    assert "build" in BLENDER_MISSION_ORDER
    assert "parts" not in BLENDER_MISSION_ORDER and "assembly" not in BLENDER_MISSION_ORDER


def test_guide_request_mode_validation():
    import uuid

    from cadx.api.schemas import GuideRequest

    r = GuideRequest(pdf_asset_id=uuid.uuid4(), mode="blender")
    assert r.mode == "blender"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GuideRequest(pdf_asset_id=uuid.uuid4(), mode="bogus")


def test_spawn_guide_stores_mode():
    import inspect as _inspect

    from cadx import spawn

    sig = _inspect.signature(spawn.spawn_guide)
    assert "mode" in sig.parameters


def test_build_tool_allowlist():
    from cadx.guide.tools import MISSION_TOOLS, merged_handlers, merged_schemas

    allow = set(MISSION_TOOLS["build"])
    schemas = merged_schemas()
    handlers = merged_handlers()
    for name in allow:
        assert name in schemas or name.startswith(("get_", "list_", "read_",
            "append_", "write_", "clear_", "add_", "ask_", "done_", "assign_")), name
    for bt in blender_tools.SCHEMAS:
        assert bt in schemas and bt in handlers


def test_unit_preamble_enforced(fake_server):
    """Every exec call carries the mm contract preamble automatically."""
    captured = []
    srv = fake_server(
        handler=lambda c: captured.append(c) or {"executed": True, "result": ""}
    )
    c = BlenderClient("127.0.0.1", srv.port)
    c.execute_code(blender_tools._UNIT_PREAMBLE + "pass")
    assert "scale_length" in captured[0]["params"]["code"]


def test_wait_for_file_byte_match(tmp_path):
    """exists() passing early must not fool the consumer — the gate waits
    for the exporter-reported byte count (partial-sync race)."""
    target = tmp_path / "part.glb"
    target.write_bytes(b"glTF")  # header visible, body not yet

    def _fill():
        time.sleep(0.3)
        target.write_bytes(b"glTF" + b"\x00" * 100)

    t = threading.Thread(target=_fill, daemon=True)
    t.start()
    assert blender_tools._wait_for_file(target, expected_bytes=104, timeout_s=3.0)
    t.join()


def test_wait_for_file_stability(tmp_path):
    """Without a byte count the gate accepts once the size holds steady."""
    target = tmp_path / "shot.png"
    assert not blender_tools._wait_for_file(target, timeout_s=0.5)
    target.write_bytes(b"pngdata")
    assert blender_tools._wait_for_file(target, timeout_s=2.0)


def test_sync_auto_registers_tagged_parts(tmp_path, monkeypatch):
    """Tag = binding contract: an unregistered cadx_part tag auto-exports
    as a part at sync — objects sharing cadx_inst group as the master."""
    sess = _sess(tmp_path, parts=[_part("casing")])
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    objects = {
        "CasingObj": {"part": "casing", "inst": "", "step": 0,
                      "premount": True, "matrix": ident},
        "ElbowM": {"part": "elbow", "inst": "elbow#1", "step": 3,
                   "premount": False, "matrix": ident},
        "ElbowFlange": {"part": "elbow", "inst": "elbow#1", "step": 3,
                        "premount": False, "matrix": ident},
        "CtxThing": {"part": "dichtring", "inst": "", "step": 0,
                     "premount": True, "matrix": ident},
    }
    sess.state.context_names = ["Dichtring"]

    async def _aw(_s):
        return objects

    monkeypatch.setattr(blender_tools, "_read_scene_objects", _aw)
    registered = {}

    async def _fake_export(s, pid, name, qty, names):
        registered[pid] = (name, qty, names)
        p = _part(pid)
        p.name = name
        p.quantity = qty
        s.state.parts.append(p)
        return p, [], None

    monkeypatch.setattr(blender_tools, "_export_register", _fake_export)

    import asyncio

    res = asyncio.run(blender_tools.blender_sync_scene(sess, {}))
    # casing already registered -> untouched; elbow auto-registered with
    # quantity=1 (one cadx_inst group of two objects); dichtring is
    # declared context -> collision error, not a registration
    assert registered.keys() == {"elbow"}
    _name, qty, names = registered["elbow"]
    assert qty == 1 and sorted(names) == ["ElbowFlange", "ElbowM"]
    assert res["ok"] is False  # the context collision is reported honestly
    assert "dichtring" in res["error"]
    assert res["auto_registered"] == ["elbow"]
    insts = {i.instance_id: i for i in sess.state.assembly.instances}
    assert "casing#1" in insts and "elbow#1" in insts


def test_overlap_solid_check_frees_socketed_pipe(tmp_path):
    """A thin pipe passing through an elbow's empty bbox corner is a
    joint, not a burial — only measured SOLID interpenetration blocks."""
    import trimesh

    elbow = trimesh.creation.annulus(r_min=8, r_max=12, height=20)  # hollow ring
    elbow_path = tmp_path / "elbow.glb"
    elbow.export(elbow_path)
    pipe_path = tmp_path / "pipe.glb"
    trimesh.creation.cylinder(radius=4, height=30).export(pipe_path)

    p_elbow = _part("elbow")
    p_elbow.mesh_path = str(elbow_path)
    p_elbow.bbox_min = [float(v) for v in elbow.bounds[0]]
    p_elbow.bbox_max = [float(v) for v in elbow.bounds[1]]
    p_pipe = _part("pipe")
    p_pipe.mesh_path = str(pipe_path)
    p_pipe.bbox_min = [-4.0, -4.0, -15.0]
    p_pipe.bbox_max = [4.0, 4.0, 15.0]

    # pipe centered in the ring's hollow — ~100% bbox overlap with the
    # elbow, ~0% solid interpenetration -> no blocker
    inst_e = _inst("elbow#1", "elbow", step=1)
    inst_p = _inst("pipe#1", "pipe", step=2)
    manifest = {"overlaps": [], "warnings": []}
    blockers = blender_check._overlap_scan(
        [inst_e, inst_p], {"elbow": p_elbow, "pipe": p_pipe}, manifest
    )
    assert blockers == []
    solid = manifest["overlaps"][0].get("solid_fraction")
    assert solid is not None and solid < blender_check._SOLID_BURY_WARN_FRACTION


def test_overlap_solid_check_flags_true_burial(tmp_path):
    import trimesh

    big_path = tmp_path / "big.glb"
    trimesh.creation.box(extents=(40, 40, 40)).export(big_path)
    small_path = tmp_path / "small.glb"
    trimesh.creation.box(extents=(20, 20, 20)).export(small_path)
    p_big, p_small = _part("big"), _part("small")
    p_big.mesh_path, p_small.mesh_path = str(big_path), str(small_path)
    p_big.bbox_min, p_big.bbox_max = [-20, -20, -20], [20, 20, 20]
    p_small.bbox_min, p_small.bbox_max = [-10, -10, -10], [10, 10, 10]

    manifest = {"overlaps": [], "warnings": []}
    blockers = blender_check._overlap_scan(
        [_inst("big#1", "big", step=1), _inst("small#1", "small", step=2)],
        {"big": p_big, "small": p_small},
        manifest,
    )
    assert len(blockers) == 1 and "small#1" in blockers[0]
    assert manifest["overlaps"][0]["solid_fraction"] == 1.0


def test_overlap_insertion_joint_warns_not_blocks(tmp_path):
    """An adapter socketed into a solid-modeled pipe interpenetrates
    ~67% of its volume — a correct insertion joint in rough-primitive
    modeling (the pipe has no hollow). Blocking it killed a real eval:
    the repair sessions applied the escape delta, saw the joint break,
    and reverted — twice, then 'unrepairable'. Deep overlap is review
    evidence, not a gate blocker; only near-invisibility blocks."""
    import trimesh

    pipe = trimesh.creation.cylinder(radius=6, height=60)
    pipe_path = tmp_path / "pipe.glb"
    pipe.export(pipe_path)
    adapter = trimesh.creation.cylinder(radius=5, height=30)
    adapter_path = tmp_path / "adapter.glb"
    adapter.export(adapter_path)

    p_pipe = _part("pipe")
    p_pipe.mesh_path = str(pipe_path)
    p_pipe.bbox_min = [float(v) for v in pipe.bounds[0]]
    p_pipe.bbox_max = [float(v) for v in pipe.bounds[1]]
    p_adapter = _part("adapter")
    p_adapter.mesh_path = str(adapter_path)
    p_adapter.bbox_min = [float(v) for v in adapter.bounds[0]]
    p_adapter.bbox_max = [float(v) for v in adapter.bounds[1]]

    inst_p = _inst("pipe#1", "pipe", step=1)
    inst_a = _inst("adapter#1", "adapter", step=2)
    inst_a.transform[2][3] = 25.0  # 20mm socketed into the pipe's top end

    manifest = {"overlaps": [], "warnings": []}
    blockers = blender_check._overlap_scan(
        [inst_p, inst_a], {"pipe": p_pipe, "adapter": p_adapter}, manifest
    )
    assert blockers == [], blockers
    solid = manifest["overlaps"][0].get("solid_fraction")
    assert (
        blender_check._SOLID_BURY_WARN_FRACTION
        <= solid
        < blender_check._SOLID_BURY_BLOCK_FRACTION
    ), solid
    assert any("adapter#1" in w for w in manifest["warnings"])


def test_checkpoint_nudge_counts_and_resets(tmp_path):
    warn, esc = blender_tools._CHECKPOINT_WARN, blender_tools._CHECKPOINT_ESCALATE
    sess = _sess(tmp_path)
    sess.tool_log = [{"name": "blender_exec"}] * (warn - 1)
    assert blender_tools._checkpoint_nudge(sess) is None
    sess.tool_log.append({"name": "blender_exec"})
    nudge = blender_tools._checkpoint_nudge(sess)
    assert nudge and str(warn) in nudge and "URGENT" not in nudge
    sess.tool_log += [{"name": "blender_exec"}] * (esc - warn)
    assert "URGENT" in blender_tools._checkpoint_nudge(sess)
    # a sync resets the counter — checkpoints are durable evidence
    sess.tool_log.append({"name": "blender_sync_scene"})
    sess.tool_log += [{"name": "blender_exec"}] * 10
    assert blender_tools._checkpoint_nudge(sess) is None


def test_forced_checkpoint_syncs_and_resets(tmp_path, monkeypatch):
    """80 unsaved calls -> blender_exec runs sync itself and records it;
    the tool result explains the checkpoint to the model."""
    sess = _sess(tmp_path, parts=[_part("pipe")])
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]

    class FakeClient:
        def execute_code(self, code, **kw):
            if "matrix_world" in code or "__CADX__" in code:
                return "__CADX__" + json.dumps(
                    {"TubeA": {"part": "pipe", "step": 1, "matrix": ident}}
                )
            return "print ok"

        def export_scene(self, path, **kw):
            Path(path).write_bytes(b"glb")

    monkeypatch.setattr(blender_client, "get_client", lambda: FakeClient())
    sess.tool_log = [{"name": "blender_exec"}] * (
        blender_tools._CHECKPOINT_ESCALATE + 5
    )

    res = asyncio.run(blender_tools.blender_exec(sess, {"code": "print('x')"}))
    assert "auto_checkpoint" in res
    # the forced sync logged itself -> counter resets -> next call only nudges late
    assert sess.tool_log[-1]["name"] == "blender_sync_scene"
    assert len(sess.state.assembly.instances) == 1
    res2 = asyncio.run(blender_tools.blender_exec(sess, {"code": "print('y')"}))
    assert "auto_checkpoint" not in res2


def test_touch_pairs_adjacency():
    """Face-contact pairs (zero overlap volume) still count as joints —
    the eps bridges them; a far instance is not a partner."""
    import numpy as np

    from cadx.guide import solver

    boxes = {
        "a#1": (np.array([0.0, 0, 0]), np.array([10.0, 10, 10])),
        "b#1": (np.array([10.5, 0, 0]), np.array([20.0, 10, 10])),
        "c#1": (np.array([100.0, 0, 0]), np.array([110.0, 10, 10])),
    }
    pairs = solver.touch_pairs(boxes, eps_mm=1.0)
    assert pairs == [("a#1", "b#1")]


def test_audit_blender_touch_graph_exempts_joints(tmp_path):
    """Without a declared joint graph the audit flagged every inserted
    part as buried and could never see disconnected clusters. Blender
    mode derives partners from physical contact: the socketed adapter is
    exempt, the floating part 400mm away is unreachable debris."""

    from cadx.guide import assembly_tools

    sess = _sess(tmp_path)
    sess.state.guide_mode = "blender"
    sess.state.parts = [
        _boxed_part("pipe", 100.0),
        _boxed_part("adapter", 20.0),
        _boxed_part("floaty", 20.0),
    ]
    sess.state.assembly.instances = [
        _inst("pipe#1", "pipe", x=0.0),
        _inst("adapter#1", "adapter", x=10.0),   # AABB inside pipe's box
        _inst("floaty#1", "floaty", x=400.0),    # touches nothing
    ]
    audit = assembly_tools.joint_geometry_audit(sess)
    assert audit["buried_instances"] == [], audit["buried_instances"]
    lost = {d["instance"] for d in audit["disconnected_instances"]}
    assert "floaty#1" in lost and "adapter#1" not in lost


def test_joint_pairs_cad_mode_keeps_declared_only(tmp_path):
    """The touch graph must not leak into cad mode — declared joints are
    the contract there; touching instances stay unpartnered."""
    import numpy as np

    from cadx.guide import assembly_tools

    boxes = {
        "a#1": (np.array([0.0, 0, 0]), np.array([10.0, 10, 10])),
        "b#1": (np.array([10.5, 0, 0]), np.array([20.0, 10, 10])),
    }
    sess = _sess(tmp_path)
    assert sess.state.guide_mode == "cad"
    assert assembly_tools._joint_pairs(sess, lambda x: x, boxes) == []
    sess.state.guide_mode = "blender"
    assert assembly_tools._joint_pairs(sess, lambda x: x, boxes) == [("a#1", "b#1")]
