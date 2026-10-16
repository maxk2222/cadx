"""Eval runner: expected-feature checks and benchmark loading (T5.1/T5.2)."""

from cadx.eval.runner import _features_ok, load_benchmarks


def test_features_ok_none_when_no_expectations() -> None:
    assert _features_ok([], {"total_holes": 0}) is None


def test_features_ok_hole_count_and_dia() -> None:
    feats = {
        "total_holes": 3,
        "hole_groups": [
            {"diameter": 5.5, "count": 2},
            {"diameter": 22.2, "count": 1},
        ],
    }
    assert _features_ok([{"type": "hole", "count": 2, "dia": 5.5}], feats) is True
    assert _features_ok([{"type": "hole", "count": 3, "dia": 5.5}], feats) is False
    assert _features_ok([{"type": "hole", "count": 3}], feats) is True


def test_features_ok_countersink_fillet_chamfer() -> None:
    feats = {"countersinks": 3, "counterbores": 0, "rounds": [3.0], "chamfer_faces": 0}
    assert _features_ok([{"type": "countersink", "count": 3}], feats) is True
    assert _features_ok([{"type": "counterbore", "count": 1}], feats) is False
    assert _features_ok([{"type": "fillet"}], feats) is True
    assert _features_ok([{"type": "chamfer"}], feats) is False


def test_features_ok_missing_features_dict() -> None:
    assert _features_ok([{"type": "hole", "count": 1}], None) is False


def test_features_ok_tunnel_axis_and_dia() -> None:
    """Mesh-mode through-bores come from detect_mesh_features as MeshHole
    entries, not BRep hole_groups — a longitudinal grip bore must match."""
    feats = {
        "holes": [
            {"kind": "tunnel", "axis": "x", "x": 0.0, "y": 16.0, "dia": 21.98, "depth": 130.0},
            {"kind": "dimple", "axis": "z", "x": 3.0, "y": 2.0, "dia": 4.0, "depth": 1.1},
        ]
    }
    assert _features_ok([{"type": "tunnel", "axis": "x", "dia": 22.2}], feats) is True
    assert _features_ok([{"type": "tunnel", "axis": "z", "dia": 22.2}], feats) is False
    assert _features_ok([{"type": "tunnel", "axis": "x", "dia": 8.0}], feats) is False
    assert _features_ok([{"type": "tunnel", "axis": "x", "dia": 22.2, "count": 2}], feats) is False
    # a dimple must never satisfy a tunnel expectation
    assert _features_ok([{"type": "tunnel", "axis": "z", "dia": 4.0}], feats) is False


def test_features_ok_hole_axis_filter() -> None:
    """A hole group on the wrong axis must not satisfy an axis-pinned entry."""
    feats = {
        "total_holes": 2,
        "hole_groups": [
            {"diameter": 5.5, "count": 2, "axis": "+z", "centers": [[-15, 0], [15, 0]]},
            {"diameter": 5.5, "count": 1, "axis": "+y", "centers": [[0, 0]]},
        ],
    }
    assert _features_ok([{"type": "hole", "count": 2, "dia": 5.5, "axis": "z"}], feats) is True
    assert _features_ok([{"type": "hole", "count": 3, "dia": 5.5, "axis": "z"}], feats) is False
    # the y-axis group alone cannot satisfy a y-axis expectation of 2
    assert _features_ok([{"type": "hole", "count": 2, "dia": 5.5, "axis": "y"}], feats) is False


def test_features_ok_hole_positions() -> None:
    feats = {
        "total_holes": 2,
        "hole_groups": [
            {"diameter": 5.5, "count": 2, "axis": "+z", "centers": [[-15, 0], [15, 0]]},
        ],
    }
    entry = {"type": "hole", "count": 2, "positions": [[-15, 0], [15, 0]]}
    assert _features_ok([entry], feats) is True
    entry2 = {"type": "hole", "count": 2, "positions": [[-15, 0], [0, 0]]}
    assert _features_ok([entry2], feats) is False


def test_features_ok_hole_pcd() -> None:
    import math

    r = 31.0
    centers = [[r * math.cos(math.pi / 3 * i), r * math.sin(math.pi / 3 * i)] for i in range(6)]
    feats = {
        "total_holes": 6,
        "hole_groups": [{"diameter": 6.6, "count": 6, "axis": "+z", "centers": centers}],
    }
    assert _features_ok([{"type": "hole", "count": 6, "dia": 6.6, "pcd": 62}], feats) is True
    assert _features_ok([{"type": "hole", "count": 6, "dia": 6.6, "pcd": 40}], feats) is False
    # fewer holes than the circle needs must fail even if dia matches
    feats["hole_groups"][0]["centers"] = centers[:3]
    feats["hole_groups"][0]["count"] = 3
    assert _features_ok([{"type": "hole", "count": 6, "dia": 6.6, "pcd": 62}], feats) is False


def test_features_ok_slot_count() -> None:
    assert _features_ok([{"type": "slot", "count": 1}], {"slots": 1}) is True
    assert _features_ok([{"type": "slot", "count": 1}], {"slots": 0}) is False


def test_load_benchmarks_parses_new_fields() -> None:
    bms = {b.name: b for b in load_benchmarks()}
    lb = bms["l_bracket_cad"]
    assert lb.gen_mode == "cad"
    assert lb.expected_features[0]["type"] == "hole"
    assert lb.expect_watertight is True
    fl = bms["flange_pcd_cad"]
    assert fl.gen_mode == "cad"
    assert fl.real_only is True
    assert any(e.get("pcd") for e in fl.expected_features)
    fs = bms["firesteel_auto"]
    assert fs.max_prompt_tokens == 4_000_000
    assert len(fs.images) == 3
    assert bms["knife_scale_hybrid"].gen_mode == "hybrid"
    assert bms["grip_mesh"].expect_watertight is True


async def test_run_benchmark_uses_runtime_budget_and_reports_usage(monkeypatch) -> None:
    """Eval must mirror production wiring: runtime token budget, explicit
    max_iterations, and usage must show up even when the run fails."""
    import cadx.eval.runner as runner
    from cadx.agent.orchestrator import RunOutcome
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    captured = {}

    async def fake_rt():
        return RuntimeSettings(agent_max_tokens_per_job=42_000)

    async def fake_gen(ctx):
        captured["ctx"] = ctx
        await ctx.trace.record(
            step="intake",
            iteration=0,
            model="m",
            messages=[],
            response_text="",
            parsed=None,
            prompt_tokens=5,
            completion_tokens=2,
            latency_ms=9,
        )
        out = RunOutcome()
        out.status = "failed"
        return out

    monkeypatch.setattr("cadx.runtime_config.runtime_from_db_or_default", fake_rt)
    monkeypatch.setattr(runner, "run_generation", fake_gen)
    llm = FakeLLM([])
    llm.total_usage.update({"prompt_tokens": 777, "completion_tokens": 3, "calls": 2})
    bm = runner.Benchmark(
        name="fail_test", prompt="box", fake_mesh=True, settings={"max_iterations": 2}
    )
    r = await runner.run_benchmark(bm, llm=llm)
    assert captured["ctx"].budget_tokens == 42_000
    assert captured["ctx"].explicit_max_iterations is True
    assert r.prompt_tokens == 777
    assert r.calls == 2
    assert r.step_stats["intake"]["prompt_tokens"] == 5
    assert r.execs == 0


async def test_run_benchmark_dumps_trace_on_crash(monkeypatch, tmp_path) -> None:
    """A crashing benchmark must still leave its trace on disk (postmortem)."""
    import cadx.eval.runner as runner
    from cadx import storage

    async def boom(ctx):
        await ctx.trace.record(
            step="intake",
            iteration=0,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            response_text="",
            parsed=None,
            prompt_tokens=1,
            completion_tokens=0,
            latency_ms=1,
        )
        raise RuntimeError("simulated benchmark crash")

    monkeypatch.setattr(runner, "run_generation", boom)
    bm = runner.Benchmark(name="crash_test", prompt="box", fake_mesh=True)
    try:
        await runner.run_benchmark(bm)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    # trace.json landed in the version dir created by _dump_trace
    hits = list(storage.data_root().glob("projects/*/versions/*/trace.json"))
    assert hits, "no trace.json dumped"

    import json

    recs = json.loads(hits[0].read_text())
    assert recs and recs[0]["step"] == "intake"
