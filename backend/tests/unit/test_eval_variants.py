"""M12: eval --variant parsing, per-variant output dirs, compare aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadx.eval import compare
from cadx.eval.runner import parse_variant, run_out_dir

# --- parse_variant -----------------------------------------------------------


def test_parse_variant_ok() -> None:
    name, overrides = parse_variant('thinking_off={"llm_thinking":"off"}')
    assert name == "thinking_off"
    assert overrides == {"llm_thinking": "off"}


def test_parse_variant_empty_object_is_valid() -> None:
    name, overrides = parse_variant("base={}")
    assert name == "base"
    assert overrides == {}


def test_parse_variant_multikey() -> None:
    name, overrides = parse_variant(
        'off={"llm_thinking":"off","agent_plan_mode":"lite","exec_timeout_s":120}'
    )
    assert name == "off"
    assert overrides["exec_timeout_s"] == 120


@pytest.mark.parametrize(
    "spec",
    [
        "noequals",  # missing =<json>
        "off=",  # empty json
        "off=notjson",  # malformed json
        "off=[1,2]",  # not an object
        'off="x"',  # not an object
        "bad name={}",  # whitespace in name
        "a/b={}",  # path separator in name
        "default={}",  # reserved
    ],
)
def test_parse_variant_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_variant(spec)


def test_parse_variant_unknown_key_names_it() -> None:
    with pytest.raises(ValueError, match=r"unknown RuntimeSettings keys.*no_such_knob"):
        parse_variant('x={"no_such_knob": 1}')


def test_parse_variant_invalid_value() -> None:
    with pytest.raises(ValueError, match="invalid values"):
        parse_variant('x={"llm_thinking": "sometimes"}')


# --- output dir layout -------------------------------------------------------


def test_run_out_dir_variant_layout(tmp_path: Path) -> None:
    assert (
        run_out_dir(tmp_path, "plate_holes", "thinking_off", "abc")
        == tmp_path / "plate_holes" / "thinking_off" / "abc"
    )
    # un-named default keeps the flat <bench>/<run_id>/ layout
    assert run_out_dir(tmp_path, "plate_holes", None, "abc") == tmp_path / "plate_holes" / "abc"


def _result_jsons(root: Path, pattern: str) -> list[Path]:
    """Sync glob helper — keeps ASYNC240 out of the async tests."""
    return sorted(root.glob(pattern))


# --- run_benchmark integration (fake orchestrator) ----------------------------


async def test_run_benchmark_merges_variant_under_benchmark_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Variant sets defaults; benchmark-local `runtime` pins win on conflict —
    and the run summary records variant + effective overrides."""
    import cadx.eval.runner as runner
    from cadx.agent.orchestrator import RunOutcome
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    captured: dict = {}

    async def fake_rt() -> RuntimeSettings:
        return RuntimeSettings()

    async def fake_gen(ctx) -> RunOutcome:
        captured["ctx"] = ctx
        out = RunOutcome()
        out.status = "succeeded"
        out.score = 90
        return out

    monkeypatch.setattr("cadx.runtime_config.runtime_from_db_or_default", fake_rt)
    monkeypatch.setattr(runner, "run_generation", fake_gen)
    # isolate the shared session DATA_DIR — run_benchmark also dumps trace.json
    monkeypatch.setattr("cadx.storage.data_root", lambda: tmp_path / "data")
    bm = runner.Benchmark(
        name="v_test",
        prompt="box",
        fake_mesh=True,
        runtime={"agent_plan_mode": "full", "llm_thinking": "always"},
    )
    r = await runner.run_benchmark(
        bm,
        llm=FakeLLM([]),
        variant="off",
        variant_overrides={"llm_thinking": "off", "agent_plan_mode": "lite"},
        eval_root=tmp_path,
    )
    rt = captured["ctx"].runtime_settings
    # benchmark-local keys win both conflicts
    assert rt.llm_thinking == "always"
    assert rt.agent_plan_mode == "full"
    assert r.variant == "off"

    hits = _result_jsons(tmp_path, "v_test/off/*/result.json")
    assert len(hits) == 1
    meta = json.loads(hits[0].read_text())
    assert meta["benchmark"] == "v_test"
    assert meta["variant"] == "off"
    assert meta["variant_overrides"] == {"llm_thinking": "off", "agent_plan_mode": "lite"}
    # effective overrides = variant merged under the benchmark's runtime
    assert meta["runtime_overrides"] == {
        "llm_thinking": "always",
        "agent_plan_mode": "full",
    }
    assert meta["result"]["status"] == "succeeded"
    assert meta["result"]["variant"] == "off"


async def test_run_benchmark_default_variant_flat_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cadx.eval.runner as runner
    from cadx.agent.orchestrator import RunOutcome
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    async def fake_rt() -> RuntimeSettings:
        return RuntimeSettings()

    async def fake_gen(ctx) -> RunOutcome:
        return RunOutcome(status="succeeded", score=80)

    monkeypatch.setattr("cadx.runtime_config.runtime_from_db_or_default", fake_rt)
    monkeypatch.setattr(runner, "run_generation", fake_gen)
    monkeypatch.setattr("cadx.storage.data_root", lambda: tmp_path / "data")
    bm = runner.Benchmark(name="d_test", prompt="box", fake_mesh=True)
    r = await runner.run_benchmark(bm, llm=FakeLLM([]), eval_root=tmp_path)
    assert r.variant == "default"
    hits = _result_jsons(tmp_path, "d_test/*/result.json")
    assert len(hits) == 1
    meta = json.loads(hits[0].read_text())
    assert meta["variant"] == "default"
    assert meta["variant_overrides"] == {}


async def test_run_benchmark_writes_crash_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cadx.eval.runner as runner
    from cadx.llm.fake import FakeLLM
    from cadx.runtime_config import RuntimeSettings

    async def fake_rt() -> RuntimeSettings:
        return RuntimeSettings()

    async def boom(ctx) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("cadx.runtime_config.runtime_from_db_or_default", fake_rt)
    monkeypatch.setattr(runner, "run_generation", boom)
    monkeypatch.setattr("cadx.storage.data_root", lambda: tmp_path / "data")
    bm = runner.Benchmark(name="c_test", prompt="box", fake_mesh=True)
    with pytest.raises(RuntimeError):
        await runner.run_benchmark(
            bm, llm=FakeLLM([]), variant="off", variant_overrides={}, eval_root=tmp_path
        )
    hits = _result_jsons(tmp_path, "c_test/off/*/result.json")
    assert len(hits) == 1
    meta = json.loads(hits[0].read_text())
    assert meta["variant"] == "off"
    assert meta["result"] is None
    assert "RuntimeError" in meta["error"]


# --- step stats thinking split ------------------------------------------------


def test_step_stats_thinking_split() -> None:
    from cadx.eval.runner import _step_stats

    records = [
        {
            "step": "codegen",
            "iteration": 1,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "latency_ms": 10,
            "parsed": {"thinking": True},
        },
        {
            "step": "codegen",
            "iteration": 2,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "latency_ms": 5,
            "parsed": {"thinking": False},
        },
        {"event": "tool_call"},  # events carry no step stats
    ]
    stats = _step_stats(records)
    cg = stats["codegen"]
    assert cg["calls"] == 2
    assert cg["prompt_tokens"] == 180
    assert cg["thinking_calls"] == 1
    assert cg["thinking_prompt_tokens"] == 100
    assert cg["thinking_completion_tokens"] == 50


# --- compare aggregation ------------------------------------------------------


def _summary(variant: str, bench: str, result: dict | None, wall_s: float = 5.0) -> dict:
    return {
        "benchmark": bench,
        "variant": variant,
        "variant_overrides": {},
        "run_id": "rid",
        "wall_s": wall_s,
        "error": None,
        "result": result,
    }


def test_aggregate_groups_by_variant_and_benchmark() -> None:
    summaries = [
        _summary(
            "off",
            "a",
            {
                "passed": True,
                "score": 90,
                "iterations": 2,
                "execs": 1,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "wall_s": 10.0,
                "exec_s": 4.0,
                "step_stats": {
                    "codegen": {
                        "calls": 1,
                        "prompt_tokens": 100,
                        "completion_tokens": 40,
                        "latency_ms": 1,
                        "thinking_calls": 1,
                        "thinking_prompt_tokens": 100,
                        "thinking_completion_tokens": 40,
                    }
                },
            },
        ),
        _summary(
            "off",
            "a",
            {
                "passed": False,
                "score": 50,
                "iterations": 3,
                "execs": 0,
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "wall_s": 8.0,
            },
        ),
        _summary(
            "always",
            "a",
            {
                "passed": True,
                "score": 70,
                "iterations": 1,
                "execs": 2,
                "prompt_tokens": 200,
                "completion_tokens": 60,
                "wall_s": 12.0,
            },
        ),
    ]
    rows = compare.aggregate(summaries)
    assert [(r.variant, r.benchmark) for r in rows] == [("always", "a"), ("off", "a")]
    off = next(r for r in rows if r.variant == "off")
    assert off.runs == 2
    assert off.passed == 1
    assert off.exec_ok == 1  # only the first run reached the sandbox
    assert off.mean_score == pytest.approx(70.0)
    assert off.mean_iterations == pytest.approx(2.5)
    assert off.prompt_tokens == 180
    assert off.completion_tokens == 60
    assert off.thinking_calls == 1
    assert off.thinking_completion_tokens == 40
    assert off.mean_wall_s == pytest.approx(9.0)


def test_aggregate_handles_crashed_and_missing_fields() -> None:
    rows = compare.aggregate(
        [
            _summary("off", "a", None, wall_s=3.0),  # crashed run
            _summary("off", "a", {"passed": True, "score": 80}),  # sparse result
        ]
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.runs == 2
    assert r.passed == 1
    assert r.exec_ok == 0
    assert r.mean_score == pytest.approx(80.0)
    assert r.mean_iterations is None


def test_render_markdown_table() -> None:
    rows = compare.aggregate(
        [_summary("off", "a", {"passed": True, "score": 90, "iterations": 2, "execs": 1})]
    )
    table = compare.render_markdown(rows)
    assert "| variant | benchmark |" in table
    assert "| off | a | 1 | 1/1 | 1/1 |" in table


def test_load_run_summaries_skips_bad_files(tmp_path: Path) -> None:
    good = tmp_path / "a" / "off" / "r1" / "result.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps(_summary("off", "a", {"passed": True})))
    bad = tmp_path / "a" / "off" / "r2" / "result.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    assert compare.load_run_summaries(tmp_path / "nope") == []
    summaries = compare.load_run_summaries(tmp_path)
    assert len(summaries) == 1
    assert summaries[0]["variant"] == "off"
