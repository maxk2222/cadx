"""SFT exporter (M13): pair shapes, dedup, secret guard, min-score filter.

Builds a synthetic ``eval_out`` + ``data/projects`` tree on tmp_path and runs
``scripts/export_sft.py``'s main() against it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = REPO_ROOT / "scripts" / "export_sft.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_sft", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("export_sft", mod)
    spec.loader.exec_module(mod)
    return mod


export_sft = pytest.fixture(_load_module, scope="module")


GOOD_CODE = "from build123d import *\nresult = Box(1, 1, 1)"
FIXED_CODE = "from build123d import *\nresult = Box(2, 2, 2)"
BROKEN_CODE = "from build123d import *\nresult = broken("


def _write_trace(
    vdir: Path,
    *,
    codegen_code: str = GOOD_CODE,
    dup_codegen: bool = False,
    repair_record: bool = False,
    broken_code: str = BROKEN_CODE,
) -> None:
    numbered = "\n".join(f"{i + 1:>5} │ {line}" for i, line in enumerate(broken_code.splitlines()))
    records = [
        {
            "step": "codegen",
            "iteration": 1,
            "model": "fake",
            "messages": [
                {"role": "system", "content": "You write build123d code."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "USER REQUEST: make a box"},
                        {"type": "image_url", "image_url": "<image>"},
                        {"type": "text", "text": "PART SPEC (JSON): {}"},
                    ],
                },
            ],
            "response": json.dumps({"code": codegen_code, "reasoning": "ok"}),
            "parsed": {"has_final": True},
        }
    ]
    if dup_codegen:
        records.append(dict(records[0]))
    if repair_record:
        records.append(
            {
                "step": "repair_1",
                "iteration": 1,
                "model": "fake",
                "messages": [
                    {"role": "system", "content": "You repair build123d code."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Fix this code — minimal edit."},
                            {
                                "type": "text",
                                "text": (
                                    f"PREVIOUS CODE (line-numbered):\n```python\n{numbered}\n```"
                                ),
                            },
                            {"type": "text", "text": "EXECUTION FAILED.\nerror: boom"},
                        ],
                    },
                ],
                "response": json.dumps({"full_code": FIXED_CODE, "reasoning": "fix"}),
                "parsed": {"has_full_code": True},
            }
        )
    (vdir / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))


def _write_attempt(vdir: Path, iteration: int, index: int, code: str, ok: bool, error=None) -> None:
    run = vdir / "iterations" / str(iteration) / f"a{index}" / "_run"
    run.mkdir(parents=True)
    (run / "model.py").write_text(code)
    res = {"ok": ok, "stdout": ""}
    if error:
        res["error"] = error
    (run / "result.json").write_text(json.dumps(res))
    (run / "input.json").write_text(json.dumps({"anchors": {"spec": {"name": "box"}}}))


def _make_run(
    root: Path,
    projects: Path,
    *,
    bench: str = "box_bench",
    variant: str = "v1",
    run_id: str = "run-1",
    pid: str = "proj-1",
    vid: str = "ver-1",
    score: float = 90,
    passed: bool = True,
    **trace_kw,
) -> Path:
    vdir = projects / pid / "versions" / vid
    vdir.mkdir(parents=True)
    _write_trace(vdir, **trace_kw)
    run_dir = root / bench / variant / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "benchmark": bench,
                "variant": variant,
                "run_id": run_id,
                "project_id": pid,
                "version_dir": f"/data/projects/{pid}/versions/{vid}",
                "result": {
                    "name": bench,
                    "score": score,
                    "passed": passed,
                    "dim_ok": True,
                    "watertight": True,
                    "version_dir": f"/data/projects/{pid}/versions/{vid}",
                },
            }
        )
    )
    return vdir


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_export(export_sft, root: Path, projects: Path, out: Path, *extra: str) -> None:
    argv = sys.argv
    sys.argv = [
        "export_sft",
        "--root",
        str(root),
        "--projects-root",
        str(projects),
        "--out",
        str(out),
        *extra,
    ]
    try:
        assert export_sft.main() == 0
    finally:
        sys.argv = argv


def test_codegen_pair_shape(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects)
    _write_attempt(vdir, 1, 1, GOOD_CODE, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--kinds", "codegen")
    rows = _rows(out)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == GOOD_CODE
    assert "<image>" in msgs[1]["content"]  # image blocks flattened
    meta = rows[0]["meta"]
    assert meta["kind"] == "codegen" and meta["benchmark"] == "box_bench"
    assert meta["score"] == 90 and meta["run_id"] == "run-1"


def test_min_score_filters(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, score=80)
    _write_attempt(vdir, 1, 1, GOOD_CODE, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--min-score", "85", "--kinds", "codegen")
    assert _rows(out) == []


def test_dedup_identical_pairs(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, dup_codegen=True)
    _write_attempt(vdir, 1, 1, GOOD_CODE, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out)
    assert len([r for r in _rows(out) if r["meta"]["kind"] == "codegen"]) == 1


def test_failed_code_not_harvested(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, codegen_code=BROKEN_CODE)
    _write_attempt(vdir, 1, 1, BROKEN_CODE, ok=False, error="boom")
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--kinds", "codegen")
    assert _rows(out) == []


def test_repair_pair_from_attempts(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    # run may even have failed overall — repair only needs a later pass
    vdir = _make_run(root, projects, passed=False, score=40, repair_record=True)
    _write_attempt(vdir, 1, 1, BROKEN_CODE, ok=False, error="boom")
    _write_attempt(vdir, 1, 2, FIXED_CODE, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--kinds", "repair")
    rows = _rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["meta"]["kind"] == "repair"
    assert row["meta"]["prompt_source"] == "trace"
    assert row["messages"][-1]["content"] == FIXED_CODE
    assert "PREVIOUS CODE" in row["messages"][1]["content"]


def test_repair_requires_later_pass(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, passed=False)
    _write_attempt(vdir, 1, 1, BROKEN_CODE, ok=False, error="boom")
    _write_attempt(vdir, 1, 2, BROKEN_CODE + "#x", ok=False, error="boom again")
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--kinds", "repair")
    assert _rows(out) == []


def test_synthesized_repair_prompt_when_no_trace_match(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, passed=False)  # no repair record at all
    _write_attempt(vdir, 1, 1, BROKEN_CODE, ok=False, error="boom")
    _write_attempt(vdir, 2, 1, FIXED_CODE, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out, "--kinds", "repair")
    rows = _rows(out)
    assert len(rows) == 1
    assert rows[0]["meta"]["prompt_source"] == "synthesized"
    assert "PART SPEC" in rows[0]["messages"][1]["content"]
    assert "boom" in rows[0]["messages"][1]["content"]


def test_secret_guard_drops_rows(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    leaked = GOOD_CODE + '\nAPI_KEY = "sk-' + "a" * 20 + '"\n'
    vdir = _make_run(root, projects, codegen_code=leaked)
    _write_attempt(vdir, 1, 1, leaked, ok=True)
    out = tmp_path / "pairs.jsonl"
    _run_export(export_sft, root, projects, out)
    assert _rows(out) == []


def test_output_is_valid_jsonl_and_deterministic(export_sft, tmp_path: Path) -> None:
    root, projects = tmp_path / "eval_out", tmp_path / "projects"
    vdir = _make_run(root, projects, repair_record=True)
    _write_attempt(vdir, 1, 1, BROKEN_CODE, ok=False, error="boom")
    _write_attempt(vdir, 1, 2, GOOD_CODE, ok=True)
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _run_export(export_sft, root, projects, out1)
    _run_export(export_sft, root, projects, out2)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    for row in _rows(out1):
        assert set(row) == {"messages", "meta"}
        for m in row["messages"]:
            assert set(m) == {"role", "content"}
