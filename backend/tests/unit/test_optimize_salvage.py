"""Optimize-session salvage — when the agent exhausts its budget without
done(), the best completed run is committed instead of discarding all
executor work and re-running the deterministic single-shot path."""

from __future__ import annotations

from cadx.agent.steps.optimize import _best_run_index
from cadx.agent.toolspecs import ToolSession


def _session_with_runs() -> ToolSession:
    from cadx.llm.fake import FakeLLM
    from tests.unit.test_session_condense import _ctx

    return ToolSession(_ctx(FakeLLM([{"content": "x"}])), kind="optimize")


def test_best_run_prefers_sf_export() -> None:
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 1.4},
                "fem": {"cases": [{"safety_factor": 3.0}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 2.9},
                "fem": {"cases": [{"safety_factor": 1.1}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1


def test_best_run_falls_back_to_worst_case_sf() -> None:
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {},
                "fem": {"cases": [{"safety_factor": 4.0}, {"safety_factor": 1.2}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {},
                "fem": {"cases": [{"safety_factor": 0.9}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 0  # worst-case SF 1.2 beats 0.9


def test_best_run_none_without_metrics() -> None:
    s = _session_with_runs()
    s.struct_runs.append({"index": 0, "res": {"topopt": {}}, "params": {}})
    assert _best_run_index(s) is None
    assert _best_run_index(_session_with_runs()) is None


def test_best_run_skips_disconnected_and_invalid() -> None:
    """A high-SF run whose export mesh is several bodies — or whose export
    verification was flagged degenerate — must never be salvaged: the SF
    describes only the loaded island, not a printable part."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 19.2, "n_components": 26},
                "fem": {"cases": [{"safety_factor": 19.2}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 3.1, "n_bodies_export": 2},
                "fem": {"cases": [{"safety_factor": 3.1}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 2,
            "res": {
                "topopt": {"sf_export": 5.5, "verification": {"valid": False}},
                "fem": {"cases": [{"safety_factor": 5.5}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 3,
            "res": {
                "topopt": {"sf_export": 2.0, "n_components": 1, "n_bodies_export": 1},
                "fem": {"cases": [{"safety_factor": 2.0}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 3  # only the connected, verified run counts


def test_best_run_skips_verify_spec_fail() -> None:
    """A canonical FAIL (e.g. envelope overshoot) disqualifies a run even
    when its safety factor is the best in the table."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 9.0},
                "fem": {"cases": [{"safety_factor": 9.0}]},
                "verify_spec": {"status": "FAIL", "checks": [{"name": "envelope_x"}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 1.8},
                "fem": {"cases": [{"safety_factor": 1.8}]},
                "verify_spec": {"status": "PASS"},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1


def test_best_run_skips_broken_load_path() -> None:
    """load_path_ok=false means at least one case cannot reach its fixtures
    through the printed part — structurally disconnected even when
    n_bodies reads 1 (smoothing bridges voxel gaps). A stellar sf_export
    on such a run describes the loaded island, not the part. None means
    'certificate skipped' and must stay eligible."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 6.34, "load_path_ok": False, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 6.34}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 2.1, "load_path_ok": None, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 2.1}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1
    s.struct_runs.append(
        {
            "index": 2,
            "res": {
                "topopt": {"sf_export": 3.0, "load_path_ok": True, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 3.0}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 2


def test_best_run_ranks_divergent_sf_by_pessimistic_bound() -> None:
    """sf_mask 0.05 / sf_export 1.37 means the load junction is one thin
    voxel wall the smoothing fattened — the part's honest bound is the
    pessimistic solve, so the divergent run must lose to a convergent one."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_mask": 0.05, "sf_export": 1.37},
                "fem": {"cases": [{"safety_factor": 0.05}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                # convergent solves — mask and export agree
                "topopt": {"sf_mask": 1.2, "sf_export": 1.3},
                "fem": {"cases": [{"safety_factor": 1.3}]},
            },
            "params": {},
        }
    )
    assert _best_run_index(s) == 1  # 1.3 > pessimistic 0.05


def test_best_run_skips_refine_verify_misses() -> None:
    """A refine run whose 'applied' bore failed the post-cut rescan (void
    cut, wrong diameter) must not be salvaged — the unrefined parent run
    stays the honest result."""
    s = _session_with_runs()
    s.struct_runs.append(
        {
            "index": 0,
            "res": {
                "topopt": {"sf_export": 2.0, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 2.0}]},
            },
            "params": {},
        }
    )
    s.struct_runs.append(
        {
            "index": 1,
            "res": {
                "topopt": {"sf_export": 9.0, "n_components": 1},
                "fem": {"cases": [{"safety_factor": 9.0}]},
                "features": {
                    "refine": {
                        "applied": [{"name": "m6"}],
                        "verify_misses": [{"x": -30.0, "y": 0.0, "dia": 6.0}],
                    }
                },
            },
            "params": {"refine_of": 0},
        }
    )
    assert _best_run_index(s) == 0


def test_topopt_images_capped_at_budget(tmp_path) -> None:
    """N4: a run result carrying 10+ renders attaches at most
    _TOPOPT_IMAGE_BUDGET and says how many were omitted — the on-disk
    files are untouched, only the context payload shrinks."""
    import types

    from cadx import storage
    from cadx.agent.steps.optimize import _TOPOPT_IMAGE_BUDGET, _topopt_images

    assert _TOPOPT_IMAGE_BUDGET == 8
    img_dir = storage.ensure(storage.data_root() / "_test_imgs" / str(tmp_path.name))
    for i in range(12):
        (img_dir / f"v{i}.png").write_bytes(b"png")
    res = {"renders": {f"view{i}": storage.relpath(img_dir / f"v{i}.png") for i in range(12)}}
    ops = types.SimpleNamespace(contact_sheet=None)
    images = _topopt_images(ops, res, tmp_path)
    assert len(images) == _TOPOPT_IMAGE_BUDGET
    assert all((img_dir / f"v{i}.png").exists() for i in range(12))
    assert any("omitted" in w for w in res.get("warnings") or [])
