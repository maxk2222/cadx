"""Persistent recognition worker: spawn amortisation, respawn, timeouts.

The worker keeps a spawned quiddity interpreter alive across calls — these
tests gate the lifecycle contract (reuse, death detection, timeout kill +
respawn) rather than quiddity's own correctness. Skips cleanly without
quiddity."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import pytest

quiddity = pytest.importorskip("quiddity")
pytest.importorskip("build123d")

from build123d import Box, export_brep  # noqa: E402

from cadx.cad import recognition  # noqa: E402


def _brep(shape) -> str:
    td = tempfile.mkdtemp(prefix="cadx-rec-test-")
    p = Path(td) / "s.brep"
    export_brep(shape, str(p))
    return str(p)


def test_worker_serves_repeated_calls():
    worker = recognition._RecognitionWorker(mp.get_context("spawn"))
    try:
        rec1 = worker.run(_brep(Box(2, 2, 2)), 60.0, 6, None)
        pid1 = worker._proc.pid if worker._proc else None
        rec2 = worker.run(_brep(Box(3, 1, 1)), 60.0, 6, None)
        pid2 = worker._proc.pid if worker._proc else None
        # same spawned interpreter answers both — boot cost paid once
        assert isinstance(rec1, dict)
        assert isinstance(rec2, dict)
        assert pid1 is not None and pid1 == pid2
    finally:
        worker._kill()


def test_worker_respawns_after_death():
    worker = recognition._RecognitionWorker(mp.get_context("spawn"))
    try:
        worker.run(_brep(Box(2, 2, 2)), 60.0, 6, None)
        pid1 = worker._proc.pid
        worker._proc.terminate()
        worker._proc.join(5.0)
        rec = worker.run(_brep(Box(2, 2, 2)), 60.0, 6, None)
        assert isinstance(rec, dict)
        assert worker._proc is not None and worker._proc.pid != pid1
    finally:
        worker._kill()


def test_worker_timeout_kills_and_recovers():
    worker = recognition._RecognitionWorker(mp.get_context("spawn"))
    try:
        # a path that cannot be read: the child answers {} quickly — but a
        # 0.2s budget still exercises the deadline path deterministically
        rec = worker.run("/nonexistent/shape.brep", 0.2, -1, None)
        assert rec == {}
        # worker is killed on timeout or survives a fast {} answer — either
        # way the next call must still be served
        rec2 = worker.run(_brep(Box(2, 2, 2)), 60.0, 6, None)
        assert isinstance(rec2, dict)
    finally:
        worker._kill()


def test_run_budgeted_amortises_spawn(monkeypatch):
    """Two consecutive evidence runs share one worker process."""
    monkeypatch.setattr(recognition, "_RECOGNITION_BUDGET_S", 60.0)
    rec1 = recognition.build_recognition_evidence(Box(2, 2, 2))
    worker = recognition._WORKERS.get(__import__("os").getpid())
    assert worker is not None
    pid1 = worker._proc.pid if worker._proc else None
    rec2 = recognition.build_recognition_evidence(Box(3, 3, 3))
    pid2 = worker._proc.pid if worker._proc else None
    assert isinstance(rec1, dict) and isinstance(rec2, dict)
    assert pid1 == pid2
    t0 = time.monotonic()
    recognition.build_recognition_evidence(Box(1, 1, 1))
    # warm worker: spawn+import (~seconds) is amortised — a trivial part
    # round-trips well under the cold-spawn cost
    assert time.monotonic() - t0 < 15.0
