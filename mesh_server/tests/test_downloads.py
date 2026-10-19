"""DownloadManager lifecycle: start → status → done, cancel, no-op on
concurrent starts, delete_weights guards."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from meshserver.adapters.llm_base import DownloadCancelled
from meshserver.adapters.fake import FakeAdapter
from meshserver.config import ServerConfig
from meshserver.downloads import DownloadManager, downloadable


def _cfg(tmp_path: Path) -> ServerConfig:
    return ServerConfig(model_dir=tmp_path / "models", job_dir=tmp_path / "jobs")


class _FakeWeights:
    """GGUF-flavored fake: weights live under model_dir/gguf/."""

    id = "fake-w"
    hf_repo = "org/fake"
    gguf_files = ["w.gguf"]
    gguf_mmproj = None
    calls = 0
    gate: threading.Event | None = None  # when set, download blocks on it

    @classmethod
    def downloaded(cls, cfg) -> bool:
        return (Path(cfg.model_dir) / "gguf" / "w.gguf").is_file()

    @classmethod
    def download_weights(cls, cfg, progress_cb=None, cancel_flag=None) -> int:
        cls.calls += 1
        dest = Path(cfg.model_dir) / "gguf"
        dest.mkdir(parents=True, exist_ok=True)
        part = dest / "w.part"
        total = 0
        for i in range(20):
            if cancel_flag is not None and cancel_flag.is_set():
                raise DownloadCancelled("stop")
            if cls.gate is not None and i > 2:
                while not cls.gate.is_set():
                    if cancel_flag is not None and cancel_flag.is_set():
                        raise DownloadCancelled("stop")
                    time.sleep(0.01)
            part.write_bytes(b"x" * (i + 1))
            total = i + 1
            if progress_cb:
                progress_cb(total, 20)
            time.sleep(0.005)
        part.rename(dest / "w.gguf")
        return total


class _BoomWeights(_FakeWeights):
    id = "boom-w"

    @classmethod
    def download_weights(cls, cfg, progress_cb=None, cancel_flag=None) -> int:
        raise ValueError("hf went sideways")


def _wait_status(mgr: DownloadManager, mid: str, want: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = mgr.status(mid)
        if s.status == want:
            return s
        time.sleep(0.02)
    raise AssertionError(f"{mid} never reached '{want}': {mgr.status(mid)}")


def setup_function():
    _FakeWeights.calls = 0
    _FakeWeights.gate = None


def test_downloadable_helper():
    assert downloadable(_FakeWeights) is True
    # mesh adapters declare no download_weights/hf_repo → not manageable here
    assert downloadable(FakeAdapter) is False


def test_start_to_done(tmp_path):
    cfg = _cfg(tmp_path)
    mgr = DownloadManager(cfg)
    st = mgr.start("fake-w", _FakeWeights)
    assert st.status in ("downloading", "done")
    st = _wait_status(mgr, "fake-w", "done")
    assert st.bytes_done > 0
    assert _FakeWeights.downloaded(cfg)
    # status() on an unknown model is idle, not an error
    assert mgr.status("nobody").status == "idle"
    assert mgr.is_downloading("fake-w") is False


def test_concurrent_start_is_noop(tmp_path):
    cfg = _cfg(tmp_path)
    gate = threading.Event()
    _FakeWeights.gate = gate
    mgr = DownloadManager(cfg)
    try:
        s1 = mgr.start("fake-w", _FakeWeights)
        s2 = mgr.start("fake-w", _FakeWeights)
        assert s1.status == "downloading" and s2.status == "downloading"
        time.sleep(0.2)
        assert _FakeWeights.calls == 1  # single thread, single download
    finally:
        gate.set()
    _wait_status(mgr, "fake-w", "done")


def test_already_downloaded_short_circuits(tmp_path):
    cfg = _cfg(tmp_path)
    (Path(cfg.model_dir) / "gguf").mkdir(parents=True)
    (Path(cfg.model_dir) / "gguf" / "w.gguf").write_bytes(b"done")
    mgr = DownloadManager(cfg)
    st = mgr.start("fake-w", _FakeWeights)
    assert st.status == "done"
    assert _FakeWeights.calls == 0


def test_cancel_returns_to_idle(tmp_path):
    cfg = _cfg(tmp_path)
    gate = threading.Event()  # never opened — cancellation is the only exit
    _FakeWeights.gate = gate
    mgr = DownloadManager(cfg)
    mgr.start("fake-w", _FakeWeights)
    _wait_status(mgr, "fake-w", "downloading", timeout=1)
    assert mgr.cancel("fake-w") is True
    _wait_status(mgr, "fake-w", "idle")
    assert mgr.cancel("fake-w") is False  # nothing running anymore
    gate.set()


def test_error_records_and_stops(tmp_path):
    mgr = DownloadManager(_cfg(tmp_path))
    mgr.start("boom-w", _BoomWeights)
    st = _wait_status(mgr, "boom-w", "error")
    assert "sideways" in (st.error or "")


def test_delete_weights_refuses_while_downloading(tmp_path):
    cfg = _cfg(tmp_path)
    gate = threading.Event()
    _FakeWeights.gate = gate
    mgr = DownloadManager(cfg)
    mgr.start("fake-w", _FakeWeights)
    _wait_status(mgr, "fake-w", "downloading", timeout=1)
    assert mgr.delete_weights("fake-w", _FakeWeights) is False
    mgr.cancel("fake-w")
    _wait_status(mgr, "fake-w", "idle")
    gate.set()


def test_delete_weights_refuses_while_loaded(tmp_path):
    mgr = DownloadManager(_cfg(tmp_path), is_loaded=lambda mid: mid == "fake-w")
    assert mgr.delete_weights("fake-w", _FakeWeights) is False


def test_delete_weights_removes_gguf(tmp_path):
    cfg = _cfg(tmp_path)
    mgr = DownloadManager(cfg)
    mgr.start("fake-w", _FakeWeights)
    _wait_status(mgr, "fake-w", "done")
    target = Path(cfg.model_dir) / "gguf" / "w.gguf"
    assert target.is_file()
    assert mgr.delete_weights("fake-w", _FakeWeights) is True
    assert not target.exists()
    assert mgr.status("fake-w").status == "idle"  # reset for a fresh download
