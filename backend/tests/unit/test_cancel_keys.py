"""Cancel-key scoping — a task timeout must never cancel the whole job.

Regression: _run_task used to set cancel:{job_id} (the CADX job) on task
timeout; every later task of that job then read the same flag and returned
'cancelled' instantly. One 480s pipeline hang pre-doomed a whole
generation run.
"""

from __future__ import annotations

from cadx.cad.tasks import _cancel_keys


def test_cancel_keys_job_only_when_no_task_id() -> None:
    assert _cancel_keys({"job_id": "j1"}) == ["cancel:j1"]


def test_cancel_keys_includes_task_scope() -> None:
    keys = _cancel_keys({"job_id": "j1", "_task_id": "execute_version-abc"})
    assert keys == ["cancel:j1", "cancel:execute_version-abc"]


def test_cancel_keys_empty_payload() -> None:
    assert _cancel_keys({}) == []


def test_task_timeout_flag_does_not_match_job_key() -> None:
    """The core invariant: the arq task id set by _run_task produces a key
    that can never collide with the user-level job cancel key."""
    keys = _cancel_keys({"job_id": "j1", "_task_id": "execute_version-abc"})
    assert "cancel:j1" in keys
    assert not any(k == "cancel:execute_version-abc" for k in keys if k.endswith("j1"))
