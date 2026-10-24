"""femverify session fixtures: report flush + shared domain cache."""

from __future__ import annotations

import pytest

from . import harness


@pytest.fixture(scope="session", autouse=True)
def _femverify_report():
    yield
    path = harness.flush_report()
    if path is not None:
        print(f"\nfemverify report → {path}")
