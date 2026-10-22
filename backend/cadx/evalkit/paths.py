"""evalkit filesystem layout — suite/baselines are committed, results and
scratch are gitignored run outputs.

No cadx imports at module level: the ``--local`` driver must be able to set
DATABASE_URL / DATA_DIR *before* ``cadx.config.get_settings()`` is first
evaluated.
"""

from __future__ import annotations

from pathlib import Path

# backend/cadx/evalkit/paths.py -> repo root is four levels up
REPO_ROOT = Path(__file__).resolve().parents[3]
EVALS_DIR = REPO_ROOT / "evals"
SUITE_DIR = EVALS_DIR / "suite"
BASELINES_DIR = EVALS_DIR / "baselines"
RESULTS_DIR = EVALS_DIR / "results"
SCRATCH_DIR = EVALS_DIR / "scratch"
REPORTS_DIR = EVALS_DIR / "reports"

# un-named eval runs are grouped under this variant in summaries
DEFAULT_VARIANT = "default"
