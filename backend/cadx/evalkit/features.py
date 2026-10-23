"""Lazy measured-feature backfill for `StageOutcome.features`.

`ExecResult.features` (extract_features output: hole_groups, rounds,
chamfers, slots…) only lives in the transient agent `RunOutcome` — it is
never persisted on `Version`, so neither driver can read it back and every
feature-family gate saw an empty measurement. When a stage's gates include
the feature family, measure the exported artifact instead:

- STEP artifact → ``import_step`` + ``extract_features`` — the exact same
  measurement the executor performs during codegen.
- Mesh-only fallback → ``detect_mesh_features`` +
  ``mesh_features_to_measured`` — recovers axis-aligned tunnels/slots;
  rounds and chamfers are not recoverable from a faceted STL.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cadx.evalkit.models import StageOutcome
from cadx.logging import get_logger

log = get_logger("evalkit.features")

# Gate types that read `outcome.features` (mirrors _gate_outcome dispatch).
FEATURE_GATE_TYPES = {
    "holes",
    "hole",
    "tunnel",
    "slot",
    "countersink",
    "counterbore",
    "fillet",
    "chamfer",
    "bore",
    "through-hole",
    "features",
}


def needs_features(specs: list[dict[str, Any]] | None) -> bool:
    """True when any gate spec consumes measured features."""
    return any(
        isinstance(g, dict) and str(g.get("type")) in FEATURE_GATE_TYPES for g in specs or []
    )


async def ensure_features(outcome: StageOutcome) -> None:
    """Populate ``outcome.features`` from stage artifacts (best-effort, in-place)."""
    if outcome.features_available is False:
        return
    if outcome.features is not None:
        outcome.features_available = True
        return
    step = _kind_path(outcome, {"step"})
    stl = _kind_path(outcome, {"stl"})
    errors: list[str] = []
    if step is not None:
        try:
            measured = await asyncio.to_thread(_measure_step, step, stl)
            if measured is not None:
                outcome.features = measured
                outcome.features_available = True
                return
        except Exception as e:  # fall through to the mesh fallback
            errors.append(f"STEP: {type(e).__name__}: {e}")
            log.warning("eval.features.step_failed", error=f"{type(e).__name__}: {e}")
    try:
        measured = await asyncio.to_thread(_measure_mesh, outcome)
        if measured is not None:
            outcome.features = measured
            outcome.features_available = True
            return
    except Exception as e:
        errors.append(f"mesh: {type(e).__name__}: {e}")
        log.warning("eval.features.mesh_failed", error=f"{type(e).__name__}: {e}")
    outcome.features_available = False
    outcome.features_error = "; ".join(errors) or (
        "no STEP/STL artifact available" if step is None and stl is None else "feature extraction returned no evidence"
    )
    log.warning(
        "eval.features.unavailable",
        version=outcome.version_id,
        reason=outcome.features_error,
    )


_KIND_EXTS = {
    "step": {".step", ".stp"},
    "brep": {".brep"},
    "stl": {".stl"},
    "glb": {".glb", ".gltf"},
    "3mf": {".3mf"},
    "obj": {".obj"},
}


def _kind_path(outcome: StageOutcome, kinds: set[str]) -> Path | None:
    """Resolve the first artifact of the given kinds to an existing file.

    Order: downloaded artifact files (http driver, real extensions), then
    storage-relative paths from the artifacts map (local driver), then the
    repo data dir — ``Settings.data_dir`` is CWD-relative, so a driver run
    from ``backend/`` resolves relpaths under ``backend/data`` and misses.
    """
    exts = {e for k in kinds for e in _KIND_EXTS.get(k, ())}
    for f in outcome.artifact_files or []:
        p = Path(f)
        if p.suffix.lower() in exts and p.exists():
            return p
    from cadx import storage
    from cadx.evalkit.paths import REPO_ROOT

    for key, rel in (outcome.artifacts or {}).items():
        if str(key).split("|", 1)[0] not in kinds:
            continue
        try:
            p = storage.resolve(rel)
        except ValueError:
            p = None
        for cand in (p, REPO_ROOT / "data" / str(rel)):
            if cand is not None and cand.exists():
                return cand
    return None


def _measure_step(step_path: Path, stl_path: Path | None) -> dict[str, Any] | None:
    """BRep measurement — identical to the executor's feature extraction."""
    from build123d import import_step

    from cadx.cad.features import extract_features

    shape = import_step(str(step_path))
    measured = extract_features(shape, stl_path)
    return measured.model_dump(mode="json")


def _measure_mesh(outcome: StageOutcome) -> dict[str, Any] | None:
    """Mesh fallback — tunnels/slots only (no rounds/chamfers on facets)."""
    from cadx.evalkit.gates import _load_mesh

    mesh = _load_mesh(outcome)
    if mesh is None:
        return None
    from cadx.cad.features import mesh_features_to_measured
    from cadx.cad.mesh_features import detect_mesh_features

    mesh_feats = detect_mesh_features(mesh).model_dump(mode="json")
    return mesh_features_to_measured(mesh_feats).model_dump(mode="json")
