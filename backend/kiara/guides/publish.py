"""Publish pipeline — internal guide run → the project's stable AR Guide.

Identity model (plan §A4): project↔guide is 1:1 — every rerun upserts into
the SAME guide document so a synced headset sees one guide that moves
forward, never duplicate/phantom guides. ``guide_id`` is derived from the
project id deterministically; ``Guide.source_version_id`` points at the
version whose output was last published; ``version.params["guide_id"]``
is the backlink.

The published document is a PROJECTION of the internal GuideState — the
internal artifact (unity_guide.json with parts/placements/joints/audit)
stays under ``output/ar_guide_generation/`` as durable evidence; the
client consumes only the GuideSchema projection.

Gate: publish is only invoked for runs that produced emit artifacts. A
failed/partial run publishes NOTHING new — the last good document stays
authoritative (the ``[FEHLER]`` convention applies when a caller
explicitly chooses to publish a partial run).
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from kiara.guide.unity_schema import GUIDE_FORMAT_VERSION, build_guide_doc
from kiara.guides import mutation
from kiara.guides.media_index import media_repository
from kiara.guides.paths import guide_dir
from kiara.logging import get_logger

log = get_logger(__name__)

ERROR_TITLE_PREFIX = "[FEHLER]"


def guide_id_for_project(project_id: Any) -> str:
    """Deterministic project↔guide identity — one stable guide per project."""
    pid = str(project_id or "").strip()
    return f"guide-{pid}" if pid else f"guide-{uuid.uuid4()}"


def _guide_tree(guide_id: str) -> Path:
    root = guide_dir(guide_id)
    for sub in ("cad_models", "images", "videos", "audio", "hands", "media",
                "output/media", "output/ar_guide_generation", "instruction"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _materialize_cad_models(state: Any, guide_id: str) -> dict[str, str]:
    """Copy each part's Unity-loadable mesh into the guide's cad_models/.

    Returns {part_id: filename}. Missing meshes keep their deterministic
    fallback filename in the doc (``{part_id}.glb``) and are recorded as
    absent — the emit must never invent a file, and the media index marks
    the honest state."""
    out_dir = guide_dir(guide_id) / "cad_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    name_by_part: dict[str, str] = {}
    used_names: dict[str, str] = {}
    for part in state.parts:
        src = getattr(part, "mesh_path", None) or getattr(part, "step_path", None) or ""
        if not src:
            name_by_part[part.id] = f"{part.id}.glb"
            continue
        src_path = Path(src)
        filename = src_path.name
        # basename collisions across different source dirs must never silently
        # overwrite — disambiguate deterministically from the part id so two
        # parts can never share one file's content
        owner = used_names.get(filename)
        if owner is not None and owner != str(src_path.resolve()):
            filename = f"{part.id}_{src_path.name}"
        used_names[filename] = str(src_path.resolve())
        dst = out_dir / filename
        try:
            if src_path.is_file():
                if not dst.exists() or dst.stat().st_size != src_path.stat().st_size:
                    shutil.copyfile(src_path, dst)
                name_by_part[part.id] = filename
            else:
                log.warning("guide.publish.missing_mesh", part=part.id, path=src)
                name_by_part[part.id] = filename or f"{part.id}.glb"
        except OSError as exc:
            log.warning("guide.publish.mesh_copy_failed", part=part.id, error=str(exc))
            name_by_part[part.id] = f"{part.id}.glb"
    return name_by_part


def _materialize_media(state: Any, guide_id: str) -> list[str]:
    """Copy step media images into output/media/ — the photoLocation URLs
    the emitter produced point at these names."""
    out_dir = guide_dir(guide_id) / "output" / "media"
    copied: list[str] = []
    for step in state.steps:
        src = Path(step.media_image) if step.media_image else None
        if src is None or not src.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / src.name
        try:
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, dst)
            copied.append(src.name)
        except OSError as exc:
            log.warning("guide.publish.media_copy_failed", file=src.name, error=str(exc))
    return copied


def _write_internal_artifact(state: Any, guide_id: str) -> Path | None:
    """Mirror the internal unity_guide.json (parts/placements/joints/audit)
    into the guide tree — durable evidence + the file translate_unity_guide
    rewrites for language variants."""
    src = Path(state.unity_guide_path) if state.unity_guide_path else None
    if src is None or not src.is_file():
        alt = Path(state.output_dir or "") / "output" / "unity_guide.json"
        src = alt if alt.is_file() else None
    if src is None:
        return None
    dst = (
        guide_dir(guide_id)
        / "output" / "ar_guide_generation" / "unity_guide.json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src, dst)
        return dst
    except OSError as exc:
        log.warning("guide.publish.artifact_copy_failed", error=str(exc))
        return None


def _update_media_index(guide_id: str, cad_names: dict[str, str], media: list[str]) -> None:
    """Register the published files so media list/delete routes and the
    attempt-commit's reference checks see them."""
    for part_id, filename in cad_names.items():
        media_repository.upsert_project_media_by_object(
            guide_id,
            f"cad_{part_id}",
            "cad_models",
            {
                "uniqueFileName": filename,
                "fileName": filename,
                "url": f"/api/assets/{guide_id}/cad_models/{filename}",
            },
        )
    for name in media:
        media_repository.upsert_project_media_by_object(
            guide_id,
            f"photo_{name}",
            "output",
            {
                "uniqueFileName": name,
                "fileName": name,
                "url": f"/api/assets/{guide_id}/output/media/{name}",
            },
        )


def publish_guide(
    state: Any,
    *,
    project_id: Any,
    version_id: Any,
    title: str = "",
    failed: bool = False,
) -> dict[str, Any]:
    """Materialize + emit + commit the project's guide document.

    SYNC (blocking FS + DB) — callers run it via run_in_threadpool /
    asyncio.to_thread. Returns the stored document."""
    guide_id = guide_id_for_project(project_id)
    _guide_tree(guide_id)

    cad_names = _materialize_cad_models(state, guide_id)
    media = _materialize_media(state, guide_id)
    _write_internal_artifact(state, guide_id)
    _update_media_index(guide_id, cad_names, media)

    doc_title = title or _title_for(state)
    if failed and not doc_title.startswith(ERROR_TITLE_PREFIX):
        doc_title = f"{ERROR_TITLE_PREFIX} {doc_title}"

    doc = build_guide_doc(
        state,
        guide_id,
        title=doc_title,
        source_file_id=str(version_id or ""),
        filename_overrides=cad_names,
    )
    doc["formatVersion"] = GUIDE_FORMAT_VERSION

    stored = mutation.replace_guide_document(
        doc,
        project_id=project_id,
        source_version_id=version_id,
    )
    log.info(
        "guide.publish.committed",
        guide_id=guide_id,
        version=stored.get("version"),
        steps=len(stored.get("steps") or []),
        failed=failed,
    )
    return stored


def _title_for(state: Any) -> str:
    doc_name = ""
    if state.document is not None and state.document.pdf_path:
        doc_name = state.document.pdf_path.rsplit("/", 1)[-1]
    return doc_name.replace(".pdf", "") or "AR Assembly Guide"


def publish_run_dir(
    state_dir: Any,
    *,
    project_id: Any,
    version_id: Any,
    status: str,
) -> dict[str, Any] | None:
    """Worker-side entry: load the saved GuideState from the run's state dir
    and publish it. SYNC — invoke via asyncio.to_thread/run_in_threadpool.

    ``status`` drives the honesty gate: ``succeeded`` publishes plainly,
    ``needs_review`` publishes flagged [FEHLER] (artifacts were emitted but
    the audit left blockers open — the client must see that), anything else
    returns None so the last good document stays authoritative."""
    if status not in ("succeeded", "needs_review"):
        return None
    try:
        from kiara.guide.state import GuideState

        state = GuideState.load(Path(str(state_dir)))
    except Exception as exc:
        log.warning("guide.publish.state_load_failed", error=str(exc), dir=str(state_dir))
        return None
    if state is None:
        return None
    return publish_guide(
        state,
        project_id=project_id,
        version_id=version_id,
        failed=(status == "needs_review"),
    )
