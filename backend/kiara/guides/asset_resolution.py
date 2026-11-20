from __future__ import annotations

import re
from pathlib import Path

from kiara.guides.asset_store import list_store_assets, readable_guides, store_model_url

VIDEO_VOICE_HAND_PREFAB_PREFIXES = ("p_videograph", "p_voicegraph", "p_lefthand", "p_righthand")

# C2 §2l: a guide's own copy, `/sync/assets/` or the generators' `/api/assets/`. That route still serves it.
_PER_GUIDE_CAD_LOCATION = re.compile(r"/(?:sync/)?assets/[^/]+/cad_models/")


def prefab_without_clone(prefab: str | None) -> str:
    """Exactly one `(Clone)` off the very end, the rule Unity and the web compare by (§2h)."""
    return str(prefab or "").removesuffix("(Clone)")


def _is_photo_object(obj: dict) -> bool:
    prefab_lower = (obj.get("prefabName") or "").strip().lower()
    is_photo_flag = (obj.get("isPhoto") is True) or bool((obj.get("photoLocation") or "").strip())
    is_photo_prefab = prefab_lower.startswith("p_photograph") or prefab_lower.startswith("p_photo")
    return is_photo_flag or is_photo_prefab


def _is_video_voice_or_hand_object(obj: dict) -> bool:
    prefab_lower = (obj.get("prefabName") or "").strip().lower()
    is_flagged = (obj.get("isVideo") is True) or (obj.get("isAudio") is True) or (obj.get("isHand") is True)
    return is_flagged or prefab_lower.startswith(VIDEO_VOICE_HAND_PREFAB_PREFIXES)


def _apply_store_record(obj: dict, record: dict) -> None:
    obj["modelLocation"] = store_model_url(record["assetId"])
    obj["filename"] = f"{record['displayName']}.glb"
    obj["is3DModel"] = True
    if not obj.get("authoredHash"):
        obj["authoredHash"] = record["currentHash"]


def hydrate_missing_model_paths(doc_to_set: dict) -> dict:
    """
    C2 §2l: resolves each model object against the shared store. Changes `doc_to_set` in place, and only
    ever reads the store: no file is copied and no index row written (B-032, B-085).

    Called from `merge_authoring_document`, AFTER the stored identity is filled in. A headset object
    always arrives empty, so resolving it any earlier would give it the current hash on every push and
    raise the version once a model changes.

    1. `assetId` set, record exists: the store URL, `<displayName>.glb`, `authoredHash` only if empty.
    2. `assetId` set, record gone: the identity stays, no location. The name is never consulted.
    3. `assetId` empty: `prefabName` without `(Clone)` finds a record: as 1, and the identity is new.
       An object holding a guide's own CAD location keeps it.
    4. None of these: nothing but `is3DModel`.

    Important:
    - Text fields must never get a CAD model path.
    - Photo objects must never get a CAD model path.
    - Video, voice and hand objects are left alone: nothing is written.
    """
    records = list_store_assets()
    records_by_id = {record["assetId"].lower(): record for record in records}
    records_by_prefab = {record["prefabName"]: record for record in records if record.get("prefabName")}

    for step in (doc_to_set.get("steps") or []):
        for obj in (step.get("objects") or []):
            if _is_video_voice_or_hand_object(obj):
                continue

            prefab = (obj.get("prefabName") or "").strip()
            prefab_lower = prefab.lower()

            is_textfield_flag = bool(obj.get("isTextField") is True)
            is_textfield_prefab = (
                prefab_lower.startswith("p_movabletextfield")
                or prefab_lower.startswith("p_movabletextfieldwithbutton")
                or prefab_lower.startswith("p_textfield")
            )
            if is_textfield_flag or is_textfield_prefab:
                obj["is3DModel"] = False
                obj["modelLocation"] = ""
                continue

            if _is_photo_object(obj):
                obj["is3DModel"] = False
                obj["modelLocation"] = ""
                continue

            asset_id = str(obj.get("assetId") or "").strip()
            if asset_id:
                record = records_by_id.get(asset_id.lower())
                if record is not None:
                    _apply_store_record(obj, record)
                else:
                    obj["modelLocation"] = ""
                    obj["is3DModel"] = True
                continue

            record = records_by_prefab.get(prefab_without_clone(prefab))
            if record is not None and not _PER_GUIDE_CAD_LOCATION.search(str(obj.get("modelLocation") or "")):
                obj["assetId"] = record["assetId"]
                obj["authoredHash"] = record["currentHash"]
                _apply_store_record(obj, record)
                continue

            missing_model = not obj.get("modelLocation")
            missing_filename = not obj.get("filename")
            if not (missing_model or missing_filename):
                continue

            if not prefab:
                continue

            if obj.get("is3DModel") is not True:
                obj["is3DModel"] = True

    return doc_to_set


def resolve_generated_models(
    steps: list[dict],
    inputs: list[dict],
    keep_unresolved_location: bool,
) -> list[dict]:
    """
    C2 §2l: the generator names a part by its file, which is all that's left of it after the run. Each
    model object is resolved by its `filename` stem, case-insensitively: this run's inputs first, then
    the store's `displayName`. It ends up exactly as hydrate would leave it, or the first push raises
    the version. Changes `steps` in place.

    An unresolved part has no identity and no location, so the web counts it as missing (A3). Its URL
    pointed into the per-run folder, which is gone. Only a guide without `cad_inputs.json` keeps what
    the generator wrote (`keep_unresolved_location`): its `cad_models/` stays, so that URL still serves.
    """
    input_names = {record["displayName"].casefold(): record for record in inputs}
    store_names = {record["displayName"].casefold(): record for record in list_store_assets()}

    for step in steps:
        for obj in (step.get("objects") or []):
            if obj.get("is3DModel") is not True or obj.get("isTextField") is True:
                continue
            if _is_photo_object(obj) or _is_video_voice_or_hand_object(obj):
                continue

            stem = Path(str(obj.get("filename") or "")).stem.casefold()
            record = input_names.get(stem) or store_names.get(stem)
            if record is None:
                obj["assetId"] = ""
                obj["authoredHash"] = ""
                if not keep_unresolved_location:
                    obj["modelLocation"] = ""
                continue

            obj["assetId"] = record["assetId"]
            obj["authoredHash"] = record["currentHash"]
            _apply_store_record(obj, record)
            if record.get("prefabName"):
                obj["prefabName"] = record["prefabName"]

    return steps


def model_changes(doc: dict, records_by_id: dict[str, dict]) -> list[dict]:
    """
    C2 §2q, D-V2-32: the models in this guide whose file was replaced after they were placed. One entry
    per asset. `records_by_id` is keyed by lower-case `assetId`. An empty `authoredHash` can't be compared,
    and a missing record is not a change: the guide already shows that model as missing. Only `steps`
    count: translations are server-derived, so "Geprüft", a client edit, could never clear one.
    `guideStepLogic.ts` `modelChangesInSteps` is the same rule.
    """
    changes: dict[str, dict] = {}
    steps = doc.get("steps")
    for step in steps if isinstance(steps, list) else []:
        objects = step.get("objects") if isinstance(step, dict) else None
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, dict) or not obj.get("authoredHash"):
                continue
            record = records_by_id.get(str(obj.get("assetId") or "").strip().lower())
            if record is None or obj["authoredHash"] == record["currentHash"]:
                continue
            changes.setdefault(
                record["assetId"],
                {
                    "assetId": record["assetId"],
                    "displayName": record["displayName"],
                    "currentHash": record["currentHash"],
                },
            )
    return list(changes.values())


def list_guide_model_changes() -> dict[str, list[dict]]:
    """`GET /sync/model-changes`: only guides with changes. Worked out each time; nothing is written."""
    records_by_id = {record["assetId"].lower(): record for record in list_store_assets()}
    result: dict[str, list[dict]] = {}
    for guide in readable_guides():
        changes = model_changes(guide, records_by_id)
        if changes:
            result[guide["guideId"]] = changes
    return result


def dedupe_step_objects_by_lineage(doc_to_set: dict) -> dict:  # noqa: C901 - ported suite contract logic
    """
    1) Dedupe by lineage (originalObjectId || objectId) - last wins.
    2) Dedupe identical CAD objects in the same step if render key and pose are equal.
       Text fields, photo, video, voice and hand objects are never merged in this pass.
    """
    from math import isclose

    def render_key(obj: dict) -> str:
        model_location = (obj.get("modelLocation") or "").strip().lower()
        if model_location:
            return f"ml::{model_location}"

        filename = (obj.get("filename") or "").strip().lower()
        prefab_name = (obj.get("prefabName") or "").strip().lower()
        return f"fn::{filename}::pf::{prefab_name}"

    def nearly_same_pose(a: dict, b: dict) -> bool:
        pa, pb = a.get("pos") or {}, b.get("pos") or {}
        ra, rb = a.get("rot") or {}, b.get("rot") or {}
        sa, sb = a.get("scale") or {}, b.get("scale") or {}

        eps_pos = 1e-4
        eps_scale = 1e-4
        eps_rot = 1e-4

        def close(x: object, y: object, eps: float) -> bool:
            try:
                return isclose(float(x), float(y), abs_tol=eps)
            except Exception:
                return False

        pos_ok = all(close(pa.get(k, 0), pb.get(k, 0), eps_pos) for k in ("x", "y", "z"))
        rot_ok = all(close(ra.get(k, 0), rb.get(k, 0), eps_rot) for k in ("x", "y", "z", "w"))
        scale_ok = all(close(sa.get(k, 1), sb.get(k, 1), eps_scale) for k in ("x", "y", "z"))

        return pos_ok and rot_ok and scale_ok

    steps = doc_to_set.get("steps") or []
    new_steps = []

    for step in steps:
        objs = step.get("objects") or []

        lineage_last: dict[str, dict] = {}
        lineage_order: list[str] = []

        for obj in objs:
            lineage_id = (obj.get("originalObjectId") or obj.get("objectId") or "").strip()
            if not lineage_id:
                lineage_id = f"__no_lineage__::{render_key(obj)}"

            if lineage_id not in lineage_last:
                lineage_order.append(lineage_id)

            lineage_last[lineage_id] = obj

        deduped = [lineage_last[lineage_id] for lineage_id in lineage_order]

        by_render_key: dict[str, dict] = {}
        order_by_render: list[str] = []

        for obj in deduped:
            if obj.get("isTextField"):
                rk = f"__textfield__::{id(obj)}"
                order_by_render.append(rk)
                by_render_key[rk] = obj
                continue

            if _is_photo_object(obj) or _is_video_voice_or_hand_object(obj):
                rk = f"__media_or_hand__::{id(obj)}"
                order_by_render.append(rk)
                by_render_key[rk] = obj
                continue

            rk = render_key(obj)
            if rk not in by_render_key:
                order_by_render.append(rk)
                by_render_key[rk] = obj
            else:
                prev = by_render_key[rk]
                if nearly_same_pose(prev, obj):
                    by_render_key[rk] = obj
                else:
                    rk2 = f"{rk}::pos::{obj.get('pos')}::rot::{obj.get('rot')}::sc::{obj.get('scale')}"
                    order_by_render.append(rk2)
                    by_render_key[rk2] = obj

        final_objs = [by_render_key[key] for key in order_by_render]

        for idx, obj in enumerate(final_objs, start=1):
            obj["objectNumber"] = idx

        new_steps.append({**step, "objects": final_objs})

    doc_to_set["steps"] = new_steps
    return doc_to_set
