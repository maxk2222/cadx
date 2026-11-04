from __future__ import annotations

import json
from typing import Any


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _plain_feature(feature: Any) -> dict[str, Any]:
    if isinstance(feature, dict):
        return feature
    dump = getattr(feature, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def _feature_signature(feature: Any) -> str:
    plain = _plain_feature(feature)
    name = str(plain.get("name") or "").strip().casefold()
    feature_type = str(plain.get("type") or "").strip().casefold()
    if name or feature_type:
        return f"{feature_type}\0{name}"
    return json.dumps(plain, sort_keys=True, separators=(",", ":"))


def dropped_feature_indices(spec: Any) -> set[int]:
    features = _get(spec, "features") or []
    drops = _get(spec, "feature_drops") or []
    signatures: dict[str, list[int]] = {}
    for index, feature in enumerate(features):
        signatures.setdefault(_feature_signature(feature), []).append(index)

    dropped: set[int] = set()
    for record in drops:
        index = _get(record, "feature_index")
        snapshot = _get(record, "feature")
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = -1
        if snapshot is None:
            if 0 <= index < len(features):
                dropped.add(index)
            continue
        signature = _feature_signature(snapshot)
        matches = signatures.get(signature) or []
        if 0 <= index < len(features) and index in matches:
            dropped.add(index)
        elif len(matches) == 1:
            dropped.add(matches[0])
    return dropped


def active_feature_rows(spec: Any) -> list[tuple[int, Any]]:
    features = _get(spec, "features") or []
    dropped = dropped_feature_indices(spec)
    return [(index, feature) for index, feature in enumerate(features) if index not in dropped]


def active_features(spec: Any) -> list[Any]:
    return [feature for _index, feature in active_feature_rows(spec)]


def feature_drop_records(spec: Any) -> list[dict[str, Any]]:
    records = _get(spec, "feature_drops") or []
    out: list[dict[str, Any]] = []
    for record in records:
        dump = getattr(record, "model_dump", None)
        out.append(dump(mode="json") if callable(dump) else dict(record))
    return out


def feature_drop_count(spec: Any) -> int:
    return len(_get(spec, "feature_drops") or [])
