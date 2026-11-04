from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cadx.cad.spec_features import active_feature_rows, dropped_feature_indices, feature_drop_count
from cadx.llm.schemas import FeatureDrop, PartSpec


def feature_drop_issues(spec: PartSpec | None) -> list[dict[str, Any]]:
    if spec is None:
        return []
    return [
        {
            "source": "spec",
            "severity": "minor",
            "what": f"agent withdrew declared feature {(drop.feature.name or drop.feature.type)!r}",
            "fix_hint": drop.reason,
            "feature_index": drop.feature_index,
            "feature_drop": drop.model_dump(mode="json"),
        }
        for drop in spec.feature_drops
    ]


def drop_spec_feature(
    spec: PartSpec, feature_index: Any, reason: Any, *, iteration: int, session: str
) -> dict[str, Any]:
    if isinstance(feature_index, bool) or not isinstance(feature_index, int):
        return {"ok": False, "error": "feature_index must be a zero-based integer"}
    features = spec.features or []
    if not 0 <= feature_index < len(features):
        available = ", ".join(
            f"{index}:{feature.name or feature.type}" for index, feature in enumerate(features)
        )
        return {
            "ok": False,
            "error": f"unknown feature_index {feature_index}; declared features: {available or '(none)'}",
        }
    if feature_index in dropped_feature_indices(spec):
        return {"ok": False, "error": f"feature {feature_index} is already dropped"}
    note = str(reason or "").strip()
    if len(note) < 12:
        return {
            "ok": False,
            "error": "reason must explain the limitation in at least 12 characters",
        }
    if len(note) > 500:
        return {"ok": False, "error": "reason must be 500 characters or fewer"}
    if session not in {"codegen", "repair", "optimize"}:
        return {"ok": False, "error": "unsupported feature-drop session"}

    feature = features[feature_index]
    record = FeatureDrop(
        feature_index=feature_index,
        feature=feature.model_copy(deep=True),
        reason=note,
        iteration=max(0, int(iteration)),
        session=session,  # type: ignore[arg-type]
        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    spec.feature_drops.append(record)
    return {
        "ok": True,
        "dropped": {
            "feature_index": feature_index,
            "name": feature.name,
            "type": feature.type,
            "reason": note,
            "iteration": record.iteration,
            "session": session,
        },
        "dropped_count": feature_drop_count(spec),
        "active_features": [
            {
                "index": index,
                "name": item.name,
                "type": item.type,
                "description": item.description,
            }
            for index, item in active_feature_rows(spec)
        ],
        "note": "The feature remains in the audit record and case-level geometry gates still apply.",
    }
