"""Data-driven mesh-adapter selection.

`mesh_server/scripts/bench_adapters.py` writes `mesh_bench.json` into
DATA_DIR with per-model/per-part-class metrics (success rate, footprint IoU,
aspect error, latency). `pick_adapter` ranks the health-reported available
models by those measured results; without a bench file it degrades to the
previous static heuristic (device-native + multiview preferred).
"""

from __future__ import annotations

import json
from typing import Any

from cadx import storage
from cadx.logging import get_logger

log = get_logger(__name__)

BENCH_FILENAME = "mesh_bench.json"


def part_class(spec: Any | None) -> str:
    """'flat' for dimensioned flat parts (plates/scales/covers), 'organic'
    otherwise — the two classes the bench corpus distinguishes."""
    if spec is None:
        return "organic"
    dims = [
        float(getattr(spec.length_x, "value", 0) or 0),
        float(getattr(spec.width_y, "value", 0) or 0),
        float(getattr(spec.height_z, "value", 0) or 0),
    ]
    L, W, H = dims
    if H > 0 and L > 0 and W > 0 and 0.3 * max(L, W) > H:
        return "flat"
    return "organic"


def load_bench() -> dict[str, Any] | None:
    """Read DATA_DIR/mesh_bench.json; None when absent/corrupt."""
    try:
        p = storage.data_root() / BENCH_FILENAME
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data.get("models"), dict) else None
    except Exception:
        log.debug("model_pick.bench_unreadable", exc_info=True)
        return None


def _score(model_stats: dict[str, Any], cls: str) -> float:
    """Higher is better. class stats fall back to the model's overall row."""
    st = model_stats.get(cls) or model_stats.get("_all") or {}
    if not st or not st.get("n"):
        return -1.0
    success = float(st.get("success", 0.0))
    iou = float(st.get("iou", 0.0))
    aspect = 1.0 - min(float(st.get("aspect_err", 1.0)), 1.0)
    height = 1.0 - min(float(st.get("height_err", 1.0)), 1.0)
    watertight = float(st.get("watertight", 0.0))
    score = 0.45 * success + 0.25 * iou + 0.15 * aspect + 0.1 * height + 0.05 * watertight
    if cls in ("flat", "detail"):
        # edge sharpness matters on technical silhouettes; organic geometry
        # legitimately has high dihedral means, so this term is flat-only
        detail = 1.0 - min(float(st.get("roughness_deg", 0.0)) / 45.0, 1.0)
        score += 0.05 * detail
    # latency is a tie-breaker flavour, not a quality signal
    return round(score, 4)


def pick_adapter(
    models: list[Any],
    cls: str,
    *,
    need_multiview: bool = False,
    bench: dict[str, Any] | None = None,
    device: str = "",
) -> Any | None:
    """Rank available mesh models for a part class. Returns the best
    MeshModelInfo or None. Falls back to the static heuristic when no bench
    data exists for a model (scored -1 → sorted last but still eligible)."""
    avail = [
        m
        for m in models
        if getattr(m, "available", False)
        and getattr(m, "kind", "mesh") == "mesh"
        and m.id != "fake"
    ]
    if need_multiview:
        multi = [m for m in avail if getattr(m, "supports_multiview", False)]
        if multi:
            avail = multi
    if not avail:
        return None
    model_stats = (bench or {}).get("models") or {}

    def key(m: Any) -> tuple[float, bool, float]:
        st = model_stats.get(m.id) or {}
        return (_score(st, cls), m.device == device, float(getattr(m, "max_images", 0)))

    return max(avail, key=key)
