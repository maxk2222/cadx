"""Optional Rust kernel module — the ONLY import point for cadx_native.

Rust accelerates the Python-heavy query paths (mesh containment, ray
batches, scanline voxelization, CSR assembly, SpMV) but never replaces the
Python implementations: every helper here returns ``None`` when the module
is missing or a call fails, and the caller falls back to the existing path.
``CADX_NATIVE=0`` disables the import entirely; ``CADX_NATIVE_GEOM=0`` /
``CADX_NATIVE_FEM=0`` bisect the two halves independently.

Determinism: results are deterministic per input (scoped-thread
partitioning, fixed probe directions, per-slot ordered accumulation).
They are NOT bit-identical to scipy's summation order — parity is defined
at rtol 1e-9 for assembled matrices, 1e-5 for solve results.
"""

from __future__ import annotations

import os
import weakref
from typing import Any

_ENABLED = os.environ.get("CADX_NATIVE", "1").strip().lower() not in ("0", "false", "off")

try:
    if not _ENABLED:
        raise ImportError("CADX_NATIVE=0")
    import cadx_native as _nx  # type: ignore[import-not-found]
except Exception:
    _nx = None


def native_enabled(kind: str = "geom") -> bool:
    """True when the kernel is importable and not kill-switched.

    `kind` bisects the rollout: "geom" covers mesh queries/voxelize,
    "fem" covers assembly/spmv. Read per call so tests can toggle."""
    if _nx is None:
        return False
    env = os.environ.get(f"CADX_NATIVE_{kind.upper()}")
    return env is None or env.strip().lower() not in ("0", "false", "off")


def native_version() -> str | None:
    return getattr(_nx, "__version__", None) if _nx is not None else None


def _threads(budget: int = 8) -> int:
    """Bounded worker count for native calls — the executor's CFS quota is
    respected upstream in resources; a modest cap keeps per-call spawn
    cost (~10µs/thread) below the ms-scale work it amortizes."""
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return max(1, min(budget, n))


# ---------------------------------------------------------------------------
# Geometry queries — trimesh-identical semantics (see mesh.rs header)
# ---------------------------------------------------------------------------

_IDX_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def mesh_index(mesh: Any) -> Any | None:
    """MeshIndex for a trimesh object — WeakKey-cached per mesh instance so
    repeated queries (contains/ray/scanline call sites in one feature pass)
    share one BVH build. `None` on any problem → caller uses mesh.*."""
    if _nx is None or mesh is None:
        return None
    try:
        idx = _IDX_CACHE.get(mesh)
        if idx is None:
            import numpy as np

            idx = _nx.MeshIndex(
                np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64)),
            )
            _IDX_CACHE[mesh] = idx
        return idx
    except Exception:
        return None


def contains(mesh: Any, pts: Any) -> Any | None:
    """trimesh.contains parity — bidirectional ray test, same fixed probe
    direction. (n,)->bool or None."""
    idx = mesh_index(mesh)
    if idx is None:
        return None
    try:
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        return np.asarray(idx.contains(arr, _threads()))
    except Exception:
        return None


def ray_any(mesh: Any, origins: Any, dirs: Any) -> Any | None:
    """trimesh.ray.intersects_any — (n,)->bool or None."""
    idx = mesh_index(mesh)
    if idx is None:
        return None
    try:
        import numpy as np

        o = np.ascontiguousarray(np.asarray(origins, dtype=np.float64))
        d = np.ascontiguousarray(np.asarray(dirs, dtype=np.float64))
        return np.asarray(idx.ray_any(o, d, _threads()))
    except Exception:
        return None


def ray_locs(mesh: Any, origins: Any, dirs: Any) -> tuple[Any, Any, Any] | None:
    """trimesh.ray.intersects_location(multiple_hits=True) — (locs, ray_idx,
    tri_idx) sorted by (ray, t), or None."""
    idx = mesh_index(mesh)
    if idx is None:
        return None
    try:
        import numpy as np

        o = np.ascontiguousarray(np.asarray(origins, dtype=np.float64))
        d = np.ascontiguousarray(np.asarray(dirs, dtype=np.float64))
        locs, rids, tids = idx.ray_locs(o, d, _threads())
        return np.asarray(locs), np.asarray(rids), np.asarray(tids)
    except Exception:
        return None


def scanline_fill(
    mesh: Any, origin: Any, dims: tuple[int, int, int], pitch: float, offset: float
) -> Any | None:
    """fem/voxelize._scanline_contains parity fill — (prod(dims),)->bool
    or None. Same eps/dedup/on-surface conventions."""
    idx = mesh_index(mesh)
    if idx is None:
        return None
    try:
        import numpy as np

        o = np.ascontiguousarray(np.asarray(origin, dtype=np.float64))
        return np.asarray(
            idx.scanline_fill(
                o, tuple(int(d) for d in dims), float(pitch), float(offset), _threads()
            )
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FEM — CSR assembly against a fixed pattern, parallel SpMV
# ---------------------------------------------------------------------------

_ASM_CACHE: dict[int, Any] = {}


def assembler_for(iK: Any, jK: Any, ndofs: int) -> Any | None:
    """Assembler for a COO (iK,jK) pattern — cached keyed on the array
    identity, matching the `ij=` cache contract in solver.assemble (the
    same ndarray objects are reused across iterations)."""
    if _nx is None:
        return None
    key = (id(iK), id(jK), int(ndofs))
    asm = _ASM_CACHE.get(key)
    if asm is not None:
        return asm
    try:
        import numpy as np

        asm = _nx.Assembler(
            np.ascontiguousarray(np.asarray(iK, dtype=np.int64)),
            np.ascontiguousarray(np.asarray(jK, dtype=np.int64)),
            int(ndofs),
        )
        # bound the cache — patterns live as long as their ij arrays
        if len(_ASM_CACHE) > 32:
            _ASM_CACHE.clear()
        _ASM_CACHE[key] = asm
        return asm
    except Exception:
        return None


def assemble_values(asm: Any, elem_E_active: Any, ke: Any, cls: Any = None) -> Any | None:
    """sK vector for `fill` — elem_E per active element times the (24,24)
    isotropic ke or the per-class (ncls,24,24) table. None on mismatch."""
    try:
        import numpy as np

        e = np.ascontiguousarray(np.asarray(elem_E_active, dtype=np.float64))
        k = np.ascontiguousarray(np.asarray(ke, dtype=np.float64))
        c = None if cls is None else np.ascontiguousarray(np.asarray(cls, dtype=np.int64))
        return np.asarray(asm.values(e, k, c, _threads()))
    except Exception:
        return None


def assemble_fill(asm: Any, sK: Any) -> Any | None:
    """CSR data vector for sK — deterministic per-slot accumulation."""
    try:
        import numpy as np

        return np.asarray(
            asm.fill(np.ascontiguousarray(np.asarray(sK, dtype=np.float64)), _threads())
        )
    except Exception:
        return None


def spmv_op(Kf: Any) -> Any | None:
    """scipy LinearOperator wrapping a parallel native matvec — the CG
    driver/preconditioner/watchdog all stay Python; only A's matvec is
    swapped. `Kf` keeps owning its arrays (SpMv copies once, O(nnz))."""
    if _nx is None:
        return None
    try:
        import numpy as np
        from scipy.sparse import linalg as spla

        K = Kf.tocsr() if hasattr(Kf, "tocsr") else Kf
        mv = _nx.SpMv(
            np.ascontiguousarray(np.asarray(K.indptr, dtype=np.int64)),
            np.ascontiguousarray(np.asarray(K.indices, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(K.data, dtype=np.float64)),
        )
        n = K.shape[0]
        return spla.LinearOperator((n, n), matvec=lambda x: mv.matvec(x, _threads()))
    except Exception:
        return None
