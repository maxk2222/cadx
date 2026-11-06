"""Concurrency slots for calls into local service backends.

Every external-service call (LLM, mesh gen, image gen) runs inside an
agent-queue job. The queue's ``max_jobs`` bounds how many jobs run at once,
not how many are inside a *local-resource* call simultaneously: a local
sidecar (mesh-server LLM / mesh gen / image gen) shares one GPU, so
unbounded fan-out just thrashes it.

Cloud providers are remote I/O — their calls must never queue behind a
local capacity cap, so they get no slot at all: the factories pass a slot
only when the resolved backend is local.

Slots are keyed by running loop so worker processes and test loops each
get their own semaphore; cap 0/None disables the slot entirely (same as
external = unbounded).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

_SLOTS: dict[tuple[str, int | None], asyncio.Semaphore] = {}


def _local_cap(kind: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(f"CADX_{kind.upper()}_LOCAL_SLOTS") or default))
    except ValueError:
        return default


def io_slot(kind: str, cap: int) -> asyncio.Semaphore | None:
    """Shared semaphore for one service kind on the current loop (None = unbounded)."""
    if cap <= 0:
        return None
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None  # sync context (CLI/tests): a loop-free semaphore
    key = (kind, loop_id)
    sem = _SLOTS.get(key)
    if sem is None:
        sem = _SLOTS[key] = asyncio.Semaphore(cap)
    return sem


def local_llm_slot() -> asyncio.Semaphore | None:
    return io_slot("llm", _local_cap("llm", 4))


def local_mesh_slot() -> asyncio.Semaphore | None:
    return io_slot("mesh", _local_cap("mesh", 2))


def local_image_slot() -> asyncio.Semaphore | None:
    return io_slot("image", _local_cap("image", 2))


class SlottedClient:
    """Mixin: wrap call bodies so local backends stay within their slot."""

    _io_slot: asyncio.Semaphore | None = None

    @contextlib.asynccontextmanager
    async def _slotted(self) -> AsyncIterator[None]:
        if self._io_slot is None:
            yield
        else:
            async with self._io_slot:
                yield
