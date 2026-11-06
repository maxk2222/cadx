"""service_slots: local backends are capped, external calls never queue."""

from __future__ import annotations

import asyncio

import pytest

from cadx.agent.image_client import make_image_client
from cadx.agent.mesh_client import make_mesh_client
from cadx.agent.service_slots import io_slot, local_llm_slot
from cadx.llm.client import LLMClient, LLMResponse, make_llm
from cadx.runtime_config import RuntimeSettings


def _runtime(**kw) -> RuntimeSettings:
    return RuntimeSettings(
        mesh_base_url="http://mesh.local:8000",
        llm_base_url="http://llm.cloud/v1",
        **kw,
    )


def test_io_slot_shares_per_kind_and_loop() -> None:
    async def run() -> None:
        a = io_slot("llm", 2)
        b = io_slot("llm", 2)
        c = io_slot("mesh", 2)
        assert a is b  # same kind+loop → shared semaphore
        assert a is not c
        assert a._value == 2

    asyncio.run(run())


def test_io_slot_zero_cap_disables() -> None:
    async def run() -> None:
        assert io_slot("llm", 0) is None

    asyncio.run(run())


def test_make_llm_external_gets_no_slot() -> None:
    client = make_llm(_runtime(), backend="external")
    assert client._io_slot is None


def test_make_llm_sidecar_gets_shared_slot() -> None:
    async def run() -> None:
        c1 = make_llm(_runtime(), backend="sidecar")
        c2 = make_llm(_runtime(), backend="sidecar")
        assert c1._io_slot is c2._io_slot
        assert c1._io_slot is local_llm_slot()

    asyncio.run(run())


def test_make_mesh_sidecar_slotted_tripo_not() -> None:
    async def run() -> None:
        sidecar = make_mesh_client(runtime_settings=_runtime(), backend="sidecar")
        tripo = make_mesh_client(runtime_settings=_runtime(), backend="tripo")
        assert sidecar is not None and sidecar._io_slot is not None
        assert tripo is not None and getattr(tripo, "_io_slot", None) is None

    asyncio.run(run())


def test_make_image_sidecar_slotted_openai_not() -> None:
    async def run() -> None:
        rt = _runtime()
        sidecar = make_image_client(runtime_settings=rt, backend="sidecar")
        cloud = make_image_client(runtime_settings=rt, backend="openai")
        assert sidecar is not None and sidecar._io_slot is not None
        assert cloud is not None and getattr(cloud, "_io_slot", None) is None

    asyncio.run(run())


@pytest.mark.asyncio
async def test_local_slot_caps_concurrency_external_unbounded() -> None:
    running = 0
    peak = 0

    async def fake_chat(self: LLMClient, role, messages, **_kw) -> LLMResponse:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return LLMResponse(text="ok")

    local = LLMClient(runtime_settings=_runtime(), io_slot=asyncio.Semaphore(2))
    external = LLMClient(runtime_settings=_runtime())
    local._chat = fake_chat.__get__(local)  # type: ignore[method-assign]
    external._chat = fake_chat.__get__(external)  # type: ignore[method-assign]

    peak = 0
    await asyncio.gather(*(local.chat("code", []) for _ in range(6)))
    assert peak <= 2

    peak = 0
    await asyncio.gather(*(external.chat("code", []) for _ in range(6)))
    assert peak == 6
