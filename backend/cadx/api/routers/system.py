"""System API — version info + proxy to the host-side updater daemon.

The api container cannot update itself (no repo mount, no docker socket, and
it dies mid-restart), so updates are driven by a stdlib-only host daemon
(`scripts/update_server.py`, started by `start.sh`, default :8766) that shells
out to `scripts/update.sh`. This router is a thin authenticated proxy plus a
server-side active-jobs gate. The daemon token lives in
CADX_UPDATE_URL/CADX_UPDATE_TOKEN env (seeded into .env / process env by
start.sh) — infrastructure config, deliberately not a DB runtime setting.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, select

from cadx.api.deps import get_session
from cadx.api.routers.health import APP_VERSION
from cadx.api.schemas import (
    RollbackIn,
    SystemVersionGit,
    SystemVersionOut,
    UpdateCheckIn,
    UpdateCheckOut,
    UpdateStartIn,
    UpdateStartOut,
    UpdateStatusOut,
)
from cadx.db.models import Job, JobStatus

router = APIRouter(prefix="/system", tags=["system"])

_OFFLINE = (
    "updater daemon offline — start the stack via scripts/start.sh "
    "or run scripts/update.sh in a terminal"
)


def _updater() -> tuple[str, str]:
    base = os.environ.get("CADX_UPDATE_URL", "http://127.0.0.1:8766").rstrip("/")
    token = os.environ.get("CADX_UPDATE_TOKEN", "")
    return base, token


async def _call(
    method: str, path: str, *, body: dict[str, Any] | None = None, timeout_s: float = 15.0
) -> Any:
    base, token = _updater()
    try:
        async with httpx.AsyncClient(base_url=base, timeout=timeout_s) as c:
            r = await c.request(
                method, path, json=body, headers={"Authorization": f"Bearer {token}"}
            )
    except httpx.HTTPError as e:
        raise HTTPException(503, _OFFLINE) from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("error") or r.json().get("detail") or r.text
        except Exception:
            detail = r.text
        raise HTTPException(r.status_code, str(detail)[:500])
    return r.json()


@router.get("/version", response_model=SystemVersionOut)
async def system_version():
    """App + git info. Git data comes from the daemon (the container has no
    repo); the endpoint still answers when the daemon is down."""
    try:
        git = await _call("GET", "/v1/version", timeout_s=5.0)
        return SystemVersionOut(version=APP_VERSION, updater_ok=True, git=SystemVersionGit(**git))
    except HTTPException:
        return SystemVersionOut(version=APP_VERSION, updater_ok=False, git=None)


@router.get("/update/check", response_model=UpdateCheckOut)
async def update_check():
    return UpdateCheckOut(**await _call("GET", "/v1/update/check"))


@router.post("/update/check", response_model=UpdateStartOut)
async def update_check_run(body: UpdateCheckIn):
    await _call("POST", "/v1/update/check", body=body.model_dump(exclude_none=True))
    return UpdateStartOut()


@router.get("/update/status", response_model=UpdateStatusOut)
async def update_status():
    return UpdateStatusOut(**await _call("GET", "/v1/update/status"))


@router.post("/update", response_model=UpdateStartOut)
async def update_start(body: UpdateStartIn, db: AsyncSession = Depends(get_session)):
    # active-jobs gate: an update restarts the whole stack and kills in-flight
    # generation jobs — require an explicit confirm_jobs so non-UI callers
    # can't miss the warning either
    n = (
        await db.execute(
            select(func.count(col(Job.id))).where(
                col(Job.status).in_(
                    [JobStatus.queued, JobStatus.running, JobStatus.waiting_for_input]
                )
            )
        )
    ).scalar_one()
    if n and not body.confirm_jobs:
        raise HTTPException(
            409, f"{n} active job(s) — confirm_jobs required (update interrupts them)"
        )
    # no force here — the script's dirty-tree refusal must keep working for
    # GUI-triggered updates; --yes (sent by the daemon) covers the confirmed
    # active-jobs warning, the api-side 409 above is the actual gate
    payload: dict[str, Any] = {}
    if body.ref:
        payload["ref"] = body.ref
    await _call("POST", "/v1/update", body=payload)
    return UpdateStartOut()


@router.post("/update/rollback", response_model=UpdateStartOut)
async def update_rollback(body: RollbackIn):
    if not body.confirm:
        raise HTTPException(400, "confirm required")
    await _call("POST", "/v1/update/rollback", body={"confirm": True})
    return UpdateStartOut()


@router.get("/stack")
async def stack_status() -> dict[str, Any]:
    """Supervised-stack snapshot from the updater watchdog (compose states,
    mesh-server supervision, recent heal actions). Returns a stub with
    updater_ok=false when the daemon is down."""
    try:
        return await _call("GET", "/v1/stack/status", timeout_s=10.0)
    except HTTPException as e:
        if e.status_code == 503:
            return {"updater_ok": False, "detail": _OFFLINE}
        raise


@router.post("/stack/restart/{service}")
async def stack_restart(service: str):
    """Manual heal — restarts a supervised service through the watchdog
    (mesh-server respawn or `docker compose restart`)."""
    return await _call("POST", f"/v1/stack/restart/{service}", timeout_s=200.0)
