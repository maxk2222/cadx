"""Runtime settings: read env config + persist LLM overrides in AppSetting."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from cadagent.api.deps import get_session
from cadagent.api.schemas import SettingsOut, SettingsUpdate
from cadagent.config import get_settings
from cadagent.db.models import AppSetting

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
async def read_settings(db: AsyncSession = Depends(get_session)):
    s = get_settings()
    row = await db.get(AppSetting, "llm")
    ov = (row.value if row else {}) or {}
    mrow = await db.get(AppSetting, "mesh")
    mov = (mrow.value if mrow else {}) or {}
    return SettingsOut(
        llm_base_url=ov.get("llm_base_url") or s.llm_base_url,
        llm_model_vision=ov.get("llm_model_vision") or s.llm_model_vision,
        llm_model_code=ov.get("llm_model_code") or s.llm_model_code,
        llm_json_mode=str(ov.get("llm_json_mode", s.llm_json_mode)),
        llm_api_key_set=bool(s.llm_api_key),
        data_dir=s.data_dir,
        agent_max_iterations=s.agent_max_iterations,
        agent_max_repairs=s.agent_max_repairs,
        agent_pass_score=s.agent_pass_score,
        agent_clarification_mode=s.agent_clarification_mode,
        agent_default_gen_mode=str(mov.get("agent_default_gen_mode") or s.agent_default_gen_mode),
        exec_timeout_s=s.exec_timeout_s,
        exec_memory_mb=s.exec_mem_limit_mb,
        mesh_base_url=mov.get("mesh_base_url") or s.mesh_base_url,
        mesh_model=mov.get("mesh_model") or s.mesh_model,
    )


@router.put("", response_model=SettingsOut)
async def update_settings(body: SettingsUpdate, db: AsyncSession = Depends(get_session)):
    d = body.model_dump(exclude_none=True)
    llm_keys = {"llm_base_url", "llm_model_vision", "llm_model_code", "llm_json_mode"}
    mesh_keys = {"mesh_base_url", "mesh_model", "agent_default_gen_mode"}
    if llm_keys & d.keys():
        row = await db.get(AppSetting, "llm")
        if row is None:
            row = AppSetting(key="llm", value={})
            db.add(row)
        val = dict(row.value or {})
        val.update({k: v for k, v in d.items() if k in llm_keys})
        row.value = val
    if mesh_keys & d.keys():
        mrow = await db.get(AppSetting, "mesh")
        if mrow is None:
            mrow = AppSetting(key="mesh", value={})
            db.add(mrow)
        val = dict(mrow.value or {})
        val.update({k: v for k, v in d.items() if k in mesh_keys})
        mrow.value = val
    await db.commit()
    return await read_settings(db)
