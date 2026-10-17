"""DB-backed application runtime settings."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.db.models import AppSetting


class RuntimeSettings(BaseModel):
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model_vision: str = "gpt-4o"
    llm_model_code: str = "gpt-4o"
    llm_json_mode: Literal["json_object", "json_schema", "none"] = "json_object"
    llm_temperature_code: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=8000, ge=1, le=2_000_000)
    llm_timeout_s: float = Field(default=180, gt=0)
    llm_stream: bool = True
    llm_idle_timeout_s: float = Field(default=180, gt=0)
    llm_total_timeout_s: float = Field(default=1800, gt=0)
    # abort a stream that only emits whitespace/indentation for this long —
    # a degeneration mode the byte-progress watchdog cannot see
    llm_degenerate_stall_s: float = Field(default=60, gt=0)
    llm_extra_body: str = ""
    llm_tools: Literal["auto", "on", "off"] = "auto"
    # adaptive reasoning escalation: "auto" merges llm_thinking_body_on into
    # the request payload for hard steps and after stall triggers, "always"
    # for every call, "off" merges llm_thinking_body_off instead. The body
    # strings are provider-specific payload dicts (Qwen3-style
    # {"chat_template_kwargs": {"enable_thinking": false}}, OpenAI-compat
    # {"reasoning_effort": "low"}) — empty means 'model default'.
    llm_thinking: Literal["off", "auto", "always"] = "auto"
    llm_thinking_body_on: str = ""
    llm_thinking_body_off: str = ""
    llm_thinking_hard_steps: str = "repair,improve"  # csv step names for "auto"
    # iterations without a new best score (or a failed repair patch / repeated
    # same-category exec error) before "auto" escalates to thinking-on — the
    # latch stays for the rest of the job
    llm_thinking_escalate_after: int = Field(default=2, ge=1, le=50)
    # hard per-job cap on calls made WITH thinking on — reasoning tokens are
    # the dominant completion cost and escalation can amplify a non-converging
    # job (observed: 260k completion tokens without convergence). Once spent,
    # thinking_on() resolves OFF for the rest of the job regardless of policy
    # or the escalation latch. 0 disables thinking entirely.
    llm_thinking_max_calls: int = Field(default=8, ge=0, le=10_000)
    agent_max_iterations: int = Field(default=3, ge=1, le=200)
    agent_max_repairs: int = Field(default=3, ge=0, le=200)
    agent_pass_score: int = Field(default=80, ge=0, le=100)
    agent_dim_tolerance_pct: float = Field(default=2, ge=0, le=100)
    agent_dim_tolerance_mm: float = Field(default=0.5, ge=0, le=1000)
    agent_clarification_mode: Literal["assume", "ask"] = "assume"
    agent_max_input_images: int = Field(default=8, ge=0, le=100)
    agent_max_tokens_per_job: int = Field(default=400_000, ge=1)
    agent_max_wall_s: float = Field(default=900, gt=0)
    agent_default_gen_mode: Literal["auto", "cad", "mesh", "hybrid", "topopt"] = "auto"
    # plan step depth: "full" = catalog-aware tool session (default),
    # "lite" = one short non-tool call (steps+params only), "auto" = lite for
    # text-only cad jobs, full otherwise
    agent_plan_mode: Literal["full", "lite", "auto"] = "full"
    agent_param_autofix: bool = True
    agent_patch_mode: Literal["auto", "always", "never"] = "auto"
    agent_patch_retries: int = Field(default=2, ge=0, le=20)
    agent_feature_feedback: bool = True
    agent_critique_reask: bool = True
    agent_tool_max_calls: int = Field(default=10, ge=1, le=100)
    agent_tool_max_calls_info: int = Field(default=6, ge=1, le=100)
    # improve sessions whose issues are only feature findings + lint minors
    # just apply bound snippets — no exploration budget needed (observed:
    # 12-call feature-improve sessions at ~16k ptok each)
    agent_tool_max_calls_improve: int = Field(default=6, ge=1, le=100)
    agent_tool_max_turns: int = Field(default=24, ge=1, le=200)
    # edit/repair sessions pay the FULL pipeline per verify — give them fewer
    # turns than info/critique sessions (observed: a 24-turn repair loop ate
    # 386k prompt tokens and the whole job budget)
    agent_tool_max_turns_edit: int = Field(default=12, ge=1, le=200)
    agent_tool_churn_limit: int = Field(default=6, ge=2, le=50)
    # history condensation (OpenHands-condenser): once a tool session's
    # history exceeds this many exchanges, the middle is replaced by a
    # deterministic one-line-per-call digest — system+seed and the last ~half
    # of exchanges stay verbatim. Bounds the per-turn resend on long staged
    # build/repair sessions where APC cannot help the growing tail.
    # 0 disables.
    agent_session_condense_turns: int = Field(default=12, ge=0, le=500)
    agent_tool_verify: bool = True
    agent_tool_verify_max: int = Field(default=2, ge=0, le=20)
    # static signature gate on generated code (AST only, pre-execution):
    # an introspected build123d/cadlib/bd_warehouse member index catches
    # hallucinated attributes / property-called-as-method before the
    # sandbox — edit tools surface findings, done/submit gate on sig errors
    agent_sigcheck: bool = True
    # staged codegen: build the part feature-by-feature inside a tool session
    # (add_block/checkpoint/verify) instead of one-shot whole-file submit.
    # "auto" = on whenever tool sessions are enabled.
    agent_staged_build: Literal["auto", "on", "off"] = "auto"
    # best-effort parametric reconstruction of the winning mesh candidate
    # (mesh -> build123d code via cad/reconstruct) — one extra executor task
    # per mesh job; off = pure mesh outcome
    agent_parametrize: bool = True
    # verify calls a staged build may make — higher than edit/repair because
    # every feature block should land on green geometry before the next one
    agent_build_verify_max: int = Field(default=8, ge=0, le=20)
    # tool-call budget for staged build sessions — ~2 calls per feature
    # (add_block + verify) plus checkpoints/done, so above the one-shot cap
    agent_tool_max_calls_build: int = Field(default=16, ge=1, le=100)
    # best-of-N first codegen attempt (AlphaCode-lite generate-then-filter):
    # N independent tool sessions sampled in parallel — they share the prompt
    # prefix under vLLM APC so cost is ~linear in completion tokens only —
    # each candidate is light-verified and the best wins. 1 = single attempt
    # (unchanged behaviour). Only the first iteration samples; repairs,
    # improves and later iterations stay single.
    agent_codegen_samples: int = Field(default=1, ge=1, le=8)
    # include a compact runtime-introspected signature reference for the
    # relevant topics in codegen/repair context (the on-demand quickref tool
    # is always available regardless of this flag)
    agent_quickref_context: bool = False
    # run a parameter-robustness audit (perturb each PARAMS value, re-execute,
    # report brittle params) as a quality signal before finishing
    agent_design_audit: bool = False
    agent_mesh_tools: bool = True
    agent_mesh_max_ops: int = Field(default=24, ge=1, le=200)
    agent_mesh_preview_max: int = Field(default=4, ge=0, le=20)
    agent_mesh_refine_rounds: int = Field(default=1, ge=0, le=10)
    # refine shape guard: an op whose result drifts below this silhouette IoU
    # vs the baseline (or loses >15% volume on a non-cut op) is refused —
    # guards the workbench against flatten/squash wreckage; 0 disables
    agent_mesh_shape_guard_iou: float = Field(default=0.8, ge=0.0, le=1.0)
    agent_mesh_hard_fail_score: int = Field(default=30, ge=0, le=100)
    agent_mesh_hard_fail_limit: int = Field(default=3, ge=1, le=20)
    agent_max_approaches: int = Field(default=2, ge=1, le=3)
    agent_stall_patience: int = Field(default=2, ge=0, le=50)  # 0 = never stall-stop
    agent_switch_score: int = Field(default=60, ge=0, le=100)
    agent_switch_after_iters: int = Field(default=4, ge=1, le=20)
    agent_struct_tools: bool = True
    agent_topopt_max_runs: int = Field(default=6, ge=0, le=20)
    agent_fem_max_runs: int = Field(default=4, ge=0, le=20)
    # chat agent: which model role drives the conversation. "vision" receives
    # tool-result images (renders, concept images); "code" is text-only.
    chat_model_role: Literal["vision", "code"] = "vision"
    chat_tool_max_calls: int = Field(default=30, ge=1, le=200)
    chat_max_turns: int = Field(default=40, ge=1, le=200)
    chat_spawn_timeout_s: int = Field(default=2700, ge=60, le=86_400)
    chat_history_messages: int = Field(default=60, ge=4, le=500)
    chat_image_max: int = Field(default=8, ge=0, le=50)  # images per user message
    fem_voxel_mm: float = Field(default=1.5, gt=0, le=100)
    fem_max_elements: int = Field(default=150_000, ge=1, le=5_000_000)
    fem_spsolve_dofs: int = Field(default=120_000, ge=1_000, le=5_000_000)
    fem_max_effective_cases: int = Field(
        default=8, ge=1, le=64
    )  # cap on load-case expansion (direction cones)
    fem_timeout_s: int = Field(default=600, ge=1, le=86_400)
    topopt_timeout_s: int = Field(default=2700, ge=1, le=86_400)
    topopt_max_iterations: int = Field(default=80, ge=1, le=1000)
    topopt_default_volfrac: float = Field(default=0.4, gt=0, le=1)
    topopt_default_rmin_mm: float = Field(default=3, gt=0, le=1000)
    topopt_smooth_iterations: int = Field(default=6, ge=0, le=30)
    topopt_min_member_mm: float = Field(
        default=0.0, ge=0, le=50
    )  # 0 = off; >0 thickens members below this size
    topopt_auto_keep: bool = True  # auto keep-mode default: hold detected fastener anchors
    topopt_auto_keep_margin_mm: float = Field(default=3.0, gt=0, le=50)
    topopt_multires_min_cells: int = Field(
        default=4000, ge=0, le=10_000_000
    )  # multires coarse pass only pays off above this solid-cell count
    # auto-enqueue a feature-finishing 'refine' child after a succeeded topopt
    # run — the standalone optimize job AND the in-job topopt phase — whenever
    # the part has features/anchors to re-cut (voxel-rough bores, destroyed
    # threads otherwise ship as-is). Worker-side only: run_refine_job loads
    # runtime settings itself, so this stays out of executor_values().
    topopt_auto_refine: bool = True
    mesh_base_url: str = ""
    mesh_model: str = "auto"
    image_model: str = "auto"
    image_edit_model: str = "auto"  # default adapter for edit/i2i requests
    image_default_strength: float = Field(default=0.6, gt=0, le=1)  # img2img denoise
    image_steps: int | None = None  # None = adapter default (FLUX:4, SDXL:30)
    image_guidance: float | None = None  # None = adapter default
    image_max_side: int = Field(default=1024, ge=64, le=4096)  # w/h clamp — OOM prevention
    image_max_count: int = Field(default=4, ge=1, le=16)  # per-request clamp
    image_autogen_mesh: bool = True  # synthesise hero image for mesh jobs w/o inputs
    image_tool_budget: int = Field(default=4, ge=0, le=50)  # concept-image calls per session
    image_timeout_s: int = Field(default=300, ge=1, le=86_400)
    mesh_timeout_s: int = Field(default=900, ge=1, le=86_400)
    # abort polling a server job whose (progress,message) pair hasn't moved
    # for this long — MPS/MLX adapters can freeze at 0% CPU and would
    # otherwise be polled until the full timeout (observed: 9.5% for >10min)
    mesh_stall_timeout_s: int = Field(default=300, ge=1, le=86_400)
    # refuse to submit a generation job when the sidecar host reports less
    # free RAM — a job launched under memory pressure stalls at 0% CPU and
    # swaps the host into a watchdog panic. 0 disables the gate.
    mesh_min_free_mb: int = Field(default=0, ge=0)
    mesh_decimate_faces: int = Field(default=30_000, ge=1)
    # face cap for the hybrid work-solid (sew+boolean in OCCT) — dominates
    # hybrid iteration latency; the export GLB keeps full detail. Keep at or
    # below mesh_decimate_faces; >60k makes every executor call minutes-slow.
    mesh_max_faces_for_cad: int = Field(default=30_000, ge=1)
    mesh_smooth_after_flatten_iters: int = Field(default=10, ge=0, le=30)
    mesh_candidates_per_iter: int = Field(default=2, ge=1, le=4)
    mesh_candidate_steps: int = Field(default=15, ge=5, le=100)
    mesh_candidate_adapters: str = ""  # csv adapter ids; "" = auto spread over available
    mesh_heightfield_flat: bool = True  # deterministic flat-part candidate
    mesh_prep_images: bool = True
    mesh_cache_enabled: bool = True
    exec_timeout_s: int = Field(default=180, ge=1, le=86_400)
    exec_mem_limit_mb: int = Field(default=4096, ge=256, le=1_048_576)
    # warm sandbox: one pre-imported zygote forks per job instead of paying
    # the ~4s build123d cold-import on every execution (incl. every failed
    # repair). Falls back to the cold subprocess path on any zygote failure.
    exec_warm_runner: bool = True

    @field_validator("llm_extra_body", "llm_thinking_body_on", "llm_thinking_body_off")
    @classmethod
    def validate_extra_body(cls, value: str) -> str:
        if not value:
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("must be a JSON object")
        return value

    def to_storage(self) -> dict[str, Any]:
        """Sparse dict: only non-default values are persisted — an absent key
        means 'use the code default'."""
        value = self.model_dump(mode="python", exclude_defaults=True)
        key = self.llm_api_key.get_secret_value()
        if key:
            value["llm_api_key"] = key
        return value

    def public_values(self) -> dict[str, Any]:
        value = self.model_dump(exclude={"llm_api_key"})
        value["llm_api_key_set"] = bool(self.llm_api_key.get_secret_value())
        return value

    def executor_values(self) -> dict[str, Any]:
        """Subset shipped to the cad-executor process inside each task payload."""
        return self.model_dump(mode="json", include=_EXECUTOR_FIELDS)


_EXECUTOR_FIELDS = {
    "exec_timeout_s",
    "exec_mem_limit_mb",
    "exec_warm_runner",
    "fem_voxel_mm",
    "fem_max_elements",
    "fem_spsolve_dofs",
    "fem_max_effective_cases",
    "topopt_default_volfrac",
    "topopt_default_rmin_mm",
    "topopt_max_iterations",
    "topopt_smooth_iterations",
    "topopt_min_member_mm",
    "topopt_auto_keep",
    "topopt_auto_keep_margin_mm",
    "topopt_multires_min_cells",
    "mesh_max_faces_for_cad",
    "mesh_decimate_faces",
    "mesh_smooth_after_flatten_iters",
}


async def load_runtime_settings(db: AsyncSession) -> RuntimeSettings:
    row = await db.get(AppSetting, "runtime")
    values = RuntimeSettings().to_storage()
    if row:
        raw = row.value or {}
        values.update(raw.get("settings", raw))
    return RuntimeSettings.model_validate(values)


async def save_runtime_settings(db: AsyncSession, settings: RuntimeSettings) -> None:
    row = await db.get(AppSetting, "runtime")
    stored = {
        "schema_version": 1,
        "settings": settings.to_storage(),
    }
    if row is None:
        db.add(AppSetting(key="runtime", value=stored))
    else:
        row.value = stored


async def runtime_from_db_or_default() -> RuntimeSettings:
    """CLI/eval entry point: DB snapshot when reachable, else code defaults."""
    try:
        from cadx.db.session import session_scope

        async with session_scope() as db:
            return await load_runtime_settings(db)
    except Exception:
        return RuntimeSettings()


async def ensure_runtime_seeded(db: AsyncSession) -> bool:
    """Create the 'runtime' row once. mesh_base_url/mesh_model/image_model are
    seeded from MESH_BASE_URL/MESH_MODEL/MESH_IMAGE_MODEL env vars (deployment
    topology set by the launcher); afterwards the DB row is the only source and
    env changes are ignored."""
    import os

    if await db.get(AppSetting, "runtime") is not None:
        return False
    settings = RuntimeSettings()
    seed_url = os.environ.get("MESH_BASE_URL", "").strip()
    if seed_url:
        settings.mesh_base_url = seed_url
    seed_model = os.environ.get("MESH_MODEL", "").strip()
    if seed_model:
        settings.mesh_model = seed_model
    seed_img = os.environ.get("MESH_IMAGE_MODEL", "").strip()
    if seed_img:
        settings.image_model = seed_img
    await save_runtime_settings(db, settings)
    return True
