"""DB-backed application runtime settings."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from cadx.db.models import AppSetting
from cadx.secrets_crypto import decrypt_stored, encrypt_secret

# write-only credentials — raw values live only in to_storage()/internal
# use; public_values() exposes "<name>_set" booleans instead. to_storage()
# encrypts them (secrets_crypto); stored "enc:v1:" values are decrypted by
# the _decrypt_secrets validator on load.
_SECRET_FIELDS = ("llm_api_key", "tripo_api_key", "image_api_key", "hf_token")


class RuntimeSettings(BaseModel):
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    # provider routing: "external" = llm_base_url provider (OpenAI-compatible
    # cloud/self-hosted), "sidecar" = the OpenAI-compatible endpoint on the
    # local mesh-server sidecar (mesh_base_url, no auth)
    llm_backend: Literal["external", "sidecar"] = "external"
    # optional single local model for both roles when llm_backend="sidecar"
    # ("" = llm_model_vision/llm_model_code apply there too)
    llm_sidecar_model: str = ""
    llm_model_vision: str = "gpt-4o"
    llm_model_code: str = "gpt-4o"
    llm_json_mode: Literal["json_object", "json_schema", "none"] = "json_object"
    llm_temperature_code: float = Field(default=0.2, ge=0, le=2)
    # context window of the served model, tokens: 0 = auto-detect via the
    # provider's /models + native APIs (llm/capabilities.py, cached in the
    # AppSetting row "llm_caps"); >0 = manual override for both roles
    llm_context_tokens: int = Field(default=0, ge=0, le=8_000_000)
    # completion budget per call: 0 = auto — derived from the detected
    # context window as clamp(ctx//4, 2048, 128_000) and additionally
    # clamped so prompt+completion fits the window; >0 = hard cap
    llm_max_tokens: int = Field(default=0, ge=0, le=2_000_000)
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
    # escalation-stage bodies: the orchestrator raises ctx.thinking_level on
    # stall/regression — level 1 merges body_medium, level 2 body_high
    # (each falls back to llm_thinking_body_on when unset/empty)
    llm_thinking_body_medium: str = ""
    llm_thinking_body_high: str = ""
    llm_thinking_hard_steps: str = ""  # csv step names that always think in "auto"
    # iterations without progress before "auto" escalates thinking one stage
    # (medium, then high on the next stall/regression) — the latch stays for
    # the rest of the job
    llm_thinking_escalate_after: int = Field(default=2, ge=1, le=50)
    # hard per-job cap on calls made WITH thinking on — reasoning tokens are
    # the dominant completion cost and escalation can amplify a non-converging
    # job (observed: 260k completion tokens without convergence). Once spent,
    # thinking_on() resolves OFF for the rest of the job regardless of policy
    # or the escalation latch. 0 disables thinking entirely.
    llm_thinking_max_calls: int = Field(default=8, ge=0, le=10_000)
    # completion budget for calls made WITH thinking on — reasoning tokens
    # count toward max_tokens, so an xhigh call can otherwise spend the whole
    # llm_max_tokens budget reasoning and return truncated/empty content
    # (parse failure → retry loop). Merged as max_tokens into the resolved
    # thinking body; 0 = keep llm_max_tokens for thinking calls too.
    llm_thinking_max_tokens: int = Field(default=0, ge=0, le=2_000_000)
    # vision payload size: tool-result renders/contact sheets are the
    # heaviest prompt item in long sessions — "low" detail and/or a smaller
    # max edge cut the per-image token cost substantially
    llm_image_detail: Literal["high", "low"] = "high"
    llm_image_max_edge: int = Field(default=1568, ge=256, le=4096)
    agent_max_iterations: int = Field(default=3, ge=1, le=200)
    agent_max_repairs: int = Field(default=3, ge=0, le=200)
    agent_pass_score: int = Field(default=80, ge=0, le=100)
    agent_dim_tolerance_pct: float = Field(default=2, ge=0, le=100)
    agent_dim_tolerance_mm: float = Field(default=0.5, ge=0, le=1000)
    agent_clarification_mode: Literal["assume", "ask"] = "assume"
    agent_max_input_images: int = Field(default=8, ge=0, le=100)
    # fraction of the detected context window a prompt may occupy before the
    # tool loop condenses history / trims tool results early — bounds the
    # resend before the provider rejects the request with a context error
    agent_context_budget_frac: float = Field(default=0.8, gt=0.1, le=0.95)
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
    # sparse-direct backend for the Kff solve: auto picks CHOLMOD when the
    # optional sksparse dep is importable (SuiteSparse inside the executor
    # image), else SuperLU. Iterative tier picks AMG (pyamg) above the DOF
    # cap, else SSOR/diagonal CG. Force a backend to benchmark/fallback.
    fem_solver_backend: str = Field(default="auto", max_length=20)  # auto|cholmod|superlu|amg|ssor
    fem_max_effective_cases: int = Field(
        default=8, ge=1, le=64
    )  # cap on load-case expansion (direction cones)
    fem_timeout_s: int = Field(default=600, ge=1, le=86_400)
    # export-verification hardening (topopt/infill pipelines): realizations
    # re-solve the export mask eroded/dilated and take the worst SF;
    # coverage weights boundary-cell E by fill fraction; the min-cut
    # certificate bounds SF from pure geometry. mesh_shift stays opt-in —
    # a second full export FEM on a half-pitch-shifted grid is expensive.
    fem_verify_realizations: bool = True
    fem_verify_realization_cells: int = Field(default=1, ge=1, le=4)
    fem_verify_coverage_e: bool = True
    fem_verify_mesh_shift: bool = False
    fem_loadpath_cert: bool = True
    # homogenized infill model: verifies the intended density field with
    # effective lattice tensors C_H(rho) next to the resolved binary mask
    fem_homogenized_infill: bool = True
    fem_homog_cell_voxels: int = Field(default=16, ge=8, le=48)
    # octet strut families for infill_style="struts_aligned": body-diagonal
    # families give 45°-dominated stress states real diagonal struts
    # instead of stair-stepped axis members
    infill_octet: bool = True
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
    # multires coarse→fine SIMP warm start — default ON (~3x faster on big
    # parts); a request/tool-level `multires: false` still opts out per run
    topopt_multires: bool = True
    # auto-enqueue a feature-finishing 'refine' child after a succeeded topopt
    # run — the standalone optimize job AND the in-job topopt phase — whenever
    # the part has features/anchors to re-cut (voxel-rough bores, destroyed
    # threads otherwise ship as-is). Worker-side only: run_refine_job loads
    # runtime settings itself, so this stays out of executor_values().
    topopt_auto_refine: bool = True
    # post-topopt refine: a thread/tap feature whose pitch is derivable
    # (thread_spec 'M8-1.25' or a standard metric dia) is re-cut with the
    # real helical V-groove, not just the tap-drill cylinder — the tap-drill
    # bore is always kept as fallback. Executor-side flag: ships via
    # executor_values() into the refine payload.
    refine_helical_threads: bool = True
    mesh_base_url: str = ""
    mesh_model: str = "auto"
    # provider routing: "sidecar" = local mesh-server, "tripo" = Tripo3D cloud
    mesh_backend: Literal["sidecar", "tripo"] = "sidecar"
    # provider routing: "sidecar" = local mesh-server, "openai" = OpenAI-
    # compatible images API (image_api_base falls back to llm_base_url)
    image_backend: Literal["sidecar", "openai"] = "sidecar"
    tripo_api_key: SecretStr = SecretStr("")
    tripo_api_base: str = "https://api.tripo3d.ai/v2/openapi"
    tripo_model: str = "v2.5-20250123"
    image_openai_model: str = "gpt-image-1"
    image_api_base: str = ""  # empty -> llm_base_url
    image_api_key: SecretStr = SecretStr("")  # empty -> llm_api_key
    # Hugging Face access token — forwarded to the mesh sidecar as the
    # X-HF-Token header; the server applies it as HF_TOKEN for downloads and
    # model loads (gated repos, higher rate limits)
    hf_token: SecretStr = SecretStr("")
    image_model: str = "auto"
    image_edit_model: str = "auto"  # default adapter for edit/i2i requests
    image_default_strength: float = Field(default=0.6, gt=0, le=1)  # img2img denoise
    # vision check on synthesised hero/concept images: extra generation
    # attempts when the verdict is "major" (worker-side, not executor)
    image_verify_retries: int = Field(default=2, ge=0, le=4)
    image_steps: int | None = None  # None = adapter default (FLUX:4, SDXL:30)
    image_guidance: float | None = None  # None = adapter default
    image_max_side: int = Field(default=1024, ge=64, le=4096)  # w/h clamp — OOM prevention
    image_max_count: int = Field(default=4, ge=1, le=16)  # per-request clamp
    image_autogen_mesh: bool = True  # synthesise hero image for mesh jobs w/o inputs
    image_tool_budget: int = Field(default=4, ge=0, le=50)  # concept-image calls per session
    image_timeout_s: int = Field(default=1800, ge=1, le=86_400)
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
    # hard wall-clock budget for the isolated analytic-STEP subprocess —
    # OCC booleans on sewn mesh solids are unbounded, so a pathological
    # solid degrades to the faceted STEP instead of stalling the job
    analytic_step_timeout_s: int = Field(default=240, ge=10, le=3600)
    mesh_smooth_after_flatten_iters: int = Field(default=10, ge=0, le=30)
    mesh_candidates_per_iter: int = Field(default=2, ge=1, le=4)
    mesh_candidate_steps: int = Field(default=15, ge=5, le=100)
    mesh_candidate_adapters: str = ""  # csv adapter ids; "" = auto spread over available
    mesh_heightfield_flat: bool = True  # deterministic flat-part candidate
    mesh_prep_images: bool = True
    mesh_cache_enabled: bool = True
    exec_timeout_s: int = Field(default=180, ge=1, le=86_400)
    # 0 = auto: 60% of the detected (cgroup-aware) memory, clamped to
    # 2048..32768 MB — cad/resources.effective_exec_mem_mb resolves it at
    # call time so the executor's own container limit wins over a stale
    # host-detection
    exec_mem_limit_mb: int = Field(default=0, ge=0, le=1_048_576)
    # warm sandbox: one pre-imported zygote forks per job instead of paying
    # the ~4s build123d cold-import on every execution (incl. every failed
    # repair). Falls back to the cold subprocess path on any zygote failure.
    exec_warm_runner: bool = True
    # 0 = auto: threads a single CPU-bound stage may use (OCC meshing,
    # BLAS, llvmpipe raster) — min(detected cpus, 8)
    exec_threads: int = Field(default=0, ge=0, le=64)
    # 0 = auto: fork/spawn pool workers for parallel FEM load cases,
    # topopt verifies and render fan-out — min(4, cpus-1); 1 disables
    exec_pool_workers: int = Field(default=0, ge=0, le=16)
    # 0 = auto: parallel perturbation runs in the design audit
    # (baseline + 2 sandbox runs per PARAM) — min(4, cpus-1)
    audit_parallel_runs: int = Field(default=0, ge=0, le=8)

    @field_validator(*_SECRET_FIELDS, mode="before")
    @classmethod
    def _decrypt_secrets(cls, value: Any) -> Any:
        return decrypt_stored(value)

    @field_validator(
        "llm_extra_body",
        "llm_thinking_body_on",
        "llm_thinking_body_off",
        "llm_thinking_body_medium",
        "llm_thinking_body_high",
    )
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
        # non-default SecretStrs dump as SecretStr objects — replace with the
        # raw secret so the JSON column stays serialisable (empty == default,
        # already excluded)
        for name in _SECRET_FIELDS:
            key = getattr(self, name).get_secret_value()
            if key:
                value[name] = encrypt_secret(key)
        return value

    def public_values(self) -> dict[str, Any]:
        value = self.model_dump(exclude=set(_SECRET_FIELDS))
        for name in _SECRET_FIELDS:
            value[f"{name}_set"] = bool(getattr(self, name).get_secret_value())
        return value

    def executor_values(self) -> dict[str, Any]:
        """Subset shipped to the cad-executor process inside each task payload."""
        return self.model_dump(mode="json", include=_EXECUTOR_FIELDS)


_EXECUTOR_FIELDS = {
    "exec_timeout_s",
    "exec_mem_limit_mb",
    "exec_warm_runner",
    "exec_threads",
    "exec_pool_workers",
    "audit_parallel_runs",
    "fem_voxel_mm",
    "fem_max_elements",
    "fem_spsolve_dofs",
    "fem_solver_backend",
    "fem_max_effective_cases",
    "fem_verify_realizations",
    "fem_verify_realization_cells",
    "fem_verify_coverage_e",
    "fem_verify_mesh_shift",
    "fem_loadpath_cert",
    "fem_homogenized_infill",
    "fem_homog_cell_voxels",
    "infill_octet",
    "topopt_default_volfrac",
    "topopt_default_rmin_mm",
    "topopt_max_iterations",
    "topopt_smooth_iterations",
    "topopt_min_member_mm",
    "topopt_auto_keep",
    "topopt_auto_keep_margin_mm",
    "topopt_multires_min_cells",
    "topopt_multires",
    "refine_helical_threads",
    "mesh_max_faces_for_cad",
    "analytic_step_timeout_s",
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
    seed_hf = os.environ.get("HF_TOKEN", "").strip()
    if seed_hf:
        settings.hf_token = SecretStr(seed_hf)
    await save_runtime_settings(db, settings)
    return True
