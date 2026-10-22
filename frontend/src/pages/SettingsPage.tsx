import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useSearchParams } from "react-router-dom";
import ModelsTab from "../components/ModelsTab";
import UpdateCard from "../components/UpdateCard";
import {
  Btn,
  Card,
  Chip,
  Field,
  PageShell,
  SectionTitle,
  Spinner,
  inputCls,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import {
  fmtMemGb,
  modelStateLabel,
  modelStateTone,
  urlHost,
} from "../lib/models";
import type { Settings, SettingsUpdate } from "../lib/types";
import { useTitle } from "../lib/useTitle";

interface FieldSpec {
  key: keyof Settings;
  label: string;
  hint?: string;
  kind?: "text" | "number" | "bool";
  options?: [string, string][];
  /** span the full row of the two-column field grid (URLs, JSON, long hints) */
  wide?: boolean;
}

/** responsive field layout — one column on narrow screens, two from sm up */
const FIELD_GRID = "grid grid-cols-1 gap-x-4 sm:grid-cols-2";

const NUM_FIELDS = new Set<string>([
  "llm_temperature_code",
  "llm_context_tokens",
  "llm_max_tokens",
  "llm_timeout_s",
  "llm_idle_timeout_s",
  "llm_degenerate_stall_s",
  "llm_total_timeout_s",
  "agent_max_iterations",
  "agent_max_repairs",
  "agent_pass_score",
  "agent_dim_tolerance_pct",
  "agent_dim_tolerance_mm",
  "agent_max_input_images",
  "agent_context_budget_frac",
  "agent_max_tokens_per_job",
  "agent_max_wall_s",
  "agent_patch_retries",
  "agent_tool_max_calls",
  "agent_tool_max_calls_info",
  "agent_tool_verify_max",
  "agent_mesh_max_ops",
  "agent_mesh_preview_max",
  "agent_mesh_refine_rounds",
  "agent_mesh_hard_fail_score",
  "agent_mesh_hard_fail_limit",
  "chat_tool_max_calls",
  "chat_max_turns",
  "chat_spawn_timeout_s",
  "chat_history_messages",
  "chat_image_max",
  "fem_voxel_mm",
  "fem_max_elements",
  "fem_timeout_s",
  "fem_verify_realization_cells",
  "fem_homog_cell_voxels",
  "topopt_timeout_s",
  "topopt_max_iterations",
  "topopt_default_volfrac",
  "topopt_default_rmin_mm",
  "mesh_timeout_s",
  "mesh_stall_timeout_s",
  "mesh_min_free_mb",
  "mesh_decimate_faces",
  "mesh_max_faces_for_cad",
  "image_steps",
  "image_guidance",
  "image_default_strength",
  "image_max_side",
  "image_max_count",
  "image_tool_budget",
  "image_timeout_s",
  "exec_timeout_s",
  "exec_mem_limit_mb",
  "exec_threads",
  "exec_pool_workers",
  "audit_parallel_runs",
]);

const BOOL_FIELDS = new Set<string>([
  "llm_stream",
  "agent_param_autofix",
  "agent_feature_feedback",
  "agent_critique_reask",
  "agent_tool_verify",
  "agent_mesh_tools",
  "image_autogen_mesh",
  "fem_verify_realizations",
  "fem_verify_coverage_e",
  "fem_verify_mesh_shift",
  "fem_loadpath_cert",
  "fem_homogenized_infill",
  "infill_octet",
]);

// mirrors the bounds in backend SettingsUpdate — keep in sync with schemas.py
const FIELD_BOUNDS: Record<
  string,
  { min?: number; max?: number; exclusiveMin?: boolean; int?: boolean }
> = {
  llm_temperature_code: { min: 0, max: 2 },
  llm_context_tokens: { min: 0, max: 8_000_000, int: true },
  llm_max_tokens: { min: 0, max: 2_000_000, int: true },
  llm_timeout_s: { min: 0, exclusiveMin: true },
  llm_idle_timeout_s: { min: 0, exclusiveMin: true },
  llm_degenerate_stall_s: { min: 0, exclusiveMin: true },
  llm_total_timeout_s: { min: 0, exclusiveMin: true },
  agent_max_iterations: { min: 1, max: 200, int: true },
  agent_max_repairs: { min: 0, max: 200, int: true },
  agent_pass_score: { min: 0, max: 100, int: true },
  agent_dim_tolerance_pct: { min: 0, max: 100 },
  agent_dim_tolerance_mm: { min: 0, max: 1000 },
  agent_max_input_images: { min: 0, max: 100, int: true },
  agent_context_budget_frac: { min: 0.1, max: 0.95, exclusiveMin: true },
  agent_max_tokens_per_job: { min: 1, int: true },
  agent_max_wall_s: { min: 0, exclusiveMin: true },
  agent_patch_retries: { min: 0, max: 20, int: true },
  agent_tool_max_calls: { min: 1, max: 100, int: true },
  agent_tool_max_calls_info: { min: 1, max: 100, int: true },
  agent_tool_verify_max: { min: 0, max: 20, int: true },
  agent_mesh_max_ops: { min: 1, max: 200, int: true },
  agent_mesh_preview_max: { min: 0, max: 20, int: true },
  agent_mesh_refine_rounds: { min: 0, max: 10, int: true },
  agent_mesh_hard_fail_score: { min: 0, max: 100, int: true },
  agent_mesh_hard_fail_limit: { min: 1, max: 20, int: true },
  chat_tool_max_calls: { min: 1, max: 200, int: true },
  chat_max_turns: { min: 1, max: 200, int: true },
  chat_spawn_timeout_s: { min: 60, max: 86_400, int: true },
  chat_history_messages: { min: 4, max: 500, int: true },
  chat_image_max: { min: 0, max: 50, int: true },
  fem_voxel_mm: { min: 0, max: 100, exclusiveMin: true },
  fem_max_elements: { min: 1, max: 5_000_000, int: true },
  fem_timeout_s: { min: 1, max: 86_400, int: true },
  fem_verify_realization_cells: { min: 1, max: 4, int: true },
  fem_homog_cell_voxels: { min: 8, max: 48, int: true },
  topopt_timeout_s: { min: 1, max: 86_400, int: true },
  topopt_max_iterations: { min: 1, max: 1000, int: true },
  topopt_default_volfrac: { min: 0, max: 1, exclusiveMin: true },
  topopt_default_rmin_mm: { min: 0, max: 1000, exclusiveMin: true },
  mesh_timeout_s: { min: 1, max: 86_400, int: true },
  mesh_stall_timeout_s: { min: 1, max: 86_400, int: true },
  mesh_min_free_mb: { min: 0, int: true },
  mesh_decimate_faces: { min: 1, int: true },
  mesh_max_faces_for_cad: { min: 1, int: true },
  image_steps: { min: 1, max: 200, int: true },
  image_guidance: { min: 0, max: 50 },
  image_default_strength: { min: 0, max: 1, exclusiveMin: true },
  image_max_side: { min: 64, max: 4096, int: true },
  image_max_count: { min: 1, max: 16, int: true },
  image_tool_budget: { min: 0, max: 50, int: true },
  image_timeout_s: { min: 1, max: 86_400, int: true },
  exec_timeout_s: { min: 1, max: 86_400, int: true },
  exec_mem_limit_mb: { min: 0, max: 1_048_576, int: true },
  exec_threads: { min: 0, max: 64, int: true },
  exec_pool_workers: { min: 0, max: 16, int: true },
  audit_parallel_runs: { min: 0, max: 8, int: true },
};

type TabId = "provider" | "models" | "agent" | "mesh" | "fem" | "system";
const TAB_IDS_LIST: TabId[] = ["provider", "models", "agent", "mesh", "fem", "system"];

const TAB_IDS = new Set<string>(TAB_IDS_LIST);

const GEN_MODE_OPTS = [["auto"], ["cad"], ["mesh"], ["hybrid"], ["topopt"]] as const;
const CLAR_OPTS = [["assume"], ["ask"]] as const;
const PATCH_OPTS = [["auto"], ["always"], ["never"]] as const;
const CHAT_ROLE_OPTS = [["vision"], ["code"]] as const;
const JSON_MODE_OPTS = [["json_object"], ["json_schema"], ["none"]] as const;
const LLM_TOOLS_OPTS = [["auto"], ["on"], ["off"]] as const;

/** compact token-count rendering for context-window chips: 131k, 262k, 1M */
const fmtCtx = (n: number | null | undefined): string => {
  if (!n || n <= 0) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
};

export default function SettingsPage() {
  const { t } = useTranslation("settings");
  useTitle(t("title"));
  const qc = useQueryClient();
  const [sp, setSp] = useSearchParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<TabId>(() => {
    const q = sp.get("tab") ?? "";
    if (TAB_IDS.has(q)) return q as TabId;
    const saved = localStorage.getItem("settings.tab") ?? "";
    return TAB_IDS.has(saved) ? (saved as TabId) : "provider";
  });

  // deep links like /settings?tab=fem while already on the page;
  // the library moved to its own route — keep old links working
  useEffect(() => {
    const q = sp.get("tab") ?? "";
    if (q === "library") {
      navigate("/library", { replace: true });
      return;
    }
    if (TAB_IDS.has(q) && q !== tab) setTab(q as TabId);
  }, [sp, navigate, tab]);
  const [form, setForm] = useState<Record<string, string> | null>(null);
  const [initial, setInitial] = useState<Record<string, string> | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [tripoKey, setTripoKey] = useState("");
  const [clearTripoKey, setClearTripoKey] = useState(false);
  const [imageKey, setImageKey] = useState("");
  const [clearImageKey, setClearImageKey] = useState(false);
  const [hfKey, setHfKey] = useState("");
  const [clearHfKey, setClearHfKey] = useState(false);
  // per-field "custom…" mode for the external-LLM model comboboxes
  const [customModel, setCustomModel] = useState<Record<string, boolean>>({});
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState("");

  const settings = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: false,
  });
  const meshHealth = useQuery({
    queryKey: ["meshHealth"],
    queryFn: api.meshHealth,
    refetchInterval: 30_000,
    retry: false,
  });
  const imageModels = useQuery({
    queryKey: ["imageModels"],
    queryFn: api.imageModels,
    refetchInterval: 30_000,
    retry: false,
  });
  const llmHealth = useQuery({
    queryKey: ["llmHealth"],
    queryFn: api.llmHealth,
    refetchInterval: 60_000,
    retry: false,
  });
  const llmModels = useQuery({
    queryKey: ["llmModels"],
    queryFn: api.llmModels,
    refetchInterval: 60_000,
    retry: false,
  });
  // full model registry (llm/mesh/image kinds) — /mesh/health only carries
  // mesh+image adapters; poll faster while a download is in flight
  const meshModelsList = useQuery({
    queryKey: ["meshModels"],
    queryFn: api.meshModels,
    refetchInterval: (q) =>
      (q.state.data ?? []).some((m) => m.downloading) ? 3000 : 30_000,
    retry: false,
  });
  const testProvider = useMutation({
    mutationFn: api.testProvider,
    // a manual test doubles as the shared health probe — keep card + sidebar in sync
    onSuccess: (d) => qc.setQueryData(["llmHealth"], d),
  });

  // seed the form once settings arrive (the api key itself is never sent back)
  useEffect(() => {
    if (form || !settings.data) return;
    const d = settings.data;
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(d)) {
      if (
        k === "data_dir" ||
        k === "llm_api_key_set" ||
        k === "tripo_api_key_set" ||
        k === "image_api_key_set" ||
        k === "hf_token_set"
      )
        continue;
      out[k] = v == null ? "" : String(v); // nullable fields show empty = default
    }
    // backend selectors need a concrete value even when an older backend
    // payload omits the keys entirely
    out.llm_backend ??= "external";
    out.mesh_backend ??= "sidecar";
    out.image_backend ??= "sidecar";
    setForm(out);
    setInitial(out);
  }, [settings.data, form]);

  const dirty = useMemo(() => {
    if (!form || !initial) return false;
    if (
      apiKey.trim() ||
      clearKey ||
      tripoKey.trim() ||
      clearTripoKey ||
      imageKey.trim() ||
      clearImageKey ||
      hfKey.trim() ||
      clearHfKey
    )
      return true;
    for (const k of new Set([...Object.keys(form), ...Object.keys(initial)])) {
      if ((form[k] ?? "") !== (initial[k] ?? "")) return true;
    }
    return false;
  }, [
    form,
    initial,
    apiKey,
    clearKey,
    tripoKey,
    clearTripoKey,
    imageKey,
    clearImageKey,
    hfKey,
    clearHfKey,
  ]);

  const revert = () => {
    setForm(initial);
    setApiKey("");
    setClearKey(false);
    setTripoKey("");
    setClearTripoKey(false);
    setImageKey("");
    setClearImageKey(false);
    setHfKey("");
    setClearHfKey(false);
    setSaveError("");
  };

  const save = useMutation({
    mutationFn: () => {
      // client-side bounds check — surfaces the field name instead of a raw 422
      const errs: string[] = [];
      for (const [k, v] of Object.entries(form!)) {
        if (!NUM_FIELDS.has(k) || v === "") continue;
        const n = Number(v);
        if (!Number.isFinite(n)) {
          errs.push(`${k}: ${t("err_nan")}`);
          continue;
        }
        const b = FIELD_BOUNDS[k];
        if (!b) continue;
        if (b.int && !Number.isInteger(n)) errs.push(`${k}: ${t("err_int")}`);
        else if (b.min != null && (b.exclusiveMin ? n <= b.min : n < b.min))
          errs.push(`${k}: ${t("err_min", { op: b.exclusiveMin ? ">" : "≥", min: b.min })}`);
        else if (b.max != null && n > b.max) errs.push(`${k}: ${t("err_max", { max: b.max })}`);
      }
      if (errs.length) throw new Error(errs.join("; "));

      const body: SettingsUpdate = {};
      for (const [k, v] of Object.entries(form!)) {
        // empty input → null → backend resets the field to its default
        if (NUM_FIELDS.has(k)) {
          const n = Number(v);
          if (v !== "" && !Number.isFinite(n)) continue;
          (body as Record<string, unknown>)[k] = v === "" ? null : n;
        } else if (BOOL_FIELDS.has(k)) {
          (body as Record<string, unknown>)[k] = v === "true";
        } else {
          (body as Record<string, unknown>)[k] = v === "" ? null : v;
        }
      }
      if (clearKey) body.clear_llm_api_key = true;
      else if (apiKey.trim()) body.llm_api_key = apiKey.trim();
      if (clearTripoKey) body.clear_tripo_api_key = true;
      else if (tripoKey.trim()) body.tripo_api_key = tripoKey.trim();
      if (clearImageKey) body.clear_image_api_key = true;
      else if (imageKey.trim()) body.image_api_key = imageKey.trim();
      if (clearHfKey) body.clear_hf_token = true;
      else if (hfKey.trim()) body.hf_token = hfKey.trim();
      return api.updateSettings(body);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["meshHealth"] });
      qc.invalidateQueries({ queryKey: ["meshModels"] });
      qc.invalidateQueries({ queryKey: ["health"] });
      qc.invalidateQueries({ queryKey: ["llmHealth"] });
      qc.invalidateQueries({ queryKey: ["llmModels"] });
      setApiKey("");
      setClearKey(false);
      setTripoKey("");
      setClearTripoKey(false);
      setImageKey("");
      setClearImageKey(false);
      setHfKey("");
      setClearHfKey(false);
      setSaveError("");
      setInitial(form);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
    onError: (e) => setSaveError(errMsg(e)),
  });

  if (settings.isLoading || !form)
    return (
      <div className="flex justify-center py-24 text-faint">
        <Spinner className="h-6 w-6" />
      </div>
    );
  const s = settings.data as Settings;

  const set = (key: string, v: string) => setForm((f) => ({ ...f!, [key]: v }));

  const field = (spec: FieldSpec) => {
    const { key, label, hint } = spec;
    const b = spec.kind === "number" ? FIELD_BOUNDS[key] : undefined;
    const el = spec.options ? (
      <Field key={key} label={label} hint={hint}>
        <select
          className={inputCls}
          value={form[key] ?? ""}
          onChange={(e) => set(key, e.target.value)}
        >
          {spec.options.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
      </Field>
    ) : spec.kind === "bool" ? (
      <Field key={key} label={label} hint={hint}>
        <label className="flex items-center gap-2 text-sm text-fg">
          <input
            type="checkbox"
            checked={form[key] === "true"}
            onChange={(e) => set(key, String(e.target.checked))}
          />
          enabled
        </label>
      </Field>
    ) : (
      <Field key={key} label={label} hint={hint}>
        <input
          className={inputCls}
          type={spec.kind === "number" ? "number" : "text"}
          min={b?.min}
          max={b?.max}
          step={spec.kind === "number" ? (b?.int ? 1 : "any") : undefined}
          value={form[key] ?? ""}
          onChange={(e) => set(key, e.target.value)}
        />
      </Field>
    );
    return spec.wide ? (
      <div key={key} className="sm:col-span-2">
        {el}
      </div>
    ) : (
      el
    );
  };

  // the registry endpoint lists llm/mesh/image kinds; fall back to the
  // health payload when the sidecar speaks the old protocol
  const models = meshModelsList.data ?? meshHealth.data?.models ?? [];
  const meshModels = models.filter((m) => m.kind !== "image" && m.kind !== "llm");
  const sidecarLlm = models.filter((m) => m.kind === "llm");
  const imageAdapters = (imageModels.data ?? []).filter((m) => m.available);
  const lp = llmHealth.data;
  const tp = testProvider.data;

  // backend selectors — missing keys (old backend payload) default sanely
  const llmBackend = form.llm_backend || "external";
  const meshBackend = form.mesh_backend || "sidecar";
  const imageBackend = form.image_backend || "sidecar";
  const llmBackendText =
    llmBackend === "sidecar"
      ? `sidecar · ${urlHost(s.mesh_base_url) || "mesh server"}`
      : `external · ${urlHost(s.llm_base_url) || "no base url"}`;
  const pickedVision = sidecarLlm.find((m) => m.id === form.llm_model_vision);

  /** write-only API-key row — mirrors the llm_api_key handling. */
  const apiKeyField = (
    label: string,
    isSet: boolean,
    value: string,
    onChange: (v: string) => void,
    cleared: boolean,
    onToggleClear: () => void,
    hint?: string,
  ) => (
    <>
      <div className="mb-3 flex items-end gap-2">
        <Field
          label={label}
          hint={
            hint ??
            (isSet
              ? t("key_isset_hint")
              : t("key_hint"))
          }
        >
          <input
            className={inputCls}
            type="password"
            autoComplete="off"
            placeholder={isSet ? "••••••••" : t("key_ph")}
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        </Field>
        {isSet && (
          <Chip tone="green" className="mb-3 shrink-0">
            {t("key_set")}
          </Chip>
        )}
        {isSet && (
          <Btn variant="ghost" className="mb-3" onClick={onToggleClear}>
            {cleared ? t("key_keep") : t("key_clear")}
          </Btn>
        )}
      </div>
      {cleared && (
        <div className="mb-3 text-xs text-amber-400">
          {t("key_removed")}
        </div>
      )}
    </>
  );

  /** external provider model picker — select from /settings/llm-models with
   *  a "custom…" escape hatch; falls back to plain text when the probe fails. */
  const extModelField = (key: string, label: string, hint?: string) => {
    const opts = llmModels.data?.models ?? [];
    const val = form[key] ?? "";
    if (llmModels.isError || customModel[key]) {
      return (
        <Field key={key} label={label} hint={hint}>
          <div className="flex items-center gap-2">
            <input
              className={inputCls}
              value={val}
              onChange={(e) => set(key, e.target.value)}
            />
            {!llmModels.isError && (
              <Btn
                variant="ghost"
                onClick={() => setCustomModel((c) => ({ ...c, [key]: false }))}
              >
                {t("list_btn")}
              </Btn>
            )}
          </div>
        </Field>
      );
    }
    return (
      <Field key={key} label={label} hint={hint}>
        <select
          className={inputCls}
          value={val}
          onChange={(e) => {
            if (e.target.value === "__custom__") {
              setCustomModel((c) => ({ ...c, [key]: true }));
            } else {
              set(key, e.target.value);
            }
          }}
        >
          {!opts.some((o) => o.id === val) && (
            <option value={val}>{val || t("pick_model")}</option>
          )}
          {opts.map((o) => (
            <option key={o.id} value={o.id}>
              {o.id}
              {o.vision ? ` · ${t("vision")}` : ""}
              {o.context_tokens ? ` · ${fmtCtx(o.context_tokens)} ctx` : ""}
              {o.downloaded === false ? ` · ${t("not_downloaded")}` : ""}
            </option>
          ))}
          <option value="__custom__">{t("custom")}</option>
        </select>
      </Field>
    );
  };

  /** sidecar model picker — /mesh/models rows of the given family; models
   *  without local weights stay visible but disabled. */
  const sidecarModelField = (spec: {
    key: string;
    label: string;
    hint?: string;
    empty?: string;
  }) => {
    const val = form[spec.key] ?? "";
    return (
      <Field key={spec.key} label={spec.label} hint={spec.hint}>
        <select
          className={inputCls}
          value={val}
          onChange={(e) => set(spec.key, e.target.value)}
        >
          {spec.empty != null && <option value="">{spec.empty}</option>}
          {sidecarLlm.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.downloaded}>
              {m.name && m.name !== m.id ? `${m.name} — ${m.id}` : m.id}
              {m.supports_vision ? ` · 👁 ${t("vision")}` : ""}
              {m.context_len ? ` · ${fmtCtx(m.context_len)} ctx` : ""}
              {!m.downloaded ? ` · ${t("not_downloaded")}` : ""}
            </option>
          ))}
          {val && !sidecarLlm.some((m) => m.id === val) && (
            <option value={val}>{val}</option>
          )}
        </select>
      </Field>
    );
  };

  const statusCard = (
    label: string,
    text: string,
    tone: "ok" | "warn" | "bad" | "idle",
    title?: string,
  ) => (
    <Card className="p-4">
      <div className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${
            tone === "ok"
              ? "bg-emerald-500"
              : tone === "warn"
                ? "bg-amber-500"
                : tone === "bad"
                  ? "bg-red-500"
                  : "bg-edge2"
          }`}
        />
        <div className="text-xs font-semibold uppercase tracking-wider text-muted">
          {label}
        </div>
      </div>
      <div
        className={`mt-1.5 truncate text-base font-medium ${
          tone === "ok"
            ? "text-emerald-400"
            : tone === "warn"
              ? "text-amber-400"
              : tone === "bad"
                ? "text-red-400"
                : "text-faint"
        }`}
        title={title}
      >
        {text}
      </div>
    </Card>
  );

  return (
    <PageShell title={t("title")} className="pb-28">

      {/* service health — 2×2 on narrow screens, one row from lg up */}
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {statusCard(
          "API",
          health.data?.ok ? t("st.online") : t("st.offline"),
          health.data?.ok ? "ok" : "bad",
        )}
        {statusCard(
          "LLM",
          lp?.ok
            ? `${t("st.online")} · ${lp.latency_ms}ms${lp.vision_ok ? ` · ${t("st.vision")}` : ""} · ${llmBackendText}`
            : llmHealth.isPending
              ? t("st.checking")
              : `${t("st.offline")} · ${llmBackendText}`,
          lp?.ok ? "ok" : llmHealth.isPending ? "idle" : "bad",
          lp?.error ??
            (llmBackend === "sidecar" || s.llm_api_key_set
              ? undefined
              : t("st.no_api_key")),
        )}
        {statusCard(
          t("st.mesh_server"),
          meshHealth.data?.ok
            ? `${t("st.online")} · ${meshHealth.data.device} · ${t("st.models_n", { count: meshModels.filter((m) => m.available).length })}`
            : t("st.offline"),
          meshHealth.data?.ok ? "ok" : "idle",
          meshHealth.data?.error ?? undefined,
        )}
        {statusCard(
          t("st.image_gen"),
          !meshHealth.data?.ok
            ? t("st.offline")
            : imageAdapters.length
              ? `${t("st.online")} · ${t("st.adapters_n", { count: imageAdapters.length })}`
              : t("st.no_adapters"),
          !meshHealth.data?.ok ? "idle" : imageAdapters.length ? "ok" : "warn",
          meshHealth.data?.error ?? undefined,
        )}
      </div>

      {/* tabs: scrollable strip on narrow screens, sticky sidebar from lg up */}
      <div className="lg:flex lg:items-start lg:gap-5">
      <div className="mb-4 flex gap-1 overflow-x-auto rounded-lg border border-edge bg-panel/60 p-1 lg:sticky lg:top-6 lg:mb-0 lg:w-44 lg:shrink-0 lg:flex-col lg:overflow-visible">
        {TAB_IDS_LIST.map((tb) => (
          <button
            key={tb}
            onClick={() => {
              setTab(tb);
              localStorage.setItem("settings.tab", tb);
              setSp({ tab: tb }, { replace: true });
            }}
            className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors lg:text-left ${
              tab === tb
                ? "bg-edge font-medium text-fg"
                : "text-muted hover:bg-panel-alt hover:text-fg"
            }`}
          >
            {t(`tab.${tb}`)}
          </button>
        ))}
      </div>

      <div className="min-w-0 flex-1">
      {tab === "provider" && (
      <Card className="p-4">
        <SectionTitle>{t("provider_title")}</SectionTitle>
        {/* backend selector — external OpenAI-compatible API or the local sidecar */}
        <div className="mb-4 flex gap-1 rounded-lg border border-edge bg-panel/80 p-1">
          {(
            [
              ["external", t("backend_external")],
              ["sidecar", t("backend_sidecar")],
            ] as const
          ).map(([v, l]) => (
            <button
              key={v}
              onClick={() => set("llm_backend", v)}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm transition-colors ${
                llmBackend === v
                  ? "bg-edge2 font-medium text-fg"
                  : "text-muted hover:bg-panel2 hover:text-fg"
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        {llmBackend === "external" ? (
          <>
            <div className="mb-3 text-xs text-muted">{t("external_hint")}</div>
            {!s.llm_api_key_set && (
              <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
                {t("no_key_warn")}
              </div>
            )}
            <div className={FIELD_GRID}>
              {field({
                key: "llm_base_url",
                label: t("f.llm_base_url"),
                hint: t("f.llm_base_url_h"),
                wide: true,
              })}
              <div className="sm:col-span-2">
                {apiKeyField(
                  t("key_api"),
                  s.llm_api_key_set,
                  apiKey,
                  setApiKey,
                  clearKey,
                  () => setClearKey((c) => !c),
                )}
              </div>
              {extModelField(
                "llm_model_vision",
                t("f.llm_model_vision"),
                t("f.llm_model_vision_h"),
              )}
              {extModelField(
                "llm_model_code",
                t("f.llm_model_code"),
                t("f.llm_model_code_h"),
              )}
            </div>
          </>
        ) : (
          <>
            <div className="mb-3 text-xs text-faint">
              {meshHealth.data?.ok
                ? t("sidecar_online", {
                    n: sidecarLlm.filter((m) => m.downloaded).length,
                    total: sidecarLlm.length,
                  })
                : t("sidecar_offline")}
            </div>
            <div className={FIELD_GRID}>
              {sidecarModelField({
                key: "llm_model_vision",
                label: t("f.llm_model_vision"),
                hint: t("f.llm_model_vision_h"),
              })}
              {sidecarModelField({
                key: "llm_model_code",
                label: t("f.llm_model_code"),
                hint: t("f.llm_model_code_h"),
              })}
              {sidecarModelField({
                key: "llm_sidecar_model",
                label: t("f.llm_sidecar_model"),
                hint: t("f.llm_sidecar_model_h"),
                empty: t("per_role"),
              })}
            </div>
            {pickedVision && !pickedVision.supports_vision && (
              <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
                {t("vision_needed")}
              </div>
            )}
          </>
        )}
        {Object.keys(s.llm_detected ?? {}).length > 0 && (
          <div className="mb-3 text-xs text-faint">
            {t("detected_ctx")}{" "}
            {Object.entries(s.llm_detected)
              .map(([id, n]) => `${id} ${fmtCtx(n)}`)
              .join(" · ")}
          </div>
        )}
        <div className={FIELD_GRID}>
          {field({
            key: "llm_json_mode",
            label: t("f.llm_json_mode"),
            options: JSON_MODE_OPTS.map(([v]) => [v, t(`opt.json_mode.${v}`)]),
          })}
          {field({
            key: "llm_tools",
            label: t("f.llm_tools"),
            hint: t("f.llm_tools_h"),
            options: LLM_TOOLS_OPTS.map(([v]) => [v, t(`opt.llm_tools.${v}`)]),
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.advanced")}
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "llm_temperature_code", label: t("f.llm_temperature_code"), kind: "number" })}
          {field({
            key: "llm_context_tokens",
            label: t("f.llm_context_tokens"),
            hint: t("f.llm_context_tokens_h"),
            kind: "number",
          })}
          {field({
            key: "llm_max_tokens",
            label: t("f.llm_max_tokens"),
            hint: t("f.llm_max_tokens_h"),
            kind: "number",
          })}
          {field({ key: "llm_stream", label: t("f.llm_stream"), kind: "bool" })}
          {field({ key: "llm_timeout_s", label: t("f.llm_timeout_s"), kind: "number" })}
          {field({ key: "llm_idle_timeout_s", label: t("f.llm_idle_timeout_s"), kind: "number" })}
          {field({ key: "llm_degenerate_stall_s", label: t("f.llm_degenerate_stall_s"), kind: "number" })}
          {field({ key: "llm_total_timeout_s", label: t("f.llm_total_timeout_s"), kind: "number" })}
          {field({
            key: "llm_extra_body",
            label: t("f.llm_extra_body"),
            hint: 'merged into every request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}',
            wide: true,
          })}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <Btn variant="ghost" onClick={() => testProvider.mutate()} busy={testProvider.isPending}>
            {t("test_provider")}
          </Btn>
          {tp && (
            <span className={`text-xs ${tp.ok ? "text-emerald-400" : "text-red-400"}`}>
              {tp.ok
                ? `${t("test_ok", { ms: tp.latency_ms })}${tp.vision_ok ? ` · ${t("vision_ok")}` : ` · ${t("vision_failed")}`}${
                    tp.context?.code?.context_tokens
                      ? ` · ctx ${fmtCtx(tp.context.code.context_tokens)} (${tp.context.code.source || t("detected")})`
                      : ""
                  }`
                : t("test_failed", { err: tp.error ?? "?" })}
            </span>
          )}
          {testProvider.isError && (
            <span className="text-xs text-red-400">{errMsg(testProvider.error)}</span>
          )}
        </div>
      </Card>
      )}

      {tab === "models" && (
        <>
          <Card className="mb-4 p-4">
            <SectionTitle>{t("hf_title")}</SectionTitle>
            {apiKeyField(
              t("key_hf"),
              s.hf_token_set,
              hfKey,
              setHfKey,
              clearHfKey,
              () => setClearHfKey((c) => !c),
              t("hf_hint"),
            )}
          </Card>
          <ModelsTab models={models} />
        </>
      )}

      {tab === "agent" && (
      <Card className="p-4">
        <SectionTitle>{t("agent_title")}</SectionTitle>
        <div className={FIELD_GRID}>
          {field({
            key: "agent_default_gen_mode",
            label: t("f.agent_default_gen_mode"),
            options: GEN_MODE_OPTS.map(([v]) => [v, t(`opt.gen_mode.${v}`)]),
          })}
          {field({ key: "agent_max_iterations", label: t("f.agent_max_iterations"), kind: "number" })}
          {field({ key: "agent_max_repairs", label: t("f.agent_max_repairs"), kind: "number" })}
          {field({ key: "agent_pass_score", label: t("f.agent_pass_score"), kind: "number" })}
          {field({ key: "agent_dim_tolerance_pct", label: t("f.agent_dim_tolerance_pct"), kind: "number" })}
          {field({ key: "agent_dim_tolerance_mm", label: t("f.agent_dim_tolerance_mm"), kind: "number" })}
          {field({
            key: "agent_clarification_mode",
            label: t("f.agent_clarification_mode"),
            options: CLAR_OPTS.map(([v]) => [v, t(`opt.clar.${v}`)]),
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.budgets")}
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "agent_max_input_images", label: t("f.agent_max_input_images"), kind: "number" })}
          {field({
            key: "agent_context_budget_frac",
            label: t("f.agent_context_budget_frac"),
            hint: t("f.agent_context_budget_frac_h"),
            kind: "number",
          })}
          {field({ key: "agent_max_tokens_per_job", label: t("f.agent_max_tokens_per_job"), kind: "number" })}
          {field({ key: "agent_max_wall_s", label: t("f.agent_max_wall_s"), kind: "number" })}
          {field({ key: "agent_param_autofix", label: t("f.agent_param_autofix"), kind: "bool" })}
          {field({
            key: "agent_patch_mode",
            label: t("f.agent_patch_mode"),
            hint: t("f.agent_patch_mode_h"),
            options: PATCH_OPTS.map(([v]) => [v, t(`opt.patch.${v}`)]),
          })}
          {field({ key: "agent_patch_retries", label: t("f.agent_patch_retries"), kind: "number" })}
          {field({ key: "agent_feature_feedback", label: t("f.agent_feature_feedback"), kind: "bool" })}
          {field({ key: "agent_critique_reask", label: t("f.agent_critique_reask"), kind: "bool" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.tools")}
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "agent_tool_max_calls", label: t("f.agent_tool_max_calls"), kind: "number" })}
          {field({ key: "agent_tool_max_calls_info", label: t("f.agent_tool_max_calls_info"), kind: "number" })}
          {field({ key: "agent_tool_verify", label: t("f.agent_tool_verify"), kind: "bool" })}
          {field({ key: "agent_tool_verify_max", label: t("f.agent_tool_verify_max"), kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.mesh_session")}
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "agent_mesh_tools",
            label: t("f.agent_mesh_tools"),
            hint: t("f.agent_mesh_tools_h"),
            kind: "bool",
          })}
          {field({ key: "agent_mesh_max_ops", label: t("f.agent_mesh_max_ops"), kind: "number" })}
          {field({ key: "agent_mesh_preview_max", label: t("f.agent_mesh_preview_max"), kind: "number" })}
          {field({ key: "agent_mesh_refine_rounds", label: t("f.agent_mesh_refine_rounds"), kind: "number" })}
          {field({
            key: "agent_mesh_hard_fail_score",
            label: t("f.agent_mesh_hard_fail_score"),
            hint: t("f.agent_mesh_hard_fail_score_h"),
            kind: "number",
          })}
          {field({
            key: "agent_mesh_hard_fail_limit",
            label: t("f.agent_mesh_hard_fail_limit"),
            hint: t("f.agent_mesh_hard_fail_limit_h"),
            kind: "number",
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.chat")}
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "chat_model_role",
            label: t("f.chat_model_role"),
            hint: t("f.chat_model_role_h"),
            options: CHAT_ROLE_OPTS.map(([v]) => [v, t(`opt.chat_role.${v}`)]),
          })}
          {field({ key: "chat_tool_max_calls", label: t("f.chat_tool_max_calls"), kind: "number" })}
          {field({ key: "chat_max_turns", label: t("f.chat_max_turns"), kind: "number" })}
          {field({
            key: "chat_spawn_timeout_s",
            label: t("f.chat_spawn_timeout_s"),
            hint: t("f.chat_spawn_timeout_s_h"),
            kind: "number",
          })}
          {field({ key: "chat_history_messages", label: t("f.chat_history_messages"), kind: "number" })}
          {field({
            key: "chat_image_max",
            label: t("f.chat_image_max"),
            kind: "number",
          })}
        </div>
      </Card>
      )}

      {tab === "mesh" && (
      <div className="grid gap-4 xl:grid-cols-2">
      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>{t("mesh_title")}</SectionTitle>
          {meshBackend === "tripo" ? (
            <Chip tone="blue">tripo api</Chip>
          ) : (
            meshHealth.data && (
              <Chip
                tone={meshHealth.data.ok ? "green" : "red"}
                title={meshHealth.data.error ?? ""}
              >
                {meshHealth.data.ok ? `online · ${meshHealth.data.device}` : "offline"}
              </Chip>
            )
          )}
        </div>
        {field({
          key: "mesh_backend",
          label: t("f.mesh_backend"),
          options: [
            ["sidecar", "Local sidecar"],
            ["tripo", "Tripo3D API"],
          ],
        })}
        {meshBackend === "tripo" ? (
          <>
            <div className="mb-3 text-xs text-muted">{t("tripo_hint")}</div>
            <div className={FIELD_GRID}>
              <div className="sm:col-span-2">
                {apiKeyField(
                  t("key_tripo"),
                  s.tripo_api_key_set,
                  tripoKey,
                  setTripoKey,
                  clearTripoKey,
                  () => setClearTripoKey((c) => !c),
                )}
              </div>
              {field({
                key: "tripo_model",
                label: t("f.tripo_model"),
                options: [
                  ["v2.5-20250123", "v2.5-20250123"],
                  ["v2.0-20240919", "v2.0-20240919"],
                  ["v1.4-20240625", "v1.4-20240625"],
                ],
              })}
              {field({
                key: "tripo_api_base",
                label: t("f.tripo_api_base"),
                hint: t("f.tripo_api_base_h"),
                wide: true,
              })}
            </div>
          </>
        ) : (
          <>
        <div className={FIELD_GRID}>
          {field({
            key: "mesh_base_url",
            label: t("f.mesh_base_url"),
            hint: t("f.mesh_base_url_h"),
            wide: true,
          })}
          <div className="sm:col-span-2">
            <Field
              label={t("f.mesh_model")}
              hint={t("f.mesh_model_h")}
            >
              <select
                className={inputCls}
                value={form.mesh_model}
                onChange={(e) => set("mesh_model", e.target.value)}
              >
                <option value="auto">{t("auto_device")}</option>
                {meshModels.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name || m.id} ({m.device}){m.available ? "" : ` — ${t("not_installed")}`}
                  </option>
                ))}
                {form.mesh_model &&
                  !["auto", ...meshModels.map((m) => m.id)].includes(form.mesh_model) && (
                    <option value={form.mesh_model}>{form.mesh_model}</option>
                  )}
              </select>
            </Field>
          </div>
          {field({ key: "mesh_timeout_s", label: t("f.mesh_timeout_s"), kind: "number" })}
          {field({ key: "mesh_stall_timeout_s", label: t("f.mesh_stall_timeout_s"), kind: "number" })}
          {field({
            key: "mesh_min_free_mb",
            label: t("f.mesh_min_free_mb"),
            kind: "number",
            hint: t("f.mesh_min_free_mb_h"),
          })}
          {field({ key: "mesh_decimate_faces", label: t("f.mesh_decimate_faces"), kind: "number" })}
          {field({ key: "mesh_max_faces_for_cad", label: t("f.mesh_max_faces_for_cad"), kind: "number" })}
        </div>
        {meshModels.length > 0 && (
          <div className="mb-2 flex flex-col gap-1.5 text-xs text-faint">
            {meshModels.map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <b className="text-fg">{m.id}</b> · {m.device}
                <Chip tone={modelStateTone(m)}>{modelStateLabel(m)}</Chip>
                {m.est_mem_mb != null && <span>{fmtMemGb(m.est_mem_mb)}</span>}
                {m.supports_multiview ? ` · ${t("multiview")}` : ""}
                {m.license_note ? (
                  <div className="basis-full text-muted">{t("license")} {m.license_note}</div>
                ) : null}
                {m.last_error ? (
                  <div className="basis-full text-amber-500">{t("last_error")} {m.last_error}</div>
                ) : null}
              </div>
            ))}
            <div>
              {t("downloads_hint_pre")}{" "}
              <b className="text-muted">{t("tab.models")}</b>
            </div>
          </div>
        )}
        {!meshHealth.data?.ok && (
          <div className="mb-2 text-xs text-faint">
            {t("mesh_offline_pre")} <code>make mesh-server</code>.{" "}
            {t("mesh_offline_post")}
          </div>
        )}
          </>
        )}
      </Card>

      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>{t("image_title")}</SectionTitle>
          {imageBackend === "openai" ? (
            <Chip tone="blue">openai api</Chip>
          ) : (
            meshHealth.data && (
              <Chip
                tone={
                  !meshHealth.data.ok
                    ? "red"
                    : imageAdapters.length
                      ? "green"
                      : "amber"
                }
                title={meshHealth.data.error ?? ""}
              >
                {!meshHealth.data.ok
                  ? "offline"
                  : imageAdapters.length
                    ? `online · ${imageAdapters.length} adapter${imageAdapters.length > 1 ? "s" : ""}`
                    : "no adapters"}
              </Chip>
            )
          )}
        </div>
        {field({
          key: "image_backend",
          label: t("f.image_backend"),
          options: [
            ["sidecar", "Local sidecar"],
            ["openai", "OpenAI API"],
          ],
        })}
        {imageBackend === "openai" ? (
          <>
            <div className="mb-3 text-xs text-muted">{t("openai_img_hint")}</div>
            <div className={FIELD_GRID}>
              {field({
                key: "image_openai_model",
                label: t("f.image_openai_model"),
                options: [
                  ["gpt-image-1", "gpt-image-1"],
                  ["dall-e-3", "dall-e-3"],
                  ["dall-e-2", "dall-e-2"],
                ],
              })}
              {field({
                key: "image_api_base",
                label: t("f.image_api_base"),
                hint: t("f.image_api_base_h"),
                wide: true,
              })}
              <div className="sm:col-span-2">
                {apiKeyField(
                  t("key_image"),
                  s.image_api_key_set,
                  imageKey,
                  setImageKey,
                  clearImageKey,
                  () => setClearImageKey((c) => !c),
                  t("img_key_hint"),
                )}
              </div>
            </div>
          </>
        ) : (
          <>
        <div className="mb-3 text-xs text-muted">
          {t("sidecar_img_hint")}
        </div>
        <div className={FIELD_GRID}>
          <Field label={t("f.image_model")}>
            <select
              className={inputCls}
              value={form.image_model}
              onChange={(e) => set("image_model", e.target.value)}
            >
              <option value="auto">{t("auto_device")}</option>
              {(imageModels.data ?? []).map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name || m.id} ({m.device}){m.available ? "" : " — not installed"}
                </option>
              ))}
              {form.image_model &&
                !["auto", ...(imageModels.data ?? []).map((m) => m.id)].includes(
                  form.image_model
                ) && <option value={form.image_model}>{form.image_model}</option>}
            </select>
          </Field>
          <Field
            label={t("f.image_edit_model")}
            hint={t("f.image_edit_model_h")}
          >
            <select
              className={inputCls}
              value={form.image_edit_model}
              onChange={(e) => set("image_edit_model", e.target.value)}
            >
              <option value="auto">{t("auto_edit")}</option>
              {(imageModels.data ?? [])
                .filter((m) => m.supports_edit)
                .map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name || m.id} ({m.device}){m.available ? "" : " — not installed"}
                  </option>
                ))}
              {form.image_edit_model &&
                !["auto", ...(imageModels.data ?? []).map((m) => m.id)].includes(
                  form.image_edit_model
                ) && <option value={form.image_edit_model}>{form.image_edit_model}</option>}
            </select>
          </Field>
        </div>
        {(imageModels.data ?? []).length > 0 ? (
          <div className="mb-2 flex flex-col gap-1.5 text-xs text-faint">
            {(imageModels.data ?? []).map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <b className="text-fg">{m.id}</b> · {m.device}
                <Chip tone={modelStateTone(m)}>{modelStateLabel(m)}</Chip>
                {m.supports_edit ? ` · ${m.requires_image ? t("edit_only") : t("edit_capable")}` : ""}
                {m.last_error ? (
                  <div className="basis-full text-amber-500">last error: {m.last_error}</div>
                ) : null}
                {m.license_note ? (
                  <div className="basis-full text-faint">license: {m.license_note}</div>
                ) : null}
              </div>
            ))}
          </div>
        ) : meshHealth.data && !meshHealth.data.ok ? (
          <div className="mb-2 text-xs text-faint">
            {t("img_offline_pre")} <code>make mesh-server</code>.
          </div>
        ) : (
          <div className="mb-2 text-xs text-faint">
            {t("img_none_pre")} <code>docs/image-generation.md</code>{" "}
            {t("img_none_post")}
          </div>
        )}
          </>
        )}
        {/* shared tuning — applies to both image backends */}
        <div className={FIELD_GRID}>
          {field({
            key: "image_default_strength",
            label: t("f.image_default_strength"),
            kind: "number",
          })}
          {field({ key: "image_steps", label: t("f.image_steps"), kind: "number" })}
          {field({ key: "image_guidance", label: t("f.image_guidance"), kind: "number" })}
          {field({ key: "image_max_side", label: t("f.image_max_side"), kind: "number" })}
          {field({ key: "image_max_count", label: t("f.image_max_count"), kind: "number" })}
          {field({ key: "image_timeout_s", label: t("f.image_timeout_s"), kind: "number" })}
          {field({
            key: "image_autogen_mesh",
            label: t("f.image_autogen_mesh"),
            hint: t("f.image_autogen_mesh_h"),
            kind: "bool",
          })}
          {field({
            key: "image_tool_budget",
            label: t("f.image_tool_budget"),
            kind: "number",
          })}
        </div>
      </Card>
      </div>
      )}

      {tab === "fem" && (
      <Card className="p-4">
        <SectionTitle>{t("fem_title")}</SectionTitle>
        <div className="mb-3 text-xs text-faint">
          {t("fem_disclaimer")}
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "fem_voxel_mm", label: t("f.fem_voxel_mm"), kind: "number" })}
          {field({ key: "fem_max_elements", label: t("f.fem_max_elements"), kind: "number" })}
          {field({ key: "fem_timeout_s", label: t("f.fem_timeout_s"), kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.topopt")}
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "topopt_timeout_s", label: t("f.topopt_timeout_s"), kind: "number" })}
          {field({ key: "topopt_max_iterations", label: t("f.topopt_max_iterations"), kind: "number" })}
          {field({ key: "topopt_default_volfrac", label: t("f.topopt_default_volfrac"), kind: "number" })}
          {field({ key: "topopt_default_rmin_mm", label: t("f.topopt_default_rmin_mm"), kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("grp.verification")}
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "fem_verify_realizations",
            label: t("f.fem_verify_realizations"),
            kind: "bool",
            hint: t("f.fem_verify_realizations_h"),
          })}
          {field({
            key: "fem_verify_realization_cells",
            label: t("f.fem_verify_realization_cells"),
            kind: "number",
            hint: t("f.fem_verify_realization_cells_h"),
          })}
          {field({
            key: "fem_verify_coverage_e",
            label: t("f.fem_verify_coverage_e"),
            kind: "bool",
            hint: t("f.fem_verify_coverage_e_h"),
          })}
          {field({
            key: "fem_verify_mesh_shift",
            label: t("f.fem_verify_mesh_shift"),
            kind: "bool",
            hint: t("f.fem_verify_mesh_shift_h"),
          })}
          {field({
            key: "fem_loadpath_cert",
            label: t("f.fem_loadpath_cert"),
            kind: "bool",
            hint: t("f.fem_loadpath_cert_h"),
          })}
          {field({
            key: "fem_homogenized_infill",
            label: t("f.fem_homogenized_infill"),
            kind: "bool",
            hint: t("f.fem_homogenized_infill_h"),
          })}
          {field({
            key: "fem_homog_cell_voxels",
            label: t("f.fem_homog_cell_voxels"),
            kind: "number",
            hint: t("f.fem_homog_cell_voxels_h"),
          })}
          {field({
            key: "infill_octet",
            label: t("f.infill_octet"),
            kind: "bool",
            hint: t("f.infill_octet_h"),
          })}
        </div>
      </Card>
      )}

      {tab === "system" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-4">
            <SectionTitle>{t("exec_title")}</SectionTitle>
            <div className={FIELD_GRID}>
              {field({ key: "exec_timeout_s", label: t("f.exec_timeout_s"), kind: "number" })}
              {field({
                key: "exec_mem_limit_mb",
                label: t("f.exec_mem_limit_mb"),
                kind: "number",
                hint: t("f.exec_mem_limit_mb_h"),
              })}
              {field({
                key: "exec_threads",
                label: t("f.exec_threads"),
                kind: "number",
                hint: t("f.exec_threads_h"),
              })}
              {field({
                key: "exec_pool_workers",
                label: t("f.exec_pool_workers"),
                kind: "number",
                hint: t("f.exec_pool_workers_h"),
              })}
              {field({
                key: "audit_parallel_runs",
                label: t("f.audit_parallel_runs"),
                kind: "number",
                hint: t("f.audit_parallel_runs_h"),
              })}
            </div>
            {Object.keys(s.resources_detected ?? {}).length > 0 && (
              <div className="mt-2 text-xs text-faint">
                {t("detected_res")} {s.resources_detected.cpus ?? "?"} cpus ·{" "}
                {Math.round((s.resources_detected.mem_mb ?? 0) / 1024)}g mem ·{" "}
                {s.resources_detected.threads ?? "?"} threads/stage ·{" "}
                {s.resources_detected.pool ?? "?"} pool workers
              </div>
            )}
            <div className="mt-2 text-xs text-muted">{t("exec_caps_hint")}</div>
          </Card>

          <Card className="p-4 text-sm text-muted">
            <SectionTitle>{t("runtime_title")}</SectionTitle>
            <div>
              {t("data_dir")} <code className="text-fg">{s.data_dir}</code>
            </div>
            <div>
              {t("runtime_hint")}
            </div>
          </Card>

          <UpdateCard />
        </div>
      )}
      </div>
      </div>

      {/* floating save bar — appears while there are unsaved changes */}
      {(dirty || save.isPending || saved || saveError) && (
        <div className="fixed inset-x-0 bottom-5 z-40 flex justify-center px-4">
          <div className="flex max-w-full items-center gap-3 rounded-full border border-edge2 bg-panel/95 py-2 pl-4 pr-2 shadow-2xl shadow-black/50 backdrop-blur">
            {saveError ? (
              <span className="max-w-md truncate text-xs text-red-400" title={saveError}>
                {saveError}
              </span>
            ) : (
              <span className="text-xs text-muted">
                {saved && !dirty
                  ? t("saved_done")
                  : t("saved_unsaved")}
              </span>
            )}
            {dirty && (
              <Btn variant="ghost" onClick={revert} disabled={save.isPending}>
                {t("common:action.revert")}
              </Btn>
            )}
            <Btn onClick={() => save.mutate()} busy={save.isPending} disabled={!dirty}>
              {saved && !dirty ? t("saved_btn") : t("save_btn")}
            </Btn>
          </div>
        </div>
      )}
    </PageShell>
  );
}

