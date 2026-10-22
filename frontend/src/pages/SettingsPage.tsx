import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import ModelsTab from "../components/ModelsTab";
import UpdateCard from "../components/UpdateCard";
import {
  Btn,
  Card,
  Chip,
  Field,
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
const TABS: { id: TabId; label: string }[] = [
  { id: "provider", label: "LLM Provider" },
  { id: "models", label: "Local models" },
  { id: "agent", label: "Agent" },
  { id: "mesh", label: "Mesh & Image" },
  { id: "fem", label: "FEM / TopOpt" },
  { id: "system", label: "System" },
];

const TAB_IDS = new Set<string>(TABS.map((t) => t.id));

/** compact token-count rendering for context-window chips: 131k, 262k, 1M */
const fmtCtx = (n: number | null | undefined): string => {
  if (!n || n <= 0) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
};

export default function SettingsPage() {
  useTitle("Settings");
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
          errs.push(`${k}: not a number`);
          continue;
        }
        const b = FIELD_BOUNDS[k];
        if (!b) continue;
        if (b.int && !Number.isInteger(n)) errs.push(`${k}: must be an integer`);
        else if (b.min != null && (b.exclusiveMin ? n <= b.min : n < b.min))
          errs.push(`${k}: must be ${b.exclusiveMin ? ">" : "≥"} ${b.min}`);
        else if (b.max != null && n > b.max) errs.push(`${k}: must be ≤ ${b.max}`);
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
      <div className="flex justify-center py-24 text-zinc-500">
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
        <label className="flex items-center gap-2 text-sm text-zinc-300">
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
  const t = testProvider.data;

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
              ? "a key is stored — enter a new one to replace it (never shown again)"
              : "stored encrypted in the app database, write-only")
          }
        >
          <input
            className={inputCls}
            type="password"
            autoComplete="off"
            placeholder={isSet ? "••••••••" : "key…"}
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        </Field>
        {isSet && (
          <Chip tone="green" className="mb-3 shrink-0">
            key set
          </Chip>
        )}
        {isSet && (
          <Btn variant="ghost" className="mb-3" onClick={onToggleClear}>
            {cleared ? "keep key" : "clear key"}
          </Btn>
        )}
      </div>
      {cleared && (
        <div className="mb-3 text-xs text-amber-400">
          the stored API key will be removed on save
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
                list
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
            <option value={val}>{val || "— pick a model —"}</option>
          )}
          {opts.map((o) => (
            <option key={o.id} value={o.id}>
              {o.id}
              {o.vision ? " · vision" : ""}
              {o.context_tokens ? ` · ${fmtCtx(o.context_tokens)} ctx` : ""}
              {o.downloaded === false ? " · not downloaded" : ""}
            </option>
          ))}
          <option value="__custom__">custom…</option>
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
              {m.supports_vision ? " · 👁 vision" : ""}
              {m.context_len ? ` · ${fmtCtx(m.context_len)} ctx` : ""}
              {!m.downloaded ? " · not downloaded" : ""}
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
                  : "bg-zinc-600"
          }`}
        />
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
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
                : "text-zinc-500"
        }`}
        title={title}
      >
        {text}
      </div>
    </Card>
  );

  return (
    <div className="mx-auto w-full max-w-xl p-4 pb-28 sm:p-6 md:max-w-3xl lg:max-w-5xl xl:max-w-6xl">
      <h1 className="mb-4 text-2xl font-bold">Settings</h1>

      {/* service health — 2×2 on narrow screens, one row from lg up */}
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {statusCard(
          "API",
          health.data?.ok ? "online" : "offline",
          health.data?.ok ? "ok" : "bad",
        )}
        {statusCard(
          "LLM",
          lp?.ok
            ? `online · ${lp.latency_ms}ms${lp.vision_ok ? " · vision" : ""} · ${llmBackendText}`
            : llmHealth.isPending
              ? "checking…"
              : `offline · ${llmBackendText}`,
          lp?.ok ? "ok" : llmHealth.isPending ? "idle" : "bad",
          lp?.error ??
            (llmBackend === "sidecar" || s.llm_api_key_set
              ? undefined
              : "no API key configured"),
        )}
        {statusCard(
          "Mesh server",
          meshHealth.data?.ok
            ? `online · ${meshHealth.data.device} · ${meshModels.filter((m) => m.available).length} models`
            : "offline",
          meshHealth.data?.ok ? "ok" : "idle",
          meshHealth.data?.error ?? undefined,
        )}
        {statusCard(
          "Image gen",
          !meshHealth.data?.ok
            ? "offline"
            : imageAdapters.length
              ? `online · ${imageAdapters.length} adapter${imageAdapters.length > 1 ? "s" : ""}`
              : "no adapters",
          !meshHealth.data?.ok ? "idle" : imageAdapters.length ? "ok" : "warn",
          meshHealth.data?.error ?? undefined,
        )}
      </div>

      {/* tabs: scrollable strip on narrow screens, sticky sidebar from lg up */}
      <div className="lg:flex lg:items-start lg:gap-5">
      <div className="mb-4 flex gap-1 overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-900/60 p-1 lg:sticky lg:top-6 lg:mb-0 lg:w-44 lg:shrink-0 lg:flex-col lg:overflow-visible">
        {TABS.map((tb) => (
          <button
            key={tb.id}
            onClick={() => {
              setTab(tb.id);
              localStorage.setItem("settings.tab", tb.id);
              setSp({ tab: tb.id }, { replace: true });
            }}
            className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors lg:text-left ${
              tab === tb.id
                ? "bg-zinc-700 font-medium text-zinc-100"
                : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            }`}
          >
            {tb.label}
          </button>
        ))}
      </div>

      <div className="min-w-0 flex-1">
      {tab === "provider" && (
      <Card className="p-4">
        <SectionTitle>LLM provider</SectionTitle>
        {/* backend selector — external OpenAI-compatible API or the local sidecar */}
        <div className="mb-4 flex gap-1 rounded-lg border border-zinc-800 bg-zinc-900/80 p-1">
          {(
            [
              ["external", "External API"],
              ["sidecar", "Local sidecar"],
            ] as const
          ).map(([v, l]) => (
            <button
              key={v}
              onClick={() => set("llm_backend", v)}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm transition-colors ${
                llmBackend === v
                  ? "bg-zinc-700 font-medium text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        {llmBackend === "external" ? (
          <>
            <div className="mb-3 text-xs text-zinc-500">
              any OpenAI-compatible endpoint — cloud provider or a local server
              (Ollama, vLLM, LM Studio)
            </div>
            {!s.llm_api_key_set && (
              <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
                No API key configured — paste a key below and save. Generation works without
                one only for providers that skip authentication.
              </div>
            )}
            <div className={FIELD_GRID}>
              {field({
                key: "llm_base_url",
                label: "Base URL",
                hint: "e.g. https://api.openai.com/v1, http://localhost:11434/v1 (Ollama), https://openrouter.ai/api/v1",
                wide: true,
              })}
              <div className="sm:col-span-2">
                {apiKeyField(
                  "API key",
                  s.llm_api_key_set,
                  apiKey,
                  setApiKey,
                  clearKey,
                  () => setClearKey((c) => !c),
                )}
              </div>
              {extModelField(
                "llm_model_vision",
                "Vision model",
                "used for image analysis & visual critique",
              )}
              {extModelField(
                "llm_model_code",
                "Code model",
                "used for CAD code generation",
              )}
            </div>
          </>
        ) : (
          <>
            <div className="mb-3 text-xs text-zinc-500">
              {meshHealth.data?.ok
                ? `sidecar online · ${sidecarLlm.filter((m) => m.downloaded).length}/${sidecarLlm.length} llm models downloaded — manage downloads under “Local models”`
                : "sidecar offline — start it with make mesh-server, then download an llm model under “Local models”"}
            </div>
            <div className={FIELD_GRID}>
              {sidecarModelField({
                key: "llm_model_vision",
                label: "Vision model",
                hint: "used for image analysis & visual critique",
              })}
              {sidecarModelField({
                key: "llm_model_code",
                label: "Code model",
                hint: "used for CAD code generation",
              })}
              {sidecarModelField({
                key: "llm_sidecar_model",
                label: "Single-model override",
                hint: "serve one model for all roles instead of separate vision/code picks",
                empty: "— per role —",
              })}
            </div>
            {pickedVision && !pickedVision.supports_vision && (
              <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
                pipeline needs a vision-capable local model — intake/critique read images
              </div>
            )}
          </>
        )}
        {Object.keys(s.llm_detected ?? {}).length > 0 && (
          <div className="mb-3 text-xs text-zinc-500">
            detected context:{" "}
            {Object.entries(s.llm_detected)
              .map(([id, n]) => `${id} ${fmtCtx(n)}`)
              .join(" · ")}
          </div>
        )}
        <div className={FIELD_GRID}>
          {field({
            key: "llm_json_mode",
            label: "JSON mode",
            options: [
              ["json_object", "json_object (default)"],
              ["json_schema", "json_schema (strict)"],
              ["none", "none (parse from text)"],
            ],
          })}
          {field({
            key: "llm_tools",
            label: "Native tool calling",
            hint: "auto: fall back per model when the provider rejects tools; on: strict; off: legacy JSON steps",
            options: [
              ["auto", "auto (recommended)"],
              ["on", "on (strict)"],
              ["off", "off"],
            ],
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          advanced
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "llm_temperature_code", label: "Temperature (code)", kind: "number" })}
          {field({
            key: "llm_context_tokens",
            label: "Context window (tokens)",
            hint: "0 = auto-detect from the provider (probed lazily, cached); >0 = manual override",
            kind: "number",
          })}
          {field({
            key: "llm_max_tokens",
            label: "Max tokens per response",
            hint: "0 = auto — derived from the detected context window and clamped so prompt + completion fits",
            kind: "number",
          })}
          {field({ key: "llm_stream", label: "Streaming", kind: "bool" })}
          {field({ key: "llm_timeout_s", label: "Request timeout (s)", kind: "number" })}
          {field({ key: "llm_idle_timeout_s", label: "Stream idle timeout (s)", kind: "number" })}
          {field({ key: "llm_degenerate_stall_s", label: "Degenerate stream abort (s)", kind: "number" })}
          {field({ key: "llm_total_timeout_s", label: "Stream total cap (s)", kind: "number" })}
          {field({
            key: "llm_extra_body",
            label: "Extra request body (JSON)",
            hint: 'merged into every request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}',
            wide: true,
          })}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <Btn variant="ghost" onClick={() => testProvider.mutate()} busy={testProvider.isPending}>
            Test provider
          </Btn>
          {t && (
            <span className={`text-xs ${t.ok ? "text-emerald-400" : "text-red-400"}`}>
              {t.ok
                ? `code ok · ${t.latency_ms}ms${t.vision_ok ? " · vision ok" : " · vision failed"}${
                    t.context?.code?.context_tokens
                      ? ` · ctx ${fmtCtx(t.context.code.context_tokens)} (${t.context.code.source || "detected"})`
                      : ""
                  }`
                : `failed: ${t.error ?? "unknown"}`}
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
            <SectionTitle>Hugging Face</SectionTitle>
            {apiKeyField(
              "Access token",
              s.hf_token_set,
              hfKey,
              setHfKey,
              clearHfKey,
              () => setClearHfKey((c) => !c),
              "hf_… token — forwarded to the sidecar for gated repos and higher rate limits; empty = anonymous; stored encrypted",
            )}
          </Card>
          <ModelsTab models={models} />
        </>
      )}

      {tab === "agent" && (
      <Card className="p-4">
        <SectionTitle>Agent</SectionTitle>
        <div className={FIELD_GRID}>
          {field({
            key: "agent_default_gen_mode",
            label: "Default generation mode",
            options: [
              ["auto", "auto (planner decides)"],
              ["cad", "cad (parametric code)"],
              ["mesh", "mesh (image → 3D)"],
              ["hybrid", "hybrid (mesh + CAD features)"],
              ["topopt", "topopt (topology optimization)"],
            ],
          })}
          {field({ key: "agent_max_iterations", label: "Max iterations", kind: "number" })}
          {field({ key: "agent_max_repairs", label: "Max repair attempts", kind: "number" })}
          {field({ key: "agent_pass_score", label: "Critique pass score (0–100)", kind: "number" })}
          {field({ key: "agent_dim_tolerance_pct", label: "Dim tolerance (%)", kind: "number" })}
          {field({ key: "agent_dim_tolerance_mm", label: "Dim tolerance (mm)", kind: "number" })}
          {field({
            key: "agent_clarification_mode",
            label: "Clarification mode",
            options: [
              ["assume", "assume (agent decides)"],
              ["ask", "ask (clarifying questions)"],
            ],
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          budgets & robustness
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "agent_max_input_images", label: "Max input images", kind: "number" })}
          {field({
            key: "agent_context_budget_frac",
            label: "Context budget fraction",
            hint: "share of the detected context window a prompt may fill before history is condensed early",
            kind: "number",
          })}
          {field({ key: "agent_max_tokens_per_job", label: "Token budget per job", kind: "number" })}
          {field({ key: "agent_max_wall_s", label: "Wall-clock budget per job (s)", kind: "number" })}
          {field({ key: "agent_param_autofix", label: "Deterministic param autofix", kind: "bool" })}
          {field({
            key: "agent_patch_mode",
            label: "Patch mode",
            hint: "auto: patch on repair, rewrite on improve",
            options: [
              ["auto", "auto"],
              ["always", "always"],
              ["never", "never"],
            ],
          })}
          {field({ key: "agent_patch_retries", label: "Patch retries", kind: "number" })}
          {field({ key: "agent_feature_feedback", label: "Feature measurement feedback", kind: "bool" })}
          {field({ key: "agent_critique_reask", label: "Critique re-ask on empty verdicts", kind: "bool" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          tool sessions
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "agent_tool_max_calls", label: "Max tool calls (edit steps)", kind: "number" })}
          {field({ key: "agent_tool_max_calls_info", label: "Max tool calls (info steps)", kind: "number" })}
          {field({ key: "agent_tool_verify", label: "Sandbox verify tool", kind: "bool" })}
          {field({ key: "agent_tool_verify_max", label: "Verify runs per session", kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          mesh transform session
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "agent_mesh_tools",
            label: "Mesh transform tools",
            hint: "MESH_REFINE session: agent fixes generated meshes with transform ops instead of blind re-generation",
            kind: "bool",
          })}
          {field({ key: "agent_mesh_max_ops", label: "Max ops per mesh session", kind: "number" })}
          {field({ key: "agent_mesh_preview_max", label: "Preview renders per session", kind: "number" })}
          {field({ key: "agent_mesh_refine_rounds", label: "Post-critique repair rounds", kind: "number" })}
          {field({
            key: "agent_mesh_hard_fail_score",
            label: "Mesh hard-fail score",
            hint: "a generation scoring ≤ this with major issues counts as a hard fail",
            kind: "number",
          })}
          {field({
            key: "agent_mesh_hard_fail_limit",
            label: "Mesh hard-fail limit",
            hint: "after this many consecutive hard fails the mesh loop stops (CAD fallback for dimensioned parts)",
            kind: "number",
          })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          chat agent
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "chat_model_role",
            label: "Chat model role",
            hint: "vision: the agent can see uploaded + generated images; code: text-only (image tools limited to captions)",
            options: [
              ["vision", "vision (multimodal)"],
              ["code", "code (text-only)"],
            ],
          })}
          {field({ key: "chat_tool_max_calls", label: "Max tool calls per turn", kind: "number" })}
          {field({ key: "chat_max_turns", label: "Max tool rounds per turn", kind: "number" })}
          {field({
            key: "chat_spawn_timeout_s",
            label: "Spawned-job wait timeout (s)",
            hint: "how long a chat turn blocks on a spawned job before returning control",
            kind: "number",
          })}
          {field({ key: "chat_history_messages", label: "History size (messages)", kind: "number" })}
          {field({
            key: "chat_image_max",
            label: "Max images per message",
            kind: "number",
          })}
        </div>
      </Card>
      )}

      {tab === "mesh" && (
      <div className="grid gap-4 xl:grid-cols-2">
      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>Mesh generation</SectionTitle>
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
          label: "Backend",
          options: [
            ["sidecar", "Local sidecar"],
            ["tripo", "Tripo3D API"],
          ],
        })}
        {meshBackend === "tripo" ? (
          <>
            <div className="mb-3 text-xs text-zinc-500">
              image → 3D via the hosted Tripo3D API — no local weights needed
            </div>
            <div className={FIELD_GRID}>
              <div className="sm:col-span-2">
                {apiKeyField(
                  "Tripo API key",
                  s.tripo_api_key_set,
                  tripoKey,
                  setTripoKey,
                  clearTripoKey,
                  () => setClearTripoKey((c) => !c),
                )}
              </div>
              {field({
                key: "tripo_model",
                label: "Tripo model",
                options: [
                  ["v2.5-20250123", "v2.5-20250123"],
                  ["v2.0-20240919", "v2.0-20240919"],
                  ["v1.4-20240625", "v1.4-20240625"],
                ],
              })}
              {field({
                key: "tripo_api_base",
                label: "Tripo API base",
                hint: "empty = default https://api.tripo3d.ai",
                wide: true,
              })}
            </div>
          </>
        ) : (
          <>
        <div className={FIELD_GRID}>
          {field({
            key: "mesh_base_url",
            label: "Mesh server URL",
            hint: "e.g. http://localhost:8765 (native) or http://mesh-server:8765 (compose gpu profile); empty = disabled",
            wide: true,
          })}
          <div className="sm:col-span-2">
            <Field
              label="Default mesh model"
              hint="auto picks by device. On Apple-silicon MPS the hunyuan3d-2mv variant can stall under memory pressure — prefer hunyuan3d-2.1-mlx or the mini/turbo variants there."
            >
              <select
                className={inputCls}
                value={form.mesh_model}
                onChange={(e) => set("mesh_model", e.target.value)}
              >
                <option value="auto">auto (device default)</option>
                {meshModels.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name || m.id} ({m.device}){m.available ? "" : " — not installed"}
                  </option>
                ))}
                {form.mesh_model &&
                  !["auto", ...meshModels.map((m) => m.id)].includes(form.mesh_model) && (
                    <option value={form.mesh_model}>{form.mesh_model}</option>
                  )}
              </select>
            </Field>
          </div>
          {field({ key: "mesh_timeout_s", label: "Generation timeout (s)", kind: "number" })}
          {field({ key: "mesh_stall_timeout_s", label: "Stall abort — no progress (s)", kind: "number" })}
          {field({
            key: "mesh_min_free_mb",
            label: "Min free host RAM (MB)",
            kind: "number",
            hint: "refuse new generation jobs when the sidecar host has less free RAM — 0 disables. Prevents MPS stalls under memory pressure.",
          })}
          {field({ key: "mesh_decimate_faces", label: "Decimate target (faces)", kind: "number" })}
          {field({ key: "mesh_max_faces_for_cad", label: "Max faces for CAD conversion", kind: "number" })}
        </div>
        {meshModels.length > 0 && (
          <div className="mb-2 flex flex-col gap-1.5 text-xs text-zinc-500">
            {meshModels.map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <b className="text-zinc-300">{m.id}</b> · {m.device}
                <Chip tone={modelStateTone(m)}>{modelStateLabel(m)}</Chip>
                {m.est_mem_mb != null && <span>{fmtMemGb(m.est_mem_mb)}</span>}
                {m.supports_multiview ? " · multiview" : ""}
                {m.license_note ? (
                  <div className="basis-full text-zinc-500">license: {m.license_note}</div>
                ) : null}
                {m.last_error ? (
                  <div className="basis-full text-amber-500">last error: {m.last_error}</div>
                ) : null}
              </div>
            ))}
            <div>
              downloads &amp; custom models live under{" "}
              <b className="text-zinc-400">Local models</b>
            </div>
          </div>
        )}
        {!meshHealth.data?.ok && (
          <div className="mb-2 text-xs text-zinc-500">
            start the sidecar with <code>make mesh-server</code>. Mesh/hybrid modes fall back
            to CAD while it is unreachable.
          </div>
        )}
          </>
        )}
      </Card>

      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>Image generation</SectionTitle>
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
          label: "Backend",
          options: [
            ["sidecar", "Local sidecar"],
            ["openai", "OpenAI API"],
          ],
        })}
        {imageBackend === "openai" ? (
          <>
            <div className="mb-3 text-xs text-zinc-500">
              text-to-image and image editing via the OpenAI images API — empty
              connection fields reuse the LLM provider config
            </div>
            <div className={FIELD_GRID}>
              {field({
                key: "image_openai_model",
                label: "OpenAI image model",
                options: [
                  ["gpt-image-1", "gpt-image-1"],
                  ["dall-e-3", "dall-e-3"],
                  ["dall-e-2", "dall-e-2"],
                ],
              })}
              {field({
                key: "image_api_base",
                label: "Image API base",
                hint: "empty = reuse LLM provider config",
                wide: true,
              })}
              <div className="sm:col-span-2">
                {apiKeyField(
                  "Image API key",
                  s.image_api_key_set,
                  imageKey,
                  setImageKey,
                  clearImageKey,
                  () => setClearImageKey((c) => !c),
                  "empty = reuse the LLM API key",
                )}
              </div>
            </div>
          </>
        ) : (
          <>
        <div className="mb-3 text-xs text-zinc-500">
          text-to-image and image editing on the same mesh-server sidecar — for concept
          images, design ideation, img2img variations and auto-generated hero images
          for mesh mode.
        </div>
        <div className={FIELD_GRID}>
          <Field label="Default image model">
            <select
              className={inputCls}
              value={form.image_model}
              onChange={(e) => set("image_model", e.target.value)}
            >
              <option value="auto">auto (device default)</option>
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
            label="Default edit model"
            hint="used when a reference image is given (img2img / instruction edit)"
          >
            <select
              className={inputCls}
              value={form.image_edit_model}
              onChange={(e) => set("image_edit_model", e.target.value)}
            >
              <option value="auto">auto (edit-capable adapter)</option>
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
          <div className="mb-2 flex flex-col gap-1.5 text-xs text-zinc-500">
            {(imageModels.data ?? []).map((m) => (
              <div key={m.id} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <b className="text-zinc-300">{m.id}</b> · {m.device}
                <Chip tone={modelStateTone(m)}>{modelStateLabel(m)}</Chip>
                {m.supports_edit ? (m.requires_image ? " · edit-only" : " · edit-capable") : ""}
                {m.last_error ? (
                  <div className="basis-full text-amber-500">last error: {m.last_error}</div>
                ) : null}
                {m.license_note ? (
                  <div className="basis-full text-zinc-500">license: {m.license_note}</div>
                ) : null}
              </div>
            ))}
          </div>
        ) : meshHealth.data && !meshHealth.data.ok ? (
          <div className="mb-2 text-xs text-zinc-500">
            mesh-server sidecar unreachable — image generation shares it; start
            it with <code>make mesh-server</code>.
          </div>
        ) : (
          <div className="mb-2 text-xs text-zinc-500">
            no image adapter installed — see <code>docs/image-generation.md</code> for
            setup (FLUX.1-schnell on MLX/CUDA, SDXL fallback).
          </div>
        )}
          </>
        )}
        {/* shared tuning — applies to both image backends */}
        <div className={FIELD_GRID}>
          {field({
            key: "image_default_strength",
            label: "Default edit strength (0–1)",
            kind: "number",
          })}
          {field({ key: "image_steps", label: "Steps (empty = adapter default)", kind: "number" })}
          {field({ key: "image_guidance", label: "Guidance (empty = adapter default)", kind: "number" })}
          {field({ key: "image_max_side", label: "Max image side (px)", kind: "number" })}
          {field({ key: "image_max_count", label: "Max images per request", kind: "number" })}
          {field({ key: "image_timeout_s", label: "Generation timeout (s)", kind: "number" })}
          {field({
            key: "image_autogen_mesh",
            label: "Auto hero image for mesh runs",
            hint: "synthesise a reference image when a mesh/hybrid job has no input images",
            kind: "bool",
          })}
          {field({
            key: "image_tool_budget",
            label: "Agent concept-image calls per session",
            kind: "number",
          })}
        </div>
      </Card>
      </div>
      )}

      {tab === "fem" && (
      <Card className="p-4">
        <SectionTitle>FEM & topology optimization</SectionTitle>
        <div className="mb-3 text-xs text-zinc-500">
          engineering feedback only — not a certification tool
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "fem_voxel_mm", label: "Voxel size (mm)", kind: "number" })}
          {field({ key: "fem_max_elements", label: "Max FEM elements", kind: "number" })}
          {field({ key: "fem_timeout_s", label: "FEM timeout (s)", kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          topopt
        </div>
        <div className={FIELD_GRID}>
          {field({ key: "topopt_timeout_s", label: "TopOpt timeout (s)", kind: "number" })}
          {field({ key: "topopt_max_iterations", label: "Max SIMP iterations", kind: "number" })}
          {field({ key: "topopt_default_volfrac", label: "Default volume fraction", kind: "number" })}
          {field({ key: "topopt_default_rmin_mm", label: "Filter radius (mm)", kind: "number" })}
        </div>
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          verification
        </div>
        <div className={FIELD_GRID}>
          {field({
            key: "fem_verify_realizations",
            label: "Verify print realizations",
            kind: "bool",
            hint: "eroded/dilated variants → worst-case SF",
          })}
          {field({
            key: "fem_verify_realization_cells",
            label: "Realization boundary cells",
            kind: "number",
            hint: "erode/dilate depth in voxels",
          })}
          {field({
            key: "fem_verify_coverage_e",
            label: "Coverage-weighted stiffness",
            kind: "bool",
            hint: "partial voxels get reduced modulus",
          })}
          {field({
            key: "fem_verify_mesh_shift",
            label: "Mesh-shift probe",
            kind: "bool",
            hint: "re-voxelize shifted ½ pitch — grid sensitivity",
          })}
          {field({
            key: "fem_loadpath_cert",
            label: "Load-path certificate",
            kind: "bool",
            hint: "geometric min-cut SF floor + bottleneck",
          })}
          {field({
            key: "fem_homogenized_infill",
            label: "Homogenized infill verify",
            kind: "bool",
            hint: "lattice C(ρ) elements → sf_homog + sf_micro",
          })}
          {field({
            key: "fem_homog_cell_voxels",
            label: "Homog cell voxels",
            kind: "number",
            hint: "voxel size of one representative cell",
          })}
          {field({
            key: "infill_octet",
            label: "Octet infill diagonals",
            kind: "bool",
            hint: "45° strut families in aligned infill",
          })}
        </div>
      </Card>
      )}

      {tab === "system" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-4">
            <SectionTitle>Executor</SectionTitle>
            <div className={FIELD_GRID}>
              {field({ key: "exec_timeout_s", label: "Code execution timeout (s)", kind: "number" })}
              {field({
                key: "exec_mem_limit_mb",
                label: "Sandbox memory limit (MB)",
                kind: "number",
                hint: "0 = auto",
              })}
              {field({
                key: "exec_threads",
                label: "Threads per stage",
                kind: "number",
                hint: "0 = auto — OCC meshing, BLAS, software raster",
              })}
              {field({
                key: "exec_pool_workers",
                label: "Process pool workers",
                kind: "number",
                hint: "0 = auto; 1 disables FEM case pool / pipeline fan-out",
              })}
              {field({
                key: "audit_parallel_runs",
                label: "Audit parallel runs",
                kind: "number",
                hint: "0 = auto — parallel ±ε design-audit perturbations",
              })}
            </div>
            {Object.keys(s.resources_detected ?? {}).length > 0 && (
              <div className="mt-2 text-xs text-zinc-500">
                detected: {s.resources_detected.cpus ?? "?"} cpus ·{" "}
                {Math.round((s.resources_detected.mem_mb ?? 0) / 1024)}g mem ·{" "}
                {s.resources_detected.threads ?? "?"} threads/stage ·{" "}
                {s.resources_detected.pool ?? "?"} pool workers
              </div>
            )}
            <div className="mt-2 text-xs text-zinc-500">
              hard outer caps are set by docker (cpus/mem_limit, auto-sized at startup); these
              apply inside the sandbox
            </div>
          </Card>

          <Card className="p-4 text-sm text-zinc-400">
            <SectionTitle>Runtime</SectionTitle>
            <div>
              data dir: <code className="text-zinc-300">{s.data_dir}</code>
            </div>
            <div>
              all runtime settings are stored in the app database — infrastructure
              (database, redis, ports) is fixed by docker compose / launcher defaults
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
          <div className="flex max-w-full items-center gap-3 rounded-full border border-zinc-700 bg-zinc-900/95 py-2 pl-4 pr-2 shadow-2xl shadow-black/50 backdrop-blur">
            {saveError ? (
              <span className="max-w-md truncate text-xs text-red-400" title={saveError}>
                {saveError}
              </span>
            ) : (
              <span className="text-xs text-zinc-400">
                {saved && !dirty
                  ? "saved — applies to new jobs"
                  : "unsaved changes — apply to new jobs; running jobs keep their snapshot"}
              </span>
            )}
            {dirty && (
              <Btn variant="ghost" onClick={revert} disabled={save.isPending}>
                revert
              </Btn>
            )}
            <Btn onClick={() => save.mutate()} busy={save.isPending} disabled={!dirty}>
              {saved && !dirty ? "saved ✓" : "Save settings"}
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

