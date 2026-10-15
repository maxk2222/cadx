import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Btn,
  Card,
  Chip,
  Field,
  SectionTitle,
  Spinner,
  inputCls,
} from "../components/ui";
import { api } from "../lib/api";
import type { Settings, SettingsUpdate } from "../lib/types";
import { useTitle } from "../lib/useTitle";

interface FieldSpec {
  key: keyof Settings;
  label: string;
  hint?: string;
  kind?: "text" | "number" | "bool";
  options?: [string, string][];
}

const NUM_FIELDS = new Set<string>([
  "llm_temperature_code",
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
  "fem_voxel_mm",
  "fem_max_elements",
  "fem_timeout_s",
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
]);

const BOOL_FIELDS = new Set<string>([
  "llm_stream",
  "agent_param_autofix",
  "agent_feature_feedback",
  "agent_critique_reask",
  "agent_tool_verify",
  "agent_mesh_tools",
  "image_autogen_mesh",
]);

// mirrors the bounds in backend SettingsUpdate — keep in sync with schemas.py
const FIELD_BOUNDS: Record<
  string,
  { min?: number; max?: number; exclusiveMin?: boolean; int?: boolean }
> = {
  llm_temperature_code: { min: 0, max: 2 },
  llm_max_tokens: { min: 1, max: 2_000_000, int: true },
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
  fem_voxel_mm: { min: 0, max: 100, exclusiveMin: true },
  fem_max_elements: { min: 1, max: 5_000_000, int: true },
  fem_timeout_s: { min: 1, max: 86_400, int: true },
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
  exec_mem_limit_mb: { min: 256, max: 1_048_576, int: true },
};

type TabId = "provider" | "agent" | "mesh" | "fem" | "system";
const TABS: { id: TabId; label: string }[] = [
  { id: "provider", label: "LLM Provider" },
  { id: "agent", label: "Agent" },
  { id: "mesh", label: "Mesh & Image" },
  { id: "fem", label: "FEM / TopOpt" },
  { id: "system", label: "System" },
];

const TAB_IDS = new Set<string>(TABS.map((t) => t.id));

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
  }, [sp]);
  const [form, setForm] = useState<Record<string, string> | null>(null);
  const [initial, setInitial] = useState<Record<string, string> | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
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
      if (k === "data_dir" || k === "llm_api_key_set") continue;
      out[k] = v == null ? "" : String(v); // nullable fields show empty = default
    }
    setForm(out);
    setInitial(out);
  }, [settings.data, form]);

  const dirty = useMemo(() => {
    if (!form || !initial) return false;
    if (apiKey.trim() || clearKey) return true;
    for (const k of new Set([...Object.keys(form), ...Object.keys(initial)])) {
      if ((form[k] ?? "") !== (initial[k] ?? "")) return true;
    }
    return false;
  }, [form, initial, apiKey, clearKey]);

  const revert = () => {
    setForm(initial);
    setApiKey("");
    setClearKey(false);
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
      return api.updateSettings(body);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["meshHealth"] });
      qc.invalidateQueries({ queryKey: ["health"] });
      qc.invalidateQueries({ queryKey: ["llmHealth"] });
      setApiKey("");
      setClearKey(false);
      setSaveError("");
      setInitial(form);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
    onError: (e) => setSaveError(String(e)),
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
    if (spec.options) {
      return (
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
      );
    }
    if (spec.kind === "bool") {
      return (
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
      );
    }
    const b = spec.kind === "number" ? FIELD_BOUNDS[key] : undefined;
    return (
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
  };

  // the sidecar hosts both adapter families — split them for display
  const models = meshHealth.data?.models ?? [];
  const meshModels = models.filter((m) => m.kind !== "image");
  const imageAdapters = (imageModels.data ?? []).filter((m) => m.available);
  const lp = llmHealth.data;
  const t = testProvider.data;

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
    <div className="mx-auto max-w-xl p-6 pb-28">
      <h1 className="mb-4 text-2xl font-bold">Settings</h1>

      {/* service health — 2×2 */}
      <div className="mb-4 grid grid-cols-2 gap-3">
        {statusCard(
          "API",
          health.data?.ok ? "online" : "offline",
          health.data?.ok ? "ok" : "bad",
        )}
        {statusCard(
          "LLM",
          lp?.ok
            ? `online · ${lp.latency_ms}ms${lp.vision_ok ? " · vision" : ""}`
            : llmHealth.isPending
              ? "checking…"
              : "offline",
          lp?.ok ? "ok" : llmHealth.isPending ? "idle" : "bad",
          lp?.error ?? (s.llm_api_key_set ? undefined : "no API key configured"),
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

      {/* category tabs */}
      <div className="mb-4 flex gap-1 overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-900/60 p-1">
        {TABS.map((tb) => (
          <button
            key={tb.id}
            onClick={() => {
              setTab(tb.id);
              localStorage.setItem("settings.tab", tb.id);
              setSp({ tab: tb.id }, { replace: true });
            }}
            className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors ${
              tab === tb.id
                ? "bg-zinc-700 font-medium text-zinc-100"
                : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            }`}
          >
            {tb.label}
          </button>
        ))}
      </div>

      {tab === "provider" && (
      <Card className="p-4">
        <SectionTitle>LLM provider (OpenAI-compatible)</SectionTitle>
        {!s.llm_api_key_set && (
          <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
            No API key configured — paste a key below and save. Generation works without
            one only for providers that skip authentication.
          </div>
        )}
        {field({
          key: "llm_base_url",
          label: "Base URL",
          hint: "e.g. https://api.openai.com/v1, http://localhost:11434/v1 (Ollama), https://openrouter.ai/api/v1",
        })}
        <div className="mb-3 flex items-end gap-2">
          <Field
            label="API key"
            hint={
              s.llm_api_key_set
                ? "a key is stored — enter a new one to replace it (never shown again)"
                : "stored in the app database, write-only"
            }
          >
            <input
              className={inputCls}
              type="password"
              autoComplete="off"
              placeholder={s.llm_api_key_set ? "••••••••" : "sk-…"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </Field>
          {s.llm_api_key_set && (
            <Btn variant="ghost" onClick={() => setClearKey((c) => !c)}>
              {clearKey ? "keep key" : "clear key"}
            </Btn>
          )}
        </div>
        {clearKey && (
          <div className="mb-3 text-xs text-amber-400">
            the stored API key will be removed on save
          </div>
        )}
        {field({ key: "llm_model_vision", label: "Vision model", hint: "used for image analysis & visual critique" })}
        {field({ key: "llm_model_code", label: "Code model", hint: "used for CAD code generation" })}
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
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
          advanced
        </div>
        {field({ key: "llm_temperature_code", label: "Temperature (code)", kind: "number" })}
        {field({ key: "llm_max_tokens", label: "Max tokens per response", kind: "number" })}
        {field({ key: "llm_stream", label: "Streaming", kind: "bool" })}
        {field({ key: "llm_timeout_s", label: "Request timeout (s)", kind: "number" })}
        {field({ key: "llm_idle_timeout_s", label: "Stream idle timeout (s)", kind: "number" })}
        {field({ key: "llm_degenerate_stall_s", label: "Degenerate stream abort (s)", kind: "number" })}
        {field({ key: "llm_total_timeout_s", label: "Stream total cap (s)", kind: "number" })}
        {field({
          key: "llm_extra_body",
          label: "Extra request body (JSON)",
          hint: 'merged into every request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}',
        })}
        <div className="mt-2 flex items-center gap-2">
          <Btn variant="ghost" onClick={() => testProvider.mutate()} busy={testProvider.isPending}>
            Test provider
          </Btn>
          {t && (
            <span className={`text-xs ${t.ok ? "text-emerald-400" : "text-red-400"}`}>
              {t.ok
                ? `code ok · ${t.latency_ms}ms${t.vision_ok ? " · vision ok" : " · vision failed"}`
                : `failed: ${t.error ?? "unknown"}`}
            </span>
          )}
          {testProvider.isError && (
            <span className="text-xs text-red-400">{String(testProvider.error)}</span>
          )}
        </div>
      </Card>
      )}

      {tab === "agent" && (
      <Card className="p-4">
        <SectionTitle>Agent</SectionTitle>
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
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
          budgets & robustness
        </div>
        {field({ key: "agent_max_input_images", label: "Max input images", kind: "number" })}
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
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
          tool sessions
        </div>
        {field({ key: "agent_tool_max_calls", label: "Max tool calls (edit steps)", kind: "number" })}
        {field({ key: "agent_tool_max_calls_info", label: "Max tool calls (info steps)", kind: "number" })}
        {field({ key: "agent_tool_verify", label: "Sandbox verify tool", kind: "bool" })}
        {field({ key: "agent_tool_verify_max", label: "Verify runs per session", kind: "number" })}
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
          mesh transform session
        </div>
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
      </Card>
      )}

      {tab === "mesh" && (
      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>Mesh generator (local models)</SectionTitle>
          {meshHealth.data && (
            <Chip
              tone={meshHealth.data.ok ? "green" : "red"}
              title={meshHealth.data.error ?? ""}
            >
              {meshHealth.data.ok ? `online · ${meshHealth.data.device}` : "offline"}
            </Chip>
          )}
        </div>
        {field({
          key: "mesh_base_url",
          label: "Mesh server URL",
          hint: "e.g. http://localhost:8765 (native) or http://mesh-server:8765 (compose gpu profile); empty = disabled",
        })}
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
        {meshModels.length > 0 && (
          <div className="mb-2 flex flex-col gap-1 text-xs text-zinc-500">
            {meshModels
              .filter((m) => m.available)
              .map((m) => (
                <div key={m.id}>
                  <b className="text-zinc-300">{m.id}</b> · {m.device}
                  {m.supports_multiview ? " · multiview" : ""}
                  {m.loaded ? " · loaded" : ""}
                  {m.license_note ? (
                    <div className="text-zinc-600">license: {m.license_note}</div>
                  ) : null}
                </div>
              ))}
          </div>
        )}
        {!meshHealth.data?.ok && (
          <div className="mb-2 text-xs text-zinc-500">
            start the sidecar with <code>make mesh-server</code>. Mesh/hybrid modes fall back
            to CAD while it is unreachable.
          </div>
        )}
      </Card>
      )}

      {tab === "mesh" && (
      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionTitle>Image generation (local models)</SectionTitle>
          {meshHealth.data && (
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
          )}
        </div>
        <div className="mb-3 text-xs text-zinc-500">
          text-to-image and image editing on the same mesh-server sidecar — for concept
          images, design ideation, img2img variations and auto-generated hero images
          for mesh mode.
        </div>
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
        {(imageModels.data ?? []).length > 0 ? (
          <div className="mb-2 flex flex-col gap-1 text-xs text-zinc-500">
            {(imageModels.data ?? []).map((m) => (
              <div key={m.id}>
                <b className="text-zinc-300">{m.id}</b> · {m.device}
                {m.loaded ? " · loaded" : ""}
                {m.supports_edit ? (m.requires_image ? " · edit-only" : " · edit-capable") : ""}
                {m.available ? "" : " · not installed"}
                {m.last_error ? (
                  <div className="text-amber-500">last error: {m.last_error}</div>
                ) : null}
                {m.license_note ? (
                  <div className="text-zinc-600">license: {m.license_note}</div>
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
      </Card>
      )}

      {tab === "fem" && (
      <Card className="p-4">
        <SectionTitle>FEM & topology optimization</SectionTitle>
        <div className="mb-3 text-xs text-zinc-500">
          engineering feedback only — not a certification tool
        </div>
        {field({ key: "fem_voxel_mm", label: "Voxel size (mm)", kind: "number" })}
        {field({ key: "fem_max_elements", label: "Max FEM elements", kind: "number" })}
        {field({ key: "fem_timeout_s", label: "FEM timeout (s)", kind: "number" })}
        <div className="mt-3 mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
          topopt
        </div>
        {field({ key: "topopt_timeout_s", label: "TopOpt timeout (s)", kind: "number" })}
        {field({ key: "topopt_max_iterations", label: "Max SIMP iterations", kind: "number" })}
        {field({ key: "topopt_default_volfrac", label: "Default volume fraction", kind: "number" })}
        {field({ key: "topopt_default_rmin_mm", label: "Filter radius (mm)", kind: "number" })}
      </Card>
      )}

      {tab === "system" && (
        <>
          <Card className="p-4">
            <SectionTitle>Executor</SectionTitle>
            {field({ key: "exec_timeout_s", label: "Code execution timeout (s)", kind: "number" })}
            {field({ key: "exec_mem_limit_mb", label: "Sandbox memory limit (MB)", kind: "number" })}
            <div className="mt-2 text-xs text-zinc-500">
              hard outer caps are set by docker (cpus/mem_limit); these apply inside the sandbox
            </div>
          </Card>

          <Card className="mt-4 p-4 text-sm text-zinc-400">
            <SectionTitle>Runtime</SectionTitle>
            <div>
              data dir: <code className="text-zinc-300">{s.data_dir}</code>
            </div>
            <div>
              all runtime settings are stored in the app database — infrastructure
              (database, redis, ports) is fixed by docker compose / launcher defaults
            </div>
          </Card>
        </>
      )}

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

