import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Package } from "lucide-react";
import { useEffect, useState } from "react";
import {
  Btn,
  Card,
  Chip,
  EmptyState,
  Field,
  SectionTitle,
  Spinner,
  inputCls,
} from "../components/ui";
import { api } from "../lib/api";
import type { GenMode, Settings } from "../lib/types";

export default function SettingsPage() {
  const qc = useQueryClient();
  const [form, setForm] = useState<Record<string, string> | null>(null);
  const [saved, setSaved] = useState(false);

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

  // seed the form once settings arrive
  useEffect(() => {
    if (form || !settings.data) return;
    const d = settings.data;
    setForm({
      llm_base_url: d.llm_base_url,
      llm_model_vision: d.llm_model_vision,
      llm_model_code: d.llm_model_code,
      llm_json_mode: d.llm_json_mode,
      mesh_base_url: d.mesh_base_url,
      mesh_model: d.mesh_model,
      agent_default_gen_mode: d.agent_default_gen_mode,
    });
  }, [settings.data, form]);

  const save = useMutation({
    mutationFn: () =>
      api.updateSettings({
        llm_base_url: form!.llm_base_url,
        llm_model_vision: form!.llm_model_vision,
        llm_model_code: form!.llm_model_code,
        llm_json_mode: form!.llm_json_mode,
        mesh_base_url: form!.mesh_base_url,
        mesh_model: form!.mesh_model,
        agent_default_gen_mode: form!.agent_default_gen_mode as GenMode,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["meshHealth"] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  if (settings.isLoading || !form)
    return (
      <div className="flex justify-center py-24 text-zinc-500">
        <Spinner className="h-6 w-6" />
      </div>
    );
  const s = settings.data as Settings;

  const field = (key: string, label: string, hint = "") => (
    <Field label={label} hint={hint}>
      <input
        className={inputCls}
        value={form[key] ?? ""}
        onChange={(e) => setForm((f) => ({ ...f!, [key]: e.target.value }))}
      />
    </Field>
  );

  const models = meshHealth.data?.models ?? [];

  return (
    <div className="mx-auto max-w-xl p-6">
      <h1 className="mb-4 text-2xl font-bold">Settings</h1>

      {/* service health */}
      <div className="mb-4 grid grid-cols-3 gap-2">
        <Card className="p-3">
          <div className="text-[11px] uppercase tracking-wider text-zinc-500">API</div>
          <div
            className={`mt-1 text-sm font-medium ${
              health.data?.ok ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {health.data?.ok ? "online" : "offline"}
          </div>
        </Card>
        <Card className="p-3">
          <div className="text-[11px] uppercase tracking-wider text-zinc-500">LLM</div>
          <div
            className={`mt-1 text-sm font-medium ${
              s.llm_api_key_set ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {s.llm_api_key_set ? "key set" : "no key"}
          </div>
        </Card>
        <Card className="p-3">
          <div className="text-[11px] uppercase tracking-wider text-zinc-500">Mesh server</div>
          <div
            className={`mt-1 text-sm font-medium ${
              meshHealth.data?.ok ? "text-emerald-400" : "text-zinc-500"
            }`}
            title={meshHealth.data?.error ?? ""}
          >
            {meshHealth.data?.ok ? `online · ${meshHealth.data.device}` : "offline"}
          </div>
        </Card>
      </div>

      <Card className="p-4">
        <SectionTitle>LLM provider (OpenAI-compatible)</SectionTitle>
        {!s.llm_api_key_set && (
          <div className="mb-3 rounded-md border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
            No API key configured — set <code className="font-mono">LLM_API_KEY</code> in your
            .env file.
          </div>
        )}
        {field(
          "llm_base_url",
          "Base URL",
          "e.g. https://api.openai.com/v1, http://localhost:11434/v1 (Ollama), https://openrouter.ai/api/v1"
        )}
        {field("llm_model_vision", "Vision model", "used for image analysis & visual critique")}
        {field("llm_model_code", "Code model", "used for CAD code generation")}
        <Field label="JSON mode">
          <select
            className={inputCls}
            value={form.llm_json_mode}
            onChange={(e) => setForm((f) => ({ ...f!, llm_json_mode: e.target.value }))}
          >
            <option value="json_object">json_object (default)</option>
            <option value="json_schema">json_schema (strict)</option>
            <option value="none">none (parse from text)</option>
          </select>
        </Field>
        <Btn onClick={() => save.mutate()} busy={save.isPending}>
          {saved ? "saved ✓" : "Save"}
        </Btn>
      </Card>

      <Card className="mt-4 p-4">
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
        {field(
          "mesh_base_url",
          "Mesh server URL",
          "e.g. http://localhost:8765 (native) or http://mesh-server:8765 (compose gpu profile); empty = disabled"
        )}
        <Field label="Default mesh model">
          <select
            className={inputCls}
            value={form.mesh_model}
            onChange={(e) => setForm((f) => ({ ...f!, mesh_model: e.target.value }))}
          >
            <option value="auto">auto (device default)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.name || m.id} ({m.device}){m.available ? "" : " — not installed"}
              </option>
            ))}
            {form.mesh_model &&
              !["auto", ...models.map((m) => m.id)].includes(form.mesh_model) && (
                <option value={form.mesh_model}>{form.mesh_model}</option>
              )}
          </select>
        </Field>
        <Field label="Default generation mode">
          <select
            className={inputCls}
            value={form.agent_default_gen_mode}
            onChange={(e) =>
              setForm((f) => ({ ...f!, agent_default_gen_mode: e.target.value }))
            }
          >
            <option value="auto">auto</option>
            <option value="cad">cad</option>
            <option value="mesh">mesh</option>
            <option value="hybrid">hybrid</option>
          </select>
        </Field>
        {models.length > 0 && (
          <div className="mb-2 flex flex-col gap-1 text-xs text-zinc-500">
            {models
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
            start the sidecar with <code>make mesh-server</code> or use{" "}
            <code>MESH_FAKE=1</code> for testing. Mesh/hybrid modes fall back to CAD while it
            is unreachable.
          </div>
        )}
        <Btn onClick={() => save.mutate()} busy={save.isPending}>
          {saved ? "saved ✓" : "Save"}
        </Btn>
      </Card>

      <CatalogBrowser />

      <Card className="mt-4 p-4 text-sm text-zinc-400">
        <SectionTitle>Runtime</SectionTitle>
        <div>
          data dir: <code className="text-zinc-300">{s.data_dir}</code>
        </div>
        <div>
          agent: {s.agent_max_iterations} iterations, {s.agent_max_repairs} repairs, pass ≥{" "}
          {s.agent_pass_score}, clarification: {s.agent_clarification_mode}
        </div>
        <div>
          executor: {s.exec_timeout_s}s timeout, {s.exec_memory_mb}MB memory cap
        </div>
      </Card>
    </div>
  );
}

function CatalogBrowser() {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const catalog = useQuery({
    queryKey: ["catalog"],
    queryFn: api.getCatalog,
    enabled: open,
    staleTime: Infinity,
  });

  const toggle = (k: string) =>
    setExpanded((s) => {
      const n = new Set(s);
      n.has(k) ? n.delete(k) : n.add(k);
      return n;
    });

  return (
    <Card className="mt-4 p-4">
      <button
        className="flex w-full items-center gap-2 text-left"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (
          <ChevronDown className="h-4 w-4 text-zinc-500" />
        ) : (
          <ChevronRight className="h-4 w-4 text-zinc-500" />
        )}
        <Package className="h-4 w-4 text-zinc-500" />
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Standard parts catalog (bd_warehouse)
        </span>
      </button>
      {open && (
        <div className="mt-3">
          {catalog.isLoading ? (
            <div className="flex justify-center py-4 text-zinc-500">
              <Spinner />
            </div>
          ) : catalog.isError ? (
            <EmptyState title="Catalog unavailable" hint="requires the executor environment" />
          ) : (
            Object.entries(catalog.data ?? {}).map(([cat, classes]) => (
              <div key={cat} className="mb-1">
                <button
                  className="flex w-full items-center gap-1.5 rounded px-1 py-1 text-sm text-zinc-300 hover:bg-zinc-800"
                  onClick={() => toggle(cat)}
                >
                  {expanded.has(cat) ? (
                    <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />
                  )}
                  {cat}
                  <span className="text-xs text-zinc-600">
                    ({Object.keys(classes).length})
                  </span>
                </button>
                {expanded.has(cat) && (
                  <div className="ml-6 flex flex-col gap-0.5 border-l border-zinc-800 pl-2">
                    {Object.entries(classes).map(([cls, sizes]) => (
                      <div key={cls} className="py-0.5 text-xs">
                        <span className="font-mono text-zinc-300">{cls}</span>
                        {Array.isArray(sizes) && sizes.length > 0 && (
                          <span className="ml-2 text-zinc-600">
                            {sizes.slice(0, 8).join(", ")}
                            {sizes.length > 8 ? `, +${sizes.length - 8}` : ""}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </Card>
  );
}
