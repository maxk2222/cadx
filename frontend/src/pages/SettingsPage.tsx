import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { GenMode, MeshHealth, Settings } from "../lib/types";

export default function SettingsPage() {
  const [s, setS] = useState<Settings | null>(null);
  const [form, setForm] = useState<Record<string, string>>({});
  const [meshHealth, setMeshHealth] = useState<MeshHealth | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings().then((d) => {
      setS(d);
      setForm({
        llm_base_url: d.llm_base_url,
        llm_model_vision: d.llm_model_vision,
        llm_model_code: d.llm_model_code,
        llm_json_mode: d.llm_json_mode,
        mesh_base_url: d.mesh_base_url,
        mesh_model: d.mesh_model,
        agent_default_gen_mode: d.agent_default_gen_mode,
      });
    });
    api.meshHealth().then(setMeshHealth).catch(() => setMeshHealth(null));
  }, []);

  if (!s) return <div className="p-6 text-zinc-500">loading…</div>;

  const save = async () => {
    await api.updateSettings({
      llm_base_url: form.llm_base_url,
      llm_model_vision: form.llm_model_vision,
      llm_model_code: form.llm_model_code,
      llm_json_mode: form.llm_json_mode,
      mesh_base_url: form.mesh_base_url,
      mesh_model: form.mesh_model,
      agent_default_gen_mode: form.agent_default_gen_mode as GenMode,
    });
    setSaved(true);
    api.meshHealth().then(setMeshHealth).catch(() => {});
    setTimeout(() => setSaved(false), 2000);
  };

  const field = (key: string, label: string, hint = "") => (
    <label className="mb-3 block">
      <span className="text-sm text-zinc-400">{label}</span>
      <input
        className="mt-1 w-full rounded bg-zinc-800 px-3 py-2 text-sm"
        value={form[key] ?? ""}
        onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
      />
      {hint && <span className="text-xs text-zinc-600">{hint}</span>}
    </label>
  );

  const models = meshHealth?.models ?? [];

  return (
    <div className="mx-auto max-w-xl p-6">
      <h1 className="mb-4 text-2xl font-bold">Settings</h1>
      <div className="rounded border border-zinc-800 p-4">
        <h2 className="mb-3 font-semibold">LLM provider (OpenAI-compatible)</h2>
        {!s.llm_api_key_set && (
          <div className="mb-3 rounded border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-300">
            No API key configured — set <code className="font-mono">LLM_API_KEY</code> in your .env file.
          </div>
        )}
        {field("llm_base_url", "Base URL", "e.g. https://api.openai.com/v1, http://localhost:11434/v1 (Ollama), https://openrouter.ai/api/v1")}
        {field("llm_model_vision", "Vision model", "used for image analysis & visual critique")}
        {field("llm_model_code", "Code model", "used for CAD code generation")}
        <label className="mb-3 block">
          <span className="text-sm text-zinc-400">JSON mode</span>
          <select
            className="mt-1 w-full rounded bg-zinc-800 px-3 py-2 text-sm"
            value={form.llm_json_mode}
            onChange={(e) => setForm((f) => ({ ...f, llm_json_mode: e.target.value }))}
          >
            <option value="json_object">json_object (default)</option>
            <option value="json_schema">json_schema (strict)</option>
            <option value="none">none (parse from text)</option>
          </select>
        </label>
        <button onClick={save} className="rounded bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500">
          {saved ? "saved ✓" : "Save"}
        </button>
      </div>

      <div className="mt-4 rounded border border-zinc-800 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold">Mesh generator (local models)</h2>
          {meshHealth && (
            <span
              className={`rounded px-2 py-0.5 text-xs ${
                meshHealth.ok ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"
              }`}
              title={meshHealth.error ?? ""}
            >
              {meshHealth.ok ? `online · ${meshHealth.device}` : "offline"}
            </span>
          )}
        </div>
        {field("mesh_base_url", "Mesh server URL", "e.g. http://localhost:8765 (native) or http://mesh-server:8765 (compose gpu profile); empty = disabled")}
        <label className="mb-3 block">
          <span className="text-sm text-zinc-400">Default mesh model</span>
          <select
            className="mt-1 w-full rounded bg-zinc-800 px-3 py-2 text-sm"
            value={form.mesh_model}
            onChange={(e) => setForm((f) => ({ ...f, mesh_model: e.target.value }))}
          >
            <option value="auto">auto (device default)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.name || m.id} ({m.device}){m.available ? "" : " — not installed"}
              </option>
            ))}
            {form.mesh_model && !["auto", ...models.map((m) => m.id)].includes(form.mesh_model) && (
              <option value={form.mesh_model}>{form.mesh_model}</option>
            )}
          </select>
        </label>
        <label className="mb-3 block">
          <span className="text-sm text-zinc-400">Default generation mode</span>
          <select
            className="mt-1 w-full rounded bg-zinc-800 px-3 py-2 text-sm"
            value={form.agent_default_gen_mode}
            onChange={(e) => setForm((f) => ({ ...f, agent_default_gen_mode: e.target.value }))}
          >
            <option value="auto">auto</option>
            <option value="cad">cad</option>
            <option value="mesh">mesh</option>
            <option value="hybrid">hybrid</option>
          </select>
        </label>
        {models.length > 0 && (
          <div className="mb-2 text-xs text-zinc-500">
            {models
              .filter((m) => m.available)
              .map((m) => (
                <div key={m.id} className="mb-1">
                  <b className="text-zinc-300">{m.id}</b> · {m.device}
                  {m.supports_multiview ? " · multiview" : ""}
                  {m.loaded ? " · loaded" : ""}
                  {m.license_note ? <div className="text-zinc-600">license: {m.license_note}</div> : null}
                </div>
              ))}
          </div>
        )}
        {!meshHealth?.ok && (
          <div className="mb-2 text-xs text-zinc-500">
            start the sidecar with <code>make mesh-server</code> or use{" "}
            <code>MESH_FAKE=1</code> for testing. Mesh/hybrid modes fall back to CAD while it is unreachable.
          </div>
        )}
        <button onClick={save} className="rounded bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500">
          {saved ? "saved ✓" : "Save"}
        </button>
      </div>

      <div className="mt-4 rounded border border-zinc-800 p-4 text-sm text-zinc-400">
        <div>data dir: <code>{s.data_dir}</code></div>
        <div>agent: {s.agent_max_iterations} iterations, {s.agent_max_repairs} repairs, pass ≥ {s.agent_pass_score}</div>
        <div>executor: {s.exec_timeout_s}s timeout, {s.exec_memory_mb}MB memory cap</div>
      </div>
    </div>
  );
}
