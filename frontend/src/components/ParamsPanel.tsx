import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { ParamDef } from "../lib/types";

export default function ParamsPanel({
  versionId,
  schema,
  params,
}: {
  versionId: string;
  schema: ParamDef[] | null;
  params: Record<string, unknown> | null;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();
  if (!schema?.length) return null;

  const apply = async () => {
    setBusy(true);
    try {
      const overrides: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(values)) {
        const def = schema.find((p) => p.name === k);
        const cur = params?.[k] ?? def?.value;
        if (v === "" || v === String(cur)) continue;
        overrides[k] = typeof cur === "number" ? Number(v) : v;
      }
      if (!Object.keys(overrides).length) return;
      const r = await api.setParams(versionId, overrides);
      nav(`/versions/${r.version_id}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-zinc-300">Parameters</h3>
      <div className="grid grid-cols-2 gap-2">
        {schema.map((p) => {
          const cur = params?.[p.name] ?? p.value;
          return (
            <label key={p.name} className="block">
              <span className="text-xs text-zinc-400">
                {p.name}
                {p.unit && p.unit !== "-" ? ` (${p.unit})` : ""}
                {p.min != null || p.max != null ? ` [${p.min ?? "−∞"}–${p.max ?? "∞"}]` : ""}
              </span>
              <input
                className="mt-0.5 w-full rounded bg-zinc-800 px-2 py-1 text-sm"
                defaultValue={String(cur)}
                onChange={(e) => setValues((v) => ({ ...v, [p.name]: e.target.value }))}
              />
            </label>
          );
        })}
      </div>
      <button
        disabled={busy}
        onClick={apply}
        className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
      >
        {busy ? "applying…" : "Apply as new version"}
      </button>
    </div>
  );
}
