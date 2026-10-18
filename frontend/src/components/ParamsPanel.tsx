import { useMutation } from "@tanstack/react-query";
import { RotateCcw } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import type { ParamDef } from "../lib/types";
import { Btn, IconBtn, SectionTitle, inputCls } from "./ui";

/** Inline error for one raw input value. Numeric params are checked against
 *  the schema min/max; empty strings and non-numeric params pass through. */
export function paramError(p: ParamDef, cur: unknown, raw: string): string | null {
  const numeric = typeof cur === "number" || (cur == null && typeof p.value === "number");
  if (!numeric || raw === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return "not a number";
  if ((p.min != null && n < p.min) || (p.max != null && n > p.max))
    return `outside ${p.min ?? "−∞"}–${p.max ?? "∞"}`;
  return null;
}

/** Diff the edited values against the current params — numeric values are
 *  parsed (finite-checked so NaN never reaches JSON null), booleans coerced
 *  from "true"/"false", everything else passed through as a string. */
export function buildOverrides(
  schema: ParamDef[],
  params: Record<string, unknown> | null,
  values: Record<string, string>
): Record<string, unknown> {
  const overrides: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(values)) {
    const def = schema.find((p) => p.name === k);
    const cur = params?.[k] ?? def?.value;
    if (v === "" || v === String(cur)) continue;
    if (typeof cur === "boolean") overrides[k] = v === "true";
    else if (typeof cur === "number") {
      const n = Number(v);
      if (Number.isFinite(n)) overrides[k] = n;
    } else overrides[k] = v;
  }
  return overrides;
}

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
  // the panel is rendered unkeyed — drop stale edits when the version changes
  const [forVersion, setForVersion] = useState(versionId);
  if (forVersion !== versionId) {
    setForVersion(versionId);
    setValues({});
  }
  const nav = useNavigate();

  const apply = useMutation({
    mutationFn: () => api.setParams(versionId, buildOverrides(schema!, params, values)),
    onSuccess: (r) => nav(`/versions/${r.version_id}`),
  });

  if (!schema?.length) return null;

  const current = (p: ParamDef) => params?.[p.name] ?? p.value;
  const edited = (p: ParamDef) => p.name in values;
  const fieldError = (p: ParamDef) =>
    edited(p) ? paramError(p, current(p), values[p.name]) : null;
  const hasErrors = schema.some((p) => fieldError(p) != null);
  const dirty = schema.some(
    (p) => edited(p) && values[p.name] !== "" && values[p.name] !== String(current(p))
  );

  return (
    <div>
      <SectionTitle
        right={
          dirty && (
            <IconBtn title="reset to current values" onClick={() => setValues({})}>
              <RotateCcw className="h-3.5 w-3.5" />
            </IconBtn>
          )
        }
      >
        Parameters
      </SectionTitle>
      <div className="grid grid-cols-2 gap-2">
        {schema.map((p) => {
          const cur = current(p);
          const raw = edited(p) ? values[p.name] : String(cur);
          const err = fieldError(p);
          return (
            <label key={p.name} className="block">
              <span className="text-xs text-zinc-400">
                {p.name}
                {p.unit && p.unit !== "-" ? ` (${p.unit})` : ""}
                {p.min != null || p.max != null
                  ? ` [${p.min ?? "−∞"}–${p.max ?? "∞"}]`
                  : ""}
              </span>
              {typeof cur === "boolean" ? (
                <span className="mt-0.5 flex items-center gap-1.5 px-1 py-1 text-xs text-zinc-400">
                  <input
                    type="checkbox"
                    className="accent-brand-500"
                    checked={raw === "true"}
                    onChange={(e) =>
                      setValues((v) => ({ ...v, [p.name]: String(e.target.checked) }))
                    }
                  />
                  {raw === "true" ? "on" : "off"}
                </span>
              ) : (
                <input
                  type={typeof cur === "number" ? "number" : "text"}
                  step="any"
                  min={typeof cur === "number" && p.min != null ? p.min : undefined}
                  max={typeof cur === "number" && p.max != null ? p.max : undefined}
                  aria-invalid={err != null || undefined}
                  className={`${inputCls} mt-0.5 px-2 py-1 ${err ? "border-red-700" : ""}`}
                  value={raw}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [p.name]: e.target.value }))
                  }
                />
              )}
              {err && <div className="mt-0.5 text-[10px] text-red-400">{err}</div>}
              {p.description && (
                <div className="mt-0.5 text-[10px] text-zinc-500">{p.description}</div>
              )}
            </label>
          );
        })}
      </div>
      {hasErrors && (
        <div className="mt-2 text-[10px] text-red-400">
          fix the out-of-range fields before applying
        </div>
      )}
      {apply.isError && (
        <div className="mt-2 text-xs text-red-400">{errMsg(apply.error)}</div>
      )}
      <Btn
        className="mt-2"
        disabled={!dirty || hasErrors}
        busy={apply.isPending}
        onClick={() => apply.mutate()}
      >
        Apply as new version
      </Btn>
    </div>
  );
}
