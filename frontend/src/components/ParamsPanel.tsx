import { useMutation } from "@tanstack/react-query";
import { RotateCcw } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { ParamDef } from "../lib/types";
import { Btn, IconBtn, SectionTitle, inputCls } from "./ui";

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
  const [key, setKey] = useState(0);
  const nav = useNavigate();

  const apply = useMutation({
    mutationFn: () => {
      const overrides: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(values)) {
        const def = schema!.find((p) => p.name === k);
        const cur = params?.[k] ?? def?.value;
        if (v === "" || v === String(cur)) continue;
        overrides[k] = typeof cur === "number" ? Number(v) : v;
      }
      return api.setParams(versionId, overrides);
    },
    onSuccess: (r) => nav(`/versions/${r.version_id}`),
  });

  if (!schema?.length) return null;

  const dirty = Object.keys(values).some(
    (k) => values[k] !== "" && values[k] !== String(params?.[k] ?? schema.find((p) => p.name === k)?.value)
  );

  return (
    <div>
      <SectionTitle
        right={
          dirty && (
            <IconBtn
              title="reset to current values"
              onClick={() => {
                setValues({});
                setKey((k) => k + 1);
              }}
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </IconBtn>
          )
        }
      >
        Parameters
      </SectionTitle>
      <div className="grid grid-cols-2 gap-2" key={key}>
        {schema.map((p) => {
          const cur = params?.[p.name] ?? p.value;
          return (
            <label key={p.name} className="block" title={p.description || undefined}>
              <span className="text-xs text-zinc-400">
                {p.name}
                {p.unit && p.unit !== "-" ? ` (${p.unit})` : ""}
                {p.min != null || p.max != null ? ` [${p.min ?? "−∞"}–${p.max ?? "∞"}]` : ""}
              </span>
              <input
                className={`${inputCls} mt-0.5 px-2 py-1`}
                defaultValue={String(cur)}
                onChange={(e) => setValues((v) => ({ ...v, [p.name]: e.target.value }))}
              />
            </label>
          );
        })}
      </div>
      {apply.isError && <div className="mt-2 text-xs text-red-400">{String(apply.error)}</div>}
      <Btn className="mt-2" disabled={!dirty} busy={apply.isPending} onClick={() => apply.mutate()}>
        Apply as new version
      </Btn>
    </div>
  );
}
