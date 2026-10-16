import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { EditableFeature } from "../lib/types";
import { Btn, Spinner, inputCls, useToast } from "./ui";

/** Parametric feature editing (M9): pick a recognised through-hole and
 *  resize its diameter — the executor gate verifies the edit and produces a
 *  child version, or refuses with a reason. */
export default function FeatureEditPanel({ versionId }: { versionId: string }) {
  const nav = useNavigate();
  const toast = useToast();
  const [featureId, setFeatureId] = useState<string>("");
  const [diameter, setDiameter] = useState<string>("");

  const feats = useQuery({
    queryKey: ["editable-features", versionId],
    queryFn: () => api.editableFeatures(versionId),
  });
  const list: EditableFeature[] = feats.data?.features ?? [];
  const editable = list.filter((f) => f.editable !== false);
  const selected = list.find((f) => String(f.feature_id) === featureId);

  const edit = useMutation({
    mutationFn: () =>
      api.editFeature(versionId, {
        feature_id: selected?.feature_id ?? featureId,
        parameter: "diameter",
        value: Number(diameter),
      }),
    onSuccess: (r) => nav(`/versions/${r.version_id}?job=${r.job_id}`),
    onError: (e) => toast(String(e), "error"),
  });

  if (feats.isLoading)
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <Spinner /> scanning for editable features…
      </div>
    );
  if (!list.length)
    return (
      <div className="text-xs text-zinc-500">
        {feats.data?.reason || "no editable features recognised on this version"}
      </div>
    );

  return (
    <div>
      <div className="mb-2 space-y-1">
        {list.map((f) => {
          const dis = f.editable === false;
          const on = String(f.feature_id) === featureId;
          return (
            <button
              key={String(f.feature_id)}
              type="button"
              disabled={dis}
              title={dis ? f.refusal || "not editable" : undefined}
              onClick={() => {
                setFeatureId(String(f.feature_id));
                setDiameter(String(f.diameter));
              }}
              className={`block w-full rounded border px-2 py-1 text-left text-xs ${
                dis
                  ? "cursor-not-allowed border-zinc-800 text-zinc-600"
                  : on
                    ? "border-brand-600 bg-brand-950/60 text-brand-200"
                    : "border-zinc-700 text-zinc-300 hover:border-zinc-500"
              }`}
            >
              #{f.feature_id} {f.kind} Ø{f.diameter} mm
              {f.axis ? ` · ${f.axis}` : ""}
              {f.depth != null ? ` · ${f.depth} mm deep` : ""}
              {dis ? ` — ${f.refusal || "not editable"}` : ""}
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="number"
          step="0.1"
          min="0"
          className={inputCls}
          placeholder="new Ø mm"
          value={diameter}
          onChange={(e) => setDiameter(e.target.value)}
        />
        <Btn
          variant="success"
          onClick={() => edit.mutate()}
          busy={edit.isPending}
          disabled={!selected || !(Number(diameter) > 0)}
        >
          Resize hole
        </Btn>
      </div>
      <div className="mt-1 text-[11px] text-zinc-600">
        verified edit — the gate refuses if the result breaks the part
      </div>
    </div>
  );
}
