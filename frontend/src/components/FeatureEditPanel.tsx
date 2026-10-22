import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { api, errMsg } from "../lib/api";
import type { EditableFeature } from "../lib/types";
import { Btn, Spinner, inputCls, useToast } from "./ui";

/** Parametric feature editing (M9): pick a recognised through-hole and
 *  resize its diameter — the executor gate verifies the edit and produces a
 *  child version, or refuses with a reason.
 *
 *  The editable-features scan is deferred behind a disclosure so the panel
 *  doesn't fire a geometry request on every page load. */
export default function FeatureEditPanel({ versionId }: { versionId: string }) {
  const { t } = useTranslation("version");
  const nav = useNavigate();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [featureId, setFeatureId] = useState<string>("");
  const [diameter, setDiameter] = useState<string>("");

  // rendered unkeyed — collapse again (and drop drafts) on version change
  useEffect(() => {
    setOpen(false);
    setFeatureId("");
    setDiameter("");
  }, [versionId]);

  const feats = useQuery({
    queryKey: ["editable-features", versionId],
    queryFn: () => api.editableFeatures(versionId),
    enabled: open,
  });
  const list: EditableFeature[] = feats.data?.features ?? [];
  const selected = list.find((f) => String(f.feature_id) === featureId);

  const edit = useMutation({
    mutationFn: () =>
      api.editFeature(versionId, {
        feature_id: selected?.feature_id ?? featureId,
        parameter: "diameter",
        value: Number(diameter),
      }),
    onSuccess: (r) => nav(`/versions/${r.version_id}?job=${r.job_id}`),
    onError: (e) => toast(errMsg(e), "error"),
  });

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 text-left text-xs text-muted hover:text-fg"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        {t("feat_title")}
        {!open && (
          <span className="text-[10px] text-muted">
            {t("feat_scan_note")}
          </span>
        )}
      </button>

      {open && (
        <div className="mt-2">
          {feats.isLoading ? (
            <div className="flex items-center gap-2 text-xs text-faint">
              <Spinner /> {t("feat_scanning")}
            </div>
          ) : feats.isError ? (
            <div className="text-xs text-red-400">{errMsg(feats.error)}</div>
          ) : !list.length ? (
            <div className="text-xs text-faint">
              {feats.data?.reason || t("feat_none")}
            </div>
          ) : (
            <>
              <div className="mb-2 space-y-1">
                {list.map((f) => {
                  const dis = f.editable === false;
                  const on = String(f.feature_id) === featureId;
                  return (
                    <button
                      key={String(f.feature_id)}
                      type="button"
                      disabled={dis}
                      title={dis ? f.refusal || t("feat_not_editable") : undefined}
                      onClick={() => {
                        setFeatureId(String(f.feature_id));
                        setDiameter(String(f.diameter));
                      }}
                      className={`block w-full rounded border px-2 py-1 text-left text-xs ${
                        dis
                          ? "cursor-not-allowed border-edge text-faint"
                          : on
                            ? "border-brand-600 bg-brand-950/60 text-brand-200"
                            : "border-edge2 text-fg hover:border-faint"
                      }`}
                    >
                      #{f.feature_id} {f.kind} Ø{f.diameter} mm
                      {f.axis ? ` · ${f.axis}` : ""}
                      {f.depth != null ? ` · ${f.depth} mm deep` : ""}
                      {dis ? ` — ${f.refusal || t("feat_not_editable")}` : ""}
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
                  placeholder={t("feat_dia_ph")}
                  value={diameter}
                  onChange={(e) => setDiameter(e.target.value)}
                />
                <Btn
                  variant="success"
                  onClick={() => edit.mutate()}
                  busy={edit.isPending}
                  disabled={!selected || !(Number(diameter) > 0)}
                >
                  {t("feat_resize")}
                </Btn>
              </div>
              <div className="mt-1 text-[11px] text-faint">
                {t("feat_verified_hint")}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
