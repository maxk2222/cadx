import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../../lib/api";
import { Btn, Card, SectionTitle, inputCls } from "../ui";

export default function SaveToLibrary({
  versionId,
  hasCode,
}: {
  versionId: string;
  hasCode: boolean;
}) {
  const { t } = useTranslation("version");
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [asTemplate, setAsTemplate] = useState(false);
  const [done, setDone] = useState(false);
  const save = useMutation({
    mutationFn: () => api.saveToLibrary(versionId, name.trim(), asTemplate),
    onSuccess: () => {
      setDone(true);
      setName("");
      qc.invalidateQueries({ queryKey: ["library"] });
    },
  });
  return (
    <Card className="p-4">
      <SectionTitle>{t("save_to_library")}</SectionTitle>
      <div className="flex flex-wrap items-center gap-2">
        <input
          className={inputCls + " w-48 text-xs"}
          placeholder={t("part_name_ph")}
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setDone(false);
          }}
        />
        {hasCode && (
          <label className="flex items-center gap-1.5 text-xs text-muted">
            <input
              type="checkbox"
              checked={asTemplate}
              onChange={(e) => setAsTemplate(e.target.checked)}
            />
            {t("as_template")}
          </label>
        )}
        <Btn
          variant="ghost"
          disabled={!name.trim()}
          busy={save.isPending}
          onClick={() => save.mutate()}
        >
          {asTemplate ? t("save_template") : t("save_step")}
        </Btn>
        {done && <span className="text-xs text-emerald-400">{t("saved")}</span>}
        {save.isError && (
          <span className="text-xs text-red-400">{errMsg(save.error)}</span>
        )}
      </div>
      <div className="mt-1.5 text-[11px] text-faint">{t("save_hint")}</div>
    </Card>
  );
}
