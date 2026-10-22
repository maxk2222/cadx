import { useTranslation } from "react-i18next";
import type { DfmWarning } from "../lib/types";

const STYLE: Record<string, string> = {
  high: "border-red-700 bg-red-950/50 text-red-200",
  medium: "border-amber-700 bg-amber-950/50 text-amber-200",
  low: "border-zinc-600 bg-zinc-900 text-zinc-300",
};

export default function DfmPanel({ warnings }: { warnings: DfmWarning[] }) {
  const { t } = useTranslation("version");
  if (!warnings.length) {
    return <div className="text-sm text-emerald-400">{t("dfm_none")}</div>;
  }
  return (
    <div className="flex flex-col gap-1.5">
      {warnings.map((w, i) => (
        <div key={i} className={`rounded border px-2 py-1.5 text-xs ${STYLE[w.severity] ?? STYLE.low}`}>
          <span className="font-semibold uppercase">{t(`dfm_sev_${w.severity}`)}</span>
          <span className="ml-2 font-mono">{w.code}</span>
          <div>{w.message}</div>
        </div>
      ))}
      <div className="text-[10px] text-zinc-500">{t("dfm_footer")}</div>
    </div>
  );
}
