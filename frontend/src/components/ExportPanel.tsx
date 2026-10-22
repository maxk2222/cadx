import { Download, RefreshCw } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../lib/api";
import { fmtBytes } from "../lib/format";
import type { Artifact } from "../lib/types";
import { Btn, SectionTitle } from "./ui";

const ORDER = ["step", "stl", "3mf", "glb", "dxf", "svg"];

export default function ExportPanel({
  versionId,
  artifacts,
  onRefresh,
}: {
  versionId: string;
  artifacts: Artifact[];
  onRefresh: () => void;
}) {
  const { t } = useTranslation("version");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const exports = ORDER.map((k) => artifacts.find((a) => a.kind === k)).filter(
    Boolean
  ) as Artifact[];
  const anyTessellated = exports.some((a) => a.meta?.tessellated);

  const reexport = async () => {
    setBusy(true);
    setError("");
    try {
      await api.reexport(versionId);
      onRefresh();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <SectionTitle
        right={
          <Btn
            variant="ghost"
            size="sm"
            busy={busy}
            onClick={reexport}
            title={t("exp_reexport_hint")}
          >
            <RefreshCw className="h-3.5 w-3.5" /> {t("exp_reexport")}
          </Btn>
        }
      >
        {t("exp_title")}
      </SectionTitle>
      <div className="flex flex-wrap gap-2">
        {exports.map((a) => (
          <a
            key={a.id}
            href={`${a.url}?download=1`}
            title={
              a.meta?.tessellated
                ? t("exp_tess_hint")
                : t("exp_download_hint", { kind: a.kind })
            }
            className="flex items-center gap-1.5 rounded-md border border-edge2 bg-panel2 px-3 py-1.5 text-sm font-medium uppercase transition-colors hover:border-edge2 hover:bg-edge2"
          >
            <Download className="h-3.5 w-3.5 text-faint" />
            {a.kind}
            {a.meta?.tessellated ? <span className="text-amber-400">*</span> : null}{" "}
            <span className="text-[10px] text-faint">{fmtBytes(a.size)}</span>
          </a>
        ))}
        {!exports.length && <span className="text-sm text-faint">{t("exp_empty")}</span>}
      </div>
      {anyTessellated && (
        <div className="mt-1.5 text-[10px] text-faint">
          <span className="text-amber-400">*</span> {t("exp_tess_note")}
        </div>
      )}
      {error && <div className="mt-1.5 text-xs text-red-400">{error}</div>}
    </div>
  );
}
