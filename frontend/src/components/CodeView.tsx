import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Download } from "lucide-react";
import { api } from "../lib/api";
import { CopyBtn, IconBtn, SectionTitle, Spinner } from "./ui";

export default function CodeView({ versionId }: { versionId: string }) {
  const { t } = useTranslation("version");
  const code = useQuery({
    queryKey: ["code", versionId],
    queryFn: () => api.getCode(versionId),
    staleTime: Infinity,
  });

  const download = () => {
    if (!code.data) return;
    const blob = new Blob([code.data], { type: "text/x-python" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "model.py";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div>
      <SectionTitle
        right={
          code.data && (
            <span className="flex gap-1">
              <CopyBtn text={code.data} />
              <IconBtn title={t("code_download")} onClick={download}>
                <Download className="h-4 w-4" />
              </IconBtn>
            </span>
          )
        }
      >
        {t("code_title")}
      </SectionTitle>
      {code.isLoading ? (
        <div className="flex justify-center py-8 text-faint">
          <Spinner />
        </div>
      ) : (
        <pre className="max-h-[560px] overflow-auto rounded-lg border border-edge bg-bg p-3 font-mono text-xs leading-relaxed text-fg">
          {code.data || "(empty)"}
        </pre>
      )}
    </div>
  );
}
