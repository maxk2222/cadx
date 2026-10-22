import { Download, Eye } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { fmtBytes } from "../lib/format";
import type { Artifact } from "../lib/types";
import { Chip, EmptyState, IconBtn, Lightbox, SectionTitle } from "./ui";

const KIND_TONE: Record<string, "zinc" | "blue" | "green" | "purple" | "amber"> = {
  step: "green",
  stl: "blue",
  "3mf": "blue",
  glb: "blue",
  dxf: "green",
  svg: "green",
  render: "zinc",
  contact_sheet: "zinc",
  thumbnail: "zinc",
  mesh_raw: "purple",
};

export default function ArtifactsTable({ artifacts }: { artifacts: Artifact[] }) {
  const { t } = useTranslation("version");
  const [lightbox, setLightbox] = useState<string | null>(null);
  if (!artifacts.length) return <EmptyState title={t("art_empty")} />;

  const isImage = (a: Artifact) =>
    ["render", "contact_sheet", "thumbnail"].includes(a.kind) ||
    /\.(png|jpe?g|webp)$/i.test(a.path);

  const sorted = [...artifacts].sort((a, b) => a.kind.localeCompare(b.kind) || (a.view ?? "").localeCompare(b.view ?? ""));

  return (
    <div>
      <SectionTitle right={<span className="normal-case tracking-normal">{t("art_files_n", { count: artifacts.length })}</span>}>
        {t("art_title")}
      </SectionTitle>
      <div className="overflow-x-auto rounded-lg border border-edge">
        <table className="w-full text-left text-sm">
          <thead className="bg-panel2/60 text-xs text-muted">
            <tr>
              <th className="px-3 py-2 font-medium">{t("art_kind")}</th>
              <th className="px-3 py-2 font-medium">{t("art_view")}</th>
              <th className="px-3 py-2 font-medium">{t("art_file")}</th>
              <th className="px-3 py-2 font-medium text-right">{t("art_size")}</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((a) => (
              <tr key={a.id} className="border-t border-edge text-fg">
                <td className="px-3 py-1.5">
                  <Chip tone={KIND_TONE[a.kind] ?? "zinc"} className="uppercase">
                    {a.kind}
                    {!!a.meta?.tessellated && (
                      <span className="text-amber-400" title={t("art_tess")}>*</span>
                    )}
                  </Chip>
                </td>
                <td className="px-3 py-1.5 text-muted">{a.view ?? "—"}</td>
                <td className="max-w-0 truncate px-3 py-1.5 font-mono text-xs text-faint">
                  {a.path.split("/").pop()}
                </td>
                <td className="px-3 py-1.5 text-right text-xs text-muted">
                  {fmtBytes(a.size)}
                </td>
                <td className="px-3 py-1.5">
                  <span className="flex justify-end gap-0.5">
                    {isImage(a) && (
                      <IconBtn title={t("art_preview")} className="h-6 w-6" onClick={() => setLightbox(a.url)}>
                        <Eye className="h-3.5 w-3.5" />
                      </IconBtn>
                    )}
                    <a
                      href={`${a.url}?download=1`}
                      title={t("common:action.download")}
                      aria-label={t("art_download_aria", { name: a.path.split("/").pop() ?? a.kind })}
                      className="inline-flex h-6 w-6 items-center justify-center rounded-md text-muted hover:bg-panel2 hover:text-fg"
                    >
                      <Download className="h-3.5 w-3.5" />
                    </a>
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
    </div>
  );
}
