import { Download, Eye } from "lucide-react";
import { useState } from "react";
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
  const [lightbox, setLightbox] = useState<string | null>(null);
  if (!artifacts.length) return <EmptyState title="No files yet" />;

  const isImage = (a: Artifact) =>
    ["render", "contact_sheet", "thumbnail"].includes(a.kind) ||
    /\.(png|jpe?g|webp)$/i.test(a.path);

  const sorted = [...artifacts].sort((a, b) => a.kind.localeCompare(b.kind) || (a.view ?? "").localeCompare(b.view ?? ""));

  return (
    <div>
      <SectionTitle right={<span className="normal-case tracking-normal">{artifacts.length} files</span>}>
        Files
      </SectionTitle>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-zinc-800/60 text-xs text-zinc-400">
            <tr>
              <th className="px-3 py-2 font-medium">kind</th>
              <th className="px-3 py-2 font-medium">view</th>
              <th className="px-3 py-2 font-medium">file</th>
              <th className="px-3 py-2 font-medium text-right">size</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((a) => (
              <tr key={a.id} className="border-t border-zinc-800 text-zinc-300">
                <td className="px-3 py-1.5">
                  <Chip tone={KIND_TONE[a.kind] ?? "zinc"} className="uppercase">
                    {a.kind}
                    {!!a.meta?.tessellated && (
                      <span className="text-amber-400" title="tessellated (mesh-derived)">*</span>
                    )}
                  </Chip>
                </td>
                <td className="px-3 py-1.5 text-zinc-400">{a.view ?? "—"}</td>
                <td className="max-w-0 truncate px-3 py-1.5 font-mono text-xs text-zinc-500">
                  {a.path.split("/").pop()}
                </td>
                <td className="px-3 py-1.5 text-right text-xs text-zinc-400">
                  {fmtBytes(a.size)}
                </td>
                <td className="px-3 py-1.5">
                  <span className="flex justify-end gap-0.5">
                    {isImage(a) && (
                      <IconBtn title="preview" className="h-6 w-6" onClick={() => setLightbox(a.url)}>
                        <Eye className="h-3.5 w-3.5" />
                      </IconBtn>
                    )}
                    <a
                      href={`${a.url}?download=1`}
                      title="download"
                      aria-label={`download ${a.path.split("/").pop() ?? a.kind}`}
                      className="inline-flex h-6 w-6 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
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
