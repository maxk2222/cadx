import { Download, RefreshCw } from "lucide-react";
import { api } from "../lib/api";
import { fmtBytes } from "../lib/format";
import type { Artifact } from "../lib/types";
import { IconBtn, SectionTitle } from "./ui";

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
  const exports = ORDER.map((k) => artifacts.find((a) => a.kind === k)).filter(
    Boolean
  ) as Artifact[];
  return (
    <div>
      <SectionTitle
        right={
          <IconBtn title="re-export all formats" onClick={() => api.reexport(versionId).then(onRefresh)}>
            <RefreshCw className="h-3.5 w-3.5" />
          </IconBtn>
        }
      >
        Exports
      </SectionTitle>
      <div className="flex flex-wrap gap-2">
        {exports.map((a) => (
          <a
            key={a.id}
            href={`${a.url}?download=1`}
            title={
              a.meta?.tessellated
                ? "STEP export is tessellated (derived from a mesh)"
                : `download .${a.kind}`
            }
            className="flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm font-medium uppercase transition-colors hover:border-zinc-500 hover:bg-zinc-700"
          >
            <Download className="h-3.5 w-3.5 text-zinc-500" />
            {a.kind}
            {a.meta?.tessellated ? <span className="text-amber-400">*</span> : null}{" "}
            <span className="text-[10px] text-zinc-500">{fmtBytes(a.size)}</span>
          </a>
        ))}
        {!exports.length && <span className="text-sm text-zinc-500">no exports yet</span>}
      </div>
    </div>
  );
}
