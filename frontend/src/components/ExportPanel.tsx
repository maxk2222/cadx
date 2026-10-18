import { Download, RefreshCw } from "lucide-react";
import { useState } from "react";
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
            title="re-generate all export formats (STEP/STL/3MF/GLB/DXF/SVG)"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Re-export
          </Btn>
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
      {anyTessellated && (
        <div className="mt-1.5 text-[10px] text-zinc-500">
          <span className="text-amber-400">*</span> tessellated export (derived
          from a mesh, not an analytic B-Rep)
        </div>
      )}
      {error && <div className="mt-1.5 text-xs text-red-400">{error}</div>}
    </div>
  );
}
