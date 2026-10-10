import { api } from "../lib/api";
import type { Artifact } from "../lib/types";

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
  const exports = ORDER.map((k) => artifacts.find((a) => a.kind === k)).filter(Boolean) as Artifact[];
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-300">Exports</h3>
        <button
          className="rounded bg-zinc-700 px-2 py-0.5 text-xs hover:bg-zinc-600"
          onClick={() => api.reexport(versionId).then(onRefresh)}
        >
          re-export
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {exports.map((a) => (
          <a
            key={a.id}
            href={`${a.url}?download=1`}
            title={a.meta?.tessellated ? "STEP export is tessellated (derived from a mesh)" : undefined}
            className="rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm font-medium uppercase hover:bg-zinc-700"
          >
            {a.kind}
            {a.meta?.tessellated ? " (tessellated)" : ""}{" "}
            <span className="text-[10px] text-zinc-500">{(a.size / 1024).toFixed(0)}k</span>
          </a>
        ))}
        {!exports.length && <span className="text-sm text-zinc-500">no exports yet</span>}
      </div>
    </div>
  );
}
