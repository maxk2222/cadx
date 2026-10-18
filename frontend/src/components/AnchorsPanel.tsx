import { useQuery } from "@tanstack/react-query";
import { Crosshair } from "lucide-react";
import { api } from "../lib/api";
import { Card, SectionTitle } from "./ui";

/** Resolved feature anchors — the table `hole_index` in FEM regions/refine
 *  refers to. Shows axis, position, diameter, depth and detection source. */
export default function AnchorsPanel({ versionId }: { versionId: string }) {
  const q = useQuery({
    queryKey: ["anchors", versionId],
    queryFn: () => api.anchors(versionId),
    staleTime: 60_000,
  });
  const anchors = q.data?.anchors ?? [];
  if (!anchors.length) return null;

  return (
    <Card className="p-4">
      <SectionTitle>
        <span className="inline-flex items-center gap-1">
          <Crosshair className="h-3 w-3" /> Feature anchors
        </span>
      </SectionTitle>
      <div className="mb-2 text-[11px] text-zinc-500">
        measured/declared holes, bores and seats — keep regions and refine
        re-cuts reference these indices
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wider text-zinc-500">
            <th className="pb-1 font-medium">#</th>
            <th className="pb-1 font-medium">name</th>
            <th className="pb-1 font-medium">kind</th>
            <th className="pb-1 font-medium">axis</th>
            <th className="pb-1 font-medium">position</th>
            <th className="pb-1 font-medium">⌀ mm</th>
            <th className="pb-1 font-medium">depth</th>
            <th className="pb-1 font-medium">source</th>
          </tr>
        </thead>
        <tbody>
          {anchors.map((a) => {
            const pos =
              a.pos?.filter((p): p is number => p != null).map((p) => p.toFixed(1)) ??
              [a.x, a.y, a.z].filter((p): p is number => p != null).map((p) => p.toFixed(1));
            return (
              <tr key={a.index} className="border-t border-zinc-800/60 text-zinc-300">
                <td className="py-1 pr-2 font-mono text-zinc-500">{a.index}</td>
                <td className="py-1 pr-2">{a.name ?? "—"}</td>
                <td className="py-1 pr-2">{a.kind}</td>
                <td className="py-1 pr-2 font-mono">{a.axis}</td>
                <td className="py-1 pr-2 font-mono text-zinc-400">
                  ({pos.join(", ")})
                </td>
                <td className="py-1 pr-2 font-mono">{a.dia?.toFixed(1) ?? "—"}</td>
                <td className="py-1 pr-2 font-mono">
                  {a.depth_mm != null ? `${a.depth_mm.toFixed(0)} mm` : "through"}
                </td>
                <td className="py-1 text-zinc-500">{a.source ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
        </table>
      </div>
    </Card>
  );
}
