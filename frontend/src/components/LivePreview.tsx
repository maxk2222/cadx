import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Spinner } from "./ui";

const HINT_AFTER_MS = 9000;

/** Provisional render overlay while the job runs — polls the live-renders
 *  manifest (available views + newest iteration) and shows the selected view
 *  with an iteration label. `?t=<mtime>` cache-busts when a render is
 *  rewritten; the browser keeps the last frame while the next image loads. */
export default function LivePreview({
  versionId,
  active = true,
}: {
  versionId: string;
  /** stop polling once the job reached a terminal state */
  active?: boolean;
}) {
  const { data } = useQuery({
    queryKey: ["live-renders", versionId],
    queryFn: () => api.liveRenders(versionId),
    refetchInterval: active ? 2500 : false,
  });
  const [view, setView] = useState<string | null>(null);
  const [ok, setOk] = useState(false);
  const [waited, setWaited] = useState(false);
  useEffect(() => {
    setOk(false);
    setWaited(false);
  }, [versionId]);
  useEffect(() => {
    if (ok) return;
    const t = setTimeout(() => setWaited(true), HINT_AFTER_MS);
    return () => clearTimeout(t);
  }, [ok]);
  const views = data?.views ?? [];
  const cur =
    view && views.includes(view) ? view : views.includes("iso") ? "iso" : views[0];
  const src = `${api.previewUrl(versionId, cur)}${cur ? "&" : "?"}t=${
    data ? Math.floor(data.mtime) : 0
  }`;
  return (
    <>
      <img
        src={src}
        alt="live render preview"
        className={`absolute inset-0 h-full w-full bg-zinc-950 object-contain transition-opacity duration-300 ${
          ok ? "opacity-100" : "opacity-0"
        }`}
        onLoad={() => setOk(true)}
        onError={() => setOk(false)}
      />
      {!ok && waited && (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-zinc-500">
          {active ? "waiting for first render…" : "no preview produced"}
        </div>
      )}
      {ok && (
        <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-brand-300">
          <Spinner className="h-3 w-3" /> live preview
        </span>
      )}
      {ok && data?.iteration != null && (
        <span className="absolute right-2 top-2 rounded bg-black/70 px-2 py-1 font-mono text-[10px] font-medium text-zinc-300">
          {data.final ? "final" : `it ${data.iteration}`}
        </span>
      )}
      {ok && views.length > 1 && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1">
          {views.map((w) => (
            <button
              key={w}
              onClick={() => setView(w)}
              className={`rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                w === cur ? "text-brand-300" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {w.replace("contact_sheet", "sheet").replace(/^fem:/, "fem ")}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
