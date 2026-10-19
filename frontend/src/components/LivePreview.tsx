import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Spinner } from "./ui";

const HINT_AFTER_MS = 9000;
const MAX_RETRIES = 3;

function viewLabel(w: string): string {
  return w
    .replace("contact_sheet", "sheet")
    .replace("orient_sheet", "orient")
    .replace(/^fem:/, "fem ");
}

/** Provisional render overlay while the job runs — polls the live-renders
 *  manifest (available views + newest iteration) and always shows the
 *  freshest frame. The next image loads off-screen and only swaps in once
 *  decoded, so a mid-write or slow render never blanks the current frame.
 *  `?t=<mtime>` cache-busts when a render is rewritten; a failed frame is
 *  retried a few times via `&r=<n>` in case the executor was mid-write. */
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
  const [shown, setShown] = useState<string | null>(null);
  const [retries, setRetries] = useState(0);
  const [waited, setWaited] = useState(false);

  useEffect(() => {
    setView(null);
    setShown(null);
    setRetries(0);
    setWaited(false);
  }, [versionId]);
  useEffect(() => {
    if (shown) return;
    const t = setTimeout(() => setWaited(true), HINT_AFTER_MS);
    return () => clearTimeout(t);
  }, [shown]);

  const views = data?.views ?? [];
  const cur =
    view && views.includes(view) ? view : views.includes("iso") ? "iso" : views[0];
  const target = cur
    ? `${api.previewUrl(versionId, cur)}&t=${data ? data.mtime : 0}`
    : null;
  // fresh retry budget whenever a genuinely new frame is targeted
  useEffect(() => setRetries(0), [target]);
  const src = target && retries ? `${target}&r=${retries}` : target;

  const onErr = () => {
    if (active && retries < MAX_RETRIES) {
      window.setTimeout(() => setRetries((r) => r + 1), 1500);
    }
  };

  return (
    <>
      {shown && (
        <img
          src={shown}
          alt="live render preview"
          className="absolute inset-0 h-full w-full bg-zinc-950 object-contain"
        />
      )}
      {src && src !== shown && (
        <img
          key={src}
          src={src}
          alt=""
          aria-hidden
          className="hidden"
          onLoad={() => setShown(src)}
          onError={onErr}
        />
      )}
      {!shown && waited && (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-zinc-500">
          {active ? "waiting for first render…" : "no preview produced"}
        </div>
      )}
      {shown && (
        <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-brand-300">
          <Spinner className="h-3 w-3" /> live preview
        </span>
      )}
      {shown && data && (data.final || data.iteration != null) && (
        <span className="absolute right-2 top-2 rounded bg-black/70 px-2 py-1 font-mono text-[10px] font-medium text-zinc-300">
          {data.final ? "final" : `it ${data.iteration}`}
        </span>
      )}
      {shown && views.length > 1 && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1">
          {views.map((w) => (
            <button
              key={w}
              onClick={() => setView(w)}
              className={`rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                w === cur ? "text-brand-300" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {viewLabel(w)}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
