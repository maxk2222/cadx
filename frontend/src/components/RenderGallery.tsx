import { useState } from "react";
import { useTranslation } from "react-i18next";
import { femViewCaption, sortViews, viewLabel } from "../lib/liveViews";
import type { Artifact } from "../lib/types";
import { Collapsible, Lightbox } from "./ui";

const IMG_RE = /\.(png|jpe?g|webp)$/i;
const isImg = (a: Artifact) => IMG_RE.test(a.path);

function Thumb({
  a,
  caption,
  onOpen,
}: {
  a: Artifact;
  caption: string;
  onOpen: (url: string) => void;
}) {
  return (
    <button
      onClick={() => onOpen(a.url)}
      title={a.path.split("/").pop()}
      className="group relative overflow-hidden rounded-md border border-zinc-800 transition-colors hover:border-zinc-600"
    >
      <img src={a.url} alt={caption} loading="lazy" className="w-full" />
      <span className="absolute bottom-1 left-1 max-w-[calc(100%-0.5rem)] truncate rounded bg-black/60 px-1 text-[10px]">
        {caption}
      </span>
    </button>
  );
}

/** fem artifact view stems look like "vonmises_export_<case>" / "regions_<case>"
 *  — group them per load case so multi-case runs stay readable */
function femCaseKey(view: string | null): { caseName: string; rank: number } {
  const m = view ? /^(vonmises|deformed|regions)(?:_export)?_(.*)$/.exec(view) : null;
  if (!m) return { caseName: view ?? "", rank: 9 };
  return {
    caseName: m[2],
    rank: m[1] === "regions" ? 0 : m[1] === "vonmises" ? 1 : 2,
  };
}

export default function RenderGallery({ artifacts }: { artifacts: Artifact[] }) {
  const { t } = useTranslation("version");
  const renderOrder = sortViews(
    artifacts.filter((a) => a.kind === "render").map((a) => a.view ?? "")
  );
  const renders = artifacts
    .filter((a) => a.kind === "render" && isImg(a))
    .sort(
      (a, b) =>
        renderOrder.indexOf(a.view ?? "") - renderOrder.indexOf(b.view ?? "")
    );
  const sheets = artifacts.filter(
    (a) => (a.kind === "contact_sheet" || a.kind === "thumbnail") && isImg(a)
  );
  const femImgs = artifacts.filter((a) => a.kind === "fem" && isImg(a));
  // any other stray image artifact not covered above (future kinds)
  const otherImgs = artifacts.filter(
    (a) =>
      isImg(a) &&
      a.kind !== "render" &&
      a.kind !== "contact_sheet" &&
      a.kind !== "thumbnail" &&
      a.kind !== "fem"
  );
  const [lightbox, setLightbox] = useState<string | null>(null);

  const femByCase = (() => {
    const m = new Map<string, { rank: number; items: Artifact[] }>();
    for (const a of femImgs) {
      const { caseName, rank } = femCaseKey(a.view);
      const g = m.get(caseName) ?? { rank, items: [] };
      g.rank = Math.min(g.rank, rank);
      g.items.push(a);
      m.set(caseName, g);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  })();

  if (!renders.length && !sheets.length && !femImgs.length && !otherImgs.length)
    return null;

  return (
    <div className="flex flex-col gap-4">
      {!!renders.length && (
        <Collapsible title={t("gal_renders")} count={renders.length}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {renders.map((r) => (
              <Thumb
                key={r.id}
                a={r}
                caption={viewLabel(r.view ?? "")}
                onOpen={setLightbox}
              />
            ))}
          </div>
        </Collapsible>
      )}

      {!!sheets.length && (
        <Collapsible title={t("gal_sheets")} count={sheets.length} defaultOpen={!renders.length}>
          <div className="grid grid-cols-1 gap-2">
            {sheets.map((s) => (
              <Thumb
                key={s.id}
                a={s}
                caption={viewLabel(s.view ?? s.kind)}
                onOpen={setLightbox}
              />
            ))}
          </div>
        </Collapsible>
      )}

      {!!femImgs.length && (
        <Collapsible
          title={t("gal_fem")}
          count={femImgs.length}
          defaultOpen={false}
        >
          <div className="flex flex-col gap-3">
            {femByCase.map(([caseName, g]) => (
              <div key={caseName || "fem"}>
                {femByCase.length > 1 && (
                  <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                    {caseName || t("gal_case")}
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {g.items
                    .sort(
                      (a, b) =>
                        femCaseKey(a.view).rank - femCaseKey(b.view).rank
                    )
                    .map((a) => (
                      <Thumb
                        key={a.id}
                        a={a}
                        caption={femViewCaption(a.view) ?? a.view ?? t("gal_fem_kind")}
                        onOpen={setLightbox}
                      />
                    ))}
                </div>
              </div>
            ))}
          </div>
        </Collapsible>
      )}

      {!!otherImgs.length && (
        <Collapsible title={t("gal_other")} count={otherImgs.length} defaultOpen={false}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {otherImgs.map((a) => (
              <Thumb
                key={a.id}
                a={a}
                caption={a.view ?? a.path.split("/").pop() ?? a.kind}
                onOpen={setLightbox}
              />
            ))}
          </div>
        </Collapsible>
      )}

      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
    </div>
  );
}
