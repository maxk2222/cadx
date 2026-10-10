import { useState } from "react";
import type { Artifact } from "../lib/types";
import { Lightbox, SectionTitle } from "./ui";

export default function RenderGallery({ artifacts }: { artifacts: Artifact[] }) {
  const renders = artifacts.filter((a) => a.kind === "render");
  const sheet = artifacts.find((a) => a.kind === "contact_sheet");
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (!renders.length && !sheet) return null;

  return (
    <div>
      <SectionTitle>Renders</SectionTitle>
      <div className="grid grid-cols-4 gap-2">
        {renders.map((r) => (
          <button
            key={r.id}
            onClick={() => setLightbox(r.url)}
            className="group relative overflow-hidden rounded-md border border-zinc-800 transition-colors hover:border-zinc-600"
          >
            <img src={r.url} alt={r.view ?? ""} className="w-full" />
            <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px]">
              {r.view}
            </span>
          </button>
        ))}
      </div>
      {sheet ? (
        <button onClick={() => setLightbox(sheet.url)} className="mt-2 block w-full">
          <img
            src={sheet.url}
            alt="contact sheet"
            className="w-full rounded-md border border-zinc-800 transition-colors hover:border-zinc-600"
          />
        </button>
      ) : null}
      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
    </div>
  );
}
