import { useState } from "react";
import type { Artifact } from "../lib/types";

export default function RenderGallery({ artifacts }: { artifacts: Artifact[] }) {
  const renders = artifacts.filter((a) => a.kind === "render");
  const sheet = artifacts.find((a) => a.kind === "contact_sheet");
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (!renders.length && !sheet) return null;

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-zinc-300">Renders</h3>
      <div className="grid grid-cols-4 gap-2">
        {renders.map((r) => (
          <button key={r.id} onClick={() => setLightbox(r.url)} className="group relative">
            <img src={r.url} alt={r.view ?? ""} className="w-full rounded border border-zinc-800" />
            <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px]">{r.view}</span>
          </button>
        ))}
      </div>
      {sheet ? (
        <button onClick={() => setLightbox(sheet.url)} className="mt-2 block w-full">
          <img src={sheet.url} alt="contact sheet" className="w-full rounded border border-zinc-800" />
        </button>
      ) : null}
      {lightbox ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-8"
          onClick={() => setLightbox(null)}
        >
          <img src={lightbox} className="max-h-full max-w-full rounded" alt="" />
        </div>
      ) : null}
    </div>
  );
}
