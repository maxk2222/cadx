import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, FileBox, ImageIcon, Pencil, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "../lib/api";
import type { Asset } from "../lib/types";
import ImageEditorModal from "./ImageEditorModal";
import { EmptyState, IconBtn, Lightbox, Spinner, inputCls } from "./ui";

const KIND_ICON: Record<string, React.ReactNode> = {
  image: <ImageIcon className="h-5 w-5" />,
};

export default function AssetManager({
  projectId,
  assets,
  selected,
  onToggle,
  onAssetReplaced,
}: {
  projectId: string;
  assets: Asset[];
  selected?: Set<string>;
  onToggle?: (id: string) => void;
  onAssetReplaced?: (srcId: string, newId: string) => void;
}) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [editing, setEditing] = useState<Asset | null>(null);
  const [editingCaption, setEditingCaption] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState("");

  const refresh = () => qc.invalidateQueries({ queryKey: ["assets", projectId] });

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    const MAX_MB = 25;
    const tooBig = Array.from(files).filter((f) => f.size > MAX_MB * 1024 * 1024);
    if (tooBig.length) {
      setErr(`${tooBig.map((f) => f.name).join(", ")} exceeds the ${MAX_MB} MB limit`);
      return;
    }
    setUploading(true);
    setErr("");
    try {
      await api.uploadAssets(projectId, Array.from(files));
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setUploading(false);
    }
  };

  const del = useMutation({
    mutationFn: (id: string) => api.deleteAsset(id),
    onSuccess: refresh,
  });
  const saveCaption = useMutation({
    mutationFn: ({ id, caption }: { id: string; caption: string }) =>
      api.updateAsset(id, caption),
    onSuccess: () => {
      setEditingCaption(null);
      refresh();
    },
  });

  return (
    <div>
      <div
        className={`rounded-lg border-2 border-dashed p-4 text-center text-sm transition-colors ${
          drag ? "border-brand-500 bg-brand-950/20 text-brand-300" : "border-zinc-700 text-zinc-400"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          upload(e.dataTransfer.files);
        }}
      >
        {uploading ? (
          <span className="inline-flex items-center gap-2">
            <Spinner /> uploading…
          </span>
        ) : (
          <>
            <Upload className="mr-1.5 inline h-4 w-4 align-[-2px]" />
            Drop reference images / STEP / STL / DXF here, or{" "}
            <button
              className="text-brand-400 underline"
              onClick={() => fileRef.current?.click()}
            >
              browse
            </button>
          </>
        )}
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept="image/*,.step,.stp,.stl,.dxf"
          onChange={(e) => upload(e.target.files)}
        />
      </div>
      {err && <div className="mt-2 text-sm text-red-400">{err}</div>}

      {assets.length === 0 ? (
        <div className="mt-3">
          <EmptyState title="No reference assets" hint="Images help the agent understand shape and features." />
        </div>
      ) : (
        <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4">
          {assets.map((a) => {
            const sel = selected?.has(a.id);
            return (
              <div
                key={a.id}
                className={`group relative overflow-hidden rounded-md border-2 transition-colors ${
                  sel ? "border-brand-500" : "border-zinc-800 hover:border-zinc-600"
                }`}
              >
                <button
                  className="block w-full"
                  onClick={() =>
                    onToggle ? onToggle(a.id) : a.kind === "image" && setLightbox(a.url)
                  }
                  onDoubleClick={() => a.kind === "image" && setLightbox(a.url)}
                  title={`${a.filename}${a.caption ? ` — ${a.caption}` : ""}`}
                >
                  {a.kind === "image" ? (
                    <img src={a.url} className="h-20 w-full object-cover" alt={a.filename} />
                  ) : (
                    <div className="flex h-20 flex-col items-center justify-center gap-1 text-zinc-500">
                      {KIND_ICON[a.kind] ?? <FileBox className="h-5 w-5" />}
                      <span className="max-w-full truncate px-1 text-[10px]">{a.filename}</span>
                    </div>
                  )}
                </button>
                <span className="absolute left-1 top-1 rounded bg-black/70 px-1 text-[9px] uppercase">
                  {a.kind.replace("reference_", "")}
                </span>
                {a.derived_from && (
                  <span
                    className="absolute left-1 top-5 rounded bg-purple-800/80 px-1 text-[9px] text-purple-200"
                    title="edited copy — original kept"
                  >
                    edited
                  </span>
                )}
                {onToggle && sel && (
                  <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-brand-500">
                    <Check className="h-3 w-3 text-white" />
                  </span>
                )}
                <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/85 to-transparent px-1 pb-0.5 pt-4 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    className="min-w-0 flex-1 truncate text-left text-[10px] text-zinc-300 hover:text-white"
                    title={a.caption ? `caption: ${a.caption} — click to edit` : "add caption"}
                    onClick={() => {
                      setEditingCaption(a.id);
                      setCaption(a.caption);
                    }}
                  >
                    {a.caption || a.filename}
                  </button>
                  {a.kind === "image" && (
                    <>
                      <IconBtn
                        title="edit image (crop / mask / remove bg)"
                        className="h-5 w-5"
                        onClick={() => setEditing(a)}
                      >
                        <Pencil className="h-3 w-3" />
                      </IconBtn>
                      <IconBtn title="view" className="h-5 w-5" onClick={() => setLightbox(a.url)}>
                        <ImageIcon className="h-3 w-3" />
                      </IconBtn>
                    </>
                  )}
                  <IconBtn
                    title="delete asset"
                    danger
                    className="h-5 w-5"
                    onClick={() => del.mutate(a.id)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </IconBtn>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {editingCaption && (
        <div className="mt-2 flex gap-2">
          <input
            className={`${inputCls} flex-1 py-1 text-xs`}
            autoFocus
            placeholder="caption (what does this image show?)"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter")
                saveCaption.mutate({ id: editingCaption, caption: caption.trim() });
              if (e.key === "Escape") setEditingCaption(null);
            }}
          />
          <button
            className="rounded bg-zinc-700 px-2 text-xs hover:bg-zinc-600"
            onClick={() =>
              saveCaption.mutate({ id: editingCaption, caption: caption.trim() })
            }
          >
            save
          </button>
        </div>
      )}
      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
      {editing && (
        <ImageEditorModal
          asset={editing}
          onClose={() => setEditing(null)}
          onSaved={(a) => {
            setEditing(null);
            refresh();
            onAssetReplaced?.(editing.id, a.id);
          }}
        />
      )}
    </div>
  );
}
