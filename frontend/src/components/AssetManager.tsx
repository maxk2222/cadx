import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  FileBox,
  ImageIcon,
  Pencil,
  PenLine,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
} from "lucide-react";
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../lib/api";
import type { Asset } from "../lib/types";
import GenerateImageModal from "./GenerateImageModal";
import ImageEditorModal from "./ImageEditorModal";
import { Btn, EmptyState, IconBtn, Lightbox, Spinner, inputCls } from "./ui";

const KIND_ICON: Record<string, React.ReactNode> = {
  image: <ImageIcon className="h-5 w-5" />,
  generated_image: <Sparkles className="h-5 w-5" />,
};

const isImage = (k: Asset["kind"]) => k === "image" || k === "generated_image";

export default function AssetManager({
  projectId,
  assets,
  selected,
  onToggle,
  onAssetReplaced,
  onDeleted,
}: {
  projectId: string;
  assets: Asset[];
  selected?: Set<string>;
  onToggle?: (id: string) => void;
  onAssetReplaced?: (srcId: string, newId: string) => void;
  /** fired after a successful delete so parents can drop the id from any
   *  selection state (otherwise a stale id is submitted on generate/edit) */
  onDeleted?: (id: string) => void;
}) {
  const { t } = useTranslation();
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [editing, setEditing] = useState<Asset | null>(null);
  const [editingCaption, setEditingCaption] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [varying, setVarying] = useState<Asset | null>(null);
  const [err, setErr] = useState("");

  const refresh = () => qc.invalidateQueries({ queryKey: ["assets", projectId] });

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    const MAX_MB = 25;
    const tooBig = Array.from(files).filter((f) => f.size > MAX_MB * 1024 * 1024);
    if (tooBig.length) {
      setErr(t("asset.too_big", { names: tooBig.map((f) => f.name).join(", "), max: MAX_MB }));
      return;
    }
    setUploading(true);
    setErr("");
    try {
      await api.uploadAssets(projectId, Array.from(files));
      refresh();
    } catch (e) {
      setErr(errMsg(e));
    } finally {
      setUploading(false);
    }
  };

  const del = useMutation({
    mutationFn: (id: string) => api.deleteAsset(id),
    onSuccess: (_d, id) => {
      onDeleted?.(id);
      refresh();
    },
    onError: (e) => setErr(errMsg(e)),
  });
  const saveCaption = useMutation({
    mutationFn: ({ id, caption }: { id: string; caption: string }) =>
      api.updateAsset(id, { caption }),
    onSuccess: () => {
      setEditingCaption(null);
      refresh();
    },
  });
  const toggleSketch = useMutation({
    mutationFn: ({ id, sketch }: { id: string; sketch: boolean }) =>
      api.updateAsset(id, { role_hint: sketch ? "sketch" : "" }),
    onSuccess: refresh,
  });

  return (
    <div>
      <div
        className={`rounded-lg border-2 border-dashed p-4 text-center text-sm transition-colors ${
          drag ? "border-brand-500 bg-brand-950/20 text-brand-300" : "border-edge2 text-muted"
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
            <Spinner /> {t("asset.uploading")}
          </span>
        ) : (
          <>
            <Upload className="mr-1.5 inline h-4 w-4 align-[-2px]" />
            {t("asset.drop_pre")}{" "}
            <button
              className="text-brand-400 underline"
              onClick={() => fileRef.current?.click()}
            >
              {t("asset.browse")}
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
      <div className="mt-2 flex justify-end">
        <Btn variant="ghost" onClick={() => setGenerating(true)}>
          <Sparkles className="mr-1 h-3.5 w-3.5" /> {t("asset.generate")}
        </Btn>
      </div>
      {err && <div className="mt-2 text-sm text-red-400">{err}</div>}

      {assets.length === 0 ? (
        <div className="mt-3">
          <EmptyState title={t("asset.empty")} hint={t("asset.empty_hint")} />
        </div>
      ) : (
        <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4">
          {assets.map((a) => {
            const sel = selected?.has(a.id);
            return (
              <div
                key={a.id}
                className={`group relative overflow-hidden rounded-md border-2 transition-colors ${
                  sel ? "border-brand-500" : "border-edge hover:border-edge2"
                }`}
              >
                <button
                  className="block w-full"
                  onClick={() =>
                    onToggle ? onToggle(a.id) : isImage(a.kind) && setLightbox(a.url)
                  }
                  onDoubleClick={() => isImage(a.kind) && setLightbox(a.url)}
                  title={`${a.filename}${a.caption ? ` — ${a.caption}` : ""}`}
                >
                  {isImage(a.kind) ? (
                    <img src={a.url} className="h-20 w-full object-cover" alt={a.filename} />
                  ) : (
                    <div className="flex h-20 flex-col items-center justify-center gap-1 text-faint">
                      {KIND_ICON[a.kind] ?? <FileBox className="h-5 w-5" />}
                      <span className="max-w-full truncate px-1 text-[10px]">{a.filename}</span>
                    </div>
                  )}
                </button>
                <span
                  className={`absolute left-1 top-1 rounded px-1 text-[9px] uppercase ${
                    a.kind === "generated_image"
                      ? "bg-brand-600/80 text-white"
                      : "bg-black/70"
                  }`}
                  title={
                    a.kind === "generated_image" && a.gen_params
                      ? t("asset.generated_by", { model: a.gen_params.model ?? t("asset.local_model"), prompt: a.gen_params.prompt ?? "" })
                      : undefined
                  }
                >
                  {a.kind === "generated_image" ? t("asset.ai") : a.kind.replace("reference_", "")}
                </span>
                {a.gen_params?.role_hint === "sketch" && (
                  <span
                    className="absolute left-1 top-9 rounded bg-emerald-800/80 px-1 text-[9px] text-emerald-200"
                    title={t("asset.sketch_hint")}
                  >
                    {t("asset.sketch")}
                  </span>
                )}
                {a.derived_from && (
                  <span
                    className="absolute left-1 top-5 rounded bg-purple-800/80 px-1 text-[9px] text-purple-200"
                    title={t("asset.edited_hint")}
                  >
                    {t("asset.edited")}
                  </span>
                )}
                {onToggle && sel && (
                  <span className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-brand-500">
                    <Check className="h-3 w-3 text-white" />
                  </span>
                )}
                <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/85 to-transparent px-1 pb-0.5 pt-4 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    className="min-w-0 flex-1 truncate text-left text-[10px] text-fg hover:text-white"
                    title={a.caption ? t("asset.caption_edit", { caption: a.caption }) : t("asset.caption_add")}
                    onClick={() => {
                      setEditingCaption(a.id);
                      setCaption(a.caption);
                    }}
                  >
                    {a.caption || a.filename}
                  </button>
                  {isImage(a.kind) && (
                    <>
                      <IconBtn
                        title={t("asset.vary_hint")}
                        className="h-5 w-5"
                        onClick={() => setVarying(a)}
                      >
                        <Wand2 className="h-3 w-3" />
                      </IconBtn>
                      <IconBtn
                        title={
                          a.gen_params?.role_hint === "sketch"
                            ? t("asset.unmark_sketch")
                            : t("asset.mark_sketch")
                        }
                        className={`h-5 w-5 ${
                          a.gen_params?.role_hint === "sketch" ? "text-emerald-400" : ""
                        }`}
                        onClick={() =>
                          toggleSketch.mutate({
                            id: a.id,
                            sketch: a.gen_params?.role_hint !== "sketch",
                          })
                        }
                      >
                        <PenLine className="h-3 w-3" />
                      </IconBtn>
                      <IconBtn
                        title={t("asset.edit_image")}
                        className="h-5 w-5"
                        onClick={() => setEditing(a)}
                      >
                        <Pencil className="h-3 w-3" />
                      </IconBtn>
                      <IconBtn title={t("common:action.open")} className="h-5 w-5" onClick={() => setLightbox(a.url)}>
                        <ImageIcon className="h-3 w-3" />
                      </IconBtn>
                    </>
                  )}
                  <IconBtn
                    title={t("asset.delete")}
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
            placeholder={t("asset.caption_ph")}
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter")
                saveCaption.mutate({ id: editingCaption, caption: caption.trim() });
              if (e.key === "Escape") setEditingCaption(null);
            }}
          />
          <button
            className="rounded bg-edge2 px-2 text-xs hover:bg-edge2"
            onClick={() =>
              saveCaption.mutate({ id: editingCaption, caption: caption.trim() })
            }
          >
            {t("common:action.save")}
          </button>
        </div>
      )}
      <Lightbox src={lightbox} onClose={() => setLightbox(null)} />
      {generating && (
        <GenerateImageModal projectId={projectId} onClose={() => setGenerating(false)} />
      )}
      {varying && (
        <GenerateImageModal
          projectId={projectId}
          referenceAsset={varying}
          onClose={() => setVarying(null)}
        />
      )}
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
