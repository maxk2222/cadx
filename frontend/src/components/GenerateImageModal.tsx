import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, errMsg } from "../lib/api";
import type { Asset } from "../lib/types";
import { Btn, Field, Modal, inputCls, useToast } from "./ui";

const SIZES = [512, 768, 1024] as const;
const isImage = (k: Asset["kind"]) => k === "image" || k === "generated_image";

export default function GenerateImageModal({
  projectId,
  referenceAsset,
  onClose,
}: {
  projectId: string;
  /** pre-selected init image (Asset "Variation" action) */
  referenceAsset?: Asset;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [model, setModel] = useState("auto");
  const [count, setCount] = useState(1);
  const [size, setSize] = useState(768);
  const [seed, setSeed] = useState("");
  const [refId, setRefId] = useState<string>(referenceAsset?.id ?? "");
  const [strength, setStrength] = useState(0.6);
  const [markSketch, setMarkSketch] = useState(false);

  const models = useQuery({ queryKey: ["imageModels"], queryFn: api.imageModels, retry: false });
  const assets = useQuery({
    queryKey: ["assets", projectId],
    queryFn: () => api.listAssets(projectId),
  });

  const edit = refId !== "";
  const refAsset = (assets.data ?? []).find((a) => a.id === refId);

  const gen = useMutation({
    mutationFn: () =>
      api.generateImages(projectId, {
        prompt: prompt.trim(),
        negative: negative.trim(),
        model,
        count,
        width: size,
        height: size,
        // digits-only input — "-" would NaN through a permissive filter
        ...(seed.trim() ? { seed: Number(seed.trim()) } : {}),
        ...(edit ? { reference_asset_id: refId, strength } : {}),
        mark_sketch: markSketch,
      }),
    onSuccess: (newAssets) => {
      qc.invalidateQueries({ queryKey: ["assets", projectId] });
      toast(
        `${newAssets.length} image${newAssets.length > 1 ? "s" : ""} ${edit ? "edited" : "generated"}`,
        "success",
      );
      onClose();
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const list = models.data ?? [];
  const imgAssets = (assets.data ?? []).filter((a) => isImage(a.kind));
  // edit mode: only adapters that accept an init image make sense; without a
  // reference the edit-only adapters (requires_image) can't run at all
  const pickable = edit
    ? list.filter((m) => m.supports_edit)
    : list.filter((m) => !m.requires_image);
  const noEditAdapter = edit && pickable.every((m) => !m.available);

  // a refId change reshapes the adapter list — never leave a stale choice
  const prevEdit = useRef(edit);
  useEffect(() => {
    if ((prevEdit.current && !edit) || (model !== "auto" && !pickable.some((m) => m.id === model)))
      setModel("auto");
    prevEdit.current = edit;
  }, [edit, model, pickable]);

  return (
    <Modal
      open
      onClose={onClose}
      title={edit ? "Edit / vary reference image" : "Generate reference image"}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (prompt.trim()) gen.mutate();
        }}
      >
        <Field
          label="Reference image"
          hint="optional — pick an asset to vary or edit it (img2img / instruction edit)"
        >
          <div className="flex items-center gap-2">
            <select
              className={`${inputCls} flex-1`}
              value={refId}
              onChange={(e) => setRefId(e.target.value)}
            >
              <option value="">none — text-to-image</option>
              {imgAssets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.caption || a.filename}
                </option>
              ))}
            </select>
            {refAsset && (
              <img
                src={refAsset.url}
                alt={refAsset.filename}
                className="h-10 w-10 rounded border border-zinc-700 object-cover"
              />
            )}
          </div>
        </Field>
        <Field
          label={edit ? "Edit instruction" : "Prompt"}
          hint={
            edit
              ? 'what to change — e.g. "make it matte black", "add cooling fins"'
              : "what should the reference show — subject, view, style"
          }
        >
          <textarea
            className={`${inputCls} h-20 resize-none`}
            value={prompt}
            autoFocus
            placeholder={
              edit
                ? "e.g. turn this into a clean black-and-white sketch"
                : "e.g. a wall-mounted headphone holder, three-quarter view, studio light"
            }
            onChange={(e) => setPrompt(e.target.value)}
          />
        </Field>
        <Field label="Negative prompt" hint="optional">
          <input
            className={inputCls}
            value={negative}
            placeholder="text, watermark, hands"
            onChange={(e) => setNegative(e.target.value)}
          />
        </Field>
        <div className="grid grid-cols-3 gap-3">
          <Field label="Model">
            <select className={inputCls} value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="auto">auto{edit ? " (edit)" : ""}</option>
              {pickable.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name || m.id}
                  {m.requires_image ? " (edit-only)" : ""}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Size">
            <select
              className={inputCls}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
            >
              {SIZES.map((s) => (
                <option key={s} value={s}>
                  {s}px
                </option>
              ))}
            </select>
          </Field>
          <Field label="Count">
            <select
              className={inputCls}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            >
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </Field>
        </div>
        {edit && (
          <Field
            label={`Strength — ${Math.round(strength * 100)}%`}
            hint="how much to change the reference (low = stay close, high = strong change)"
          >
            <input
              type="range"
              className="w-full accent-brand-500"
              min={0.05}
              max={0.95}
              step={0.05}
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
            />
          </Field>
        )}
        <label className="mb-3 flex items-center gap-2 text-xs text-zinc-300">
          <input
            type="checkbox"
            className="accent-brand-500"
            checked={markSketch}
            onChange={(e) => setMarkSketch(e.target.checked)}
          />
          Mark result as sketch — feeds it to the sketch-conditioned mesh adapter
          (TripoSG-scribble)
        </label>
        <Field label="Seed" hint="optional — empty = random">
          <input
            className={inputCls}
            value={seed}
            placeholder="e.g. 42"
            onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
          />
        </Field>
        {models.isSuccess && list.length === 0 && (
          <div className="mb-2 text-xs text-amber-400">
            no image adapter reported by the mesh server — install one (see docs/image-generation)
          </div>
        )}
        {noEditAdapter && (
          <div className="mb-2 text-xs text-amber-400">
            no edit-capable image adapter available — qwen-image-edit is the default choice on
            Apple Silicon (see docs/image-generation)
          </div>
        )}
        <div className="flex justify-end gap-2">
          <Btn variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Btn>
          <Btn type="submit" busy={gen.isPending} disabled={!prompt.trim() || noEditAdapter}>
            {edit ? "Generate edit" : "Generate"}
          </Btn>
        </div>
        {gen.isPending && (
          <div className="mt-2 text-xs text-zinc-500">
            local generation can take a moment — the model loads on first use
          </div>
        )}
      </form>
    </Modal>
  );
}
