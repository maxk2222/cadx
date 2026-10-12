import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../lib/api";
import { Btn, Field, Modal, inputCls, useToast } from "./ui";

const SIZES = [512, 768, 1024] as const;

export default function GenerateImageModal({
  projectId,
  onClose,
}: {
  projectId: string;
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

  const models = useQuery({ queryKey: ["imageModels"], queryFn: api.imageModels, retry: false });

  const gen = useMutation({
    mutationFn: () =>
      api.generateImages(projectId, {
        prompt: prompt.trim(),
        negative: negative.trim(),
        model,
        count,
        width: size,
        height: size,
        ...(seed.trim() ? { seed: Number(seed.trim()) } : {}),
      }),
    onSuccess: (assets) => {
      qc.invalidateQueries({ queryKey: ["assets", projectId] });
      toast(`${assets.length} image${assets.length > 1 ? "s" : ""} generated`, "success");
      onClose();
    },
    onError: (e) => toast(String(e), "error"),
  });

  const list = models.data ?? [];

  return (
    <Modal open onClose={onClose} title="Generate reference image">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (prompt.trim()) gen.mutate();
        }}
      >
        <Field label="Prompt" hint="what should the reference show — subject, view, style">
          <textarea
            className={`${inputCls} h-20 resize-none`}
            value={prompt}
            autoFocus
            placeholder="e.g. a wall-mounted headphone holder, three-quarter view, studio light"
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
              <option value="auto">auto</option>
              {list.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name || m.id}
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
        <Field label="Seed" hint="optional — empty = random">
          <input
            className={inputCls}
            value={seed}
            placeholder="e.g. 42"
            onChange={(e) => setSeed(e.target.value.replace(/[^0-9-]/g, ""))}
          />
        </Field>
        {models.isSuccess && list.length === 0 && (
          <div className="mb-2 text-xs text-amber-400">
            no image adapter reported by the mesh server — install one (see docs/image-generation)
          </div>
        )}
        <div className="flex justify-end gap-2">
          <Btn variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Btn>
          <Btn type="submit" busy={gen.isPending} disabled={!prompt.trim()}>
            Generate
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
