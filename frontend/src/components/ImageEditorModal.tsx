import { Eraser, Lasso, Paintbrush, Scissors, Undo2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../lib/api";
import type { Asset } from "../lib/types";
import { Btn, IconBtn, Spinner } from "./ui";

type Tool = "crop" | "brush" | "lasso";
type Rect = { x: number; y: number; w: number; h: number };

const ASPECTS: { label: string; value: number | null }[] = [
  { label: "free", value: null },
  { label: "1:1", value: 1 },
  { label: "4:3", value: 4 / 3 },
  { label: "3:2", value: 3 / 2 },
  { label: "16:9", value: 16 / 9 },
];

export default function ImageEditorModal({
  asset,
  onSaved,
  onClose,
}: {
  asset: Asset;
  onSaved: (a: Asset) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation("common");
  const viewRef = useRef<HTMLCanvasElement>(null);
  const workRef = useRef<HTMLCanvasElement | null>(null); // image-resolution bitmap
  const maskRef = useRef<HTMLCanvasElement | null>(null); // keep-region alpha mask
  const undoRef = useRef<ImageData[]>([]);
  const dragRef = useRef<{ mode: string; sx: number; sy: number; rect?: Rect } | null>(null);

  const [ready, setReady] = useState(false);
  const [tool, setTool] = useState<Tool>("crop");
  const [crop, setCrop] = useState<Rect | null>(null);
  const [aspect, setAspect] = useState<number | null>(null);
  const [brush, setBrush] = useState(40);
  const [erase, setErase] = useState(false);
  const [lassoPts, setLassoPts] = useState<[number, number][]>([]);
  const [hasMask, setHasMask] = useState(false);
  const [undoN, setUndoN] = useState(0);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  const work = () => workRef.current!;
  const wctx = () => work().getContext("2d")!;

  const toImg = useCallback((e: React.PointerEvent): [number, number] => {
    const c = viewRef.current!;
    const r = c.getBoundingClientRect();
    return [
      Math.round(((e.clientX - r.left) / r.width) * c.width),
      Math.round(((e.clientY - r.top) / r.height) * c.height),
    ];
  }, []);

  const draw = useCallback(() => {
    const view = viewRef.current;
    const w = workRef.current;
    if (!view || !w) return;
    const ctx = view.getContext("2d")!;
    ctx.clearRect(0, 0, view.width, view.height);
    ctx.drawImage(w, 0, 0);

    // dim everything outside the keep-mask
    const m = maskRef.current;
    if (m && hasMask) {
      const dim = document.createElement("canvas");
      dim.width = view.width;
      dim.height = view.height;
      const dc = dim.getContext("2d")!;
      dc.fillStyle = "rgba(0,0,0,0.55)";
      dc.fillRect(0, 0, dim.width, dim.height);
      dc.globalCompositeOperation = "destination-out";
      dc.drawImage(m, 0, 0);
      ctx.drawImage(dim, 0, 0);
    }

    if (crop) {
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath();
      ctx.rect(0, 0, view.width, view.height);
      ctx.rect(crop.x, crop.y, crop.w, crop.h);
      ctx.fill("evenodd");
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = Math.max(2, view.width / 500);
      ctx.strokeRect(crop.x, crop.y, crop.w, crop.h);
      const hs = Math.max(6, view.width / 120);
      ctx.fillStyle = "#3b82f6";
      for (const [hx, hy] of [
        [crop.x, crop.y],
        [crop.x + crop.w, crop.y],
        [crop.x, crop.y + crop.h],
        [crop.x + crop.w, crop.y + crop.h],
      ])
        ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
      ctx.restore();
    }

    if (lassoPts.length) {
      ctx.save();
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = Math.max(2, view.width / 500);
      ctx.beginPath();
      lassoPts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      ctx.fillStyle = "#3b82f6";
      for (const [x, y] of lassoPts) {
        ctx.beginPath();
        ctx.arc(x, y, Math.max(3, view.width / 300), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }, [crop, lassoPts, hasMask]);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const w = document.createElement("canvas");
      w.width = img.naturalWidth;
      w.height = img.naturalHeight;
      w.getContext("2d")!.drawImage(img, 0, 0);
      workRef.current = w;
      const view = viewRef.current!;
      view.width = w.width;
      view.height = w.height;
      maskRef.current = null;
      setHasMask(false);
      undoRef.current = [];
      setUndoN(0);
      setReady(true);
    };
    img.onerror = () => setErr(t("asset.ed.err_load"));
    img.src = `${asset.url}?t=${asset.created_at}`;
  }, [asset, t]);

  useEffect(draw, [draw, ready]);

  const pushUndo = () => {
    undoRef.current.push(wctx().getImageData(0, 0, work().width, work().height));
    if (undoRef.current.length > 20) undoRef.current.shift();
    setUndoN(undoRef.current.length);
  };

  const undo = () => {
    const d = undoRef.current.pop();
    setUndoN(undoRef.current.length);
    if (!d) return;
    const view = viewRef.current!;
    const w = document.createElement("canvas");
    w.width = d.width;
    w.height = d.height;
    w.getContext("2d")!.putImageData(d, 0, 0);
    workRef.current = w;
    view.width = d.width;
    view.height = d.height;
    maskRef.current = null;
    setHasMask(false);
    setCrop(null);
    setLassoPts([]);
    draw();
  };

  const replaceWork = (next: HTMLCanvasElement) => {
    workRef.current = next;
    const view = viewRef.current!;
    view.width = next.width;
    view.height = next.height;
    maskRef.current = null;
    setHasMask(false);
    setCrop(null);
    setLassoPts([]);
    draw();
  };

  const normRect = (r: Rect): Rect => ({
    x: Math.round(Math.max(0, Math.min(r.x, r.x + r.w))),
    y: Math.round(Math.max(0, Math.min(r.y, r.y + r.h))),
    w: Math.round(Math.abs(r.w)),
    h: Math.round(Math.abs(r.h)),
  });

  const applyCrop = () => {
    if (!crop || crop.w < 4 || crop.h < 4) return;
    pushUndo();
    const r = normRect(crop);
    const next = document.createElement("canvas");
    next.width = r.w;
    next.height = r.h;
    next.getContext("2d")!.drawImage(work(), r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
    replaceWork(next);
  };

  const mask = () => {
    if (!maskRef.current || maskRef.current.width !== work().width) {
      const m = document.createElement("canvas");
      m.width = work().width;
      m.height = work().height;
      maskRef.current = m;
    }
    return maskRef.current!;
  };

  const applyMask = () => {
    if (!hasMask || !maskRef.current) return;
    pushUndo();
    const w = work();
    const next = document.createElement("canvas");
    next.width = w.width;
    next.height = w.height;
    const nctx = next.getContext("2d")!;
    nctx.drawImage(w, 0, 0);
    nctx.globalCompositeOperation = "destination-in";
    nctx.drawImage(maskRef.current, 0, 0);
    // autocrop to content bounding box
    const d = nctx.getImageData(0, 0, next.width, next.height).data;
    let minX = next.width, minY = next.height, maxX = -1, maxY = -1;
    for (let y = 0; y < next.height; y += 2)
      for (let x = 0; x < next.width; x += 2)
        if (d[(y * next.width + x) * 4 + 3] > 8) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
    if (maxX < 0) {
      setErr(t("asset.ed.err_empty"));
      undoRef.current.pop();
      setUndoN(undoRef.current.length);
      return;
    }
    const pad = Math.round(Math.max(maxX - minX, maxY - minY) * 0.03) + 2;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(next.width, maxX + pad);
    maxY = Math.min(next.height, maxY + pad);
    const cropped = document.createElement("canvas");
    cropped.width = maxX - minX;
    cropped.height = maxY - minY;
    cropped.getContext("2d")!.drawImage(next, -minX, -minY);
    setErr("");
    replaceWork(cropped);
  };

  const closeLasso = () => {
    if (lassoPts.length < 3) return;
    const mctx = mask().getContext("2d")!;
    mctx.fillStyle = "#fff";
    mctx.beginPath();
    lassoPts.forEach(([x, y], i) => (i ? mctx.lineTo(x, y) : mctx.moveTo(x, y)));
    mctx.closePath();
    mctx.fill();
    setLassoPts([]);
    setHasMask(true);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!ready || busy) return;
    const [x, y] = toImg(e);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (tool === "crop") {
      const handle = crop ? nearCorner(crop, x, y) : null;
      dragRef.current = { mode: handle ?? "new", sx: x, sy: y, rect: crop ?? undefined };
      if (!handle) setCrop({ x, y, w: 0, h: 0 });
    } else if (tool === "brush") {
      dragRef.current = { mode: "brush", sx: x, sy: y };
      paintBrush(x, y);
    } else {
      setLassoPts((p) => [...p, [x, y]]);
    }
  };

  const nearCorner = (r: Rect, x: number, y: number): string | null => {
    const tol = Math.max(10, work().width / 60);
    for (const [name, cx, cy] of [
      ["nw", r.x, r.y],
      ["ne", r.x + r.w, r.y],
      ["sw", r.x, r.y + r.h],
      ["se", r.x + r.w, r.y + r.h],
    ] as const)
      if (Math.abs(x - cx) < tol && Math.abs(y - cy) < tol) return name;
    // inside → move
    if (x > r.x && x < r.x + r.w && y > r.y && y < r.y + r.h) return "move";
    return null;
  };

  const paintBrush = (x: number, y: number) => {
    const mctx = mask().getContext("2d")!;
    mctx.fillStyle = "#fff";
    mctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    mctx.beginPath();
    mctx.arc(x, y, brush, 0, Math.PI * 2);
    mctx.fill();
    mctx.globalCompositeOperation = "source-over";
    setHasMask(true);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const [x, y] = toImg(e);
    if (d.mode === "brush") {
      paintBrush(x, y);
      draw();
      return;
    }
    if (tool !== "crop") return;
    if (d.mode === "move" && d.rect) {
      const r = d.rect;
      setCrop({ ...r, x: r.x + x - d.sx, y: r.y + y - d.sy });
      return;
    }
    if (d.mode !== "new" && d.rect) {
      // resize from the dragged corner, anchored on the opposite one
      const r = d.rect;
      const ax = d.mode.includes("w") ? r.x + r.w : r.x;
      const ay = d.mode.includes("n") ? r.y + r.h : r.y;
      setCrop(rectFrom(ax, ay, x, y));
      return;
    }
    setCrop(rectFrom(d.sx, d.sy, x, y));
  };

  const rectFrom = (x1: number, y1: number, x2: number, y2: number): Rect => {
    let w = Math.abs(x2 - x1);
    let h = Math.abs(y2 - y1);
    if (aspect && w > 0 && h > 0) {
      // grow the smaller dimension to match the aspect
      if (w / h > aspect) h = w / aspect;
      else w = h * aspect;
    }
    const x = x2 >= x1 ? x1 : x1 - w;
    const y = y2 >= y1 ? y1 : y1 - h;
    return { x, y, w, h };
  };

  const onPointerUp = () => {
    dragRef.current = null;
    setCrop((c) => (c ? normRect(c) : c));
  };

  const removeBg = async () => {
    setBusy(t("asset.ed.busy_bg"));
    setErr("");
    try {
      const blob = await api.removeAssetBg(asset.id);
      const bmp = await createImageBitmap(blob);
      pushUndo();
      const next = document.createElement("canvas");
      next.width = bmp.width;
      next.height = bmp.height;
      next.getContext("2d")!.drawImage(bmp, 0, 0);
      replaceWork(next);
    } catch (e) {
      setErr(errMsg(e));
    } finally {
      setBusy("");
    }
  };

  const save = async () => {
    setBusy(t("asset.ed.busy_save"));
    setErr("");
    try {
      const blob = await new Promise<Blob | null>((res) =>
        work().toBlob(res, "image/png"),
      );
      if (!blob) throw new Error("canvas export failed");
      onSaved(await api.deriveAsset(asset.id, blob));
    } catch (e) {
      setErr(errMsg(e));
      setBusy("");
    }
  };

  const toolBtn = (t: Tool, icon: React.ReactNode, label: string) => (
    <button
      className={`flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs ${
        tool === t ? "bg-brand-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
      }`}
      onClick={() => {
        setTool(t);
        setLassoPts([]);
      }}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/80 backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900 px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">{t("asset.ed.title")}</span>
          <span className="max-w-48 truncate text-xs text-zinc-500">{asset.filename}</span>
        </div>
        <IconBtn title={t("action.close")} onClick={onClose}>
          <X className="h-4 w-4" />
        </IconBtn>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-900/80 px-4 py-2">
        {toolBtn("crop", <Scissors className="h-3.5 w-3.5" />, t("asset.ed.crop"))}
        {toolBtn("brush", <Paintbrush className="h-3.5 w-3.5" />, t("asset.ed.brush"))}
        {toolBtn("lasso", <Lasso className="h-3.5 w-3.5" />, t("asset.ed.lasso"))}
        <span className="mx-1 h-5 w-px bg-zinc-700" />
        {tool === "crop" && (
          <>
            <select
              className="rounded bg-zinc-800 px-2 py-1.5 text-xs text-zinc-300"
              value={aspect === null ? "free" : String(aspect)}
              onChange={(e) =>
                setAspect(e.target.value === "free" ? null : Number(e.target.value))
              }
            >
              {ASPECTS.map((a) => (
                <option key={a.label} value={a.value === null ? "free" : String(a.value)}>
                  {a.label}
                </option>
              ))}
            </select>
            <Btn variant="primary" onClick={applyCrop} disabled={!crop || crop.w < 4}>
              {t("asset.ed.apply_crop")}
            </Btn>
          </>
        )}
        {tool === "brush" && (
          <>
            <label className="flex items-center gap-1.5 text-xs text-zinc-400">
              {t("asset.ed.size")}
              <input
                type="range"
                min={5}
                max={200}
                value={brush}
                onChange={(e) => setBrush(Number(e.target.value))}
                className="w-24"
              />
            </label>
            <button
              className={`flex items-center gap-1 rounded px-2 py-1.5 text-xs ${
                erase ? "bg-amber-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
              }`}
              onClick={() => setErase((v) => !v)}
            >
              <Eraser className="h-3.5 w-3.5" />
              {t("asset.ed.erase")}
            </button>
          </>
        )}
        {tool === "lasso" && lassoPts.length > 2 && (
          <Btn variant="primary" onClick={closeLasso}>
            {t("asset.ed.close_sel")}
          </Btn>
        )}
        {hasMask && (
          <Btn variant="primary" onClick={applyMask}>
            {t("asset.ed.keep_sel")}
          </Btn>
        )}
        <span className="mx-1 h-5 w-px bg-zinc-700" />
        <button
          className="flex items-center gap-1.5 rounded bg-purple-700 px-2.5 py-1.5 text-xs text-white hover:bg-purple-600 disabled:opacity-50"
          onClick={removeBg}
          disabled={!!busy}
        >
          <Wand2 className="h-3.5 w-3.5" />
          {t("asset.ed.remove_bg")}
        </button>
        <button
          className="flex items-center gap-1.5 rounded bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
          onClick={undo}
          disabled={!undoN || !!busy}
        >
          <Undo2 className="h-3.5 w-3.5" />
          {t("asset.ed.undo")}
        </button>
      </div>

      <div
        className="flex flex-1 items-center justify-center overflow-auto p-4"
        style={{
          backgroundImage:
            "repeating-conic-gradient(#27272a 0% 25%, #18181b 0% 50%)",
          backgroundSize: "24px 24px",
        }}
      >
        <canvas
          ref={viewRef}
          role="img"
          aria-label={t("asset.ed.aria", { name: asset.filename })}
          className="max-h-full max-w-full cursor-crosshair shadow-2xl"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onDoubleClick={tool === "lasso" ? closeLasso : undefined}
        />
      </div>

      <div className="flex items-center justify-between border-t border-zinc-800 bg-zinc-900 px-4 py-2.5">
        <div className="text-xs text-zinc-500">
          {busy ? (
            <span className="inline-flex items-center gap-2 text-zinc-300">
              <Spinner className="h-3.5 w-3.5" /> {busy}
            </span>
          ) : err ? (
            <span className="text-red-400">{err}</span>
          ) : (
            <span>
              {tool === "crop" && t("asset.ed.hint_crop")}
              {tool === "brush" && t("asset.ed.hint_brush")}
              {tool === "lasso" && t("asset.ed.hint_lasso")}
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <Btn variant="ghost" onClick={onClose}>
            {t("action.cancel")}
          </Btn>
          <Btn variant="primary" onClick={save} disabled={!ready || !!busy}>
            {t("asset.ed.save_copy")}
          </Btn>
        </div>
      </div>
    </div>
  );
}
