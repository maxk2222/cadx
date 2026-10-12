import {
  Boxes,
  Camera,
  Grid3x3,
  Maximize,
  Minimize,
  RotateCw,
  Scan,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import ModelCanvas, { VIEWS, type ViewPreset } from "./editor/ModelCanvas";
import { IconBtn } from "./ui";

export default function Viewer3D({ url }: { url: string | null }) {
  const [autoRotate, setAutoRotate] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [grid, setGrid] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewCmd, setViewCmd] = useState<{ view: ViewPreset; n: number }>({ view: "iso", n: 0 });
  const [fitKey, setFitKey] = useState(0);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", h);
    return () => document.removeEventListener("fullscreenchange", h);
  }, []);

  // reset transient view state when the model changes
  useEffect(() => {
    setFitKey((k) => k + 1);
    setWireframe(false);
  }, [url]);

  const screenshot = () => {
    glRef.current?.domElement.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "model.png";
      a.click();
      URL.revokeObjectURL(a.href);
    });
  };

  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else wrapRef.current?.requestFullscreen();
  };

  if (!url) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-zinc-500">
        <Boxes className="h-8 w-8 text-zinc-700" />
        no 3D preview yet
      </div>
    );
  }

  const toolBtn = (
    title: string,
    active: boolean,
    onClick: () => void,
    icon: ReactNode,
    key?: string
  ) => (
    <IconBtn
      key={key}
      title={title}
      onClick={onClick}
      className={`bg-zinc-900/80 backdrop-blur ${active ? "text-brand-400" : ""}`}
    >
      {icon}
    </IconBtn>
  );

  return (
    <div ref={wrapRef} className="relative h-full w-full bg-zinc-900">
      <ModelCanvas
        url={url}
        autoRotate={autoRotate}
        wireframe={wireframe}
        grid={grid}
        viewCmd={viewCmd}
        fitKey={fitKey}
        glRef={glRef}
      />

      {/* view presets */}
      <div className="absolute left-2 top-2 flex gap-1">
        {VIEWS.map((v) =>
          toolBtn(
            v.title,
            false,
            () => setViewCmd({ view: v.id, n: Date.now() }),
            <span className="px-0.5 text-[10px] font-semibold uppercase">{v.id}</span>,
            v.id
          )
        )}
      </div>

      {/* toolbar */}
      <div className="absolute right-2 top-2 flex gap-1">
        {toolBtn("fit / reset view", false, () => setFitKey((k) => k + 1), <Scan className="h-4 w-4" />)}
        {toolBtn("auto-rotate", autoRotate, () => setAutoRotate((v) => !v), <RotateCw className="h-4 w-4" />)}
        {toolBtn("wireframe", wireframe, () => setWireframe((v) => !v), <Boxes className="h-4 w-4" />)}
        {toolBtn("toggle grid", grid, () => setGrid((v) => !v), <Grid3x3 className="h-4 w-4" />)}
        {toolBtn("screenshot (PNG)", false, screenshot, <Camera className="h-4 w-4" />)}
        {toolBtn(
          fullscreen ? "exit fullscreen" : "fullscreen",
          fullscreen,
          toggleFullscreen,
          fullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />
        )}
      </div>
    </div>
  );
}
