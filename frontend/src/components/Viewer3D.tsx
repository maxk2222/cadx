import { Line } from "@react-three/drei";
import {
  Boxes,
  Camera,
  Grid3x3,
  Maximize,
  Minimize,
  RotateCw,
  Ruler,
  Scan,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import { distMm, nextMeasurePts, type Pt3 } from "../lib/measure";
import ModelCanvas, {
  VIEWS,
  type PickInfo,
  type ViewPreset,
} from "./editor/ModelCanvas";
import { IconBtn, OpacitySlider } from "./ui";

/** surface marker sphere in displayed-model mm space — same look as
 *  EditPage's measure/drill pins */
function Marker({ p, color = "#f59e0b" }: { p: Pt3; color?: string }) {
  return (
    <mesh position={p}>
      <sphereGeometry args={[0.8, 16, 12]} />
      <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
    </mesh>
  );
}

export default function Viewer3D({ url }: { url: string | null }) {
  const [autoRotate, setAutoRotate] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [opacity, setOpacity] = useState(1);
  const [grid, setGrid] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewCmd, setViewCmd] = useState<{ view: ViewPreset; n: number }>({ view: "iso", n: 0 });
  const [fitKey, setFitKey] = useState(0);
  const [measuring, setMeasuring] = useState(false);
  const [measurePts, setMeasurePts] = useState<Pt3[]>([]);
  const [hover, setHover] = useState<PickInfo | null>(null);
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
    setMeasuring(false);
  }, [url]);

  // disengaging the measure tool (toggle or Esc) clears its markers
  useEffect(() => {
    if (!measuring) {
      setMeasurePts([]);
      setHover(null);
      return;
    }
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMeasuring(false);
    };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [measuring]);

  const screenshot = () => {
    glRef.current?.domElement.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "model.png";
      a.click();
      // deferred: revoking in the same task can abort the download in
      // some browsers before it starts
      setTimeout(() => URL.revokeObjectURL(a.href), 0);
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

  const measureDist =
    measurePts.length === 2 ? distMm(measurePts[0], measurePts[1]) : null;

  return (
    <div ref={wrapRef} className="relative h-full w-full bg-zinc-900">
      <ModelCanvas
        url={url}
        autoRotate={autoRotate}
        wireframe={wireframe}
        opacity={opacity}
        grid={grid}
        viewCmd={viewCmd}
        fitKey={fitKey}
        glRef={glRef}
        preserveDrawingBuffer // PNG screenshots read the framebuffer
        pickEnabled={measuring}
        // click-only tool — ignore drag-move repeats
        onPick={(i, viaDrag) => {
          if (!viaDrag) setMeasurePts((pts) => nextMeasurePts(pts, i.point));
        }}
        onHover={setHover}
      >
        {/* measure overlays render in displayed-model space and never
            intercept picks (ModelCanvas wraps children in NO_RAYCAST) */}
        {measuring && hover && (
          <mesh position={hover.point}>
            <sphereGeometry args={[0.7, 16, 12]} />
            <meshBasicMaterial color="#f59e0b" transparent opacity={0.85} depthTest={false} />
          </mesh>
        )}
        {measurePts.map((p, i) => (
          <Marker key={i} p={p} />
        ))}
        {measureDist != null && (
          <Line points={measurePts} color="#f59e0b" lineWidth={1.5} depthTest={false} />
        )}
      </ModelCanvas>

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
      <div className="absolute right-2 top-2 flex flex-wrap justify-end gap-1">
        <OpacitySlider value={opacity} onChange={setOpacity} />
        {toolBtn("fit / reset view", false, () => setFitKey((k) => k + 1), <Scan className="h-4 w-4" />)}
        {toolBtn("auto-rotate", autoRotate, () => setAutoRotate((v) => !v), <RotateCw className="h-4 w-4" />)}
        {toolBtn("wireframe", wireframe, () => setWireframe((v) => !v), <Boxes className="h-4 w-4" />)}
        {toolBtn("toggle grid", grid, () => setGrid((v) => !v), <Grid3x3 className="h-4 w-4" />)}
        {toolBtn(
          "measure distance — click two surface points (Esc exits)",
          measuring,
          () => setMeasuring((v) => !v),
          <Ruler className="h-4 w-4" />
        )}
        {toolBtn("screenshot (PNG)", false, screenshot, <Camera className="h-4 w-4" />)}
        {toolBtn(
          fullscreen ? "exit fullscreen" : "fullscreen",
          fullscreen,
          toggleFullscreen,
          fullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />
        )}
      </div>

      {/* measure status chip — hint while picking, distance once both
          surface points are placed (same styling as EditPage's) */}
      {measuring && (
        <div className="absolute bottom-3 left-3 flex items-center rounded-lg border border-amber-800/60 bg-zinc-950/90 px-3 py-1.5 text-sm text-amber-300 backdrop-blur">
          <Ruler className="mr-1.5 h-3.5 w-3.5" />
          {measureDist != null ? (
            <>
              {measureDist.toFixed(2)} mm
              <button
                type="button"
                aria-label="clear measurement"
                className="ml-2 text-xs text-zinc-500 hover:text-zinc-300"
                onClick={() => setMeasurePts([])}
              >
                clear
              </button>
            </>
          ) : (
            <span className="text-xs">
              {measurePts.length === 1
                ? "click a second surface point"
                : "click two surface points"}{" "}
              · right-drag orbits · Esc exits
            </span>
          )}
        </div>
      )}
    </div>
  );
}
