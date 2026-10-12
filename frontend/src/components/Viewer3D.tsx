import { Bounds, Center, Environment, Gltf, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import {
  Boxes,
  Camera,
  Grid3x3,
  Loader2,
  Maximize,
  Minimize,
  RotateCw,
  Scan,
} from "lucide-react";
import {
  Component,
  Suspense,
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { IconBtn } from "./ui";

type ViewPreset = "iso" | "front" | "top" | "side";

const VIEWS: { id: ViewPreset; dir: [number, number, number]; title: string }[] = [
  { id: "iso", dir: [1, 0.8, 1], title: "isometric view" },
  { id: "front", dir: [0, 0.15, 1], title: "front view" },
  { id: "top", dir: [0.001, 1, 0.001], title: "top view" },
  { id: "side", dir: [1, 0.15, 0], title: "side view" },
];

function CameraRig({
  cmd,
  model,
  controls,
}: {
  cmd: { view: ViewPreset; n: number };
  model: RefObject<THREE.Group | null>;
  controls: RefObject<OrbitControlsImpl | null>;
}) {
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    if (!cmd.n) return;
    const dir = VIEWS.find((v) => v.id === cmd.view)?.dir ?? VIEWS[0].dir;
    let dist = 150;
    if (model.current) {
      const box = new THREE.Box3().setFromObject(model.current);
      const size = box.getSize(new THREE.Vector3()).length();
      if (size > 0) dist = size * 1.4;
    }
    const d = new THREE.Vector3(...dir).normalize().multiplyScalar(dist);
    camera.position.copy(d);
    camera.lookAt(0, 0, 0);
    controls.current?.target.set(0, 0, 0);
    controls.current?.update();
  }, [cmd, camera, model, controls]);
  return null;
}

function Wireframe({ on, model }: { on: boolean; model: RefObject<THREE.Group | null> }) {
  useEffect(() => {
    model.current?.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => {
          if ("wireframe" in m) (m as THREE.MeshStandardMaterial).wireframe = on;
        });
      }
    });
  }, [on, model]);
  return null;
}

function GlCapture({
  glRef,
}: {
  glRef: { current: THREE.WebGLRenderer | null };
}) {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    glRef.current = gl;
  }, [gl, glRef]);
  return null;
}

class GlBoundary extends Component<{ children: ReactNode }, { err: boolean }> {
  state = { err: false };
  static getDerivedStateFromError() {
    return { err: true };
  }
  render() {
    if (this.state.err)
      return (
        <Html center>
          <div className="whitespace-nowrap text-sm text-red-400">
            failed to load 3D preview
          </div>
        </Html>
      );
    return this.props.children;
  }
}

export default function Viewer3D({ url }: { url: string | null }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const modelRef = useRef<THREE.Group | null>(null);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const [viewCmd, setViewCmd] = useState({ view: "iso" as ViewPreset, n: 0 });
  const [fitKey, setFitKey] = useState(0);
  const [autoRotate, setAutoRotate] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [grid, setGrid] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);

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
      <Canvas
        camera={{ position: [60, 45, 60], fov: 40 }}
        gl={{ preserveDrawingBuffer: true }}
      >
        <color attach="background" args={["#18181b"]} />
        <Suspense
          fallback={
            <Html center>
              <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
            </Html>
          }
        >
          <GlBoundary>
            <Bounds key={fitKey} fit clip observe margin={1.2}>
              <Center>
                <group ref={modelRef}>
                  <Gltf src={url} />
                </group>
              </Center>
            </Bounds>
          </GlBoundary>
          <Environment preset="city" />
        </Suspense>
        <OrbitControls ref={controlsRef} makeDefault autoRotate={autoRotate} />
        {grid && (
          <gridHelper args={[200, 40, "#3f3f46", "#27272a"]} position={[0, -0.01, 0]} />
        )}
        <CameraRig cmd={viewCmd} model={modelRef} controls={controlsRef} />
        <Wireframe on={wireframe} model={modelRef} />
        <GlCapture glRef={glRef} />
      </Canvas>

      {/* view presets */}
      <div className="absolute left-2 top-2 flex gap-1">
        {VIEWS.map((v) =>
          toolBtn(v.title, false, () => setViewCmd({ view: v.id, n: Date.now() }),
            <span className="px-0.5 text-[10px] font-semibold uppercase">
              {v.id}
            </span>,
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
