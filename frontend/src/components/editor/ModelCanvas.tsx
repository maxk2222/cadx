import { Bounds, Center, Environment, Gltf, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Loader2 } from "lucide-react";
import {
  Component,
  Suspense,
  useEffect,
  useRef,
  type ReactNode,
  type RefObject,
} from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

export type ViewPreset = "iso" | "front" | "top" | "side";

export const VIEWS: { id: ViewPreset; dir: [number, number, number]; title: string }[] = [
  { id: "iso", dir: [1, 0.8, 1], title: "isometric view" },
  { id: "front", dir: [0, 0.15, 1], title: "front view" },
  { id: "top", dir: [0.001, 1, 0.001], title: "top view" },
  { id: "side", dir: [1, 0.15, 0], title: "side view" },
];

export interface PickInfo {
  /** hit point in model space — GLB coords are the mm frame (X=len, Y=wid, Z=up). */
  point: [number, number, number];
  /** face normal in model space. */
  normal: [number, number, number];
}

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

function GlCapture({ glRef }: { glRef: { current: THREE.WebGLRenderer | null } }) {
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

/** Converts an r3f intersection into model-space mm coordinates. */
function pickFrom(e: ThreeEvent<PointerEvent>): PickInfo | null {
  const p = e.object.worldToLocal(e.point.clone());
  const n = e.face?.normal;
  return {
    point: [p.x, p.y, p.z],
    normal: n ? [n.x, n.y, n.z] : [0, 0, 1],
  };
}

const NO_RAYCAST = () => null;

export default function ModelCanvas({
  url,
  autoRotate = false,
  wireframe = false,
  grid = true,
  viewCmd,
  fitKey = 0,
  glRef,
  pickEnabled = false,
  onPick,
  onHover,
  onModel,
  children,
}: {
  url: string;
  autoRotate?: boolean;
  wireframe?: boolean;
  grid?: boolean;
  viewCmd?: { view: ViewPreset; n: number };
  fitKey?: number;
  glRef?: { current: THREE.WebGLRenderer | null };
  /** enable pointer picking on the model surface (mm coords). */
  pickEnabled?: boolean;
  /** pointerdown + dragging-move picks (brush strokes). */
  onPick?: (info: PickInfo) => void;
  /** hover picks (brush cursor) — null when the pointer leaves the model. */
  onHover?: (info: PickInfo | null) => void;
  /** called once when the model group ref is populated. */
  onModel?: (g: THREE.Group) => void;
  children?: ReactNode;
}) {
  const modelRef = useRef<THREE.Group | null>(null);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const dragging = useRef(false);

  const pickHandlers = pickEnabled
    ? {
        onPointerDown: (e: ThreeEvent<PointerEvent>) => {
          const info = pickFrom(e);
          if (!info) return;
          dragging.current = true;
          if (controlsRef.current) controlsRef.current.enabled = false;
          e.stopPropagation();
          onPick?.(info);
        },
        onPointerMove: (e: ThreeEvent<PointerEvent>) => {
          const info = pickFrom(e);
          onHover?.(info);
          if (dragging.current && info) onPick?.(info);
        },
        onPointerUp: () => {
          dragging.current = false;
          if (controlsRef.current) controlsRef.current.enabled = true;
        },
        onPointerLeave: () => {
          dragging.current = false;
          if (controlsRef.current) controlsRef.current.enabled = true;
          onHover?.(null);
        },
      }
    : {};

  return (
    <Canvas camera={{ position: [60, 45, 60], fov: 40 }} gl={{ preserveDrawingBuffer: true }}>
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
              <group
                ref={(g) => {
                  modelRef.current = g;
                  if (g) onModel?.(g);
                }}
                {...pickHandlers}
              >
                <Gltf src={url} />
                {/* overlays render in model space (mm) but must not intercept picks */}
                <group raycast={NO_RAYCAST}>{children}</group>
              </group>
            </Center>
          </Bounds>
        </GlBoundary>
        <Environment preset="city" />
      </Suspense>
      <OrbitControls ref={controlsRef} makeDefault autoRotate={autoRotate} />
      {grid && <gridHelper args={[200, 40, "#3f3f46", "#27272a"]} position={[0, -0.01, 0]} />}
      {viewCmd && <CameraRig cmd={viewCmd} model={modelRef} controls={controlsRef} />}
      <Wireframe on={wireframe} model={modelRef} />
      {glRef && <GlCapture glRef={glRef} />}
    </Canvas>
  );
}
