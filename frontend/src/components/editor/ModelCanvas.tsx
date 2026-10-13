import { Bounds, Center, Environment, Gltf, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Loader2 } from "lucide-react";
import {
  Component,
  Suspense,
  useEffect,
  useMemo,
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
  /** hit point in displayed-model space (mm — includes the preview matrix). */
  point: [number, number, number];
  /** face normal in the same space, flipped toward the camera. */
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

const IDENTITY = new THREE.Matrix4();
const NO_RAYCAST = () => null;

/**
 * Converts an r3f intersection into displayed-model mm coordinates: the pick
 * frame wraps the GLB (and carries the instant-preview matrix), so picked
 * points/normals always describe the visible surface — the same frame new
 * ops/region points are written in. The normal is flipped toward the camera
 * ray so painted load normals point outward, not into the body.
 */
function pickFrom(e: ThreeEvent<PointerEvent>, frame: THREE.Object3D): PickInfo | null {
  frame.updateWorldMatrix(true, false);
  const p = frame.worldToLocal(e.point.clone());
  const n = e.face?.normal.clone() ?? new THREE.Vector3(0, 0, 1);
  n.transformDirection(e.object.matrixWorld); // world space
  if (n.dot(e.ray.direction) > 0) n.negate(); // face the camera
  n.transformDirection(frame.matrixWorld.clone().invert()); // -> frame local
  return {
    point: [p.x, p.y, p.z],
    normal: [n.x, n.y, n.z],
  };
}

export default function ModelCanvas({
  url,
  autoRotate = false,
  wireframe = false,
  grid = true,
  viewCmd,
  fitKey = 0,
  glRef,
  pickEnabled = false,
  modelMatrix,
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
  /** instant-preview transform applied to the model only (never overlays). */
  modelMatrix?: THREE.Matrix4 | null;
  /** pointerdown + dragging-move picks (brush strokes). */
  onPick?: (info: PickInfo) => void;
  /** hover picks (brush cursor) — rAF-throttled; null when leaving the model. */
  onHover?: (info: PickInfo | null) => void;
  /** called once when the model content group (inside the pick frame) is ready —
   *  use it for static analysis like the AABB evaluator. */
  onModel?: (g: THREE.Group) => void;
  children?: ReactNode;
}) {
  const modelRef = useRef<THREE.Group | null>(null);
  const frameRef = useRef<THREE.Group | null>(null);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const dragging = useRef(false);
  const hoverPending = useRef<PickInfo | null>(null);
  const raf = useRef(0);

  const emitHover = () => {
    raf.current = 0;
    onHover?.(hoverPending.current);
  };
  const queueHover = (info: PickInfo | null) => {
    hoverPending.current = info;
    if (!raf.current) raf.current = requestAnimationFrame(emitHover);
  };

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const handlers = pickEnabled
    ? {
        onPointerDown: (e: ThreeEvent<PointerEvent>) => {
          const frame = frameRef.current;
          if (!frame) return;
          const info = pickFrom(e, frame);
          if (!info) return;
          dragging.current = true;
          if (controlsRef.current) controlsRef.current.enabled = false;
          e.stopPropagation();
          onPick?.(info);
        },
        onPointerMove: (e: ThreeEvent<PointerEvent>) => {
          const frame = frameRef.current;
          if (!frame) return;
          const info = pickFrom(e, frame);
          queueHover(info);
          if (dragging.current && info) onPick?.(info);
        },
        onPointerUp: () => {
          dragging.current = false;
          if (controlsRef.current) controlsRef.current.enabled = true;
        },
        onPointerLeave: () => {
          dragging.current = false;
          if (controlsRef.current) controlsRef.current.enabled = true;
          queueHover(null);
        },
      }
    : {};

  const matrix = useMemo(() => modelMatrix ?? IDENTITY, [modelMatrix]);

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
                }}
                {...handlers}
              >
                {/* pick frame: carries the instant-preview matrix — picks and
                    the AABB evaluator share this displayed-model space */}
                <group ref={frameRef} matrixAutoUpdate={false} matrix={matrix}>
                  <group ref={(g) => g && onModel?.(g)}>
                    <Gltf src={url} />
                  </group>
                </group>
                {/* overlays render in displayed-model space but must not
                    intercept picks */}
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
