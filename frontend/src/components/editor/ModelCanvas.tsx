import {
  Bounds,
  Center,
  Environment,
  Gltf,
  Html,
  Lightformer,
  OrbitControls,
} from "@react-three/drei";
import { Canvas, useLoader, useThree, type ThreeEvent } from "@react-three/fiber";
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
import { STLLoader } from "three-stdlib";

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
  /** triangle index of the hit face in `object`'s geometry (facet picking). */
  faceIndex?: number | null;
  /** the mesh inside the pick frame that was hit. */
  object?: THREE.Object3D;
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
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    model.current?.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => {
          if ("wireframe" in m) (m as THREE.MeshStandardMaterial).wireframe = on;
        });
      }
    });
    invalidate(); // material flag mutation is invisible to the reconciler
  }, [on, model, invalidate]);
  return null;
}

/** demand-mode catch-all: re-render whenever inputs that drive imperative
 *  scene changes mutate (url, overlay children, preview matrix). Declarative
 *  JSX commits already invalidate in demand mode — this covers the paths
 *  React cannot observe. */
function InvalidateOn({ deps }: { deps: readonly unknown[] }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps is the API
  }, [invalidate, ...deps]);
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

/** STL loader for non-GLB previews (library parts). STLs are Z-up — rotate
 * into the Y-up display frame used by the GLB exports. */
function StlModel({ url }: { url: string }) {
  const geom = useLoader(STLLoader, url);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    // mutates the loader-cached geometry — idempotent, but it must not run
    // during render (same object may be shared by other canvases)
    geom.computeVertexNormals();
    invalidate();
  }, [geom, invalidate]);
  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial color="#8b93a1" metalness={0.1} roughness={0.7} />
    </mesh>
  );
}

/** true when `obj` is `root` or a descendant of it. */
function isDescendantOf(obj: THREE.Object3D, root: THREE.Object3D): boolean {
  let o: THREE.Object3D | null = obj;
  while (o) {
    if (o === root) return true;
    o = o.parent;
  }
  return false;
}

/**
 * Converts an r3f event into displayed-model mm coordinates: the pick frame
 * wraps the GLB (and carries the instant-preview matrix), so picked
 * points/normals always describe the visible surface — the same frame new
 * ops/region points are written in. The normal is flipped toward the camera
 * ray so painted load normals point outward, not into the body.
 *
 * Only intersections with objects INSIDE the pick frame count — r3f raycasts
 * the handler group recursively, so overlay children (markers, gizmo arrows,
 * the brush cursor sphere, ghost planes) would otherwise produce phantom
 * picks floating in front of the real surface.
 */
function pickFrom(e: ThreeEvent<PointerEvent>, frame: THREE.Object3D): PickInfo | null {
  const hit = e.intersections.find((i) => isDescendantOf(i.object, frame));
  if (!hit) return null;
  frame.updateWorldMatrix(true, false);
  const p = frame.worldToLocal(hit.point.clone());
  const n = hit.face?.normal.clone() ?? new THREE.Vector3(0, 0, 1);
  n.transformDirection(hit.object.matrixWorld); // world space
  if (n.dot(e.ray.direction) > 0) n.negate(); // face the camera
  n.transformDirection(frame.matrixWorld.clone().invert()); // -> frame local
  return {
    point: [p.x, p.y, p.z],
    normal: [n.x, n.y, n.z],
    faceIndex: hit.faceIndex ?? null,
    object: hit.object,
  };
}

export default function ModelCanvas({
  url,
  format = "glb",
  autoRotate = false,
  wireframe = false,
  grid = true,
  viewCmd,
  fitKey = 0,
  glRef,
  pickEnabled = false,
  modelMatrix,
  preserveDrawingBuffer = false,
  onPick,
  onHover,
  onModel,
  children,
}: {
  url: string;
  /** model file format — GLB/GLTF by default; "stl" for library part previews */
  format?: "glb" | "stl";
  autoRotate?: boolean;
  wireframe?: boolean;
  grid?: boolean;
  viewCmd?: { view: ViewPreset; n: number };
  fitKey?: number;
  glRef?: { current: THREE.WebGLRenderer | null };
  /** keep the framebuffer readable after present — required for canvas
   *  screenshots (Viewer3D); costs a copy everywhere else, so it's opt-in */
  preserveDrawingBuffer?: boolean;
  /** enable pointer picking on the model surface (mm coords). */
  pickEnabled?: boolean;
  /** instant-preview transform applied to the model only (never overlays). */
  modelMatrix?: THREE.Matrix4 | null;
  /** pointerdown + dragging-move picks (brush strokes). `viaDrag` is true for
   *  the pointermove repeats — click-style tools should ignore those. */
  onPick?: (info: PickInfo, viaDrag?: boolean) => void;
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

  // releasing the button off-model must still end the drag — the group's
  // onPointerUp only fires for hits inside it. NOTE: we never touch
  // OrbitControls.enabled here — disabling it mid-gesture freezes an
  // in-progress right-drag orbit; the mouseButtons remap keeps LEFT out of
  // the controls entirely, so no disabling is needed.
  useEffect(() => {
    const up = () => {
      dragging.current = false;
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, []);

  const handlers = pickEnabled
    ? {
        onPointerDown: (e: ThreeEvent<PointerEvent>) => {
          // left button only — right belongs to orbit while a tool is armed;
          // on touch only the primary finger paints (a second finger is the
          // start of an orbit/pinch, not a second brush)
          if (e.button !== 0) return;
          if (e.pointerType === "touch" && !e.isPrimary) return;
          const frame = frameRef.current;
          if (!frame) return;
          const info = pickFrom(e, frame);
          if (!info) return;
          dragging.current = true;
          e.stopPropagation();
          onPick?.(info);
        },
        onPointerMove: (e: ThreeEvent<PointerEvent>) => {
          const frame = frameRef.current;
          if (!frame) return;
          const info = pickFrom(e, frame);
          queueHover(info);
          if (dragging.current && info) onPick?.(info, true);
        },
        onPointerUp: () => {
          dragging.current = false;
        },
        onPointerLeave: () => {
          // do NOT clear `dragging` here — the synthesized leave fires when
          // the ray misses the model mid-stroke, and the stroke should
          // resume when the pointer re-enters. The window pointerup/
          // pointercancel listener ends drags on release.
          queueHover(null);
        },
      }
    : {};

  const matrix = useMemo(() => modelMatrix ?? IDENTITY, [modelMatrix]);

  // while a pick/paint tool is armed the LEFT button belongs to the tool —
  // orbit moves to right-drag (middle = pan, wheel = zoom; touch: one finger
  // paints, two fingers orbit). -1 matches no mouseAction -> STATE.NONE.
  const mouseButtons = useMemo(
    () =>
      pickEnabled
        ? {
            LEFT: -1 as unknown as THREE.MOUSE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }
        : {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
          },
    [pickEnabled]
  );
  const touches = useMemo(
    () => ({
      ONE: (pickEnabled ? -1 : THREE.TOUCH.ROTATE) as THREE.TOUCH,
      TWO: THREE.TOUCH.DOLLY_ROTATE,
    }),
    [pickEnabled]
  );

  return (
    <Canvas
      // demand frameloop — render only on invalidation. Declarative commits
      // and drei OrbitControls/Bounds invalidate themselves; imperative
      // mutations (wireframe flag, STL normals, RegionFill colours) call
      // invalidate() explicitly, with InvalidateOn as the catch-all.
      // NOTE: no useLoader cache cleanup on unmount — the suspense cache is
      // shared across canvases and transient preview URLs are not re-fetchable.
      frameloop="demand"
      camera={{ position: [60, 45, 60], fov: 40 }}
      gl={{ preserveDrawingBuffer }}
      style={{ cursor: pickEnabled ? "crosshair" : undefined }}
    >
      <color attach="background" args={["#18181b"]} />
      <InvalidateOn deps={[url, format, wireframe, grid, matrix, children]} />
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
                    {format === "stl" ? <StlModel url={url} /> : <Gltf src={url} />}
                  </group>
                </group>
                {/* overlays render in displayed-model space but must not
                    intercept picks */}
                <group raycast={NO_RAYCAST}>{children}</group>
              </group>
            </Center>
          </Bounds>
          {/* local procedural environment — preset="city" fetched an HDR
              from a CDN at runtime (breaks offline/self-hosted); Lightformer
              children render the env map locally instead: soft top light +
              cool/warm side strips ≈ the city look */}
          <Environment resolution={256}>
            <Lightformer
              intensity={2}
              rotation-x={Math.PI / 2}
              position={[0, 5, 0]}
              scale={[10, 10, 1]}
            />
            <Lightformer
              intensity={1.4}
              rotation-y={Math.PI / 2}
              position={[-5, 1, -1]}
              scale={[12, 2.5, 1]}
              color="#cfe2ff"
            />
            <Lightformer
              intensity={0.7}
              rotation-y={-Math.PI / 2}
              position={[5, 0.5, -1]}
              scale={[8, 3, 1]}
              color="#ffddbb"
            />
          </Environment>
        </GlBoundary>
      </Suspense>
      <OrbitControls
        ref={controlsRef}
        makeDefault
        autoRotate={autoRotate}
        mouseButtons={mouseButtons}
        touches={touches}
      />
      {grid && <gridHelper args={[200, 40, "#3f3f46", "#27272a"]} position={[0, -0.01, 0]} />}
      {viewCmd && <CameraRig cmd={viewCmd} model={modelRef} controls={controlsRef} />}
      <Wireframe on={wireframe} model={modelRef} />
      {glRef && <GlCapture glRef={glRef} />}
    </Canvas>
  );
}
