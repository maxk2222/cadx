import { Canvas } from "@react-three/fiber";
import { Bounds, Center, Environment, Gltf, OrbitControls } from "@react-three/drei";
import { Suspense } from "react";

export default function Viewer3D({ url }: { url: string | null }) {
  if (!url) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500 text-sm">
        no 3D preview yet
      </div>
    );
  }
  return (
    <Canvas camera={{ position: [60, 45, 60], fov: 40 }}>
      <color attach="background" args={["#18181b"]} />
      <Suspense fallback={null}>
        <Bounds fit clip observe margin={1.2}>
          <Center>
            <Gltf src={url} />
          </Center>
        </Bounds>
        <Environment preset="city" />
      </Suspense>
      <OrbitControls makeDefault />
      <gridHelper args={[200, 40, "#3f3f46", "#27272a"]} position={[0, -0.01, 0]} />
    </Canvas>
  );
}
