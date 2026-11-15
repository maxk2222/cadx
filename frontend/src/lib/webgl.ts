import { useCallback, useState } from "react";

/**
 * Synchronous WebGL probe — mounting a Three.js Canvas without this check
 * throws "Error creating WebGL context" into the ErrorBoundary whenever the
 * browser's GPU process is disabled/crashed (Chrome: GL_VENDOR=Disabled),
 * which users read as a crash of the whole page. Probe first, render a
 * legible fallback instead.
 */
export function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return Boolean(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** [available, retry] — retry re-probes so a recovered GPU process can
 * remount the canvas without a full page reload. */
export function useWebGLAvailable(): [boolean, () => void] {
  const [ok, setOk] = useState(webglAvailable);
  const retry = useCallback(() => setOk(webglAvailable()), []);
  return [ok, retry];
}
