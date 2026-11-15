import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

import i18n from "../../i18n";
import { useWebGLAvailable } from "../../lib/webgl";

/**
 * Gate for react-three-fiber canvases: probes WebGL BEFORE Three.js mounts.
 * A disabled/crashed browser GPU process throws "Error creating WebGL
 * context" deep inside THREE.WebGLRenderer — the ErrorBoundary then makes
 * it look like the whole page crashed. This renders a legible fallback
 * with a retry that re-probes once the GPU process recovered.
 */
export default function WebGLGate({ children }: { children: ReactNode }) {
  const [ok, retry] = useWebGLAvailable();
  if (!ok) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 text-center">
        <TriangleAlert className="h-8 w-8 text-amber-400" />
        <div className="text-sm font-medium text-fg">
          {i18n.t("viewer.webgl_unavailable")}
        </div>
        <div className="max-w-xs text-xs text-muted">
          {i18n.t("viewer.webgl_unavailable_hint")}
        </div>
        <button
          type="button"
          className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500"
          onClick={retry}
        >
          {i18n.t("viewer.context_lost_reload")}
        </button>
      </div>
    );
  }
  return <>{children}</>;
}
