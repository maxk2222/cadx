import {
  buildFacetJob,
  facetSetToWire,
  type FacetJob,
  type FacetWireResult,
} from "./facets";
import { runFillRequest, type FillRequest } from "./regionFill";

/**
 * Off-thread compute for the viewer: facet segmentation (weld + crease
 * clustering) and painted-region vertex fills. Both are pure buffer math —
 * the main thread sends copied attribute arrays (transferred) and gets
 * back typed arrays it rehydrates into geometry. Results are identical to
 * the synchronous paths; the callers keep sync fallbacks for small meshes
 * and for environments without Worker support.
 */

interface JobMsg {
  id: number;
  kind: "facets" | "fills";
  payload: unknown;
}

// `self` is typed as Window under the DOM lib — postMessage needs the
// worker-side (message, transfer) signature.
const post = self as unknown as Pick<Worker, "postMessage">;

self.onmessage = (e: MessageEvent<JobMsg>) => {
  const { id, kind, payload } = e.data;
  if (kind === "facets") {
    const results = (payload as FacetJob[]).map((j): FacetWireResult | null => {
      try {
        const fs = buildFacetJob(j);
        return fs ? facetSetToWire(fs) : null;
      } catch {
        return null;
      }
    });
    const t: Transferable[] = [];
    for (const r of results) {
      if (r) t.push(r.positions.buffer, r.index.buffer, r.facetOf.buffer);
    }
    post.postMessage({ id, result: results }, t);
    return;
  }
  if (kind === "fills") {
    const results = runFillRequest(payload as FillRequest);
    const t = results.filter((r): r is Float32Array => !!r).map((r) => r.buffer);
    post.postMessage({ id, result: results }, t);
    return;
  }
  post.postMessage({ id, result: null, error: `unknown job kind ${kind}` });
};
