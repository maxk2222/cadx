import type { FacetJob, FacetWireResult } from "./facets";
import type { FillRequest } from "./regionFill";

/**
 * Lazy-singleton client for viewer.worker.ts. `null` return = workers
 * unavailable or failed to spawn — callers fall back to the synchronous
 * code path. Requests are tagged so multiple kinds can share one worker;
 * an onerror rejects every in-flight call and drops the instance (the next
 * call respawns).
 */
let worker: Worker | null = null;
let failed = false;
let seq = 0;
const pending = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: unknown) => void }
>();

function ensure(): Worker | null {
  if (failed) return null;
  if (worker) return worker;
  if (typeof Worker === "undefined") {
    failed = true;
    return null;
  }
  try {
    worker = new Worker(new URL("./viewer.worker.ts", import.meta.url), {
      type: "module",
    });
  } catch {
    failed = true;
    return null;
  }
  worker.onmessage = (e: MessageEvent<{ id: number; result: unknown }>) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    p.resolve(e.data.result);
  };
  worker.onerror = (e) => {
    const ps = [...pending.values()];
    pending.clear();
    worker = null;
    ps.forEach((p) => p.reject(e));
  };
  return worker;
}

function call<T>(kind: string, payload: unknown, transfer: Transferable[]): Promise<T> | null {
  const w = ensure();
  if (!w) return null;
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage({ id, kind, payload }, transfer);
  });
}

export const viewerWorker = {
  /** facet-segment every job — resolve order matches the input array. */
  facets(jobs: FacetJob[]): Promise<(FacetWireResult | null)[]> | null {
    const t: Transferable[] = [];
    for (const j of jobs) {
      t.push(j.pos.array.buffer as ArrayBuffer);
      if (j.index) t.push(j.index.buffer as ArrayBuffer);
    }
    return call("facets", jobs, t);
  },
  /** painted-region vertex colours for every mesh — order matches input. */
  fills(req: FillRequest): Promise<(Float32Array | null)[]> | null {
    const t: Transferable[] = [];
    for (const e of req.entries) t.push(e.points.buffer as ArrayBuffer);
    for (const m of req.meshes) t.push(m.pos.array.buffer as ArrayBuffer);
    return call("fills", req, t);
  },
};
