import type { MeshModelInfo } from "./types";

/** Lifecycle state of a sidecar model, ordered most-specific first:
 *  an active download wins over pinned/loaded/install state. */
export type ModelState =
  | "downloading"
  | "pinned"
  | "loaded"
  | "installed"
  | "not_installed";

export function modelState(m: MeshModelInfo): ModelState {
  if (m.downloading) return "downloading";
  if (m.loaded) return m.pinned ? "pinned" : "loaded";
  // `available` is the legacy install flag — `downloaded` supersedes it
  if (m.downloaded ?? m.available) return "installed";
  return "not_installed";
}

export function modelStateLabel(m: MeshModelInfo): string {
  const s = modelState(m);
  return s === "not_installed" ? "not installed" : s;
}

export function modelStateTone(m: MeshModelInfo): "blue" | "green" | "purple" | "zinc" {
  const s = modelState(m);
  if (s === "downloading") return "blue";
  if (s === "loaded") return "green";
  if (s === "pinned") return "purple";
  return "zinc";
}

/** Megabytes → "~12 GB" for est_mem_mb / queue rows; null → "—". */
export function fmtMemGb(mb: number | null | undefined): string {
  if (mb == null) return "—";
  const gb = mb / 1024;
  return `~${gb >= 10 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

/** Plain GiB figure (weights on disk); null → "—". */
export function fmtGb(gb: number | null | undefined): string {
  if (gb == null) return "—";
  return `${gb.toFixed(1)} GB`;
}

/** Downloaded bytes so far — HF weights run into the GB range. */
export function fmtDlBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${(n / 1e3).toFixed(0)} KB`;
}

/** "https://api.openai.com/v1" → "api.openai.com"; empty/garbage → "". */
export function urlHost(u: string | null | undefined): string {
  if (!u) return "";
  try {
    return new URL(u).host;
  } catch {
    return "";
  }
}
