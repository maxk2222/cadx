import type { FeatureAnchor, MeshOp } from "./types";

/**
 * Shared restore helpers for the persisted editor draft and the measured
 * feature-anchor table. The three editors (structural, organic, mesh-ops
 * page) all restore the same shapes — keep the parsing in one place.
 *
 * `ui_editor` is a single blob shared by every editor (saveEditorState
 * merges server-side), so readers must test key PRESENCE, not truthiness:
 * a draft written by another editor legitimately lacks your keys, and a
 * persisted empty array is a deliberate "user cleared it" that must not
 * fall back to spec defaults.
 */

/** permissive shape of the persisted ui_editor blob — each editor narrows
 *  it via the generic into its own draft interface. */
export type UiDraft = Record<string, unknown>;

type VersionLike = {
  kind?: string;
  resolved_mode?: string | null;
  options?: unknown;
  params?: unknown;
  analysis?: unknown;
} | null | undefined;

/** version.options.ui_editor, or null when nothing was persisted. */
export function readUiEditor<T extends UiDraft = UiDraft>(version: VersionLike): T | null {
  const opts = version?.options;
  const ue = (opts as Record<string, unknown> | null | undefined)?.ui_editor;
  return ue && typeof ue === "object" ? (ue as T) : null;
}

/** analysis.anchors.holes — the anchor table stored by the run, before any
 *  on-demand /versions/{id}/anchors scan replaces it. */
export function readStoredAnchors(version: VersionLike): FeatureAnchor[] {
  const a = (version?.analysis as { anchors?: { holes?: FeatureAnchor[] } } | null | undefined)
    ?.anchors?.holes;
  return Array.isArray(a) ? a : [];
}

/** recorded mesh ops of a mesh-mode parent — replayed on the raw generator
 *  output (mesh_edit versions have their ops baked already). */
export function recordedMeshOps(version: VersionLike): MeshOp[] {
  if (!version || version.resolved_mode !== "mesh" || version.kind === "mesh_edit")
    return [];
  const recorded = (version.params as { ops?: MeshOp[] } | null | undefined)?.ops;
  return Array.isArray(recorded) ? recorded : [];
}

/** the ops stack a version was opened with: a persisted draft wins — even an
 *  empty one (the user deliberately dropped the ops). Falls back to the
 *  recorded parent ops for mesh-mode versions, else the empty stack. */
export function savedOps(version: VersionLike): MeshOp[] {
  if (!version) return [];
  const draft = readUiEditor<{ ops?: MeshOp[] }>(version);
  if (draft && "ops" in draft) return draft.ops ?? [];
  return recordedMeshOps(version);
}

/** coerce a string to a finite number with fallback (slider/range inputs). */
export const num = (s: string, fb = 0) => {
  const n = Number(s);
  return Number.isFinite(n) ? n : fb;
};
