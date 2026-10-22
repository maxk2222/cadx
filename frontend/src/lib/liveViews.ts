/** View-name helpers for the live render preview + render gallery.
 *
 * View names are render-file stems ("iso", "front", …) or "fem:<stem>" for
 * structural plots (vonmises_*, deformed_*, regions_*). Model renders are
 * the better default — the freshest manifest write may carry fem plots only,
 * so fem views are always last in the implicit pick order. */

/** preferred default order — matches the backend's _PREVIEW_PREFER first */
export const VIEW_PREFER = [
  "iso",
  "contact_sheet",
  "orient_sheet",
  "thumbnail",
  "front",
  "top",
  "back",
  "left",
  "right",
  "bottom",
  "iso_ghost",
  "front_ghost",
] as const;

export const isFemView = (v: string) => v.startsWith("fem:");

/** Pick the view to display: an explicit user selection wins, then the
 * canonical model views, then any non-fem view, fem plots last. */
export function pickView(views: string[], selected: string | null): string | null {
  if (selected && views.includes(selected)) return selected;
  for (const w of VIEW_PREFER) if (views.includes(w)) return w;
  return views.find((w) => !isFemView(w)) ?? views[0] ?? null;
}

/** stable sort for the view switcher: model views in prefer order, then fem
 * views grouped (vonmises < deformed < regions, export variants adjacent) */
export function sortViews(views: string[]): string[] {
  const rank = (w: string): [number, string] => {
    if (!isFemView(w)) {
      const i = (VIEW_PREFER as readonly string[]).indexOf(w);
      return [i === -1 ? 50 : i, w];
    }
    const stem = w.slice(4);
    const sub = stem.startsWith("vonmises")
      ? 0
      : stem.startsWith("deformed")
        ? 1
        : stem.startsWith("regions")
          ? 2
          : 3;
    return [100 + sub, stem];
  };
  return [...views].sort((a, b) => {
    const [ra, sa] = rank(a);
    const [rb, sb] = rank(b);
    return ra - rb || sa.localeCompare(sb);
  });
}

/** compact human label for the view switcher / gallery captions */
export function viewLabel(w: string): string {
  if (w.startsWith("fem:")) {
    const s = w
      .slice(4)
      .replace(/^(vonmises|deformed|regions)_(export_)?/, "$1 $2")
      .replaceAll("_", " ")
      .trim();
    return `fem ${s}`;
  }
  return w
    .replace("contact_sheet", "sheet")
    .replace("orient_sheet", "orient")
    .replaceAll("_", " ");
}

/** parse a fem view stem ("vonmises_export_chain pull") into a friendly
 * caption like "von mises (export) · chain pull" — falls back to the raw stem */
export function femViewCaption(view: string | null): string | null {
  if (!view) return null;
  const m = /^(vonmises|deformed|regions)(?:_(export))?_(.*)$/.exec(view);
  if (!m) return view.replaceAll("_", " ");
  const kind =
    m[1] === "vonmises" ? "von mises" : m[1] === "deformed" ? "deformed" : "regions";
  return `${kind}${m[2] ? " (export)" : ""} · ${m[3].replaceAll("_", " ")}`;
}
