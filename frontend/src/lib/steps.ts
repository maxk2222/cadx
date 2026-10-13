/** Pipeline step ids → display labels (shared by JobLog, JobsPage, VersionTree). */
export const STEP_LABEL: Record<string, string> = {
  intake: "intake",
  plan: "plan",
  meshgen: "mesh gen",
  imagegen: "image gen",
  orient: "orient",
  meshprocess: "mesh process",
  meshrefine: "mesh refine",
  meshedit: "mesh edit",
  codegen: "codegen",
  improve: "improve",
  autofix: "autofix",
  fem: "fem",
  repair: "repair",
  execute: "build",
  critique: "critique",
  dfm: "dfm",
  finalize: "export",
  budget: "budget",
  optimize: "topopt",
  params: "params",
  reexport: "re-export",
  done: "done",
};

export function stepLabel(s: string | null | undefined): string {
  return s ? (STEP_LABEL[s] ?? s) : "";
}
